"""The NeuroDB HTTP API (FastAPI/ASGI).

Run with ``python -m neurodb`` or ``uvicorn neurodb.server:app``.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import Settings, get_settings
from .embedding import embed_text
from .models import (
    AnomalyRequest,
    CompleteRequest,
    CreateMemoryRequest,
    EmbedRequest,
    SearchRequest,
    TextSearchRequest,
    TextWriteRequest,
    WriteRequest,
)
from .observability import CONTENT_TYPE_LATEST, Metrics, request_id_var
from .store import MemoryError_, NeuroStore, NotFoundError, StoreError

logger = logging.getLogger("neurodb")

STATIC_DIR = Path(__file__).resolve().parent / "static"

DESCRIPTION = (
    "NeuroDB is a content-addressable store powered by **Modern Hopfield "
    "networks**. Writing a pattern is appending a vector; retrieval is a single "
    "attention step. It offers pattern completion, per-field anomaly detection "
    "and similarity search, with single-file persistence."
)


_RATE_LIMIT_EXEMPT = frozenset({"/", "/health", "/version", "/metrics", "/ready"})
_LEGACY_DATA_PREFIXES = ("/memories", "/stats", "/flush", "/embed")


def _is_legacy_data_path(path: str) -> bool:
    """True for an unversioned data path (i.e. not already under /v1)."""

    return any(path == p or path.startswith(p + "/") for p in _LEGACY_DATA_PREFIXES)


def _error_response(request, status: int, code: str, message, **extra) -> JSONResponse:
    """Consistent error envelope carrying a request id.

    Keeps a top-level ``detail`` mirror for backward compatibility with
    existing clients (and the dashboard) that read ``detail``.
    """

    request_id = getattr(getattr(request, "state", None), "request_id", None)
    body = {
        "error": {"code": code, "message": message, "request_id": request_id, **extra},
        "detail": message,
    }
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(status_code=status, content=body, headers=headers)


class _FixedWindowLimiter:
    """A small in-process fixed-window rate limiter (per client key).

    Adequate for the single-node deployment target; swap for a shared backend
    if NeuroDB is ever run as multiple instances behind a load balancer.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, int], int] = {}

    def allow(self, key: str, now: float) -> bool:
        if self.limit <= 0:
            return True
        window = int(now // 60)
        with self._lock:
            count = self._counts.get((key, window), 0) + 1
            self._counts[(key, window)] = count
            if len(self._counts) > 10_000:  # opportunistic cleanup
                self._counts = {
                    k: v for k, v in self._counts.items() if k[1] == window
                }
            return count <= self.limit


async def _autosave_loop(store: NeuroStore, interval: float, metrics: Metrics) -> None:
    if interval <= 0:
        return
    while True:
        try:
            await asyncio.sleep(interval)
            if await run_in_threadpool(store.flush):
                metrics.record_save(True)
                logger.debug("autosave: persisted store")
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - keep the loop alive
            metrics.record_save(False)
            logger.exception("autosave loop error")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = NeuroStore(settings.data_file, fail_on_corrupt_load=settings.fail_on_corrupt_load)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail-closed: never serve an internet-facing instance without auth
        # unless the operator explicitly opted into anonymous access.
        if not settings.api_key and not settings.allow_anonymous:
            raise RuntimeError(
                "Refusing to start without authentication. Set NEURODB_API_KEY, "
                "or set NEURODB_ALLOW_ANONYMOUS=1 to allow anonymous access."
            )
        if not settings.api_key:
            logger.warning(
                "NeuroDB is running WITHOUT authentication "
                "(NEURODB_ALLOW_ANONYMOUS); do not expose it to untrusted networks."
            )
        logger.info("NeuroDB %s starting (data_file=%s)", __version__, settings.data_file)
        task = asyncio.create_task(_autosave_loop(store, settings.autosave_interval, metrics))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await run_in_threadpool(store.save_all)
            logger.info("NeuroDB stopped; store persisted")

    app = FastAPI(
        title="NeuroDB",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        contact={"name": "createif labs", "url": "https://github.com/createiflabs/NeuroDB"},
        license_info={"name": "MIT"},
    )
    app.state.store = store
    app.state.settings = settings

    # CORS is closed by default; only enabled for explicitly configured origins.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    limiter = _FixedWindowLimiter(settings.rate_limit_per_minute)

    @app.middleware("http")
    async def _security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'",
        )
        return response

    @app.middleware("http")
    async def _rate_limit(request, call_next):
        if (
            settings.rate_limit_per_minute > 0
            and request.url.path not in _RATE_LIMIT_EXEMPT
        ):
            client = request.client.host if request.client else "anon"
            key = request.headers.get("X-API-Key") or client
            if not limiter.allow(key, time.monotonic()):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded."},
                    headers={"Retry-After": "60"},
                )
        return await call_next(request)

    @app.middleware("http")
    async def _limit_body_size(request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > settings.max_request_bytes:
                    return JSONResponse(
                        status_code=413, content={"detail": "Request body too large."}
                    )
            except ValueError:
                pass
        return await call_next(request)

    @app.middleware("http")
    async def _request_context(request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = rid
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        label = getattr(route, "path", "unmatched")
        metrics.observe_request(request.method, label, response.status_code, duration)
        response.headers["X-Request-ID"] = rid
        # Nudge clients off the unversioned (legacy) data routes.
        if _is_legacy_data_path(request.url.path):
            response.headers["Deprecation"] = "true"
            response.headers["Link"] = '</v1>; rel="successor-version"'
        return response

    @app.exception_handler(NotFoundError)
    async def _not_found(request, exc: NotFoundError):
        return _error_response(request, 404, "not_found", str(exc))

    @app.exception_handler(StoreError)
    async def _store_error(request, exc: StoreError):
        return _error_response(request, 400, "bad_request", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc: RequestValidationError):
        return _error_response(
            request,
            422,
            "validation_error",
            "Request validation failed.",
            errors=jsonable_encoder(exc.errors()),
        )

    @app.exception_handler(HTTPException)
    async def _http_error(request, exc: HTTPException):
        return _error_response(request, exc.status_code, "http_error", exc.detail)

    @app.exception_handler(Exception)
    async def _unhandled(request, exc: Exception):
        logger.exception(
            "unhandled error [request_id=%s]", getattr(request.state, "request_id", None)
        )
        return _error_response(request, 500, "internal_error", "Internal server error.")

    async def require_api_key(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
        authorization: str | None = Header(None),
    ) -> None:
        if not settings.api_key:
            return
        token = x_api_key
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:]
        # Constant-time comparison to avoid leaking the key via response timing.
        if not hmac.compare_digest(token or "", settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    # -- public endpoints -------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"name": "NeuroDB", "version": __version__})

    @app.get("/health", tags=["system"])
    async def health():
        # Liveness + lightweight counts (used by the dashboard).
        stats = store.stats()
        return {
            "status": "ok",
            "version": __version__,
            "memories": stats["memories"],
            "patterns": stats["patterns"],
        }

    @app.get("/ready", tags=["system"])
    async def ready():
        # Readiness for load balancers: 503 until the last persist succeeded.
        if store.last_save_ok:
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not ready"})

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        stats = store.stats()
        body = metrics.render(stats["memories"], stats["patterns"])
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)

    @app.get("/version", tags=["system"])
    async def version():
        return {
            "name": "NeuroDB",
            "version": __version__,
            "engine": "modern-hopfield",
            "embedding_dim": settings.embedding_dim,
            "auth_required": bool(settings.api_key),
        }

    # -- data API (auth-protected) ---------------------------------------
    api = APIRouter(dependencies=[Depends(require_api_key)])

    @api.get("/stats", tags=["system"])
    async def stats():
        return store.stats()

    @api.post("/flush", tags=["system"])
    async def flush():
        """Synchronously persist all dirty memories (fsync-durable) and report
        how many were written."""

        persisted = await run_in_threadpool(store.flush)
        return {"persisted": persisted, "durable": True}

    @api.post("/memories", tags=["memories"], status_code=201)
    async def create_memory(body: CreateMemoryRequest):
        mem = store.create_memory(
            body.name, body.dimension, body.beta, body.fields, body.normalize
        )
        return mem.info()

    @api.get("/memories", tags=["memories"])
    async def list_memories(
        limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)
    ):
        return {
            "memories": store.list_memories(limit, offset),
            "total": store.count_memories(),
            "limit": limit,
            "offset": offset,
        }

    @api.get("/memories/{name}", tags=["memories"])
    async def get_memory(name: str):
        return store.get_memory(name).info()

    @api.delete("/memories/{name}", tags=["memories"])
    async def delete_memory(name: str):
        store.delete_memory(name)
        return {"deleted": name}

    # -- patterns (writing is appending a vector) ------------------------
    @api.post("/memories/{name}/patterns", tags=["patterns"])
    async def write_patterns(name: str, body: WriteRequest):
        mem = store.get_memory(name)
        items = [item.model_dump() for item in body.items]
        affected = await run_in_threadpool(mem.write, items)
        return {"written": len(affected), "ids": affected}

    @api.get("/memories/{name}/patterns/{pattern_id}", tags=["patterns"])
    async def get_pattern(name: str, pattern_id: str):
        return store.get_memory(name).get(pattern_id)

    @api.delete("/memories/{name}/patterns/{pattern_id}", tags=["patterns"])
    async def delete_pattern(name: str, pattern_id: str):
        mem = store.get_memory(name)
        removed = await run_in_threadpool(mem.delete, [pattern_id])
        if removed == 0:
            raise NotFoundError(f"Pattern {pattern_id!r} not found in memory {name!r}.")
        return {"deleted": pattern_id}

    # -- content-addressable operations ----------------------------------
    @api.post("/memories/{name}/complete", tags=["recall"])
    async def complete(name: str, body: CompleteRequest):
        mem = store.get_memory(name)
        return await run_in_threadpool(
            mem.complete, body.query, body.beta, body.mask, body.steps, body.top_k
        )

    @api.post("/memories/{name}/search", tags=["recall"])
    async def search(name: str, body: SearchRequest):
        mem = store.get_memory(name)
        results = await run_in_threadpool(
            mem.search, body.query, body.k, body.filter, body.include_vectors, body.metric
        )
        return {"results": results, "count": len(results)}

    @api.post("/memories/{name}/anomaly", tags=["recall"])
    async def anomaly(name: str, body: AnomalyRequest):
        mem = store.get_memory(name)
        return await run_in_threadpool(mem.anomaly, body.query, body.beta, body.top_k)

    # -- text convenience endpoints (built-in embedder) ------------------
    def _ensure_text_dim(mem) -> None:
        if mem.dimension != settings.embedding_dim:
            raise MemoryError_(
                f"Memory {mem.name!r} has dimension {mem.dimension}, but the built-in "
                f"text embedder produces {settings.embedding_dim}-d vectors. Create the "
                "memory with that dimension to use /texts endpoints."
            )

    @api.post("/memories/{name}/texts", tags=["text"])
    async def write_texts(name: str, body: TextWriteRequest):
        mem = store.get_memory(name)
        _ensure_text_dim(mem)
        items = []
        for item in body.items:
            meta = dict(item.metadata)
            meta.setdefault("text", item.text)
            items.append(
                {
                    "id": item.id,
                    "vector": embed_text(item.text, settings.embedding_dim).tolist(),
                    "metadata": meta,
                }
            )
        affected = await run_in_threadpool(mem.write, items)
        return {"written": len(affected), "ids": affected}

    @api.post("/memories/{name}/search/text", tags=["text"])
    async def search_text(name: str, body: TextSearchRequest):
        mem = store.get_memory(name)
        _ensure_text_dim(mem)
        vector = embed_text(body.text, settings.embedding_dim).tolist()
        results = await run_in_threadpool(
            mem.search, vector, body.k, body.filter, body.include_vectors
        )
        return {"results": results, "count": len(results)}

    @api.post("/memories/{name}/recall/text", tags=["text"])
    async def recall_text(name: str, body: TextSearchRequest):
        """Hopfield recall over text memories: returns the attention distribution
        across stored patterns for the embedded query."""

        mem = store.get_memory(name)
        _ensure_text_dim(mem)
        vector = embed_text(body.text, settings.embedding_dim).tolist()
        return await run_in_threadpool(mem.complete, vector, None, None, 1, body.k)

    @api.post("/embed", tags=["text"])
    async def embed(body: EmbedRequest):
        vector = embed_text(body.text, settings.embedding_dim)
        return {"vector": vector.tolist(), "dimension": settings.embedding_dim}

    # Versioned API plus an unversioned legacy alias (deprecation headers added
    # by the request-context middleware). The legacy mount is hidden from the
    # OpenAPI schema so /v1 is the single documented surface.
    app.include_router(api, prefix="/v1")
    app.include_router(api, include_in_schema=False)
    return app


# Module-level ASGI app for `uvicorn neurodb.server:app` and the container CMD.
app = create_app()
