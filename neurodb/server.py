"""The NeuroDB HTTP API (FastAPI/ASGI).

Run with ``python -m neurodb`` or ``uvicorn neurodb.server:app``.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import Settings, get_settings
from .embedding import embed_text
from .models import (
    CreateCollectionRequest,
    EmbedRequest,
    SearchRequest,
    TextSearchRequest,
    TextUpsertRequest,
    UpsertRequest,
)
from .store import CollectionError, NotFoundError, StoreError, VectorStore

logger = logging.getLogger("neurodb")

STATIC_DIR = Path(__file__).resolve().parent / "static"

DESCRIPTION = (
    "NeuroDB is a lightweight, container-native **vector database** for AI memory "
    "and semantic search. Store embeddings with JSON metadata and run fast "
    "nearest-neighbour queries over a clean REST API."
)


async def _autosave_loop(store: VectorStore, interval: float) -> None:
    """Periodically persist collections that have unsaved changes."""

    if interval <= 0:
        return
    while True:
        try:
            await asyncio.sleep(interval)
            flushed = await run_in_threadpool(store.flush)
            if flushed:
                logger.debug("autosave: flushed %d collection(s)", flushed)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive, keep the loop alive
            logger.exception("autosave loop error")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = VectorStore(settings.data_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("NeuroDB %s starting (data_dir=%s)", __version__, settings.data_dir)
        task = asyncio.create_task(_autosave_loop(store, settings.autosave_interval))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await run_in_threadpool(store.save_all)
            logger.info("NeuroDB stopped; all collections persisted")

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- error handling ---------------------------------------------------
    @app.exception_handler(NotFoundError)
    async def _not_found(_request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(StoreError)
    async def _store_error(_request, exc: StoreError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # -- auth -------------------------------------------------------------
    async def require_api_key(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
        authorization: str | None = Header(None),
    ) -> None:
        if not settings.api_key:
            return
        token = x_api_key
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:]
        if token != settings.api_key:
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
        stats = store.stats()
        return {
            "status": "ok",
            "version": __version__,
            "collections": stats["collections"],
            "vectors": stats["vectors"],
        }

    @app.get("/version", tags=["system"])
    async def version():
        return {
            "name": "NeuroDB",
            "version": __version__,
            "embedding_dim": settings.embedding_dim,
            "auth_required": bool(settings.api_key),
        }

    # -- data API (auth-protected) ---------------------------------------
    api = APIRouter(dependencies=[Depends(require_api_key)])

    @api.get("/stats", tags=["system"])
    async def stats():
        return store.stats()

    @api.post("/collections", tags=["collections"], status_code=201)
    async def create_collection(body: CreateCollectionRequest):
        col = store.create_collection(body.name, body.dimension, body.metric)
        return col.info()

    @api.get("/collections", tags=["collections"])
    async def list_collections():
        return {"collections": store.list_collections()}

    @api.get("/collections/{name}", tags=["collections"])
    async def get_collection(name: str):
        return store.get_collection(name).info()

    @api.delete("/collections/{name}", tags=["collections"])
    async def delete_collection(name: str):
        store.delete_collection(name)
        return {"deleted": name}

    @api.post("/collections/{name}/persist", tags=["collections"])
    async def persist_collection(name: str):
        col = store.get_collection(name)
        await run_in_threadpool(col.save, store._dir(name))
        return {"persisted": name, "count": col.count}

    @api.post("/collections/{name}/vectors", tags=["vectors"])
    async def upsert_vectors(name: str, body: UpsertRequest):
        col = store.get_collection(name)
        items = [item.model_dump() for item in body.items]
        affected = await run_in_threadpool(col.upsert, items)
        return {"upserted": len(affected), "ids": affected}

    @api.get("/collections/{name}/vectors/{vector_id}", tags=["vectors"])
    async def get_vector(name: str, vector_id: str):
        return store.get_collection(name).get(vector_id)

    @api.delete("/collections/{name}/vectors/{vector_id}", tags=["vectors"])
    async def delete_vector(name: str, vector_id: str):
        col = store.get_collection(name)
        removed = await run_in_threadpool(col.delete, [vector_id])
        if removed == 0:
            raise NotFoundError(f"Vector {vector_id!r} not found in collection {name!r}.")
        return {"deleted": vector_id}

    @api.post("/collections/{name}/search", tags=["search"])
    async def search(name: str, body: SearchRequest):
        col = store.get_collection(name)
        results = await run_in_threadpool(
            col.search, body.vector, body.k, body.filter, body.include_vectors
        )
        return {"results": results, "count": len(results)}

    # -- text convenience endpoints (use the built-in embedder) -----------
    def _ensure_text_dim(col) -> None:
        if col.dimension != settings.embedding_dim:
            raise CollectionError(
                f"Collection {col.name!r} has dimension {col.dimension}, but the "
                f"built-in text embedder produces {settings.embedding_dim}-d vectors. "
                "Create the collection with that dimension to use /texts endpoints."
            )

    @api.post("/collections/{name}/texts", tags=["text"])
    async def upsert_texts(name: str, body: TextUpsertRequest):
        col = store.get_collection(name)
        _ensure_text_dim(col)
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
        affected = await run_in_threadpool(col.upsert, items)
        return {"upserted": len(affected), "ids": affected}

    @api.post("/collections/{name}/search/text", tags=["text"])
    async def search_text(name: str, body: TextSearchRequest):
        col = store.get_collection(name)
        _ensure_text_dim(col)
        vector = embed_text(body.text, settings.embedding_dim).tolist()
        results = await run_in_threadpool(
            col.search, vector, body.k, body.filter, body.include_vectors
        )
        return {"results": results, "count": len(results)}

    @api.post("/embed", tags=["text"])
    async def embed(body: EmbedRequest):
        vector = embed_text(body.text, settings.embedding_dim)
        return {"vector": vector.tolist(), "dimension": settings.embedding_dim}

    app.include_router(api)
    return app


# Module-level ASGI app for `uvicorn neurodb.server:app` and the container CMD.
app = create_app()
