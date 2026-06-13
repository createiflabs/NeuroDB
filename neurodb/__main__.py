"""``python -m neurodb`` entrypoint: serve the API with uvicorn."""

from __future__ import annotations

import os

from .config import get_settings
from .observability import configure_logging


def main() -> None:
    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    # The store keeps all state in this process — multiple workers would each
    # get their own copy and corrupt the shared data file. Enforce one worker.
    workers = os.environ.get("WEB_CONCURRENCY")
    if workers and workers != "1":
        raise SystemExit(
            "NeuroDB must run with a single worker (in-process store); "
            f"WEB_CONCURRENCY={workers} is not supported."
        )
    uvicorn.run(
        "neurodb.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
