"""``python -m neurodb`` entrypoint: serve the API with uvicorn."""

from __future__ import annotations

import logging

from .config import get_settings


def main() -> None:
    import uvicorn

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "neurodb.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
