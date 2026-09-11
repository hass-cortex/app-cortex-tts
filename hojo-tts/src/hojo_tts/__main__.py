"""Entry point: ``python -m hojo_tts``."""

from __future__ import annotations

import logging
import os

import uvicorn

from . import config


def _configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "info").upper()
    # bashio's `fatal` has no logging equivalent; treat it as CRITICAL.
    level = {"FATAL": "CRITICAL", "WARNING": "WARNING", "TRACE": "DEBUG"}.get(
        level, level
    )
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # These log one line per HTTP request, which during a model download buries
    # everything worth reading under a few hundred redirect traces. Their
    # warnings still get through, and LOG_LEVEL=debug restores the detail.
    if logging.getLogger().level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "huggingface_hub", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    """Run the HTTP server."""
    _configure_logging()
    settings = config.load()
    uvicorn.run(
        "hojo_tts.app:app",
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
        # One worker: the engines hold hundreds of megabytes each and a second
        # worker would duplicate them without adding throughput, since the
        # decode loop is CPU-bound and already serialised per engine.
        workers=1,
    )


if __name__ == "__main__":
    main()
