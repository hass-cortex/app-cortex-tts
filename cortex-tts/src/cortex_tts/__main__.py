"""Entry point: ``python -m cortex_tts``."""

from __future__ import annotations

import logging
import os

import uvicorn

from . import config, preferences


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


def _cap_numeric_threads(threads: int) -> None:
    """Hold BLAS and OpenMP to the same budget ONNX Runtime is given.

    Both default to one thread per core and would oversubscribe the host
    during a synthesis ORT is already threading — measured on MOSS-TTS-Nano,
    four ORT threads against two made it 70% slower, and these libraries can
    do the same damage without appearing in any setting.

    It has to happen here. They read the environment when they are imported,
    so by the time anything has touched numpy the value is already fixed —
    which is also why this cannot live in the settings the UI writes without
    a restart, and why `num_threads` is the one stored setting whose full
    effect waits for one.
    """
    if threads <= 0:
        return
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))


def main() -> None:
    """Run the HTTP server."""
    _configure_logging()
    settings = config.load()
    _cap_numeric_threads(preferences.load(settings.data_dir).num_threads)
    uvicorn.run(
        "cortex_tts.app:app",
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
