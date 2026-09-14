"""Application assembly: state, lifespan, routes, ingress UI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from cortex_speech import (
    BY_ID,
    SpeechConfig,
    SpeechService,
    subscribe_models_changed,
)

from . import __version__, config, discovery, events, preferences
from .api.deps import AppState
from .api.routes import api, compat, public
from .stats import FILE_NAME as STATS_FILE
from .stats import StatsStore

_LOGGER = logging.getLogger(__name__)


class _RevalidatingStatic(StaticFiles):
    """Serve the UI so a replaced file is actually picked up.

    A hot-deploy swaps index.html under a URL that never changes, so the
    response has to force revalidation. The ETag keeps that a 304.
    """

    def file_response(self, *args: object, **kwargs: object) -> Response:
        """Return the file with revalidation forced."""
        response = super().file_response(*args, **kwargs)  # type: ignore[arg-type]
        response.headers["cache-control"] = "no-cache"
        return response


async def _preload(state: AppState) -> None:
    """Make the default model usable, fetching it first if this is a fresh install.

    Runs in the background: the port must accept connections while a
    quarter-gigabyte bundle is still arriving.
    """
    spec = BY_ID.get(state.preferences.default_model)
    if spec is None:
        return

    if not state.speech.model(spec).downloaded:
        _LOGGER.info("default model %s is not present, downloading it now", spec.id)
        progress = state.downloads.start(spec)
        task = state.downloads.task_for(spec.id)
        if task is not None:
            await task
        if progress.state != "done":
            _LOGGER.warning(
                "could not download %s: %s — pick a model in the app's UI",
                spec.id,
                progress.error or progress.state,
            )
            return

    try:
        await state.registry.acquire(spec.id)
    except Exception as err:  # noqa: BLE001 - preload is best-effort
        _LOGGER.warning("preload of %s failed: %s", spec.id, err)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build application state, announce to Supervisor, warm the default model."""
    settings = config.load()
    # What the user last chose, which is everything except what has to be
    # settled before the process starts. See preferences.py.
    prefs = preferences.load(settings.data_dir)

    # The app's options are translated into the library's vocabulary exactly
    # here, and nowhere else.
    speech = SpeechService(
        SpeechConfig(
            data_dir=settings.data_dir,
            num_threads=prefs.num_threads,
            max_loaded_models=prefs.max_loaded_models,
            temperature=prefs.temperature,
            execution_provider=prefs.execution_provider,
        )
    )
    # The library reports that the voice set changed; turning that into an
    # event on the Home Assistant bus is the app's job, not the library's.
    unsubscribe = subscribe_models_changed(events.fire_models_changed)

    state = AppState(
        settings=settings,
        speech=speech,
        version=__version__,
        preferences_path=settings.data_dir,
        stats=StatsStore(settings.data_dir / STATS_FILE),
        preferences=prefs,
    )
    app.state.cortex = state

    _LOGGER.info(
        "cortex-tts %s listening on %s:%d "
        "(data=%s, threads=%d, max_loaded=%d, temp=%.2f, provider=%s)",
        state.version,
        settings.host,
        settings.port,
        settings.data_dir,
        prefs.num_threads,
        prefs.max_loaded_models,
        prefs.temperature,
        prefs.execution_provider,
    )

    # A model that left the catalog leaves its weights behind; nothing else
    # would ever mention them again.
    for model_id, size in speech.orphans().items():
        _LOGGER.warning(
            "%s is downloaded but no longer in the catalog, using %.0f MB — "
            "delete %s to reclaim it",
            model_id,
            size / 1_000_000,
            model_id,
        )

    await discovery.announce(port=settings.port, api_key=settings.api_key)
    preload: asyncio.Task[None] | None = None
    if prefs.preload:
        # Off the critical path: the port should accept connections while a
        # multi-hundred-megabyte bundle is still being mapped into memory.
        # Held, so the loop cannot collect it half-way, and cancelled at
        # shutdown so it cannot load into a registry being emptied.
        preload = asyncio.create_task(_preload(state))

    yield

    unsubscribe()
    if preload is not None and not preload.done():
        preload.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await preload
    for model_id in list(state.registry.loaded_ids):
        await state.registry.unload(model_id)


def create_app() -> FastAPI:
    """Build the ASGI application."""
    app = FastAPI(
        title="Cortex TTS",
        description="CPU text-to-speech for Home Assistant, with a Chinese "
        "text pipeline the model cannot do for itself.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # Every error the API sends is `{"code", "message"}`: what a route raised,
    # what the router could not match, and what pydantic refused. A client
    # parses one shape or it parses none of them.
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = detail
        else:
            body = {"code": "ERROR", "message": str(detail)}
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err.get('loc', ()) if part != 'body')}: "
            f"{err.get('msg', 'invalid')}"
            for err in exc.errors()
        )
        return JSONResponse(
            status_code=422, content={"code": "VALIDATION", "message": problems}
        )

    app.include_router(public)
    app.include_router(api)
    app.include_router(compat)

    # The lifespan builds its own; this one only needs where the UI lives.
    static_dir = config.load().static_dir
    if static_dir.is_dir():
        # Mounted last so it never shadows an API route. `html=True` serves
        # index.html for the ingress root.
        app.mount(
            "/", _RevalidatingStatic(directory=str(static_dir), html=True), name="ui"
        )
    else:
        _LOGGER.warning("no static UI at %s; API only", static_dir)

        @app.get("/")
        async def _root() -> JSONResponse:
            return JSONResponse({"service": "cortex-tts", "ui": "not bundled"})

    return app


app = create_app()

__all__ = ["app", "create_app"]
