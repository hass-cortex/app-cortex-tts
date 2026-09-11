"""Application assembly: state, lifespan, routes, ingress UI."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, config, discovery
from .api.deps import AppState
from .api.routes import api, compat, public
from .catalog import BY_ID, inspect
from .download import DownloadManager
from .engine.registry import EngineRegistry
from .refs import ReferenceStore

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
    spec = BY_ID.get(state.settings.default_model)
    if spec is None:
        return

    if not inspect(state.settings.data_dir, spec).downloaded:
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
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.references_dir.mkdir(parents=True, exist_ok=True)

    references = ReferenceStore(settings.references_dir)
    state = AppState(
        settings=settings,
        registry=EngineRegistry(
            settings.data_dir,
            references,
            num_threads=settings.num_threads,
            max_loaded=settings.max_loaded_models,
            temperature=settings.temperature,
        ),
        references=references,
        downloads=DownloadManager(settings.data_dir),
        version=__version__,
    )
    app.state.hojo = state

    _LOGGER.info(
        "hojo-tts %s listening on %s:%d (data=%s, threads=%d, max_loaded=%d, temp=%.2f)",
        state.version,
        settings.host,
        settings.port,
        settings.data_dir,
        settings.num_threads,
        settings.max_loaded_models,
        settings.temperature,
    )

    await discovery.announce(port=settings.port, api_key=settings.api_key)
    if settings.preload:
        # Off the critical path: the port should accept connections while a
        # multi-hundred-megabyte bundle is still being mapped into memory.
        asyncio.create_task(_preload(state))

    yield

    for model_id in list(state.registry.loaded_ids):
        await state.registry.unload(model_id)


def create_app() -> FastAPI:
    """Build the ASGI application."""
    app = FastAPI(
        title="Hojo TTS",
        description="CPU text-to-speech for Home Assistant, with a Chinese "
        "text pipeline the model cannot do for itself.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        """Render errors in one shape whether raised with a dict or a string."""
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = detail
        else:
            body = {"code": "ERROR", "message": str(detail)}
        return JSONResponse(status_code=exc.status_code, content=body)

    app.include_router(public)
    app.include_router(api)
    app.include_router(compat)

    settings = config.load()
    static_dir = settings.static_dir
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
            return JSONResponse({"service": "hojo-tts", "ui": "not bundled"})

    return app


app = create_app()

__all__ = ["app", "create_app"]
