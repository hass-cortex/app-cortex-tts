"""Shared application state and the auth dependency."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path

from fastapi import Header, HTTPException, Request, status

from cortex_speech import (
    DownloadManager,
    EngineRegistry,
    ReferenceStore,
    SpeechService,
)

from ..config import Settings
from ..preferences import Preferences


@dataclass
class AppState:
    """Everything the routes need, assembled once at startup.

    The speech library arrives as one object. The app holds a handle to it and
    reads what it needs off that handle, rather than keeping its own references
    to the library's internals — which is what lets the library rearrange them.
    """

    settings: Settings
    speech: SpeechService
    version: str
    preferences_path: Path
    """Where the stored settings live, resolved once at the boundary.

    The app owns `data_dir` and translates it in `app.py`; carrying the
    resolved path here keeps the routes from deriving a layout twice."""
    preferences: Preferences
    """How the app behaves, as the user last set it.

    Mutable on purpose: `PUT /api/settings` replaces it and the next request
    reads the new value. The three that are bound when a session is created
    are applied by dropping what is resident, not by restarting."""

    @property
    def registry(self) -> EngineRegistry:
        """The engine registry owned by the speech library."""
        return self.speech.registry

    @property
    def references(self) -> ReferenceStore:
        """The reference-recording store owned by the speech library."""
        return self.speech.references

    @property
    def downloads(self) -> DownloadManager:
        """The download manager owned by the speech library."""
        return self.speech.downloads


def get_state(request: Request) -> AppState:
    """Return the application state attached to the running app."""
    return request.app.state.cortex


async def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject requests without the configured bearer token.

    An empty configured key disables the check, which is the sane default
    behind Home Assistant ingress where the Supervisor has already
    authenticated the user. Ingress requests are recognised by the header the
    Supervisor adds and skip the check regardless.
    """
    state: AppState = request.app.state.cortex
    expected = state.settings.api_key
    if not expected:
        return
    if request.headers.get("X-Ingress-Path") is not None:
        return

    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    elif x_api_key:
        supplied = x_api_key.strip()

    # Constant-time compare so a wrong key cannot be narrowed by timing.
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_REQUIRED", "message": "authentication required"},
        )
