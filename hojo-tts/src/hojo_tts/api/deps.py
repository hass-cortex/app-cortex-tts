"""Shared application state and the auth dependency."""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from ..config import Settings
from ..download import DownloadManager
from ..engine.registry import EngineRegistry
from ..refs import ReferenceStore


@dataclass
class AppState:
    """Everything the routes need, assembled once at startup."""

    settings: Settings
    registry: EngineRegistry
    references: ReferenceStore
    downloads: DownloadManager
    version: str


def get_state(request: Request) -> AppState:
    """Return the application state attached to the running app."""
    return request.app.state.hojo


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
    state: AppState = request.app.state.hojo
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
