"""Shared application state and the auth dependency."""

from __future__ import annotations

import asyncio
import hmac
import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import (
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketException,
    status,
)
from starlette.requests import HTTPConnection

from cortex_speech import (
    DownloadManager,
    EngineRegistry,
    ReferenceStore,
    SpeechService,
)

from ..config import Settings
from ..preferences import Preferences
from ..stats import StatsStore
from ..updates import Hub

_LOGGER = logging.getLogger(__name__)


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
    stats: StatsStore
    """What this host has measured, per model.

    Held beside the settings rather than inside the speech library: the
    library renders audio and the app is what decides a measurement is worth
    remembering."""
    preferences: Preferences
    """How the app behaves, as the user last set it.

    Mutable on purpose: `PUT /api/settings` replaces it and the next request
    reads the new value. The two that are bound when a session is created —
    `num_threads` and `execution_provider`, the pair `rebuild_needed` compares
    — are applied by dropping what is resident, not by restarting."""
    updates: Hub = field(default_factory=Hub)
    """Where a route says the UI's picture of something went stale."""
    settings_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """Serialises `PUT /api/settings`.

    Replacing `preferences` is a read-modify-write that spans an await, so
    without this two saves overlapping in that window each write over the
    other's fields and both answer 200."""

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


def get_state_ws(websocket: WebSocket) -> AppState:
    """The same state for a WebSocket route; FastAPI resolves the two by type."""
    return websocket.app.state.cortex


INGRESS_PEER = "172.30.32.2"
"""The Supervisor's address on the hassio network, the only source of ingress."""


def is_ingress(request: HTTPConnection) -> bool:
    """Whether the Supervisor's ingress proxy sent this request.

    The `X-Ingress-Path` header alone proves nothing — any client on a
    published port can add it — so the peer address must be the Supervisor's
    as well. The ingress proxy is the only thing that connects from there.
    """
    if request.headers.get("X-Ingress-Path") is None:
        return False
    return request.client is not None and request.client.host == INGRESS_PEER


WS_SUBPROTOCOL = "cortex-tts"
"""What a browser offers first when it carries the key as a subprotocol.

A `WebSocket` constructor cannot set a header, so a page served from a
published port has no way to send `X-API-Key` on the handshake. The
subprotocol list is the one field it can set, and it travels in the same
handshake the header would have — so `["cortex-tts", "<key>"]` says the same
thing to the same reader. API clients, which can set headers, still do.
"""


def _subprotocol_key(conn: HTTPConnection) -> str:
    """The key a browser put in the subprotocol list, if it did."""
    offered = conn.scope.get("subprotocols") or []
    if len(offered) < 2 or offered[0] != WS_SUBPROTOCOL:
        return ""
    return str(offered[1]).strip()


def _supplied_key(
    authorization: str | None,
    x_api_key: str | None,
    conn: HTTPConnection | None = None,
) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    if x_api_key:
        return x_api_key.strip()
    if conn is not None:
        return _subprotocol_key(conn)
    return ""


def _key_accepted(
    conn: HTTPConnection, authorization: str | None, x_api_key: str | None
) -> bool:
    """Whether this connection may proceed; logs a refusal.

    An empty configured key disables the check, which is only sane when the
    port is not published. Ingress connections skip it because the
    Supervisor has already authenticated the user; see `is_ingress`.
    """
    state: AppState = conn.app.state.cortex
    expected = state.settings.api_key
    if not expected or is_ingress(conn):
        return True

    supplied = _supplied_key(authorization, x_api_key, conn)
    # Constant-time compare so a wrong key cannot be narrowed by timing. Bytes,
    # because the str form raises on a non-ASCII token instead of rejecting it.
    if supplied and hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        return True
    # Say so. A refused request is invisible otherwise — there is no
    # access log — and the admin UI reacts to a 401 by asking for a key,
    # so "why is it asking?" has to be answerable from here.
    _LOGGER.warning(
        "refused %s %s: %s (peer=%s, ingress-path=%s)",
        conn.scope.get("method", "WEBSOCKET"),
        conn.url.path,
        "no key supplied" if not supplied else "key did not match",
        conn.client.host if conn.client else "unknown",
        conn.headers.get("X-Ingress-Path") is not None,
    )
    return False


async def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject requests without the configured bearer token."""
    if not _key_accepted(request, authorization, x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_REQUIRED", "message": "authentication required"},
        )


async def require_api_key_ws(
    websocket: WebSocket,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """The same check for a WebSocket, which has no status code to refuse with.

    The handshake is completed and the socket closed with policy-violation,
    because a 403 during the handshake reaches most clients as an opaque
    connection error and the close code at least names the reason.
    """
    if not _key_accepted(websocket, authorization, x_api_key):
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="authentication required"
        )
