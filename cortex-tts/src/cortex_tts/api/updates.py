"""`/api/events`: one socket that says which of the UI's reads went stale.

Each frame is ``{"type": kind}`` with kind one of ``models``, ``voices``,
``references``, ``settings``; the client re-reads that endpoint. Nothing else
travels here — the data stays on the routes that own it. A ``ping`` goes out
while idle so a proxy in between does not close a quiet socket.
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from .deps import WS_SUBPROTOCOL, AppState, get_state_ws, require_api_key_ws

events = APIRouter(prefix="/api", dependencies=[Depends(require_api_key_ws)])

# Quiet this long and a keep-alive frame goes out.
PING_S = 25.0


@events.websocket("/events")
async def watch_events(
    websocket: WebSocket, state: AppState = Depends(get_state_ws)
) -> None:
    """Stream change notices until the client goes away."""
    offered = websocket.scope.get("subprotocols") or []
    await websocket.accept(
        subprotocol=WS_SUBPROTOCOL if WS_SUBPROTOCOL in offered else None
    )
    queue = state.updates.subscribe()
    # The only thing a client sends is a close; reading is how it is heard.
    reader = asyncio.create_task(websocket.receive())
    try:
        while True:
            waiter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {reader, waiter}, timeout=PING_S, return_when=asyncio.FIRST_COMPLETED
            )
            if reader in done:
                waiter.cancel()
                break
            if waiter in done:
                await websocket.send_json({"type": waiter.result()})
            else:
                waiter.cancel()
                await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        state.updates.unsubscribe(queue)
        reader.cancel()
        with contextlib.suppress(BaseException):
            await reader
