"""Telling whoever is hosting the library that the set of voices changed.

Downloading a model or adding a reference recording changes which voices exist.
The library knows *that* it happened; it must not know that the host answers by
putting an event on a Home Assistant bus. So it publishes to listeners and the
host registers one.

This is the whole reason the boundary exists: the previous version of this
module imported the Supervisor client directly, which put a Home Assistant
concept inside the speech library.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress

_LOGGER = logging.getLogger(__name__)

# The return value is ignored: a listener reports back to its own world, not
# to the library. Typing it `object` lets a host return whatever it wants.
Listener = Callable[[str], Awaitable[object]]

_listeners: list[Listener] = []


def subscribe(listener: Listener) -> Callable[[], None]:
    """Register a coroutine to be called whenever the voice set changes.

    Args:
        listener: Called with a short reason tag. It must not raise; a failing
            listener is logged and the others still run.

    Returns:
        A callable that removes the listener again.
    """
    _listeners.append(listener)

    def unsubscribe() -> None:
        with suppress(ValueError):
            _listeners.remove(listener)

    return unsubscribe


async def notify_models_changed(reason: str) -> None:
    """Tell every listener that the set of usable voices changed.

    Args:
        reason: Short tag describing what changed, passed through to listeners
            for debugging. Listeners are expected to reconcile full state, so
            the tag is advisory.
    """
    for listener in list(_listeners):
        try:
            await listener(reason)
        except Exception:  # noqa: BLE001 - one bad listener must not stop the rest
            _LOGGER.exception("models-changed listener failed (%s)", reason)
