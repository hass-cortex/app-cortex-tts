"""Telling Home Assistant when the set of usable voices changed.

Downloading a model or uploading a reference recording changes which voices
exist. Without a nudge the integration would only notice on a config-entry
reload, so a voice just uploaded would not appear until something restarted.
One event on the HA bus closes that gap.
"""

from __future__ import annotations

import logging

from . import supervisor

_LOGGER = logging.getLogger(__name__)

EVENT_MODELS_CHANGED = "cortex_tts_models_changed"


async def fire_models_changed(reason: str) -> bool:
    """Fire the models-changed event on the Home Assistant bus.

    Args:
        reason: Short tag describing what changed, carried in the payload for
            debugging. The integration reconciles the full state regardless,
            so the payload is advisory.

    Returns:
        Whether the event was accepted. A failure is not fatal: the
        integration still reconciles on its next reload.
    """
    # Supervisor proxies only /core/api/* through to Home Assistant.
    fired = await supervisor.post(
        f"/core/api/events/{EVENT_MODELS_CHANGED}",
        {"reason": reason},
        what=f"fire {EVENT_MODELS_CHANGED}",
    )
    if fired:
        _LOGGER.info("fired %s (%s)", EVENT_MODELS_CHANGED, reason)
    return fired
