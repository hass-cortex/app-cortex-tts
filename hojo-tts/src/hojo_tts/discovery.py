"""Announcing this app to Home Assistant through the Supervisor.

Publishing a discovery record is what makes the integration appear on the
Devices page by itself, carrying the host, port and API key so the user never
has to copy a token by hand. Outside the Supervisor (local development) there
is nothing to announce to, and the absence of a token says so.
"""

from __future__ import annotations

import logging
import os

from . import supervisor

_LOGGER = logging.getLogger(__name__)

SERVICE = "hojo_tts"


async def announce(*, port: int, api_key: str) -> bool:
    """Publish (or refresh) this app's Supervisor discovery record.

    Args:
        port: Port the API listens on, as seen from Home Assistant.
        api_key: Bearer token the integration should use.

    Returns:
        Whether a record was published. ``False`` means there was no
        Supervisor to talk to, which is normal outside the addon.
    """
    # Supervisor resolves the addon's own hostname for Home Assistant.
    hostname = os.environ.get("HOSTNAME", "").strip() or "local-hojo-tts"
    published = await supervisor.post(
        "/discovery",
        {
            "service": SERVICE,
            "config": {"host": hostname, "port": port, "api_key": api_key},
        },
        what="announce to Supervisor",
    )
    if published:
        _LOGGER.info(
            "announced %s to Home Assistant (host=%s port=%d)", SERVICE, hostname, port
        )
    return published
