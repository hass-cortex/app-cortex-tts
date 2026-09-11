"""Posting to the Supervisor, and what it means when there is none.

Outside the addon there is no ``SUPERVISOR_TOKEN`` and nothing to talk to,
which is normal in local development. Every call here is best-effort: a
failure is logged and reported, never raised, because none of them are worth
failing a synthesis over.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)

URL = "http://supervisor"
_TIMEOUT = aiohttp.ClientTimeout(total=10)


async def post(path: str, payload: dict[str, Any], *, what: str) -> bool:
    """POST to the Supervisor, returning whether it accepted.

    Args:
        path: Path under the Supervisor root, e.g. ``/discovery``.
        payload: JSON body.
        what: Short label for the log line.

    Returns:
        Whether the Supervisor accepted it. ``False`` when there is no
        Supervisor, when it refused, or when it could not be reached.
    """
    token = os.environ.get("SUPERVISOR_TOKEN", "").strip()
    if not token:
        _LOGGER.debug("no SUPERVISOR_TOKEN, skipping %s", what)
        return False

    try:
        async with (
            aiohttp.ClientSession(timeout=_TIMEOUT) as session,
            session.post(
                f"{URL}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            ) as response,
        ):
            if response.status >= 400:
                body = await response.text()
                _LOGGER.warning(
                    "%s rejected (%s): %s", what, response.status, body[:200]
                )
                return False
    except (aiohttp.ClientError, TimeoutError) as err:
        _LOGGER.warning("could not %s: %s", what, err)
        return False
    return True
