"""Telling the admin UI that something it shows has changed.

The UI reads everything over REST; this only says *which* of those reads is
stale, so a page keeps one socket open instead of polling. Four kinds:
``models`` (a card, a download, a load, a measurement), ``voices``,
``references`` and ``settings``. A download in flight is the one thing that
changes without anyone calling a route, so the hub watches the download
manager and ticks ``models`` while its progress moves.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from cortex_speech import DownloadManager

_LOGGER = logging.getLogger(__name__)

# How often a running download is looked at for the progress bar.
DOWNLOAD_TICK_S = 0.5


class Hub:
    """Fan-out of change notices to every open events socket."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._watcher: asyncio.Task[None] | None = None

    def subscribe(self) -> asyncio.Queue[str]:
        """A queue that receives every kind published from now on."""
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._subscribers.discard(queue)

    def publish(self, *kinds: str) -> None:
        """Tell every listener that these reads are stale."""
        for queue in self._subscribers:
            for kind in kinds:
                queue.put_nowait(kind)

    def watch(self, downloads: DownloadManager) -> None:
        """Tick ``models`` while any download's progress is moving."""
        if self._watcher is None:
            self._watcher = asyncio.create_task(self._watch(downloads))

    async def close(self) -> None:
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
            self._watcher = None

    async def _watch(self, downloads: DownloadManager) -> None:
        seen: list[tuple[str, str, float]] = []
        while True:
            await asyncio.sleep(DOWNLOAD_TICK_S)
            if not self._subscribers:
                continue
            now = [(p.model_id, p.state, p.percent) for p in downloads.all()]
            if now != seen:
                seen = now
                self.publish("models")
