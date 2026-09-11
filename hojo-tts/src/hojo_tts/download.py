"""Fetching model bundles from Hugging Face, with progress the UI can watch.

Downloads run in a worker thread; the API returns immediately and callers poll
:meth:`DownloadManager.status`. A failed download leaves the partial bundle in
place — ``catalog.inspect`` reports it as not downloaded, and retrying resumes
whichever files are still missing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import ModelSpec, model_dir
from .events import fire_models_changed

_LOGGER = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """State of one model download.

    Attributes:
        model_id: Which model is being fetched.
        state: ``running``, ``done`` or ``failed``.
        completed_files: Bundle members already on disk.
        total_files: Bundle members in total.
        current: Member currently being fetched.
        error: Failure message when ``state`` is ``failed``.
        started: Unix timestamp when the download began.
    """

    model_id: str
    state: str = "running"
    completed_files: int = 0
    total_files: int = 0
    current: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)

    @property
    def percent(self) -> float:
        """Rough completion, by file count rather than bytes."""
        if not self.total_files:
            return 0.0
        return round(self.completed_files / self.total_files * 100, 1)


class DownloadManager:
    """Runs at most one download per model and reports progress."""

    def __init__(self, data_dir: Path) -> None:
        """Create a manager writing bundles under ``data_dir/models``."""
        self._data_dir = data_dir
        self._progress: dict[str, DownloadProgress] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = threading.Lock()

    def status(self, model_id: str) -> DownloadProgress | None:
        """Return progress for a model, or ``None`` if it was never started."""
        with self._lock:
            return self._progress.get(model_id)

    def task_for(self, model_id: str) -> asyncio.Task[None] | None:
        """Return the in-flight download task for a model, if there is one."""
        return self._tasks.get(model_id)

    def is_running(self, model_id: str) -> bool:
        """Whether a download for this model is in flight."""
        task = self._tasks.get(model_id)
        return task is not None and not task.done()

    def start(self, spec: ModelSpec) -> DownloadProgress:
        """Begin (or rejoin) a download for ``spec``.

        Returns the progress record; calling this while a download is already
        running is a no-op that returns the in-flight record.
        """
        if self.is_running(spec.id):
            existing = self.status(spec.id)
            if existing is not None:
                return existing

        progress = DownloadProgress(
            model_id=spec.id, total_files=len(spec.files), state="running"
        )
        with self._lock:
            self._progress[spec.id] = progress

        self._tasks[spec.id] = asyncio.create_task(self._run(spec, progress))
        return progress

    async def _run(self, spec: ModelSpec, progress: DownloadProgress) -> None:
        try:
            await asyncio.to_thread(self._fetch, spec, progress)
        except Exception as err:  # noqa: BLE001 - surfaced verbatim to the UI
            _LOGGER.exception("download of %s failed", spec.id)
            with self._lock:
                progress.state = "failed"
                progress.error = str(err)
        else:
            with self._lock:
                progress.state = "done"
                progress.current = ""
            # A newly-downloaded model means new voices; tell Home Assistant so
            # its entities appear without a reload.
            await fire_models_changed(f"downloaded:{spec.id}")

    def _fetch(self, spec: ModelSpec, progress: DownloadProgress) -> None:
        """Download every bundle member, skipping ones already present."""
        from huggingface_hub import hf_hub_download

        target = model_dir(self._data_dir, spec.id)
        target.mkdir(parents=True, exist_ok=True)

        for name in spec.files:
            destination = target / name
            if destination.is_file():
                with self._lock:
                    progress.completed_files += 1
                continue

            with self._lock:
                progress.current = name
            _LOGGER.info("downloading %s/%s", spec.repo_id, name)

            # Fetch into the bundle directory directly. `local_dir` copies the
            # blob out of the hub cache so /data holds exactly one copy — the
            # cache would otherwise double the disk cost of every model.
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=name,
                local_dir=str(target),
            )
            with self._lock:
                progress.completed_files += 1

        # huggingface_hub leaves its lock/metadata directory behind.
        cache_marker = target / ".cache"
        if cache_marker.is_dir():
            import shutil

            shutil.rmtree(cache_marker, ignore_errors=True)


def remove_bundle(data_dir: Path, spec: ModelSpec) -> int:
    """Delete a downloaded bundle. Returns how many files were removed."""
    target = model_dir(data_dir, spec.id)
    if not target.is_dir():
        return 0
    import shutil

    count = sum(1 for _ in target.rglob("*") if _.is_file())
    shutil.rmtree(target, ignore_errors=True)
    _LOGGER.info("removed bundle %s (%d files)", spec.id, count)
    return count
