"""Fetching model bundles from Hugging Face, with progress the UI can watch.

Downloads run in a worker thread; the API returns immediately and callers poll
:meth:`DownloadManager.status`. At most ``MAX_CONCURRENT`` fetch at once and
the rest wait as ``queued``, in the order they were asked for: bundles are
hundreds of megabytes each, and two of them sharing one link finish later
than the same two one after the other. A failed download leaves the partial
bundle in place — ``catalog.inspect`` reports it as not downloaded, and
retrying resumes whichever files are still missing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import ModelSpec, model_dir
from .notifications import notify_models_changed

_LOGGER = logging.getLogger(__name__)

# How many bundles fetch at once; every further request waits its turn.
MAX_CONCURRENT = 2

# How often a running download's partial size is read for the progress bar.
PROGRESS_POLL_S = 0.5


@dataclass
class DownloadProgress:
    """State of one model download.

    Attributes:
        model_id: Which model is being fetched.
        state: ``queued``, ``running``, ``done`` or ``failed``.
        completed_files: Bundle members already on disk.
        total_files: Bundle members in total.
        completed_bytes: Size of the members already on disk.
        current_bytes: How much of the member being fetched has arrived.
        total_bytes: Size of the whole bundle, once the hub has said; 0 until.
        current: Member currently being fetched.
        error: Failure message when ``state`` is ``failed``.
        started: Unix timestamp when the download began.
    """

    model_id: str
    state: str = "queued"
    completed_files: int = 0
    total_files: int = 0
    completed_bytes: int = 0
    current_bytes: int = 0
    total_bytes: int = 0
    current: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)

    @property
    def percent(self) -> float:
        """Completion by bytes, or by file count until the sizes are known.

        A bundle is a handful of files and one of them is most of it, so a
        count moves in jumps of a third; bytes move while the file arrives.
        """
        if self.total_bytes:
            done = min(self.completed_bytes + self.current_bytes, self.total_bytes)
            return round(done / self.total_bytes * 100, 1)
        if not self.total_files:
            return 0.0
        return round(self.completed_files / self.total_files * 100, 1)


class DownloadManager:
    """Runs at most one download per model, ``MAX_CONCURRENT`` at a time."""

    def __init__(self, data_dir: Path) -> None:
        """Create a manager writing bundles under ``data_dir/models``."""
        self._data_dir = data_dir
        self._progress: dict[str, DownloadProgress] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = threading.Lock()
        self._slots: asyncio.Semaphore | None = None

    def _slot(self) -> asyncio.Semaphore:
        # Made on first use so the manager can be built before the loop runs.
        if self._slots is None:
            self._slots = asyncio.Semaphore(MAX_CONCURRENT)
        return self._slots

    def all(self) -> list[DownloadProgress]:
        """Every download this process has been asked for, in request order."""
        with self._lock:
            return list(self._progress.values())

    def status(self, model_id: str) -> DownloadProgress | None:
        """Return progress for a model, or ``None`` if it was never started."""
        with self._lock:
            return self._progress.get(model_id)

    def task_for(self, model_id: str) -> asyncio.Task[None] | None:
        """Return the in-flight download task for a model, if there is one."""
        return self._tasks.get(model_id)

    def is_running(self, model_id: str) -> bool:
        """Whether a download for this model is in flight, queued included."""
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

        progress = DownloadProgress(model_id=spec.id, total_files=len(spec.files))
        with self._lock:
            self._progress[spec.id] = progress

        self._tasks[spec.id] = asyncio.create_task(self._run(spec, progress))
        return progress

    async def _run(self, spec: ModelSpec, progress: DownloadProgress) -> None:
        try:
            async with self._slot():
                with self._lock:
                    progress.state = "running"
                    progress.started = time.time()
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
            await notify_models_changed(f"downloaded:{spec.id}")

    def _fetch(self, spec: ModelSpec, progress: DownloadProgress) -> None:
        """Download every bundle member, skipping ones already present."""
        target = model_dir(self._data_dir, spec.id)
        target.mkdir(parents=True, exist_ok=True)

        sizes = self._sizes(spec)
        with self._lock:
            progress.total_bytes = sum(sizes.values())

        for source in spec.sources:
            # A bundle can span repositories: MOSS publishes its weights and
            # its audio codec separately and needs both.
            into = target / source.subdir if source.subdir else target
            into.mkdir(parents=True, exist_ok=True)

            for name in source.files:
                destination = into / name
                if destination.is_file():
                    with self._lock:
                        progress.completed_files += 1
                        progress.completed_bytes += destination.stat().st_size
                    continue

                with self._lock:
                    progress.current = name
                _LOGGER.info("downloading %s/%s", source.repo_id, name)
                self._fetch_one(source.repo_id, name, into, progress)
                with self._lock:
                    progress.completed_files += 1
                    progress.completed_bytes += destination.stat().st_size
                    progress.current_bytes = 0

        # huggingface_hub leaves its lock/metadata directory behind, once per
        # directory it was pointed at.
        import shutil

        for cache_marker in target.rglob(".cache"):
            if cache_marker.is_dir():
                shutil.rmtree(cache_marker, ignore_errors=True)

    @staticmethod
    def _sizes(spec: ModelSpec) -> dict[tuple[str, str], int]:
        """What each bundle member weighs, from the hub; missing where it fails.

        One HEAD per file. A bundle the hub cannot describe still downloads,
        with progress by file count instead.
        """
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        sizes: dict[tuple[str, str], int] = {}
        for source in spec.sources:
            for name in source.files:
                try:
                    meta = get_hf_file_metadata(hf_hub_url(source.repo_id, name))
                except Exception as err:  # noqa: BLE001 - progress only, never fatal
                    _LOGGER.debug("no size for %s/%s: %s", source.repo_id, name, err)
                    return {}
                if not meta.size:
                    return {}
                sizes[source.repo_id, name] = meta.size
        return sizes

    def _fetch_one(
        self, repo_id: str, name: str, into: Path, progress: DownloadProgress
    ) -> None:
        """Fetch one member, reading its partial size while it arrives.

        `hf_hub_download` reports nothing while it runs, so it runs on its
        own thread and this one watches the `.incomplete` blob the hub writes
        under `into/.cache`, which grows as the bytes land.
        """
        from huggingface_hub import constants, hf_hub_download

        # Over xet the hub collects chunks in its own cache and writes the
        # file out at the end, so nothing grows while the bytes arrive; plain
        # HTTP streams into the `.incomplete` blob and the bar can follow it.
        constants.HF_HUB_DISABLE_XET = True

        failure: list[BaseException] = []

        def fetch() -> None:
            try:
                # Fetch into the bundle directory directly. `local_dir`
                # copies the blob out of the hub cache so /data holds exactly
                # one copy — the cache would otherwise double every model's
                # disk cost.
                hf_hub_download(repo_id=repo_id, filename=name, local_dir=str(into))
            except BaseException as err:  # noqa: BLE001 - re-raised below
                failure.append(err)

        worker = threading.Thread(target=fetch, name=f"download:{name}", daemon=True)
        worker.start()
        partial = into / ".cache" / "huggingface" / "download"
        while worker.is_alive():
            worker.join(PROGRESS_POLL_S)
            arrived = (
                sum(
                    f.stat().st_size
                    for f in partial.rglob("*.incomplete")
                    if f.is_file()
                )
                if partial.is_dir()
                else 0
            )
            with self._lock:
                progress.current_bytes = arrived
        if failure:
            raise failure[0]


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
