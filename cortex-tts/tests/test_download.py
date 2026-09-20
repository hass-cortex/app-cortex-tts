"""Downloads run one at a time; the rest wait their turn as `queued`."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from cortex_speech import BY_ID
from cortex_speech.download import MAX_CONCURRENT, DownloadManager


class _Gate:
    """A fetch that blocks until released, so ordering can be observed."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started: list[str] = []

    def fetch(self, spec, progress) -> None:
        self.started.append(spec.id)
        self.release.wait(timeout=5)


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0.01)


async def test_one_more_than_the_slots_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _Gate()
    manager = DownloadManager(tmp_path)
    monkeypatch.setattr(manager, "_fetch", gate.fetch)
    ids = ["hojo-40m", "moss-nano", "omnivoice"][: MAX_CONCURRENT + 1]
    started = [manager.start(BY_ID[m]) for m in ids]
    await _settle()
    assert gate.started == ids[:MAX_CONCURRENT]
    assert [p.state for p in started] == ["running"] * MAX_CONCURRENT + ["queued"]
    assert manager.is_running(ids[-1]), "queued counts as in flight"

    gate.release.set()
    await asyncio.gather(*(manager.task_for(m) for m in ids))
    assert gate.started == ids
    assert all(p.state == "done" for p in started)


async def test_asking_again_while_queued_rejoins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _Gate()
    manager = DownloadManager(tmp_path)
    monkeypatch.setattr(manager, "_fetch", gate.fetch)
    ids = ["hojo-40m", "moss-nano", "omnivoice"][: MAX_CONCURRENT + 1]
    last = [manager.start(BY_ID[m]) for m in ids][-1]
    await _settle()
    assert last.state == "queued"
    assert manager.start(BY_ID[ids[-1]]) is last
    gate.release.set()
    await asyncio.gather(*(manager.task_for(m) for m in ids))


async def test_a_failure_frees_the_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = DownloadManager(tmp_path)
    calls: list[str] = []

    def fetch(spec, progress) -> None:
        calls.append(spec.id)
        if spec.id == "hojo-40m":
            raise RuntimeError("no network")

    monkeypatch.setattr(manager, "_fetch", fetch)
    failed = manager.start(BY_ID["hojo-40m"])
    done = manager.start(BY_ID["moss-nano"])
    await asyncio.gather(*(manager.task_for(m) for m in ("hojo-40m", "moss-nano")))
    assert calls == ["hojo-40m", "moss-nano"]
    assert (failed.state, failed.error) == ("failed", "no network")
    assert done.state == "done"


class TestProgressIsByBytes:
    def test_bytes_move_while_a_file_arrives(self) -> None:
        from cortex_speech.download import DownloadProgress

        progress = DownloadProgress("m", total_files=3, total_bytes=1000)
        progress.completed_files, progress.completed_bytes = 1, 100
        assert progress.percent == 10.0
        progress.current_bytes = 450
        assert progress.percent == 55.0

    def test_file_count_until_the_sizes_are_known(self) -> None:
        from cortex_speech.download import DownloadProgress

        progress = DownloadProgress("m", total_files=4, completed_files=1)
        assert progress.percent == 25.0

    def test_never_over_a_hundred(self) -> None:
        from cortex_speech.download import DownloadProgress

        progress = DownloadProgress("m", total_bytes=10, completed_bytes=10)
        progress.current_bytes = 5
        assert progress.percent == 100.0
