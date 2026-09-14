"""What has to hold when two requests overlap.

One uvicorn worker means one event loop, so the only real parallelism is the
worker threads `asyncio.to_thread` hands out — model builds, synthesis, and
storing a reference recording. Everything here is a regression: each test
failed before the lock or the ordering it pins was put in.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cortex_speech import BY_ID
from cortex_speech.catalog import model_dir
from cortex_speech.engine.base import Delivery, Synthesis
from cortex_speech.engine.registry import EngineRegistry
from cortex_speech.references import ReferenceStore

AUTH = {"Authorization": "Bearer test-key"}


class _Blocking:
    """Renders silence, but only once told to."""

    sample_rate = 24000
    provider = "cpu"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.forgotten: list[str] = []

    def synthesize(
        self, segments: list[str], voice: str, *, delivery: Delivery = Delivery()
    ) -> Synthesis:
        del voice, delivery
        self.entered.set()
        self.release.wait(timeout=5)
        return Synthesis(
            audio=np.zeros(240, dtype=np.float32),
            sample_rate=self.sample_rate,
            segments=len(segments),
            inference_ms=1.0,
        )

    def forget(self, reference_id: str) -> None:
        self.forgotten.append(reference_id)


def _pretend_downloaded(data_dir: Path, model_id: str) -> None:
    spec = BY_ID[model_id]
    root = model_dir(data_dir, spec.id)
    for name in spec.files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


@pytest.fixture
def engines() -> list[_Blocking]:
    return []


@pytest.fixture
def registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engines: list[_Blocking]
) -> EngineRegistry:
    def build(backend: object, context: object) -> _Blocking:
        del backend, context
        engine = _Blocking()
        engine.release.set()  # synthesis is free unless a test says otherwise
        engines.append(engine)
        return engine

    monkeypatch.setattr("cortex_speech.engine.registry.build", build)
    for model_id in ("hojo-40m", "moss-nano"):
        _pretend_downloaded(tmp_path, model_id)
    return EngineRegistry(
        tmp_path,
        ReferenceStore(tmp_path / "references"),
        max_loaded=2,
        num_threads=2,
        temperature=0.8,
    )


class TestASettingsChangeCannotBeOutrunByABuild:
    """`reconfigure` used to sweep an empty `_slots` while a build was inside
    `to_thread`, so the engine landed afterwards carrying the thread count and
    provider that had just been replaced — and the caller was told nothing
    needed dropping."""

    async def test_an_engine_built_during_the_change_is_dropped_too(
        self, registry: EngineRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        building = threading.Event()
        finish = threading.Event()

        def slow_build(backend: object, context: object) -> _Blocking:
            del backend, context
            building.set()
            finish.wait(timeout=5)
            engine = _Blocking()
            engine.release.set()
            return engine

        monkeypatch.setattr("cortex_speech.engine.registry.build", slow_build)

        load = asyncio.create_task(registry.acquire("hojo-40m"))
        await asyncio.to_thread(building.wait, 5)

        change = asyncio.create_task(registry.reconfigure(execution_provider="cuda"))
        await asyncio.sleep(0)
        finish.set()

        dropped = await change
        await load

        assert dropped, "the engine built during the change was not dropped"
        assert registry.loaded_ids == set()

    async def test_the_provider_it_reports_is_the_one_that_was_asked_for(
        self, registry: EngineRegistry
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        await registry.reconfigure(execution_provider="cuda")
        assert registry.providers_in_use == {}


class TestForgettingAReferenceWaitsForTheEngine:
    """The conditioning cache is read on a worker thread. Forgetting between a
    miss and the insert that follows it puts the stale encoding straight back,
    so the engine's lock has to be held."""

    async def test_it_does_not_reach_an_engine_mid_synthesis(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        engine = engines[0]
        engine.release.clear()
        engine.entered.clear()

        speaking = asyncio.create_task(registry.synthesize("hojo-40m", ["x"], "v"))
        await asyncio.to_thread(engine.entered.wait, 5)

        forgetting = asyncio.create_task(registry.forget_reference("wanwan"))
        await asyncio.sleep(0.05)
        assert engine.forgotten == [], "forget reached an engine that was rendering"

        engine.release.set()
        await speaking
        await forgetting
        assert engine.forgotten == ["wanwan"]


class TestUnloadingReleasesTheEngine:
    """A dropped slot is where a card's memory comes back. OmniVoice bound its
    ONNX `forward` to its own instance, so the model outlived every name for it
    and kept 618 MiB of a 4096 MiB card until something else triggered a
    collection."""

    async def test_the_engine_is_gone_once_its_slot_is(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        alive = weakref.ref(engines[0])
        engines.clear()

        assert await registry.unload("hojo-40m")
        assert alive() is None

    async def test_an_engine_that_outlives_its_slot_is_collected_and_reported(
        self,
        registry: EngineRegistry,
        engines: list[_Blocking],
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cycle is the case a weakref cannot settle by itself, and the one
        OmniVoice was in. Pinned on the registry's response rather than on the
        object vanishing: whether the last reference has actually gone by then
        depends on the thread pool, not on anything this code decides."""
        await registry.synthesize("hojo-40m", ["x"], "v")
        held = engines[0]  # a cycle, standing in for the bound `forward`
        collections = 0

        def counted() -> int:
            nonlocal collections
            collections += 1
            return 0

        monkeypatch.setattr("cortex_speech.engine.registry.gc.collect", counted)

        with caplog.at_level("WARNING"):
            assert await registry.unload("hojo-40m")

        assert collections == 1, "a surviving engine was not collected"
        assert "outlived its slot" in caplog.text
        assert held is engines[0]


class TestConcurrentReferenceUploads:
    """`add` runs on a worker thread while `update` and `remove` stay on the
    loop, so the store is reached from more than one thread. Six same-named
    uploads used to return 200 apiece and leave three references."""

    def test_same_named_uploads_all_survive(
        self, tmp_path: Path, reference_wav: bytes
    ) -> None:
        store = ReferenceStore(tmp_path / "references")
        with ThreadPoolExecutor(max_workers=6) as pool:
            stored = list(
                pool.map(
                    lambda _: store.add(
                        name="dup probe",
                        transcript="這是一段測試錄音的文字內容。",
                        audio=reference_wav,
                    ),
                    range(6),
                )
            )

        ids = {reference.id for reference in stored}
        assert len(ids) == 6, "two uploads agreed on an id and one overwrote the other"
        assert {reference.id for reference in store.list()} == ids
        for reference in stored:
            assert reference.audio_path.is_file()

    def test_adding_and_removing_at_once_keeps_the_index_readable(
        self, tmp_path: Path, reference_wav: bytes
    ) -> None:
        """Both call `_save`; a shared temporary let one truncate the other's
        bytes before either renamed."""
        store = ReferenceStore(tmp_path / "references")
        doomed = [
            store.add(name=f"gone {i}", transcript="測試錄音。", audio=reference_wav)
            for i in range(6)
        ]

        with ThreadPoolExecutor(max_workers=12) as pool:
            adds = [
                pool.submit(
                    store.add,
                    name=f"kept {i}",
                    transcript="測試錄音。",
                    audio=reference_wav,
                )
                for i in range(6)
            ]
            removals = [pool.submit(store.remove, ref.id) for ref in doomed]
            for task in (*adds, *removals):
                task.result()

        reopened = ReferenceStore(tmp_path / "references")
        assert {r.id for r in reopened.list()} == {r.id for r in store.list()}
        assert len(reopened.list()) == 6


class TestOverlappingSettingsWrites:
    """`PUT /api/settings` is a read-modify-write over `state.preferences` that
    spans an await, so a second save arriving inside the wait read what the
    first had replaced and wrote it back, both answering 200."""

    def test_the_new_settings_are_published_before_the_registry_waits(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = client.app.state.cortex  # type: ignore[attr-defined]
        seen: dict[str, object] = {}
        original = EngineRegistry.reconfigure

        async def spy(self: EngineRegistry, **kwargs: object) -> bool:
            seen["threads"] = state.preferences.num_threads
            seen["locked"] = state.settings_lock.locked()
            return await original(self, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(EngineRegistry, "reconfigure", spy)

        assert (
            client.put("/api/settings", json={"num_threads": 6}, headers=AUTH).json()[
                "settings"
            ]["num_threads"]
            == 6
        )
        assert seen["threads"] == 6, "a concurrent reader would still see the old value"
        assert seen["locked"], "a second writer could enter while this one waits"
