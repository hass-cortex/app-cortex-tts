"""What has to hold when two requests overlap.

One uvicorn worker means one event loop, so the only real parallelism is the
worker threads `asyncio.to_thread` hands out — model builds, synthesis, and
storing a reference recording. Everything here is a regression: each test
failed before the lock or the ordering it pins was put in.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cortex_speech import BY_ID
from cortex_speech.catalog import model_dir
from cortex_speech.device import Memory
from cortex_speech.engine.base import Delivery, Synthesis, UnknownVoiceError
from cortex_speech.engine.registry import EngineRegistry, OutOfMemoryError
from cortex_speech.references import ReferenceStore

AUTH = {"Authorization": "Bearer test-key"}


class _Witness:
    """Something weakref-able to leave in a frame's locals."""


class _Blocking:
    """Renders silence, but only once told to."""

    sample_rate = 24000
    provider = "cpu"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.forgotten: list[str] = []
        self.calls = 0
        self.closed = False

    def synthesize(
        self, segments: list[str], voice: str, *, delivery: Delivery = Delivery()
    ) -> Synthesis:
        del voice, delivery
        self.calls += 1
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

    def close(self) -> None:
        self.closed = True


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
        execution_provider="cpu",
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
    """A dropped slot is where a card's memory comes back — by closing the
    engine, not by hoping nothing else holds it. OmniVoice bound its ONNX
    `forward` to its own instance, and two streams failing together each kept
    the other's engine in a traceback; both kept the card until a collection
    happened to run."""

    async def test_the_engine_is_closed_once_its_slot_is(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        assert await registry.unload("hojo-40m")
        assert engines[0].closed

    async def test_an_engine_held_elsewhere_is_closed_all_the_same(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        """Whoever still holds it holds a shell; the sessions are gone."""
        await registry.synthesize("hojo-40m", ["x"], "v")
        held = engines[0]  # a cycle, standing in for the bound `forward`
        assert await registry.unload("hojo-40m")
        assert held.closed
        assert registry.loaded_ids == set()

    async def test_the_unload_line_carries_the_figure_after_the_close(
        self,
        registry: EngineRegistry,
        engines: list[_Blocking],
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        registry._execution_provider = "cuda"  # noqa: SLF001
        await registry.acquire("hojo-40m")
        # Read before the close, then after it.
        readings = iter([Memory(3716, 4096), Memory(110, 4096)])
        monkeypatch.setattr(
            "cortex_speech.engine.registry.memory", lambda: next(readings)
        )
        with caplog.at_level("INFO"):
            await registry.unload("hojo-40m")
        assert "unloaded (asked), device 110/4096 MiB (-3606 MiB)" in caplog.text


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


class TestRunningTheDeviceOutOfMemory:
    """An ONNX Runtime arena only grows, and only the session dying gives it
    back — so a render that exhausts the card leaves it exhausted, and every
    request after it fails the same way until somebody restarts the app.
    Measured on a 4 GB GTX 1650 with MOSS-TTS-Nano: 3716 MiB standing after
    the failure, 110 MiB once the engine was closed."""

    _MESSAGE = (
        "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code "
        "returned while running Add node. Name:'/Add_60' Status Message: "
        "bfc_arena.cc:358 Failed to allocate memory for requested buffer "
        "of size 6537216"
    )

    @classmethod
    def _starve(cls, engine: _Blocking) -> None:
        def fail(*_args: object, **_kwargs: object) -> Synthesis:
            engine.calls += 1
            engine.entered.set()
            engine.release.wait(timeout=5)
            raise RuntimeError(cls._MESSAGE)

        engine.synthesize = fail  # type: ignore[method-assign]

    async def test_the_engine_is_closed_so_the_next_request_starts_clean(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        self._starve(engines[0])

        with pytest.raises(OutOfMemoryError):
            await registry.synthesize("hojo-40m", ["x"], "v")

        assert engines[0].closed, "the exhausted arena was kept"
        assert registry.loaded_ids == set()
        await registry.synthesize("hojo-40m", ["x"], "v")
        assert len(engines) == 2, "the next request did not get a fresh engine"

    async def test_a_stream_drops_it_too(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        """Where it actually bit: the failing traces were `/api/speak/stream`."""
        await registry.synthesize("hojo-40m", ["x"], "v")
        self._starve(engines[0])

        with pytest.raises(OutOfMemoryError):
            async for _ in registry.synthesize_stream("hojo-40m", ["x"], "v"):
                pass

        assert engines[0].closed
        assert registry.loaded_ids == set()

    async def test_the_engine_is_closed_before_the_error_reaches_the_caller(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        """Release is a line in the code, not a consequence of what the
        caller does with the exception afterwards. The first fix cleared the
        failing frames and hoped; two streams failing together still kept
        3716 MiB for as long as anyone cared to watch."""
        await registry.synthesize("hojo-40m", ["x"], "v")
        self._starve(engines[0])

        try:
            await registry.synthesize("hojo-40m", ["x"], "v")
        except OutOfMemoryError as err:
            assert err.__traceback__ is not None, "the trace must still print"
            assert engines[0].closed
        else:  # pragma: no cover - the engine was told to fail
            raise AssertionError("expected the render to fail")

    async def test_a_request_queued_behind_the_failure_gets_a_fresh_engine(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        """Seen in production as two streams failing 100 ms apart: the second
        was already waiting for the lock when the first ran the card out, took
        the lock before the drop could, and rendered on the same full arena."""
        await registry.synthesize("hojo-40m", ["x"], "v")
        first = engines[0]
        self._starve(first)
        first.release.clear()

        failing = asyncio.create_task(registry.synthesize("hojo-40m", ["x"], "v"))
        await asyncio.to_thread(first.entered.wait, 5)
        queued = asyncio.create_task(registry.synthesize("hojo-40m", ["y"], "v"))
        await asyncio.sleep(0)  # let it reach the lock
        first.release.set()

        with pytest.raises(OutOfMemoryError):
            await failing
        await queued

        # One warm-up render and the failing one; the queued request is neither.
        assert first.calls == 2, "the queued request ran on the exhausted engine"
        assert len(engines) == 2 and engines[1].calls == 1

    async def test_an_out_of_memory_unload_reports_the_figure(
        self,
        registry: EngineRegistry,
        engines: list[_Blocking],
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The engine is closed before the line is written, so the figure
        is the card after release rather than a stale reading."""
        registry._execution_provider = "cuda"  # noqa: SLF001
        await registry.synthesize("hojo-40m", ["x"], "v")
        self._starve(engines[0])
        readings = iter([Memory(3716, 4096), Memory(110, 4096)])
        monkeypatch.setattr(
            "cortex_speech.engine.registry.memory", lambda: next(readings)
        )

        with caplog.at_level("INFO"), pytest.raises(OutOfMemoryError):
            await registry.synthesize("hojo-40m", ["x"], "v")

        assert (
            "unloaded (out of memory), device 110/4096 MiB (-3606 MiB)" in caplog.text
        )

    async def test_an_ordinary_failure_keeps_the_engine(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        """Only exhaustion costs the engine; a bad voice must not."""

        def fail(*_args: object, **_kwargs: object) -> Synthesis:
            raise UnknownVoiceError("no such voice")

        await registry.acquire("hojo-40m")
        engines[0].synthesize = fail  # type: ignore[method-assign]
        with pytest.raises(UnknownVoiceError):
            await registry.synthesize("hojo-40m", ["x"], "v")
        assert registry.loaded_ids == {"hojo-40m"}
        assert not engines[0].closed

    async def test_no_room_to_load_is_reported_as_such(
        self, registry: EngineRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half: the card can be too full to build the sessions at
        all, which surfaced as a raw runtime error rather than something a
        caller could act on."""

        def starved_build(backend: object, context: object) -> _Blocking:
            del backend, context
            raise RuntimeError("CUDA failure 2: out of memory")

        monkeypatch.setattr("cortex_speech.engine.registry.build", starved_build)
        with pytest.raises(OutOfMemoryError, match="not enough memory"):
            await registry.synthesize("hojo-40m", ["x"], "v")
        assert registry.loaded_ids == set()


class TestAWaiterHandedAnEvictedEngine:
    """`acquire` hands out the slot before the lock is held. In between, an
    eviction ahead in the queue can retire that very slot — and a waiter that
    went on to use it would be rendering on an engine the registry no longer
    counts, alongside the one that replaced it. On a 4 GB card that is two
    models where there is room for one."""

    async def test_the_waiter_loads_afresh_instead(
        self, registry: EngineRegistry, engines: list[_Blocking]
    ) -> None:
        await registry.reconfigure(max_loaded=1)
        await registry.synthesize("hojo-40m", ["x"], "v")
        hojo = engines[0]
        hojo.release.clear()

        running = asyncio.create_task(registry.synthesize("hojo-40m", ["a"], "v"))
        await asyncio.to_thread(hojo.entered.wait, 5)
        # The eviction queues for hojo's lock first, then the waiter does.
        evicting = asyncio.create_task(registry.synthesize("moss-nano", ["b"], "v"))
        await asyncio.sleep(0)
        waiting = asyncio.create_task(registry.synthesize("hojo-40m", ["c"], "v"))
        await asyncio.sleep(0)
        hojo.release.set()

        await asyncio.gather(running, evicting, waiting)

        assert hojo.closed
        # The warm-up render and the one that was running; not the waiter's.
        assert hojo.calls == 2, "the waiter rendered on the evicted engine"
        assert len(engines) == 3, "the waiter did not load the model again"
        assert registry.loaded_ids == {"hojo-40m"}


class TestLoadingIsAState:
    """A bundle takes seconds to become a session. For the whole of that the
    registry used to answer "nothing resident", which reads exactly like idle
    to `/health`, to a card and to anyone reading the log."""

    async def test_a_model_being_built_is_visible(
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

        assert registry.loading_ids == {"hojo-40m"}
        assert registry.loaded_ids == set(), "loading and loaded must not overlap"

        finish.set()
        await load
        assert registry.loading_ids == set()
        assert registry.loaded_ids == {"hojo-40m"}

    async def test_a_build_that_fails_does_not_stay_loading(
        self, registry: EngineRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_build(backend: object, context: object) -> _Blocking:
            del backend, context
            raise RuntimeError("CUDA failure 2: out of memory")

        monkeypatch.setattr("cortex_speech.engine.registry.build", broken_build)
        with pytest.raises(OutOfMemoryError):
            await registry.acquire("hojo-40m")
        assert registry.loading_ids == set()


class TestTheLifecycleIsReadableFromTheLog:
    """Every diagnosis this file exists because of needed `nvidia-smi` in one
    window and the log in another, lined up by timestamp. These are the lines
    that make the log enough on its own."""

    async def test_every_transition_says_what_happened_and_why(
        self,
        registry: EngineRegistry,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "cortex_speech.engine.registry.memory",
            lambda: Memory(2916, 4096),
        )
        registry._execution_provider = "cuda"  # noqa: SLF001 - reading its own log

        with caplog.at_level("INFO"):
            await registry.reconfigure(max_loaded=1)
            await registry.synthesize("hojo-40m", ["x"], "v")
            await registry.synthesize("moss-nano", ["x"], "v")
            await registry.unload("moss-nano")

        lines = [r.getMessage() for r in caplog.records if "model " in r.getMessage()]
        assert "model hojo-40m loading (0/1 resident), device 2916/4096 MiB" in lines
        assert "model hojo-40m resident (1/1), device 2916/4096 MiB (+0 MiB)" in lines
        assert any("unloaded (evicted for moss-nano)" in line for line in lines)
        assert any("model moss-nano unloaded (asked)" in line for line in lines)

    async def test_an_unload_names_which_of_the_four_reasons(
        self,
        registry: EngineRegistry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        with caplog.at_level("INFO"):
            await registry.reconfigure(execution_provider="cuda")
        assert any(
            "unloaded (settings changed)" in r.getMessage() for r in caplog.records
        )

    async def test_a_cpu_host_does_not_shell_out_for_a_figure_it_has_no_use_for(
        self, registry: EngineRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading it costs a subprocess, and on a CPU host it describes
        nothing this app did."""
        calls = 0

        def counted() -> Memory | None:
            nonlocal calls
            calls += 1
            return None

        monkeypatch.setattr("cortex_speech.engine.registry.memory", counted)
        await registry.synthesize("hojo-40m", ["x"], "v")
        assert calls == 0

    async def test_a_failure_records_what_was_being_said(
        self,
        registry: EngineRegistry,
        engines: list[_Blocking],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The traceback says where it broke, never what was asked for."""
        await registry.acquire("hojo-40m")

        def fail(*_args: object, **_kwargs: object) -> Synthesis:
            raise RuntimeError("something in the runtime")

        engines[0].synthesize = fail  # type: ignore[method-assign]
        with caplog.at_level("WARNING"), pytest.raises(RuntimeError):
            await registry.synthesize("hojo-40m", ["一二三。", "四五。"], "some-voice")

        assert (
            "render failed on hojo-40m/some-voice (2 segment(s), 7 chars): "
            "something in the runtime"
        ) in caplog.text

    async def test_an_answer_to_the_caller_is_not_logged_as_an_incident(
        self,
        registry: EngineRegistry,
        engines: list[_Blocking],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await registry.acquire("hojo-40m")

        def fail(*_args: object, **_kwargs: object) -> Synthesis:
            raise UnknownVoiceError("no such voice")

        engines[0].synthesize = fail  # type: ignore[method-assign]
        with caplog.at_level("WARNING"), pytest.raises(UnknownVoiceError):
            await registry.synthesize("hojo-40m", ["x"], "v")
        assert "failed on" not in caplog.text
