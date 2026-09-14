"""The registry's lifecycle contracts, driven with a fake backend.

What is pinned: the resident bound and its LRU eviction, and that a settings
change drops engines only for the two values a session cannot adopt.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from cortex_speech import BY_ID
from cortex_speech.catalog import model_dir
from cortex_speech.engine.base import Delivery, Synthesis
from cortex_speech.engine.registry import EngineRegistry
from cortex_speech.references import ReferenceStore


class _Fake:
    """Records what it was asked, renders silence."""

    sample_rate = 24000
    provider = "cpu"

    def __init__(self) -> None:
        self.temperatures: list[float | None] = []
        self.deliveries: list[Delivery] = []

    def synthesize(
        self, segments: list[str], voice: str, *, delivery: Delivery = Delivery()
    ) -> Synthesis:
        self.temperatures.append(delivery.temperature)
        self.deliveries.append(delivery)
        return Synthesis(
            audio=np.zeros(240, dtype=np.float32),
            sample_rate=self.sample_rate,
            segments=len(segments),
            inference_ms=1.0,
        )

    def forget(self, reference_id: str) -> None:
        del reference_id

    def close(self) -> None:
        pass


def _pretend_downloaded(data_dir: Path, model_id: str) -> None:
    spec = BY_ID[model_id]
    root = model_dir(data_dir, spec.id)
    for name in spec.files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineRegistry:
    monkeypatch.setattr(
        "cortex_speech.engine.registry.build", lambda backend, context: _Fake()
    )
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


class TestResidentBound:
    async def test_loading_past_the_bound_evicts_the_least_recently_used(
        self, registry: EngineRegistry
    ) -> None:
        await registry.reconfigure(max_loaded=1)
        await registry.synthesize("hojo-40m", ["x"], "v")
        await registry.synthesize("moss-nano", ["x"], "v")
        assert registry.loaded_ids == {"moss-nano"}


class TestReconfigure:
    async def test_a_temperature_change_keeps_residents_and_reaches_them(
        self, registry: EngineRegistry
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        engine = (await registry.acquire("hojo-40m")).engine
        assert isinstance(engine, _Fake)

        dropped = await registry.reconfigure(temperature=0.2)

        assert not dropped
        assert registry.is_loaded("hojo-40m")
        await registry.synthesize("hojo-40m", ["x"], "v")
        assert engine.temperatures[-1] == 0.2

    async def test_a_per_call_temperature_still_wins(
        self, registry: EngineRegistry
    ) -> None:
        await registry.synthesize(
            "hojo-40m", ["x"], "v", delivery=Delivery(temperature=0.0)
        )
        engine = (await registry.acquire("hojo-40m")).engine
        assert isinstance(engine, _Fake)
        assert engine.temperatures == [0.0]

    async def test_the_same_thread_count_drops_nothing(
        self, registry: EngineRegistry
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        dropped = await registry.reconfigure(num_threads=2, execution_provider="cpu")
        assert not dropped
        assert registry.is_loaded("hojo-40m")

    @pytest.mark.parametrize(
        "change", [{"num_threads": 4}, {"execution_provider": "cuda"}]
    )
    async def test_a_session_bound_change_drops_every_resident(
        self, registry: EngineRegistry, change: dict[str, object]
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        await registry.synthesize("moss-nano", ["x"], "v")
        assert await registry.reconfigure(**change)  # type: ignore[arg-type]
        assert registry.loaded_ids == set()

    async def test_a_smaller_bound_evicts_down_to_it(
        self, registry: EngineRegistry
    ) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        await registry.synthesize("moss-nano", ["x"], "v")
        assert await registry.reconfigure(max_loaded=1)
        assert registry.loaded_ids == {"moss-nano"}


class TestIdleUnload:
    """A model nobody has asked for in a while is dropped, so a card shared
    with another workload is not held for a reply that is not coming."""

    async def test_an_idle_model_is_dropped_after_the_bound(
        self, registry: EngineRegistry, caplog: pytest.LogCaptureFixture
    ) -> None:
        await registry.reconfigure(idle_seconds=0.2)
        with caplog.at_level("INFO"):
            await registry.synthesize("hojo-40m", ["x"], "v")
            assert registry.loaded_ids == {"hojo-40m"}
            await asyncio.sleep(0.6)
        assert registry.loaded_ids == set()
        assert "unloaded (idle for 0s)" in caplog.text

    async def test_a_request_resets_the_clock(self, registry: EngineRegistry) -> None:
        await registry.reconfigure(idle_seconds=0.4)
        await registry.synthesize("hojo-40m", ["x"], "v")
        await asyncio.sleep(0.25)
        await registry.synthesize("hojo-40m", ["x"], "v")
        await asyncio.sleep(0.25)
        assert registry.loaded_ids == {"hojo-40m"}, "dropped mid-conversation"
        await asyncio.sleep(0.5)
        assert registry.loaded_ids == set()

    async def test_zero_keeps_the_model(self, registry: EngineRegistry) -> None:
        await registry.synthesize("hojo-40m", ["x"], "v")
        await asyncio.sleep(0.3)
        assert registry.loaded_ids == {"hojo-40m"}

    async def test_turning_it_off_stops_the_sweep(
        self, registry: EngineRegistry
    ) -> None:
        await registry.reconfigure(idle_seconds=0.2)
        await registry.synthesize("hojo-40m", ["x"], "v")
        await registry.reconfigure(idle_seconds=0)
        await asyncio.sleep(0.5)
        assert registry.loaded_ids == {"hojo-40m"}

    async def test_close_unloads_everything_and_stops(
        self, registry: EngineRegistry, caplog: pytest.LogCaptureFixture
    ) -> None:
        await registry.reconfigure(idle_seconds=60)
        await registry.synthesize("hojo-40m", ["x"], "v")
        with caplog.at_level("INFO"):
            await registry.close()
        assert registry.loaded_ids == set()
        assert "unloaded (shutting down)" in caplog.text
        assert registry._reaper is None  # noqa: SLF001
