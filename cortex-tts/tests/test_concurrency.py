"""What happens to an engine when its slot goes away.

Dropping the last name for one is where a card's memory is supposed to come
back, and the registry is the only thing in a position to check that it did.
"""

from __future__ import annotations

import threading
import weakref
from pathlib import Path

import numpy as np
import pytest

from cortex_speech import BY_ID
from cortex_speech.catalog import model_dir
from cortex_speech.engine.base import Delivery, Synthesis
from cortex_speech.engine.registry import EngineRegistry
from cortex_speech.references import ReferenceStore


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
