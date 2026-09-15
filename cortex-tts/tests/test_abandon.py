"""A render whose listener has gone must stop, not finish for nobody.

The engine protocol carries a `stop` check, asked between the units a model
produces. What is pinned here is the contract around it: the registry passes
it through, asks it itself between streamed chunks, releases the lock, and
lets `AbandonedError` out unchanged — it is the caller's own news coming back.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from cortex_speech import BY_ID
from cortex_speech.catalog import model_dir
from cortex_speech.engine.base import (
    AbandonedError,
    Delivery,
    StopCheck,
    Synthesis,
    check_stop,
)
from cortex_speech.engine.registry import EngineRegistry
from cortex_speech.references import ReferenceStore


class _Whole:
    """Renders in three units and asks between them, as the real engines do."""

    sample_rate = 24000
    provider = "cpu"

    def __init__(self) -> None:
        self.units = 0

    def synthesize(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Synthesis:
        del voice, delivery
        for _ in range(3):
            check_stop(stop)
            self.units += 1
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


class _Chunked(_Whole):
    """Streams five chunks and never looks at `stop`: the registry must."""

    def synthesize_stream(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Generator[np.ndarray, None, None]:
        del segments, voice, delivery, stop
        for _ in range(5):
            self.units += 1
            yield np.zeros(240, dtype=np.float32)


def _pretend_downloaded(data_dir: Path, model_id: str) -> None:
    spec = BY_ID[model_id]
    root = model_dir(data_dir, spec.id)
    for name in spec.files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine
) -> EngineRegistry:
    monkeypatch.setattr(
        "cortex_speech.engine.registry.build", lambda backend, context: engine
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


class TestCheckStop:
    def test_no_check_costs_nothing(self) -> None:
        check_stop(None)

    def test_a_listening_caller_lets_the_render_continue(self) -> None:
        check_stop(lambda: False)

    def test_a_gone_caller_raises(self) -> None:
        with pytest.raises(AbandonedError):
            check_stop(lambda: True)


class TestWholeRender:
    async def test_stop_reaches_the_engine_and_ends_the_render_early(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _Whole()
        registry = _registry(tmp_path, monkeypatch, engine)
        gone = threading.Event()

        def stop() -> bool:
            # Gone after the first unit: the second must never be produced.
            if engine.units >= 1:
                gone.set()
            return gone.is_set()

        with pytest.raises(AbandonedError):
            await registry.synthesize("hojo-40m", ["x"], "v", stop=stop)
        assert engine.units == 1

    async def test_the_lock_is_released_afterwards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An abandoned render must not hold the engine for the next caller."""
        engine = _Whole()
        registry = _registry(tmp_path, monkeypatch, engine)
        with pytest.raises(AbandonedError):
            await registry.synthesize("hojo-40m", ["x"], "v", stop=lambda: True)
        result = await registry.synthesize("hojo-40m", ["x"], "v")
        assert result.segments == 1

    async def test_the_engine_stays_resident(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Abandonment is not a device failure; nothing should be unloaded."""
        registry = _registry(tmp_path, monkeypatch, _Whole())
        with pytest.raises(AbandonedError):
            await registry.synthesize("hojo-40m", ["x"], "v", stop=lambda: True)
        assert registry.is_loaded("hojo-40m")


class TestStreamedRender:
    async def test_the_registry_asks_between_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chunk-streaming engine need not look at `stop` itself."""
        engine = _Chunked()
        registry = _registry(tmp_path, monkeypatch, engine)
        received = 0

        def stop() -> bool:
            return received >= 2

        with pytest.raises(AbandonedError):
            async for _ in registry.synthesize_stream(
                "moss-nano", ["x"], "v", stop=stop
            ):
                received += 1
        assert received == 2
        # The generator was closed at the third pull: at most one unit past
        # what was consumed, never the remaining three.
        assert engine.units <= 3

    async def test_a_whole_engine_on_the_stream_path_gets_the_check_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _Whole()
        registry = _registry(tmp_path, monkeypatch, engine)
        with pytest.raises(AbandonedError):
            async for _ in registry.synthesize_stream(
                "hojo-40m", ["x"], "v", stop=lambda: True
            ):
                pass
        assert engine.units == 0


class TestEngineGuardsDoNotSwallowIt:
    """`AbandonedError` is a `RuntimeError`; an engine's own guard against the
    model producing nothing must let it through, or a listener leaving is
    reported as NO_AUDIO and nothing is logged as an abandonment."""

    def test_the_40m_render_guard_reraises_abandonment(self) -> None:
        from cortex_speech.engine.preset import PresetEngine

        class _Model:
            sample_rate = 24000

            def generate(self, text, *, voice, temperature, seed, on_step):
                on_step()
                raise AssertionError("unreachable")

        engine = PresetEngine.__new__(PresetEngine)
        engine._model = _Model()  # noqa: SLF001 - no bundle on disk
        with pytest.raises(AbandonedError):
            engine._render("x", "v", 0.8, lambda: True)  # noqa: SLF001


class TestTheWireMappingLetsItThrough:
    def test_engine_errors_does_not_turn_it_into_a_500(self) -> None:
        from cortex_tts.api.routes import engine_errors

        with pytest.raises(AbandonedError), engine_errors():
            raise AbandonedError("the listener went away")
