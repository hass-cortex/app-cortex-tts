"""The MOSS stream's thread and queue, driven with a fake runtime.

The runtime calls back from inside its decode loop, so the render runs on a
worker thread. What is pinned here is that a consumer who stops reading stops
the worker too: one that stayed blocked on a full queue would keep the
runtime's per-stream state alive under the next request.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cortex_speech.engine.base import NoAudioError
from cortex_speech.engine.moss import MossEngine


class _Runtime:
    """Emits `chunks` audio chunks per segment, or raises after `fail_after`."""

    codec_meta = {"codec_config": {"sample_rate": 48000}}

    def __init__(self, chunks: int = 50, fail_after: int | None = None) -> None:
        self.chunks = chunks
        self.fail_after = fail_after
        self.emitted = 0

    def resolve_prompt_audio_codes(self, **kwargs: Any) -> list[list[int]]:
        return []

    def synthesize_single_chunk(
        self,
        *,
        text: str,
        prompt_audio_codes: list[list[int]],
        streaming: bool,
        on_audio_chunk: Callable[[np.ndarray], None],
    ) -> None:
        del text, prompt_audio_codes, streaming
        for index in range(self.chunks):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("codec fell over")
            time.sleep(0.002)
            self.emitted += 1
            on_audio_chunk(np.full((480, 2), 0.1, dtype=np.float32))


def _engine(runtime: _Runtime) -> MossEngine:
    engine = object.__new__(MossEngine)
    engine._runtime = runtime  # type: ignore[assignment]
    engine._builtin_ids = {"v"}
    return engine


def _stream_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "moss-stream"]


class TestStreaming:
    def test_chunks_arrive_mono_and_in_full(self) -> None:
        runtime = _Runtime(chunks=5)
        chunks = list(_engine(runtime).synthesize_stream(["a。", "b。"], "v"))
        # 5 per segment plus the gap between segments.
        assert len(chunks) == 11
        assert all(c.ndim == 1 for c in chunks)

    def test_a_runtime_failure_reaches_the_consumer(self) -> None:
        runtime = _Runtime(chunks=5, fail_after=2)
        with pytest.raises(RuntimeError, match="codec fell over"):
            list(_engine(runtime).synthesize_stream(["a。"], "v"))
        assert not _stream_threads()

    def test_nothing_produced_is_an_error(self) -> None:
        with pytest.raises(NoAudioError):
            list(_engine(_Runtime(chunks=0)).synthesize_stream(["a。"], "v"))


class TestAbandonment:
    def test_closing_the_generator_stops_the_worker_promptly(self) -> None:
        """The failure this exists for: a consumer gone, a worker blocked."""
        runtime = _Runtime(chunks=200)
        stream = _engine(runtime).synthesize_stream(["a。"], "v")
        next(stream)
        next(stream)

        started = time.perf_counter()
        stream.close()
        elapsed = time.perf_counter() - started

        assert elapsed < 2.0, f"close() blocked for {elapsed:.1f}s"
        deadline = time.perf_counter() + 2.0
        while _stream_threads() and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert not _stream_threads(), "render thread outlived its consumer"
        assert runtime.emitted < 200, "the worker rendered on after the close"


class TestSampling:
    """Every segment starts from the same sampler state.

    The runtime seeds once at construction and draws from that generator for
    every frame, so consecutive syntheses came out differently: the same line
    measured 6.80 s, 7.52 s and 9.60 s on three requests. A reply split across
    requests is several syntheses, which is how one half ended up planned
    differently from the other.
    """

    class _Sampling(_Runtime):
        """Records what the sampler would have drawn for each segment."""

        def __init__(self) -> None:
            super().__init__(chunks=1)
            self.rng = np.random.default_rng(999)
            self.draws: list[float] = []

        def synthesize_single_chunk(self, **kwargs: Any) -> None:
            self.draws.append(float(self.rng.random()))
            super().synthesize_single_chunk(**kwargs)

    def test_each_segment_draws_from_the_same_state(self) -> None:
        runtime = self._Sampling()
        list(_engine(runtime).synthesize_stream(["a。", "b。", "c。"], "v"))
        assert len(runtime.draws) == 3
        assert len(set(runtime.draws)) == 1, runtime.draws

    def test_a_second_reply_starts_where_the_first_did(self) -> None:
        """Two requests are what a split reply becomes; they must match."""
        runtime = self._Sampling()
        engine = _engine(runtime)
        list(engine.synthesize_stream(["a。"], "v"))
        list(engine.synthesize_stream(["b。"], "v"))
        assert runtime.draws[0] == runtime.draws[1]
