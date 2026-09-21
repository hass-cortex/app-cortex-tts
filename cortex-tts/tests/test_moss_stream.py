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
        self.texts: list[str] = []

    def resolve_prompt_audio_codes(self, **kwargs: Any) -> list[list[int]]:
        return []

    def count_text_tokens(self, text: str) -> int:
        """One character, one token: the real tokenizer is the bundle's."""
        return len(text)

    def split_voice_clone_text(self, text: str, max_tokens: int = 75) -> list[str]:
        """Cut into pieces of at most `max_tokens`, as the runtime's does."""
        return [text[i : i + max_tokens] for i in range(0, len(text), max_tokens)]

    def synthesize_single_chunk(
        self,
        *,
        text: str,
        prompt_audio_codes: list[list[int]],
        streaming: bool,
        on_audio_chunk: Callable[[np.ndarray], None],
    ) -> dict[str, Any]:
        del prompt_audio_codes, streaming
        self.texts.append(text)
        emitted: list[np.ndarray] = []
        for index in range(self.chunks):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("codec fell over")
            time.sleep(0.002)
            self.emitted += 1
            chunk = np.full((480, 2), 0.1, dtype=np.float32)
            emitted.append(chunk)
            on_audio_chunk(chunk)
        # The batch path reads this; the streaming one ignores it.
        return {"waveform": np.concatenate(emitted) if emitted else np.zeros((0, 2))}


def _engine(runtime: _Runtime, max_text_tokens: int | None = None) -> MossEngine:
    engine = object.__new__(MossEngine)
    engine._runtime = runtime  # type: ignore[assignment]
    engine._builtin_ids = {"v"}
    engine._max_text_tokens = max_text_tokens  # type: ignore[assignment]
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


class TestTokenBudget:
    """What one call into the model may carry, counted in its own tokens.

    The pipeline's segments are sized in characters against an audio ceiling;
    this is the bound only the engine can apply, and the reason it exists is
    that stopping is sampled — a longer chunk is more chances to sample the
    end early. Measured on one 411-character reply at the pinned seed: whole,
    12.00 s of the 28 s it needed; cut to this budget, all of it.
    """

    def test_an_over_budget_segment_is_cut(self) -> None:
        runtime = _Runtime(chunks=1)
        list(_engine(runtime, max_text_tokens=5).synthesize_stream(["abcdefghij"], "v"))
        assert runtime.texts == ["abcde", "fghij"]

    def test_a_segment_within_budget_is_passed_through_untouched(self) -> None:
        """Not round-tripped: the runtime's splitter also rewrites its input."""
        runtime = _Runtime(chunks=1)
        list(_engine(runtime, max_text_tokens=50).synthesize_stream(["a。"], "v"))
        assert runtime.texts == ["a。"]

    def test_no_budget_leaves_every_segment_whole(self) -> None:
        runtime = _Runtime(chunks=1)
        list(_engine(runtime).synthesize_stream(["abcdefghij"], "v"))
        assert runtime.texts == ["abcdefghij"]

    def test_the_batch_path_cuts_the_same_way(self) -> None:
        """Both paths must agree, or a reply changes with the delivery."""
        runtime = _Runtime(chunks=1)
        _engine(runtime, max_text_tokens=5).synthesize(["abcdefghij"], "v")
        assert runtime.texts == ["abcde", "fghij"]


class _Seeded(_Runtime):
    """Comes up short at every seed but `good`, recording the seeds it got."""

    def __init__(self, good: int, *, short: int = 5, full: int = 200) -> None:
        super().__init__(chunks=full)
        self.good = good
        self.short = short
        self.full = full
        self.seeds: list[int] = []
        self._rng: Any = None

    @property
    def rng(self) -> Any:
        return self._rng

    @rng.setter
    def rng(self, value: Any) -> None:
        self._rng = value
        self.seeds.append(int(value.bit_generator.seed_seq.entropy))

    def synthesize_single_chunk(self, **kwargs: Any) -> dict[str, Any]:
        self.chunks = self.full if self.seeds[-1] == self.good else self.short
        return super().synthesize_single_chunk(**kwargs)


# 30 Latin characters: `overrun` wants 2.14 s for them, so the 2.0 s a full
# render produces here passes and the 0.05 s a short one produces does not.
_JUDGEABLE = "a" * 30


class TestRetries:
    """Stopping is sampled, so only another seed changes a short generation.

    Measured on one 411-character chunk: 12.00 s at the seed this engine pins
    and 26.64 s at the next one tried, of the 28 s the text needed.
    """

    def test_a_short_generation_is_retried_at_another_seed(self) -> None:
        runtime = _Seeded(good=7)
        _engine(runtime).synthesize([_JUDGEABLE], "v")
        assert runtime.seeds == [1234, 7]

    def test_the_pinned_seed_is_never_tried_twice(self) -> None:
        """It is also in `RETRY_SEEDS`, and a repeat would be identical."""
        runtime = _Seeded(good=-1)
        _engine(runtime).synthesize([_JUDGEABLE], "v")
        assert runtime.seeds == [1234, 7]

    def test_the_streamed_path_does_not_retry(self) -> None:
        """Its chunks are already gone; it says so instead."""
        runtime = _Seeded(good=7)
        list(_engine(runtime).synthesize_stream([_JUDGEABLE], "v"))
        assert runtime.seeds == [1234]

    def test_the_streamed_path_says_a_chunk_stopped_short(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runtime = _Seeded(good=7)
        with caplog.at_level("WARNING", logger="cortex_speech.engine.moss"):
            list(_engine(runtime).synthesize_stream([_JUDGEABLE], "v"))
        assert "stopped at 0.1s" in caplog.text

    def test_a_full_generation_is_left_alone(self) -> None:
        runtime = _Seeded(good=1234)
        _engine(runtime).synthesize([_JUDGEABLE], "v")
        assert runtime.seeds == [1234]
