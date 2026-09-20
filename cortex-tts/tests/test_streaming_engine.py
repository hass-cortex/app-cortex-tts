"""The streaming protocol, and the failure it is shaped to prevent.

`StreamingEngine` is deliberately separate from `Engine`: two of three engines
cannot emit mid-segment audio, and a method they would have to decline is worse
than not being asked. The registry asks with `isinstance`.

These tests use a fake engine. Exercising the real one needs a 729 MB bundle,
and what is worth pinning here is the contract, not the model.
"""

from __future__ import annotations

import time
from collections.abc import Generator

import numpy as np

from cortex_speech.engine.base import StreamingEngine


class _Streaming:
    """Emits three chunks, slowly, so ordering and timing are observable."""

    sample_rate = 48000

    def synthesize_stream(
        self, segments: list[str], voice: str
    ) -> Generator[np.ndarray, None, None]:
        del segments, voice
        for value in (0.1, 0.2, 0.3):
            time.sleep(0.01)
            yield np.full(480, value, dtype=np.float32)


class _NotStreaming:
    """A whole-utterance engine: the shape the 40M and OmniVoice have."""

    sample_rate = 24000

    def synthesize(self, segments: list[str], voice: str) -> None:
        del segments, voice


class TestProtocolMembership:
    def test_an_engine_that_streams_is_recognised(self) -> None:
        assert isinstance(_Streaming(), StreamingEngine)

    def test_an_engine_that_does_not_is_not(self) -> None:
        """The registry must be able to tell without calling anything."""
        assert not isinstance(_NotStreaming(), StreamingEngine)


class TestChunkContract:
    def test_chunks_are_mono_float32(self) -> None:
        """A stereo chunk would reach a consumer expecting one channel.

        The real engine's codec emits `(samples, channels)`; downmixing on the
        way out is what keeps this true, and forgetting it is silent — the
        audio plays, at double length and half speed.
        """
        for chunk in _Streaming().synthesize_stream(["x"], "v"):
            assert chunk.ndim == 1
            assert chunk.dtype == np.float32

    def test_chunks_arrive_before_the_whole_is_rendered(self) -> None:
        """The point of the feature, and the bug it is easy to write instead.

        Collecting chunks and yielding them after the render finishes produces
        a generator that satisfies every other assertion here and streams
        nothing. So this measures that the first chunk beats the last.
        """
        started = time.perf_counter()
        stream = _Streaming().synthesize_stream(["x"], "v")
        first = next(stream)
        first_at = time.perf_counter() - started
        for _ in stream:
            pass
        last_at = time.perf_counter() - started
        assert first.size
        assert first_at < last_at / 2, (
            f"first chunk at {first_at * 1000:.0f}ms of {last_at * 1000:.0f}ms "
            "— chunks look collected rather than streamed"
        )
