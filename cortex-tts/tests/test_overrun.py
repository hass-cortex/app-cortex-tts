"""Judging a generation that stopped short, and what is retried after it.

The seed order is the contract: a model whose own seed is already among the
retries must not spend a whole render repeating an attempt that sampling being
seeded makes identical.
"""

from __future__ import annotations

import numpy as np

from cortex_speech.engine.overrun import (
    TRUNCATION_RATIO,
    expected_seconds,
    looks_truncated,
    render_with_retries,
)

_SAMPLE_RATE = 48000

# Long enough to judge (`MIN_JUDGEABLE_CHARS`), and Latin so the rate is 14.0.
_TEXT = "a" * 30


def _seconds(count: float) -> np.ndarray:
    return np.zeros(int(count * _SAMPLE_RATE), dtype=np.float32)


def _record(short_at: set[int]) -> tuple[list[int], object]:
    """A generator that comes up short at the given seeds, recording each."""
    seen: list[int] = []

    def generate(seed: int) -> np.ndarray:
        seen.append(seed)
        return _seconds(0.1 if seed in short_at else 10.0)

    return seen, generate


class TestLooksTruncated:
    def test_a_generation_near_its_estimate_is_not_short(self) -> None:
        assert not looks_truncated(_TEXT, expected_seconds(_TEXT) * 0.9)

    def test_the_default_ratio_is_unchanged(self) -> None:
        """Hojo's judgement must not move; only engines that ask get another."""
        just_under = expected_seconds(_TEXT) * (TRUNCATION_RATIO - 0.01)
        just_over = expected_seconds(_TEXT) * (TRUNCATION_RATIO + 0.01)
        assert looks_truncated(_TEXT, just_under)
        assert not looks_truncated(_TEXT, just_over)

    def test_a_tighter_ratio_catches_what_the_default_misses(self) -> None:
        seconds = expected_seconds(_TEXT) * 0.7
        assert not looks_truncated(_TEXT, seconds)
        assert looks_truncated(_TEXT, seconds, ratio=0.8)

    def test_text_too_short_to_judge_is_never_short(self) -> None:
        assert not looks_truncated("abc", 0.01)


class TestRetrySeeds:
    def test_a_seed_outside_the_retries_gets_all_three(self) -> None:
        """Hojo's 42: its own, then both retries."""
        seen, generate = _record(short_at={42, 7, 1234})
        render_with_retries(_TEXT, _SAMPLE_RATE, generate, seed=42)  # type: ignore[arg-type]
        assert seen == [42, 7, 1234]

    def test_a_seed_already_among_the_retries_is_not_repeated(self) -> None:
        """MOSS's 1234: the third attempt would repeat the first exactly."""
        seen, generate = _record(short_at={1234, 7})
        render_with_retries(_TEXT, _SAMPLE_RATE, generate, seed=1234)  # type: ignore[arg-type]
        assert seen == [1234, 7]

    def test_a_full_first_attempt_stops_there(self) -> None:
        seen, generate = _record(short_at=set())
        render_with_retries(_TEXT, _SAMPLE_RATE, generate, seed=42)  # type: ignore[arg-type]
        assert seen == [42]

    def test_the_ratio_reaches_the_judgement(self) -> None:
        """A generation the default accepts, a tighter ratio retries."""
        seconds = expected_seconds(_TEXT) * 0.7

        def generate(seed: int) -> np.ndarray:
            seen.append(seed)
            return _seconds(seconds)

        seen: list[int] = []
        render_with_retries(_TEXT, _SAMPLE_RATE, generate, seed=42)
        assert seen == [42]

        seen = []
        render_with_retries(_TEXT, _SAMPLE_RATE, generate, seed=42, ratio=0.8)
        assert seen == [42, 7, 1234]
