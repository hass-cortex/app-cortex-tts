"""The gate that refuses a reply too long to be worth rendering.

A synthesis is one uninterruptible worker thread, so a reply that renders past
the caller's timeout is not merely late — its audio finishes into a socket
nobody reads, and the model was held the whole time. Measured in production:
a 514-character story at over seven minutes on a CPU rendering OmniVoice at
4.6x. The guard estimates from the text's length and the speed this host has
measured, and refuses before the render starts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cortex_speech import BY_ID, ModelSpec
from cortex_speech.engine.overrun import estimated_audio_seconds
from cortex_tts.api.routes import _guard_render_length
from cortex_tts.preferences import Preferences
from cortex_tts.stats import ModelStats

SPEC = BY_ID["omnivoice"]


class TestTheEstimate:
    def test_mandarin_is_summed_across_segments(self) -> None:
        # 4.5 Mandarin chars a second; two nine-character segments -> 4 s.
        segments = ["一二三四五六七八九", "一二三四五六七八九"]
        assert estimated_audio_seconds(segments) == pytest.approx(4.0)

    def test_empty_is_zero(self) -> None:
        assert estimated_audio_seconds([]) == 0.0


def _state(limit: int, samples: dict[str, tuple[float, ...]]) -> SimpleNamespace:
    stats = SimpleNamespace(
        get=lambda model_id: [
            ModelStats(kind=kind, samples=s) for kind, s in samples.items()
        ]
    )
    return SimpleNamespace(
        preferences=Preferences(max_synthesis_seconds=limit),
        stats=stats,
        references=SimpleNamespace(get=lambda voice_id: None),
    )


def _designed_voice(spec: ModelSpec) -> str:
    # OmniVoice's own voices are `designed`; _voice_kind reads the catalog.
    return "wanwan-xiaohe"


class TestTheGuard:
    """`_voice_kind` reads the catalog, so a designed OmniVoice voice is the
    `designed` kind — the key its measurements are filed under."""

    _LONG = ["一二三四五六七八九十" * 12]  # ~120 chars -> ~27 s of audio

    def test_a_reply_over_the_limit_is_refused(self) -> None:
        state = _state(120, {"designed": (4.6, 4.6)})  # 27 s * 4.6 ~= 123 s
        with pytest.raises(HTTPException) as caught:
            _guard_render_length(state, SPEC, self._LONG, _designed_voice(SPEC))
        assert caught.value.status_code == 413
        assert caught.value.detail["code"] == "RENDER_TOO_LONG"

    def test_a_short_reply_passes(self) -> None:
        state = _state(120, {"designed": (4.6,)})
        _guard_render_length(state, SPEC, ["一二三四五。"], _designed_voice(SPEC))

    def test_zero_accepts_any_length(self) -> None:
        state = _state(0, {"designed": (4.6,)})
        _guard_render_length(state, SPEC, self._LONG, _designed_voice(SPEC))

    def test_an_unmeasured_model_is_not_judged(self) -> None:
        """Nothing to estimate from; one long render teaches the store."""
        state = _state(1, {})
        _guard_render_length(state, SPEC, self._LONG, _designed_voice(SPEC))

    def test_a_fast_model_passes_the_same_text(self) -> None:
        state = _state(120, {"designed": (0.4,)})  # 27 s * 0.4 ~= 11 s
        _guard_render_length(state, SPEC, self._LONG, _designed_voice(SPEC))
