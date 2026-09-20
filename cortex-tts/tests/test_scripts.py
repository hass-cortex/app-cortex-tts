"""What is decided by the character, whoever wrote it.

Sentence ends and speech rates are keyed by script in `text/scripts.py`, not
by the request's language: 「।」 closes a sentence in an untagged request, in
mixed text and in a reply arriving in pieces alike. The locale keeps what
depends on the reader.
"""

from __future__ import annotations

import pytest

from cortex_speech.pacing import SentenceBuffer
from cortex_speech.text.pipeline import segment
from cortex_speech.text.scripts import (
    CLAUSE_MARKS,
    ENDINGS,
    HAN_RATE,
    LATIN_RATE,
    TERMINATORS,
    UNKNOWN_RATE,
    chars_within,
    spoken_seconds,
)


class TestATerminatorEndsASentenceWhateverTheScript:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Devanagari danda
            ("आज मौसम अच्छा है। कल बारिश होगी।", ["आज मौसम अच्छा है।", "कल बारिश होगी।"]),
            # Arabic question mark and full stop (Urdu)
            ("كيف حالك؟ أنا بخير۔", ["كيف حالك؟", "أنا بخير۔"]),
            # Ethiopic full stop
            ("ሰላም ነው። ደህና ነኝ።", ["ሰላም ነው።", "ደህና ነኝ።"]),
        ],
    )
    def test_the_batch_splitter_breaks_after_it(
        self, text: str, expected: list[str]
    ) -> None:
        """A bound that one sentence meets and two do not: the cut lands on
        the terminator, not on the length."""
        assert segment(text, limit=len(expected[0]) + 3) == expected

    def test_the_live_buffer_agrees(self) -> None:
        """One declaration: what `segment` cuts, the pacer hands out."""
        buffer = SentenceBuffer()
        buffer.feed("आज मौसम अच्छा है। कल")
        assert buffer.sentences == ["आज मौसम अच्छा है।"]
        assert buffer.tail.strip() == "कल"

    def test_a_segment_already_terminated_gains_no_stop(self) -> None:
        """An untagged Hindi request falls to the generic locale, whose stop is
        the ASCII one; its own danda must still count as an ending."""
        assert segment("कल बारिश होगी।") == ["कल बारिश होगी।"]

    def test_thai_has_no_terminator_and_splits_on_length(self) -> None:
        """UAX #29 lists no Thai sentence terminator; ICU and espeak-ng split
        none, and neither does this. The length bound still holds."""
        text = "วันนี้อากาศดี " * 20
        pieces = segment(text, limit=60)
        assert len(pieces) > 1
        assert all(len(p) <= 60 for p in pieces)

    def test_the_three_sets_are_one_declaration(self) -> None:
        assert set(TERMINATORS) < set(ENDINGS)
        assert not set(TERMINATORS) & set(CLAUSE_MARKS)


class TestASpeechRateIsTheScripts:
    def test_kana_and_hangul_speak_at_the_han_rate(self) -> None:
        """One character is one syllable or mora there too."""
        assert spoken_seconds("きょうはいいてんき") == pytest.approx(9 / HAN_RATE)
        assert spoken_seconds("오늘은날씨가") == pytest.approx(6 / HAN_RATE)

    def test_latin_speaks_at_its_own(self) -> None:
        assert spoken_seconds("Hello there") == pytest.approx(10 / LATIN_RATE)

    def test_an_unmeasured_script_is_assumed_slow(self) -> None:
        """Over-admitting past a model's audio ceiling truncates; under-admitting
        only splits. So the floor is the slowest measured rate, never the
        Latin one an unlisted script used to fall into."""
        assert min(HAN_RATE, LATIN_RATE) == UNKNOWN_RATE
        thai = "วันนี้อากาศดี"
        assert spoken_seconds(thai) == pytest.approx(len(thai) / UNKNOWN_RATE)
        assert chars_within(30.0, thai) == int(30.0 * UNKNOWN_RATE)

    def test_mixed_text_is_counted_per_character(self) -> None:
        assert spoken_seconds("溫度 26 度") == pytest.approx(
            3 / HAN_RATE + 2 / LATIN_RATE
        )
