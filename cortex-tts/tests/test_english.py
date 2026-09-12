"""The Latin normaliser: the same job as `normalize`, in the other script."""

from __future__ import annotations

import pytest

from cortex_speech.text import english
from cortex_speech.text.pipeline import TextOptions, prepare


class TestCardinal:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "zero"),
            (13, "thirteen"),
            (48, "forty-eight"),
            (72, "seventy-two"),
            (105, "one hundred and five"),
            (1000, "one thousand"),
            (12345, "twelve thousand three hundred and forty-five"),
        ],
    )
    def test_integers(self, value: int, expected: str) -> None:
        assert english.cardinal(value) == expected

    def test_a_decimal_reads_its_fraction_digit_by_digit(self) -> None:
        assert english.decimal("26.5") == "twenty-six point five"
        assert english.decimal("-3.25") == "minus three point two five"

    def test_a_year_is_not_a_quantity(self) -> None:
        assert english.year(2026) == "twenty twenty-six"
        assert english.year(2005) == "two thousand and five"


class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Turn on 3 lights", "Turn on three lights"),
            ("humidity 68%", "humidity sixty-eight percent"),
            ("It is 26.5°C", "It is twenty-six point five degrees Celsius"),
            ("at 8:30", "at eight thirty"),
            ("at 8:00", "at eight o'clock"),
        ],
    )
    def test_constructs(self, raw: str, expected: str) -> None:
        assert english.normalize(raw) == expected

    @pytest.mark.parametrize("dash", ["-", "~", "\u2013", "\u2014"])
    def test_a_dash_between_numbers_is_a_range(self, dash: str) -> None:
        assert english.normalize(f"48 {dash} 72 hours") == (
            "forty-eight to seventy-two hours"
        )

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The battery is at 80.", "The battery is at eighty."),
            ("Wait 15, then 20.", "Wait fifteen, then twenty."),
        ],
    )
    def test_a_number_ending_a_sentence_still_expands(
        self, raw: str, expected: str
    ) -> None:
        assert english.normalize(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # A label is read digit by digit, not as a count.
            ("except P0+ fixes", "except P zero+ fixes"),
            ("v1.2 shipped", "v one point two shipped"),
            ("PM2.5 is high", "PM two point five is high"),
            # Digits before a letter carry a unit, so they stay a quantity.
            ("a 24V supply", "a twenty-four V supply"),
        ],
    )
    def test_a_digit_welded_to_a_letter_is_still_spoken(
        self, raw: str, expected: str
    ) -> None:
        assert english.normalize(raw) == expected


class TestPipelineChoosesByScript:
    def test_latin_text_is_normalised_into_english(self) -> None:
        assert prepare(
            "It usually takes 48 - 72 hours", TextOptions(convert_script=False)
        ) == ["It usually takes forty-eight to seventy-two hours."]

    def test_chinese_text_still_gets_the_chinese_normaliser(self) -> None:
        assert prepare("室外溫度 25-30 度") == ["室外温度二十五到三十度。"]

    def test_one_chinese_word_does_not_make_a_sentence_chinese(self) -> None:
        assert prepare(
            "Turn on 3 lights in the 客廳", TextOptions(convert_script=False)
        ) == ["Turn on three lights in the 客廳."]

    def test_a_latin_acronym_does_not_make_a_sentence_english(self) -> None:
        assert prepare("目前 CPU 使用率 42%") == ["目前 CPU 使用率百分之四十二。"]

    def test_a_short_acronym_does_not_outvote_three_chinese_words(self) -> None:
        # Counting letters made this a 3-3 tie and picked the ASCII stop.
        assert prepare("請檢查 CPU") == ["请检查 CPU。"]
