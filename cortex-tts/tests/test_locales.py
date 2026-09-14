"""One locale per language, and a language the pipeline was never told about."""

from __future__ import annotations

import pytest

from cortex_speech.text.locales import LOCALES, primary
from cortex_speech.text.options import NormalizeOptions
from cortex_speech.text.pipeline import (
    TextOptions,
    locale_for,
    plan,
    prepare,
    sniff_language,
)


class TestResolution:
    """Which locale a request lands in."""

    @pytest.mark.parametrize(
        ("tag", "code"),
        [("zh-TW", "zh"), ("zh-Hant-TW", "zh"), ("EN-us", "en"), ("de", "de")],
    )
    def test_the_primary_subtag_picks_the_locale(self, tag: str, code: str) -> None:
        assert primary(tag) == code
        assert locale_for(tag).code == code

    def test_written_locales_are_registered_by_importing_the_pipeline(self) -> None:
        assert {"zh", "en"} <= set(LOCALES)

    def test_a_language_without_a_locale_gets_the_generic_one(self) -> None:
        locale = locale_for("de-DE")
        assert locale.code == "de"
        assert locale.rewrites == ()

    @pytest.mark.parametrize(
        ("text", "tag"),
        [
            ("客廳的燈已經打開了", "zh-Hant"),
            ("客厅的灯已经打开了", "zh"),
            ("温度は二十六度です", "ja"),
            ("It is warm", "en"),
            ("80%。", "zh"),
        ],
    )
    def test_sniffing_names_the_most_specific_tag_it_can(
        self, text: str, tag: str
    ) -> None:
        assert sniff_language(text) == tag

    def test_the_request_s_tag_beats_the_sniff(self) -> None:
        assert plan("客廳的燈", language="zh-CN").language == "zh-CN"

    def test_an_underscore_tag_is_read_as_the_hyphenated_one(self) -> None:
        assert plan("垃圾", language="zh_TW").rewrites["taiwan_readings"] is True

    def test_a_bare_zh_is_refined_by_the_script(self) -> None:
        # "zh" from a voice or a picker says only "Chinese"; the glyphs say
        # which. A region or script in the tag is left alone.
        assert plan("客廳的燈", language="zh").language == "zh-Hant"
        assert plan("客厅的灯", language="zh").language == "zh"
        assert plan("客廳的燈", language="zh-CN").language == "zh-CN"


# Bare numbers are read only when asked; the shapes with a unit, a clock
# or a date around them are read by default.
WITH_NUMBERS = TextOptions(normalize_options=NormalizeOptions(expand_numbers=True))


class TestGeneric:
    """What every other language gets: its fixed shapes, and a stop."""

    def test_a_bare_number_is_left_as_digits_unless_asked(self) -> None:
        # Zimmer 302 is a room, not three hundred and two; the locale does not
        # know which, so it does not guess.
        assert "".join(prepare("Zimmer 302", language="de")) == "Zimmer 302."
        assert "".join(prepare("Zimmer 302", WITH_NUMBERS, "de")) == (
            "Zimmer dreihundertzwei."
        )

    def test_a_letter_unit_keeps_its_space_when_unnamed(self) -> None:
        assert "".join(prepare("5 ppm", WITH_NUMBERS, "de")) == "fünf ppm."

    def test_numbers_are_read_in_the_language_s_own_words(self) -> None:
        assert "".join(prepare("Es sind 26.5 Grad.", WITH_NUMBERS, "de")) == (
            "Es sind sechsundzwanzig Komma fünf Grad."
        )
        assert "".join(prepare("Il y a 1,234 personnes", WITH_NUMBERS, "fr")) == (
            "Il y a mille deux cent trente-quatre personnes."
        )

    def test_units_are_named_the_way_cldr_names_them(self) -> None:
        assert "".join(prepare("Es sind 26.5°C und 68%.", language="de")) == (
            "Es sind sechsundzwanzig Komma fünf Grad Celsius und achtundsechzig Prozent."
        )
        assert "".join(prepare("Vent 12 km/h", language="fr")) == (
            "Vent douze kilomètres par heure."
        )

    def test_a_unit_cldr_has_no_name_for_keeps_its_symbol(self) -> None:
        # Japanese CLDR composes no "per" unit; "リットル毎duration-minute"
        # read aloud would be worse than the symbol.
        assert "".join(prepare("湿度 68%、2 L/min", language="ja")) == (
            "湿度六十八パーセント、二 L/min。"
        )
        assert "".join(prepare("濃度 5 ppm", WITH_NUMBERS, "de")) == "濃度 fünf ppm."

    def test_home_assistant_s_units_are_named_by_cldr(self) -> None:
        assert "".join(prepare("2 L/min und 12 mV", language="de")) == (
            "zwei Liter pro Minute und zwölf mV."
        )
        assert "".join(prepare("2 µg/m³", language="fr")) == (
            "deux microgrammes par mètre cube."
        )

    def test_a_model_that_reads_numerals_keeps_the_generic_locale_out(self) -> None:
        decided = plan("Es sind 26.5°C", language="de", reads_numerals=True)
        assert decided.normalize_text is False
        assert "".join(
            prepare("Es sind 26.5°C", language="de", reads_numerals=True)
        ) == ("Es sind 26.5°C.")

    def test_a_written_locale_is_kept_whatever_the_model_reads(self) -> None:
        assert plan("溫度 26.5°C", language="zh", reads_numerals=True).normalize_text
        assert plan("It is 26.5°C", language="en", reads_numerals=True).normalize_text

    def test_a_decimal_comma_is_read_where_the_language_writes_one(self) -> None:
        assert "".join(prepare("26,5°C", language="de")) == (
            "sechsundzwanzig Komma fünf Grad Celsius."
        )
        # Home Assistant's own 26.5 is a decimal in every language, and a
        # comma before three digits is its thousands separator.
        assert "".join(prepare("26.5°C und 1,234 kWh", language="de")) == (
            "sechsundzwanzig Komma fünf Grad Celsius und "
            "eintausendzweihundertvierunddreißig Kilowattstunden."
        )

    def test_an_iso_date_is_laid_out_by_cldr_and_read_in_words(self) -> None:
        assert "".join(prepare("2026-09-14", language="de")) == (
            "vierzehnte September zweitausendsechsundzwanzig."
        )
        assert "".join(prepare("2026-09-14", language="fr")) == (
            "quatorze septembre deux mille vingt-six."
        )
        # An era year (令和八年) is not what a sensor date should sound like.
        assert (
            "".join(prepare("2026-09-14", language="ja")) == "二千二十六年九月十四日。"
        )

    def test_a_date_that_is_not_one_is_left_alone(self) -> None:
        assert "".join(prepare("2026-13-40", language="de")) == "2026-13-40."
        assert "".join(prepare("2026-13-40", language="xx")) == "2026-13-40."

    def test_a_clock_literal_is_hour_words_then_minute_words(self) -> None:
        assert "".join(prepare("um 14:35", language="de")) == (
            "um vierzehn fünfunddreißig."
        )
        # A score is not a time, and a bare number is not read unasked.
        assert "".join(prepare("3:2", language="de")) == "3:2."

    def test_japanese_closes_the_gap_an_expansion_leaves(self) -> None:
        assert "".join(prepare("温度は 26.5 度です", WITH_NUMBERS, "ja")) == (
            "温度は二十六点五度です。"
        )

    def test_korean_keeps_its_word_spaces(self) -> None:
        assert "".join(prepare("안녕 3 명", WITH_NUMBERS, "ko")) == "안녕 삼 명."

    def test_a_language_num2words_lacks_keeps_its_digits(self) -> None:
        # Read wrong is worse than read as digits: the digits at least are
        # not another language's words.
        assert "".join(prepare("Dua 12 buah", WITH_NUMBERS, "xx")) == "Dua 12 buah."

    def test_the_switch_leaves_the_numbers_alone(self) -> None:
        options = TextOptions(normalize_text=False)
        assert "".join(prepare("Es sind 26.5 Grad", options, "de")) == (
            "Es sind 26.5 Grad."
        )


class TestChineseRewrites:
    """The rewrites only Chinese has, and how the tag decides them."""

    @pytest.mark.parametrize("tag", ["zh-TW", "zh-Hant", "zh-Hant-TW", "ZH-tw"])
    def test_taiwan_s_tags_turn_the_readings_on(self, tag: str) -> None:
        assert plan("垃圾", language=tag).rewrites["taiwan_readings"] is True

    @pytest.mark.parametrize("tag", ["zh", "zh-CN", "zh-Hans", "zh-HK", "zh-Hant-HK"])
    def test_other_chinese_tags_leave_them_off(self, tag: str) -> None:
        # Hong Kong writes Traditional but reads Cantonese; Taiwan readings are
        # Taiwan's, so only the region or the script that means Taiwan counts.
        assert plan("垃圾", language=tag).rewrites["taiwan_readings"] is False

    def test_an_explicit_answer_beats_the_tag(self) -> None:
        options = TextOptions(taiwan_readings=True)
        assert plan("垃圾", options, "zh-CN").rewrites["taiwan_readings"] is True

    def test_readings_need_the_conversion_they_are_keyed_on(self) -> None:
        # The table is keyed by Simplified words: with conversion off the
        # pass would match nothing and still be reported as running.
        options = TextOptions(convert_script=False, taiwan_readings=True)
        decided = plan("企業", options, "zh-TW")
        assert decided.rewrites == {"convert_script": False, "taiwan_readings": False}

    def test_conversion_is_always_on_for_chinese(self) -> None:
        assert plan("垃圾", language="zh-CN").rewrites["convert_script"] is True

    def test_no_other_language_has_them(self) -> None:
        assert plan("Hello", language="en").rewrites == {}
        assert plan("研究は楽しい", language="ja").rewrites == {}
