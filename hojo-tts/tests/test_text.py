"""Tests for the text path.

The cases below are the ones measured against a real ASR round trip: raw
Traditional Chinese and unnormalised sensor text scored 32-85% character error
rate, and the same sentences through this pipeline scored 0-4%. If these
regress, the voice stops being intelligible.
"""

from __future__ import annotations

import pytest

from hojo_tts.engine.overrun import trim_trailing_babble
from hojo_tts.text.normalize import NormalizeOptions, normalize
from hojo_tts.text.numbers import cardinal, decimal, digit_string
from hojo_tts.text.pipeline import TextOptions, prepare, segment, to_simplified


class TestNumbers:
    """Chinese numeral rendering."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "零"),
            (7, "七"),
            (10, "十"),
            (15, "十五"),
            (26, "二十六"),
            (48, "四十八"),
            (100, "一百"),
            (148, "一百四十八"),
            (350, "三百五十"),
            (1013, "一千零一十三"),
            (10000, "一萬"),
            (48000, "四萬八千"),
        ],
    )
    def test_cardinal(self, value: int, expected: str) -> None:
        assert cardinal(value) == expected

    def test_cardinal_negative(self) -> None:
        assert cardinal(-52) == "負五十二"

    @pytest.mark.parametrize(
        ("literal", "expected"),
        [
            ("26.5", "二十六點五"),
            ("3.2", "三點二"),
            ("-3.5", "負三點五"),
            ("100", "一百"),
        ],
    )
    def test_decimal(self, literal: str, expected: str) -> None:
        assert decimal(literal) == expected

    def test_digit_string_reads_identifiers_one_by_one(self) -> None:
        assert digit_string("2026") == "二零二六"


class TestNormalize:
    """Expanding what Home Assistant actually emits."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("26.5°C", "攝氏二十六點五度"),
            ("68%", "百分之六十八"),
            ("14:35", "十四點三十五分"),
            ("7:00", "七點整"),
            ("2026-09-06", "二零二六年九月六日"),
            ("48 W", "四十八瓦"),
            ("3.2 kWh", "三點二度電"),
            ("12 km/h", "十二公里每小時"),
            ("1013 hPa", "一千零一十三百帕"),
            ("-3.5°C", "攝氏負三點五度"),
        ],
    )
    def test_units_and_literals(self, raw: str, expected: str) -> None:
        assert normalize(raw) == expected

    def test_range_reads_as_a_span_not_a_subtraction(self) -> None:
        assert normalize("25-30 度") == "二十五到三十 度"

    def test_version_reads_digit_by_digit(self) -> None:
        assert "二零二六點九" in normalize("更新到 2026.9 版本")

    def test_latin_words_are_left_alone(self) -> None:
        assert "Home Assistant" in normalize("你的 Home Assistant 已更新")

    def test_temperature_prefix_can_be_dropped(self) -> None:
        options = NormalizeOptions(temperature_prefix=False)
        assert normalize("26.5°C", options) == "二十六點五度"

    def test_no_arabic_digits_survive_a_sensor_sentence(self) -> None:
        raw = "現在室內溫度是 26.5°C，濕度 68%，PM2.5 是 12 微克。"
        assert not any(char.isdigit() for char in normalize(raw))


class TestScriptConversion:
    """The conversion the model cannot do without."""

    def test_traditional_becomes_simplified(self) -> None:
        assert to_simplified("客廳的燈") == "客厅的灯"

    def test_taiwanese_vocabulary_survives(self) -> None:
        # t2s converts glyphs only. The phrase-aware variants would rewrite
        # 設定 to 设置 and 訊號 to 信号, changing the words spoken aloud.
        assert to_simplified("設定") == "设定"
        assert to_simplified("訊號") == "讯号"

    def test_latin_is_untouched(self) -> None:
        assert to_simplified("Wi-Fi 已經連線") == "Wi-Fi 已经连线"


class TestSegmentation:
    """Splitting long text into synthesis-sized chunks."""

    def test_short_text_is_one_segment(self) -> None:
        assert segment("客厅的灯已经打开了。") == ["客厅的灯已经打开了。"]

    def test_empty_text_yields_nothing(self) -> None:
        assert segment("   ") == []

    def test_long_text_splits_on_sentences(self) -> None:
        text = "。".join(["这是一个很长的句子用来测试分段功能" * 2] * 6) + "。"
        segments = segment(text, limit=120)
        assert len(segments) > 1
        assert all(len(s) <= 120 for s in segments)

    def test_unpunctuated_text_is_still_bounded(self) -> None:
        segments = segment("字" * 500, limit=120)
        assert all(len(s) <= 120 for s in segments)
        # Each piece gains a full stop; the words themselves are unchanged.
        assert "".join(s.rstrip("。") for s in segments) == "字" * 500


class TestPipeline:
    """The whole path, in the order that matters."""

    def test_normalisation_runs_before_conversion(self) -> None:
        # Normalisation emits Traditional numerals (二十六點五度); if the
        # script pass ran first they would reach the model unconverted.
        prepared = "".join(prepare("溫度 26.5°C"))
        assert prepared == "温度摄氏二十六点五度。"

    def test_gap_left_by_an_expanded_number_is_closed(self) -> None:
        # The model reads a space as a pause, and expanding "26.5°C" strands
        # the space that preceded the digits between two Chinese characters.
        assert " " not in "".join(prepare("溫度是 26.5°C。"))

    def test_space_around_latin_is_kept(self) -> None:
        prepared = "".join(prepare("你的 Home Assistant 已經更新。"))
        assert " Home Assistant " in prepared

    def test_switches_can_be_turned_off(self) -> None:
        options = TextOptions(normalize_text=False, convert_script=False)
        assert "".join(prepare("溫度 26.5°C", options)) == "溫度 26.5°C。"

    def test_punctuation_only_input_yields_no_segments(self) -> None:
        assert prepare("。。。") == []

    @pytest.mark.parametrize(
        "raw",
        [
            "好的，客廳的燈已經打開了。",
            "現在室內溫度是 26.5°C，濕度 68%，PM2.5 是 12 微克。",
            "現在時間是 14:35，距離下一個行程還有 25 分鐘。",
            "今天是 2026-09-06，最高溫 31°C，降雨機率 40%。",
        ],
    )
    def test_ha_sentences_reach_the_model_pronounceable(self, raw: str) -> None:
        prepared = "".join(prepare(raw))
        assert prepared
        # No Arabic digits and no Traditional-only glyphs left to mispronounce.
        assert not any(char.isdigit() for char in prepared)
        assert prepared == to_simplified(prepared)


class TestVoiceIds:
    """Deriving a voice id from a display name.

    A deployment that names its voices in Chinese folds to nothing under
    ASCII, so the fallback is the normal path here, not the edge case.
    """

    def test_latin_name_stays_readable(self) -> None:
        from hojo_tts.refs import slugify

        assert slugify("Anna Su") == "anna-su"

    def test_chinese_name_falls_back_to_a_stable_id(self) -> None:
        from hojo_tts.refs import slugify

        assert slugify("雅玶").startswith("voice-")
        assert slugify("雅玶") == slugify("雅玶")

    def test_a_trailing_digit_does_not_become_the_whole_id(self) -> None:
        from hojo_tts.refs import slugify

        # "灣灣小何2" folded to "2" — unreadable, and colliding with every
        # other name ending in the same digit.
        assert slugify("灣灣小何2").startswith("voice-")

    def test_names_differing_only_by_digit_get_distinct_ids(self) -> None:
        from hojo_tts.refs import slugify

        ids = {slugify(n) for n in ("雅玶", "雅玶2", "雅玶3", "雅玶4")}
        assert len(ids) == 4


class TestTranscriptEditing:
    """Correcting the transcript of a stored reference.

    A transcript that does not match the recording degrades the clone with no
    error, so editing it has to be possible without re-uploading the audio.
    """

    def _store(self, tmp_path):
        import numpy as np
        import soundfile as sf

        from hojo_tts.refs import ReferenceStore

        store = ReferenceStore(tmp_path)
        tone = (0.2 * np.sin(np.linspace(0, 900, 24000 * 5))).astype("float32")
        buffer = __import__("io").BytesIO()
        sf.write(buffer, tone, 24000, format="WAV", subtype="PCM_16")
        reference = store.add(
            name="Test Voice", transcript="客廳的燈打開了。", audio=buffer.getvalue()
        )
        return store, reference

    def test_edit_replaces_the_prepared_and_raw_forms(self, tmp_path) -> None:
        store, reference = self._store(tmp_path)

        updated = store.update(reference.id, transcript="溫度是 26.5°C。")

        assert updated.raw_transcript == "溫度是 26.5°C。"
        assert updated.transcript == "温度是摄氏二十六点五度。"

    def test_edit_keeps_the_audio_fingerprint(self, tmp_path) -> None:
        # Cached encodings derive from the recording alone; changing the text
        # must not force the engine to spend seconds re-encoding it.
        store, reference = self._store(tmp_path)

        updated = store.update(reference.id, transcript="完全不同的一句話。")

        assert updated.fingerprint == reference.fingerprint

    def test_edit_survives_a_reload(self, tmp_path) -> None:
        from hojo_tts.refs import ReferenceStore

        store, reference = self._store(tmp_path)
        store.update(reference.id, transcript="改過的逐字稿。")

        reloaded = ReferenceStore(tmp_path).get(reference.id)
        assert reloaded is not None
        assert reloaded.raw_transcript == "改過的逐字稿。"

    def test_editing_an_unknown_reference_raises(self, tmp_path) -> None:
        store, _ = self._store(tmp_path)

        with pytest.raises(KeyError):
            store.update("nope", transcript="x")

    def test_an_unpronounceable_transcript_is_rejected(self, tmp_path) -> None:
        from hojo_tts.refs import ReferenceError

        store, reference = self._store(tmp_path)

        with pytest.raises(ReferenceError):
            store.update(reference.id, transcript="。。。")


class TestTrailingBabble:
    """Dropping speech the model invents after finishing a short sentence.

    Measured: "好了" (2 chars, ~0.7 s of speech) came back as 2.06 s — the
    sentence, a pause, then an unrelated syllable the model made up.
    """

    SR = 24000

    def _clip(self, spans, total_s):
        """Build audio with speech in `spans` (seconds) and silence elsewhere."""
        import numpy as np

        audio = np.zeros(int(total_s * self.SR), dtype="float32")
        for start, end in spans:
            a, b = int(start * self.SR), int(end * self.SR)
            audio[a:b] = 0.3 * np.sin(np.linspace(0, 400, b - a)).astype("float32")
        return audio

    def test_invented_tail_is_dropped(self) -> None:
        # The measured shape of the "好了" failure.
        audio = self._clip([(0.1, 0.75), (1.0, 1.55)], 2.06)
        trimmed = trim_trailing_babble(audio, self.SR, characters=2)

        assert len(trimmed) / self.SR < 1.0

    def test_speech_matching_the_text_is_untouched(self) -> None:
        # 8 chars needs ~1.8 s; this clip delivers it in one run.
        audio = self._clip([(0.05, 1.8)], 1.9)
        trimmed = trim_trailing_babble(audio, self.SR, characters=8)

        assert len(trimmed) == len(audio)

    def test_real_pauses_in_long_speech_survive(self) -> None:
        # 60 chars needs ~13 s. Three clauses with real pauses between them
        # must all be kept — the clip does not overrun, so nothing is cut.
        audio = self._clip([(0.1, 4.5), (4.9, 9.0), (9.4, 13.5)], 13.8)
        trimmed = trim_trailing_babble(audio, self.SR, characters=60)

        assert len(trimmed) == len(audio)

    def test_never_cuts_away_most_of_the_sentence(self) -> None:
        # Even wildly over-long, what is kept still covers the bulk of the
        # text — the estimate is loose, so the floor is only a backstop.
        audio = self._clip([(0.1, 2.5), (3.0, 4.0), (4.5, 6.0)], 6.2)
        trimmed = trim_trailing_babble(audio, self.SR, characters=10)

        assert len(trimmed) / self.SR >= (10 / 4.5) * 0.7

    def test_tail_after_a_fast_sentence_is_still_dropped(self) -> None:
        # Measured "大燈已關閉": 5 chars spoken in 0.90 s — faster than the
        # estimate — then a 0.34 s pause and an invented syllable. Anchoring
        # the floor on the estimate itself would have kept the tail.
        audio = self._clip([(0.14, 1.04), (1.38, 1.66)], 1.74)
        trimmed = trim_trailing_babble(audio, self.SR, characters=5)

        assert len(trimmed) / self.SR < 1.2


class TestSentenceTermination:
    """Every segment reaches the model with sentence-final punctuation.

    Measured: without it the model does not emit end-of-speech promptly and
    appends an invented syllable — "客廳的燈已經打開了" came back with a
    trailing "哈", and the identical text with a full stop did not.
    """

    def test_a_bare_sentence_gains_a_full_stop(self) -> None:
        assert prepare("客廳的燈已經打開了") == ["客厅的灯已经打开了。"]

    def test_existing_punctuation_is_left_alone(self) -> None:
        assert prepare("客廳的燈已經打開了。") == ["客厅的灯已经打开了。"]

    @pytest.mark.parametrize("ending", ["！", "？", "；"])
    def test_other_terminators_are_accepted(self, ending: str) -> None:
        assert prepare(f"真的嗎{ending}") == [f"真的吗{ending}"]

    def test_a_trailing_comma_becomes_a_full_stop(self) -> None:
        # A comma tells the model to carry on, which is the problem.
        assert prepare("好了，") == ["好了。"]

    def test_every_segment_of_a_split_is_terminated(self) -> None:
        text = "。".join(["这是一个很长的句子用来测试分段" * 3] * 4) + "。"
        assert all(s.endswith("。") for s in segment(text, limit=100))

    def test_an_english_sentence_splits_on_its_full_stop(self) -> None:
        # The ASCII stop was not a sentence boundary, so an English paragraph
        # reached the hard wrap as one chunk and was cut mid-word.
        assert prepare(
            "It is done. Nothing is left.",
            TextOptions(normalize_text=False, convert_script=False),
        ) == ["It is done. Nothing is left."]

    def test_a_decimal_is_not_a_sentence_boundary(self) -> None:
        assert prepare(
            "It is 26.5 degrees",
            TextOptions(normalize_text=False, convert_script=False),
        ) == ["It is 26.5 degrees."]

    def test_a_hard_wrap_falls_on_a_space(self) -> None:
        # "exciting" came back as "exciti." and "ng".
        text = "word " * 40
        for piece in segment(text.strip(), limit=60):
            assert not piece.rstrip(".").endswith("wor"), piece

    def test_an_english_full_stop_counts_as_terminated(self) -> None:
        # It was missing from the terminator set, so an already-finished
        # English sentence collected a "。" the English voice cannot say.
        assert prepare(
            "Turn on the lights.",
            TextOptions(normalize_text=False, convert_script=False),
        ) == ["Turn on the lights."]

    def test_an_english_sentence_gains_an_ascii_stop(self) -> None:
        assert prepare(
            "Turn on the lights",
            TextOptions(normalize_text=False, convert_script=False),
        ) == ["Turn on the lights."]
