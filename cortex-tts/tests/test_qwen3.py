"""Qwen3-TTS's voice list and its generation ceiling.

Neither needs weights: the speaker table is read from `config.json`, and the
ceiling is arithmetic on the text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex_speech.engine.qwen3 import (
    _SPEAKERS,
    FRAMES_PER_SECOND,
    _frame_budget,
    own_voices,
)

# The nine speakers `Qwen3-TTS-12Hz-0.6B-CustomVoice` ships, as its model card
# lists them. Pinned so a table entry that loses its language or gender is a
# failure rather than a voice the UI labels "unknown".
SHIPPED_SPEAKERS = (
    "vivian",
    "serena",
    "uncle_fu",
    "dylan",
    "eric",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
)


def _config(tmp_path: Path, speakers: dict[str, int]) -> Path:
    (tmp_path / "config.json").write_text(
        json.dumps({"talker_config": {"spk_id": speakers}}), encoding="utf-8"
    )
    return tmp_path


class TestTheSpeakerTable:
    def test_every_shipped_speaker_is_described(self) -> None:
        assert set(SHIPPED_SPEAKERS) <= set(_SPEAKERS)

    def test_no_speaker_is_described_twice_over(self) -> None:
        """Ids are matched lower-cased; a capitalised duplicate never matches."""
        assert all(name == name.lower() for name in _SPEAKERS)

    @pytest.mark.parametrize("speaker", SHIPPED_SPEAKERS)
    def test_each_carries_a_language_and_a_sex(self, speaker: str) -> None:
        language, tag, gender, label = _SPEAKERS[speaker]
        assert language and tag and label
        assert gender in {"female", "male", "unknown"}


class TestListingVoices:
    def test_it_reads_the_checkpoint_rather_than_the_table(
        self, tmp_path: Path
    ) -> None:
        """`config.json` is what the talker itself reads the ids from."""
        directory = _config(tmp_path, {"vivian": 3065, "ryan": 3061})
        assert [voice.id for voice in own_voices(directory)] == ["vivian", "ryan"]

    def test_it_labels_them_from_the_table(self, tmp_path: Path) -> None:
        directory = _config(tmp_path, {"vivian": 3065})
        voice = own_voices(directory)[0]
        assert voice.language == "zh"
        assert voice.gender == "female"
        assert voice.source == "builtin"
        assert "Vivian" in voice.name

    def test_a_speaker_the_table_has_not_heard_of_still_lists(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint may gain a voice before this repo does; it must show."""
        directory = _config(tmp_path, {"newcomer": 3070})
        voice = own_voices(directory)[0]
        assert voice.id == "newcomer"
        assert voice.language is None
        assert voice.gender == "unknown"

    def test_a_checkpoint_with_no_speakers_lists_none(self, tmp_path: Path) -> None:
        """The cloning bundle declares `own_voices=False` and has none."""
        assert own_voices(_config(tmp_path, {})) == []


class TestTheGenerationCeiling:
    """The talker stops by sampling end-of-speech, and can fail to.

    Measured: at temperature 0 a ten-character line ran to the model's own
    2048-frame limit — 2.7 minutes of invented audio for one clause. The
    ceiling is what turns that into a clipped reply instead of a hung request.
    """

    def test_it_grows_with_the_text(self) -> None:
        assert _frame_budget("客廳的燈已經打開了。" * 4) > _frame_budget(
            "客廳的燈已經打開了。"
        )

    def test_short_text_still_gets_room(self) -> None:
        """The floor matters most where the estimate is least reliable."""
        assert _frame_budget("好。") >= 4 * FRAMES_PER_SECOND

    def test_empty_text_is_not_a_zero_budget(self) -> None:
        assert _frame_budget("") > 0

    def test_it_stays_under_the_models_own_limit_for_one_segment(self) -> None:
        """A segment is at most `MAX_CHARS_PER_SEGMENT`; the cap has to bite first."""
        from cortex_speech.text.pipeline import MAX_CHARS_PER_SEGMENT

        assert _frame_budget("字" * MAX_CHARS_PER_SEGMENT) < 2048
