"""OmniVoice's designed voices.

A designed voice is a string handed to the model, and the model validates it
against a closed vocabulary — so a typo here is not a worse voice, it is a
`ValueError` raised in the middle of a reply. That vocabulary ships in the
vendored code, which makes the check exact and free: `voice_design.py` imports
nothing but `re`, so none of this loads torch.
"""

from __future__ import annotations

import re

import pytest

from cortex_speech.engine.omni import _DESIGNED_VOICES, own_voices
from cortex_speech.vendor.omnivoice.voice_design import (
    _INSTRUCT_MUTUALLY_EXCLUSIVE,
    _INSTRUCT_VALID_EN,
)

VOICE_IDS = sorted(_DESIGNED_VOICES)


class TestEveryDesignedVoiceIsSayable:
    @pytest.mark.parametrize("voice_id", VOICE_IDS)
    def test_its_attributes_are_in_the_models_vocabulary(self, voice_id: str) -> None:
        instruct = _DESIGNED_VOICES[voice_id][0]
        unknown = [
            item.strip()
            for item in instruct.split(",")
            if item.strip() not in _INSTRUCT_VALID_EN
        ]
        assert not unknown, f"{voice_id}: {unknown} not in the instruct vocabulary"

    @pytest.mark.parametrize("voice_id", VOICE_IDS)
    def test_it_names_each_category_at_most_once(self, voice_id: str) -> None:
        """Two ages or two pitches in one instruct is refused by the model."""
        items = [item.strip() for item in _DESIGNED_VOICES[voice_id][0].split(",")]
        for category in _INSTRUCT_MUTUALLY_EXCLUSIVE:
            assert len([item for item in items if item in category]) <= 1

    @pytest.mark.parametrize("voice_id", VOICE_IDS)
    def test_its_id_is_a_slug(self, voice_id: str) -> None:
        """Voice ids travel in URLs and in Home Assistant entity ids."""
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", voice_id)


class TestListingThem:
    def test_it_opens_nothing(self, tmp_path) -> None:
        """The set is chosen here, not read from the bundle — so an empty
        directory still lists every voice."""
        assert len(own_voices(tmp_path)) == len(_DESIGNED_VOICES)

    def test_they_claim_no_language(self, tmp_path) -> None:
        """The attributes say nothing about one; the text decides."""
        assert all(voice.language is None for voice in own_voices(tmp_path))

    def test_they_are_offered_as_designed(self, tmp_path) -> None:
        """Not `builtin`: a designed voice costs about half what a clone does
        on this model, and a caller comparing them has to be able to tell."""
        assert all(voice.source == "designed" for voice in own_voices(tmp_path))

    def test_both_sexes_are_offered(self, tmp_path) -> None:
        genders = {voice.gender for voice in own_voices(tmp_path)}
        assert genders == {"female", "male"}

    def test_each_carries_a_label_that_is_not_its_id(self, tmp_path) -> None:
        assert all(
            voice.name and voice.name != voice.id for voice in own_voices(tmp_path)
        )
