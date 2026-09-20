"""What a request may ask for beyond the words and the voice.

Two capabilities, declared per model so a caller hears "this model has no
such knob" rather than having the field quietly ignored — the reason MOSS's
missing temperature is an error and not a silent default.
"""

from __future__ import annotations

import dataclasses

import pytest

from cortex_speech import BY_ID, CATALOG, Delivery
from cortex_speech.engine.base import narrowing


class TestTheDeliveryItself:
    def test_it_asks_for_nothing_by_default(self) -> None:
        """An engine that reads it must see "unset", not a guess."""
        delivery = Delivery()
        assert delivery.temperature is None
        assert delivery.language is None
        assert delivery.instruct is None

    def test_it_is_frozen(self) -> None:
        """The registry fills the temperature in with `replace`, not in place:
        one request's default must not leak into the next."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            Delivery().temperature = 0.5  # type: ignore[misc]


class TestWhoTakesWhat:
    """Declared, not inferred. A caller reads this before spending a synthesis."""

    def test_only_models_that_read_several_languages_take_one(self) -> None:
        for spec in CATALOG:
            if spec.language_choice:
                assert len(spec.languages) > 2, spec.id

    def test_a_voice_bound_model_takes_no_language(self) -> None:
        """On these the voice is the language; a request naming one would be
        accepted and ignored, which is indistinguishable from working."""
        for model_id in ("hojo-40m", "moss-nano"):
            assert not BY_ID[model_id].language_choice

    def test_no_model_here_takes_an_instruction(self) -> None:
        """OmniVoice's is a closed vocabulary already surfaced as designed
        voices, and nothing else reads free text."""
        assert not {spec.id for spec in CATALOG if spec.style_instruction}

    def test_an_instruction_implies_a_language_choice(self) -> None:
        """Both come from the same checkpoint feature set; one without the
        other would mean the catalog and the engine disagree."""
        for spec in CATALOG:
            if spec.style_instruction:
                assert spec.language_choice, spec.id


class TestNarrowingALanguageTag:
    """A whole tag arrives and the engine decides how much of it it knows.

    Reducing `zh-TW` to `zh` in the caller would throw away a distinction
    before any model got to say whether it mattered — and a model does name
    things narrower than a base code: OmniVoice names 646 languages including
    Cantonese.

    One function, in `engine/base.py`, because a tag cannot mean different
    amounts depending on which model reads it. It used to be copied into both
    engines with a test pinning the copies to each other, which is a test that
    keeps duplication honest rather than removing it.
    """

    def test_it_tries_the_whole_tag_first(self) -> None:
        assert narrowing("zh-Hant-TW")[0] == "zh-hant-tw"

    def test_it_falls_back_towards_the_base(self) -> None:
        assert narrowing("zh-Hant-TW") == ["zh-hant-tw", "zh-hant", "zh"]

    def test_an_underscore_tag_is_the_same_tag(self) -> None:
        """Home Assistant writes `zh-TW`; other callers write `zh_TW`."""
        assert narrowing("en_US") == narrowing("en-US")

    def test_a_base_code_narrows_to_itself(self) -> None:
        assert narrowing("ja") == ["ja"]

    def test_the_engine_uses_the_one_function(self) -> None:
        """Not "it agrees" — the same object, so it cannot disagree."""
        from cortex_speech.engine import omni

        assert omni.narrowing is narrowing
