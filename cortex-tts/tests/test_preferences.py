"""Settings the user changes while the app runs.

These were addon options, where every change cost a restart and a change to
the list of choices cost a rebuild — the kind that takes a service offline if
the build then fails. None of them need that: most are read afresh on the next
request, and the three bound when a session is created are adopted by dropping
what is resident.

What stayed an addon option is what must be settled before the process starts:
the log level, and the key the Supervisor pushes through discovery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex_speech import CATALOG
from cortex_tts.preferences import FILE_NAME, Preferences, TextRule, load, save


class TestStoring:
    def test_nothing_stored_yet_is_the_shipped_defaults(self, tmp_path: Path) -> None:
        assert load(tmp_path) == Preferences()

    def test_what_was_saved_comes_back(self, tmp_path: Path) -> None:
        save(tmp_path, Preferences(default_model="moss-nano", num_threads=4))
        restored = load(tmp_path)
        assert restored.default_model == "moss-nano"
        assert restored.num_threads == 4

    def test_a_corrupt_file_does_not_stop_the_app(self, tmp_path: Path) -> None:
        """Starting on defaults is recoverable from the UI; refusing to start
        leaves someone with no voice and no way to fix it."""
        (tmp_path / FILE_NAME).write_text("{ this is not json")
        assert load(tmp_path) == Preferences()

    def test_the_write_is_atomic(self, tmp_path: Path) -> None:
        """A half-written file is the corrupt one above, on every restart."""
        save(tmp_path, Preferences())
        assert not list(tmp_path.glob("*.tmp"))


class TestValidation:
    def test_one_bad_field_leaves_the_others_alone(self) -> None:
        """The whole reason merges are per-field: a typo in a number must not
        discard the model someone chose in another box."""
        merged = Preferences().merged(
            {"num_threads": "lots", "default_model": "moss-nano"}
        )
        assert merged.num_threads == Preferences().num_threads
        assert merged.default_model == "moss-nano"

    @pytest.mark.parametrize("value", [-1, 17])
    def test_threads_outside_the_range_are_refused(self, value: int) -> None:
        assert Preferences().merged({"num_threads": value}).num_threads == 2

    @pytest.mark.parametrize("offset", [-1, 1])
    def test_resident_models_outside_the_range_are_refused(self, offset: int) -> None:
        """The range is 1..catalog size, so the bounds move when it grows."""
        value = 1 + offset if offset < 0 else len(CATALOG) + offset
        assert Preferences().merged({"max_loaded_models": value}).max_loaded_models == 1

    @pytest.mark.parametrize("value", [0.05, 3.5])
    def test_a_threshold_outside_the_range_is_refused(self, value: float) -> None:
        assert Preferences().merged({"stream_rtf": value}).stream_rtf == 0.8

    def test_a_threshold_within_it_is_kept(self) -> None:
        assert Preferences().merged({"stream_rtf": 1.2}).stream_rtf == 1.2

    @pytest.mark.parametrize("value", [-1, 86401])
    def test_idle_unload_outside_the_range_is_refused(self, value: int) -> None:
        assert (
            Preferences().merged({"idle_unload_seconds": value}).idle_unload_seconds
            == 0
        )

    def test_idle_unload_is_off_unless_asked_for(self) -> None:
        assert Preferences().idle_unload_seconds == 0
        assert (
            Preferences().merged({"idle_unload_seconds": 300}).idle_unload_seconds
            == 300
        )

    def test_every_model_may_be_resident_at_once(self) -> None:
        assert Preferences().merged(
            {"max_loaded_models": len(CATALOG)}
        ).max_loaded_models == len(CATALOG)

    def test_an_unknown_model_is_refused(self) -> None:
        """It would leave the app defaulting to something that cannot exist."""
        assert Preferences().merged({"default_model": "gpt-9"}).default_model == (
            "hojo-40m"
        )

    def test_an_unknown_execution_provider_is_refused(self) -> None:
        assert (
            Preferences().merged({"execution_provider": "tpu"}).execution_provider
            == "auto"
        )

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_temperature_outside_the_range_is_refused(self, value: float) -> None:
        assert Preferences().merged({"temperature": value}).temperature == 0.8

    def test_a_voice_is_taken_as_given(self) -> None:
        """A voice only exists once its model is downloaded; refusing an
        unknown one would make the field impossible to set beforehand."""
        assert Preferences().merged({"default_voice": "Yuewen"}).default_voice == (
            "Yuewen"
        )

    def test_a_field_this_version_does_not_know_is_ignored(self) -> None:
        """A newer UI must not be able to inject keys into the dataclass."""
        assert Preferences().merged({"invented": 1}) == Preferences()


class TestWhatNeedsARebuild:
    """Which changes a loaded engine cannot adopt.

    Thread count and execution provider are bound when ONNX Runtime creates
    a session. The rest are read again on the next request, so dropping
    engines for them would be a rebuild nobody asked for.
    """

    @pytest.mark.parametrize(
        "change",
        [
            {"num_threads": 4},
            {"execution_provider": "cuda"},
        ],
    )
    def test_session_bound_settings_need_one(self, change: dict[str, object]) -> None:
        base = Preferences()
        assert base.rebuild_needed(base.merged(change))

    @pytest.mark.parametrize(
        "change",
        [
            {"default_model": "moss-nano"},
            {"default_voice": "Yuewen"},
            {"temperature": 0.0},
            {"preload": False},
            {"max_loaded_models": 2},
            {"idle_unload_seconds": 300},
        ],
    )
    def test_the_rest_take_effect_on_the_next_request(
        self, change: dict[str, object]
    ) -> None:
        base = Preferences()
        assert not base.rebuild_needed(base.merged(change))

    def test_changing_nothing_needs_nothing(self) -> None:
        base = Preferences()
        assert not base.rebuild_needed(base.merged({}))


class TestHandEditedFiles:
    """settings.json is a file people open in an editor."""

    def test_a_non_utf8_file_starts_with_defaults(self, tmp_path: Path) -> None:
        (tmp_path / FILE_NAME).write_bytes(b"\xff\xfe{}")
        assert load(tmp_path) == Preferences()

    @pytest.mark.parametrize(
        ("raw", "expected"), [("false", False), ("true", True), (False, False)]
    )
    def test_preload_reads_the_strings_an_editor_leaves(
        self, raw: object, expected: bool
    ) -> None:
        assert Preferences().merged({"preload": raw}).preload is expected

    def test_a_word_that_is_not_a_boolean_is_refused(self) -> None:
        prefs, ignored = Preferences().validated({"preload": "maybe"})
        assert prefs.preload is True
        assert ignored == ["preload"]


class TestTextRules:
    """What the switches default to, per model and language."""

    def test_no_rules_leave_every_switch_to_the_pipeline(self) -> None:
        assert Preferences().text_defaults("hojo-40m", "zh-TW") == TextRule()

    def test_the_most_specific_rule_wins_per_switch(self) -> None:
        prefs = Preferences().merged(
            {
                "text_rules": [
                    {"expand_numbers": True, "taiwan_readings": False},
                    {"language": "zh", "taiwan_readings": True},
                    {"model": "hojo-40m", "expand_numbers": False},
                    {"model": "hojo-40m", "language": "zh-TW", "normalize_text": False},
                ]
            }
        )
        settled = prefs.text_defaults("hojo-40m", "zh-TW")
        assert settled == TextRule(
            normalize_text=False, expand_numbers=False, taiwan_readings=True
        )
        # Another model sees only the rules that cover it.
        assert prefs.text_defaults("moss-nano", "zh-CN") == TextRule(
            expand_numbers=True, taiwan_readings=True
        )
        assert prefs.text_defaults("moss-nano", "en") == TextRule(
            expand_numbers=True, taiwan_readings=False
        )

    def test_a_language_covers_the_tags_it_prefixes(self) -> None:
        rule = TextRule(language="zh-Hant")
        assert rule.covers("moss-nano", "zh-Hant-TW")
        assert rule.covers("moss-nano", "ZH-HANT")
        assert not rule.covers("moss-nano", "zh-Hans")
        assert not rule.covers("moss-nano", "zh")

    def test_a_later_rule_beats_an_equal_one(self) -> None:
        prefs = Preferences().merged(
            {"text_rules": [{"expand_numbers": True}, {"expand_numbers": False}]}
        )
        assert prefs.text_defaults("hojo-40m", "en").expand_numbers is False

    def test_a_rule_naming_an_unknown_model_refuses_the_list(self) -> None:
        prefs, ignored = Preferences().validated(
            {"text_rules": [{"model": "nope", "expand_numbers": True}]}
        )
        assert ignored == ["text_rules"]
        assert prefs.text_rules == ()

    def test_hand_edited_values_are_read(self) -> None:
        prefs = Preferences().merged(
            {
                "text_rules": [
                    {"language": "zh_TW", "expand_numbers": "true", "model": ""}
                ]
            }
        )
        assert prefs.text_rules == (TextRule(language="zh-TW", expand_numbers=True),)

    def test_rules_survive_a_round_trip(self, tmp_path: Path) -> None:
        prefs = Preferences(
            text_rules=(TextRule(model="hojo-40m", expand_numbers=False),)
        )
        save(tmp_path, prefs)
        assert load(tmp_path) == prefs

    def test_every_setting_can_actually_be_written(self) -> None:
        """A stored field is only real if all three layers carry it.

        The dataclass holds it, `SettingsUpdate` has to accept it over the
        wire, and `_VALIDATORS` has to let it through — a field missing from
        either of the last two is stored as its default forever, and the PUT
        answers 200 with `ignored: []` while dropping the value. That is how
        the opening-wait setting shipped broken for one deploy, before it was
        measured to decide nothing and removed.
        """
        from dataclasses import fields

        from cortex_tts.api.schemas import SettingsUpdate
        from cortex_tts.preferences import _VALIDATORS

        stored = {f.name for f in fields(Preferences)}
        assert stored <= set(SettingsUpdate.model_fields), (
            "not settable over the API: "
            f"{sorted(stored - set(SettingsUpdate.model_fields))}"
        )
        assert stored <= set(_VALIDATORS), (
            f"silently dropped by validated(): {sorted(stored - set(_VALIDATORS))}"
        )
