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
from cortex_tts.preferences import FILE_NAME, Preferences, load, save


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

    @pytest.mark.parametrize("value", [-1, 3601])
    def test_max_synthesis_outside_the_range_is_refused(self, value: int) -> None:
        assert (
            Preferences().merged({"max_synthesis_seconds": value}).max_synthesis_seconds
            == 0
        )

    def test_max_synthesis_is_off_unless_asked_for(self) -> None:
        assert Preferences().max_synthesis_seconds == 0
        assert (
            Preferences().merged({"max_synthesis_seconds": 120}).max_synthesis_seconds
            == 120
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
            {"max_synthesis_seconds": 120},
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
