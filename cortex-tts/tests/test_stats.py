"""The measurements a model card rests on.

Three things matter: a figure is this host's or absent — never a
plausible-looking number from somewhere else; one odd reply cannot define it;
and a clone is never averaged with a designed or bundled voice, because on the
same model they cost about twice one another.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex_tts.stats import MAX_SAMPLES, MIN_AUDIO_SECONDS, StatsStore


@pytest.fixture
def store(tmp_path: Path) -> StatsStore:
    return StatsStore(tmp_path / "stats.json")


class TestNothingMeasuredIsNothingShown:
    def test_an_unmeasured_model_has_no_figure(self, store: StatsStore) -> None:
        assert store.get("hojo-40m") == []

    def test_a_refused_sample_leaves_it_unmeasured(self, store: StatsStore) -> None:
        """Better no number than one the host did not really produce."""
        store.record("hojo-40m", "builtin", 3.0, MIN_AUDIO_SECONDS / 2)
        assert store.get("hojo-40m") == []


class TestWhatCounts:
    def test_a_real_synthesis_is_kept(self, store: StatsStore) -> None:
        store.record("hojo-40m", "builtin", 0.55, 4.0)
        measured = store.get("hojo-40m")
        assert len(measured) == 1
        assert measured[0].kind == "builtin"
        assert measured[0].rtf == 0.55
        assert measured[0].count == 1

    def test_a_clip_too_short_to_be_typical_is_not(self, store: StatsStore) -> None:
        """Fixed per-synthesis cost dominates a very short reply."""
        store.record("hojo-40m", "builtin", 0.55, 4.0)
        store.record("hojo-40m", "builtin", 9.0, 0.4)
        assert store.get("hojo-40m")[0].count == 1

    def test_a_nonsense_rtf_is_not(self, store: StatsStore) -> None:
        store.record("hojo-40m", "builtin", 0.0, 4.0)
        assert store.get("hojo-40m") == []


class TestTheFigureItself:
    def test_it_is_the_median_not_the_last(self, store: StatsStore) -> None:
        """One slow reply — a busy host, a cold cache — must not define it."""
        for rtf in (0.50, 0.52, 0.54, 9.00):
            store.record("hojo-40m", "builtin", rtf, 4.0)
        assert store.get("hojo-40m")[0].rtf == 0.53

    def test_it_follows_a_host_that_changed(self, store: StatsStore) -> None:
        """Older samples fall off, so a thread or provider change shows."""
        for _ in range(MAX_SAMPLES):
            store.record("moss-nano", "builtin", 1.07, 4.0)
        for _ in range(MAX_SAMPLES):
            store.record("moss-nano", "builtin", 0.37, 4.0)
        measured = store.get("moss-nano")[0]
        assert measured.count == MAX_SAMPLES
        assert measured.rtf == 0.37


class TestAcrossRestarts:
    def test_measurements_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "stats.json"
        StatsStore(path).record("omnivoice", "designed", 3.46, 7.0)
        assert StatsStore(path).get("omnivoice")[0].rtf == 3.46

    def test_an_unreadable_file_is_not_fatal(self, tmp_path: Path) -> None:
        """Losing measurements is recoverable; refusing to start is not."""
        path = tmp_path / "stats.json"
        path.write_text("{not json", encoding="utf-8")
        assert StatsStore(path).get("omnivoice") == []

    def test_a_file_written_by_something_else_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "stats.json"
        path.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
        assert StatsStore(path).get("omnivoice") == []


class TestDeletingAModel:
    def test_it_forgets_what_was_measured(self, store: StatsStore) -> None:
        store.record("omnivoice", "designed", 3.46, 7.0)
        store.forget("omnivoice")
        assert store.get("omnivoice") == []

    def test_forgetting_one_leaves_the_others(self, store: StatsStore) -> None:
        store.record("omnivoice", "designed", 3.46, 7.0)
        store.record("hojo-40m", "builtin", 0.55, 4.0)
        store.forget("omnivoice")
        assert store.get("hojo-40m") != []


class TestResettingEverything:
    def test_clear_forgets_every_model(self, store: StatsStore) -> None:
        store.record("omnivoice", "designed", 3.46, 7.0)
        store.record("hojo-40m", "builtin", 0.55, 4.0)
        store.clear()
        assert store.get("omnivoice") == []
        assert store.get("hojo-40m") == []

    def test_a_cleared_store_stays_cleared_across_a_restart(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "stats.json"
        first = StatsStore(path)
        first.record("omnivoice", "designed", 3.46, 7.0)
        first.clear()
        assert StatsStore(path).get("omnivoice") == []


class TestOneFigurePerKind:
    """A clone and a designed voice are different work on the same model.

    Measured on OmniVoice, one host: designed 3.46, a clone of a ten-second
    recording 7.17. Averaged, the card would describe neither.
    """

    def test_kinds_are_counted_apart(self, store: StatsStore) -> None:
        store.record("omnivoice", "designed", 3.46, 7.0)
        store.record("omnivoice", "reference", 7.17, 7.0)
        measured = {m.kind: m.rtf for m in store.get("omnivoice")}
        assert measured == {"designed": 3.46, "reference": 7.17}

    def test_they_are_returned_in_a_stable_order(self, store: StatsStore) -> None:
        """A card that reorders itself between refreshes is unreadable."""
        store.record("moss-nano", "reference", 1.4, 7.0)
        store.record("moss-nano", "builtin", 1.07, 7.0)
        assert [m.kind for m in store.get("moss-nano")] == ["builtin", "reference"]

    def test_forgetting_a_model_drops_every_kind(self, store: StatsStore) -> None:
        store.record("omnivoice", "designed", 3.46, 7.0)
        store.record("omnivoice", "reference", 7.17, 7.0)
        store.forget("omnivoice")
        assert store.get("omnivoice") == []

    def test_a_flat_file_from_the_earlier_layout_is_dropped(
        self, tmp_path: Path
    ) -> None:
        """It averaged the kinds, which is what this replaced."""
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"omnivoice": [3.4, 7.2]}), encoding="utf-8")
        assert StatsStore(path).get("omnivoice") == []
