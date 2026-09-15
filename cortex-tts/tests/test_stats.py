"""The measurements a model card and the planner both rest on.

Three things matter: a figure is this host's or absent — never a
plausible-looking number from somewhere else; it is fitted rather than
averaged, so neither a short request nor a reply split into many can move it;
and a clone is never mixed with a designed or bundled voice, because on the
same model they cost about twice one another.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex_speech import RenderSample
from cortex_tts.stats import MAX_RENDERS, StatsStore


def _sample(audio: float, fixed: float = 0.3, factor: float = 0.5) -> RenderSample:
    return RenderSample(
        audio_s=audio, wall_s=fixed + factor * audio, cjk=int(audio * 4), latin=0
    )


def _measure(store: StatsStore, model: str, kind: str, **shape: float) -> None:
    """Enough requests, of different lengths, to fit a line."""
    for audio in (2.0, 4.0, 8.0):
        store.record(model, kind, _sample(audio, **shape))


@pytest.fixture
def store(tmp_path: Path) -> StatsStore:
    return StatsStore(tmp_path / "stats.json")


class TestNothingMeasuredIsNothingShown:
    def test_an_unmeasured_model_has_no_figure(self, store: StatsStore) -> None:
        assert store.get("hojo-40m") == []

    def test_one_request_is_not_a_line(self, store: StatsStore) -> None:
        """Better no number than one drawn through a single point."""
        store.record("hojo-40m", "builtin", _sample(3.0))
        store.record("hojo-40m", "builtin", _sample(5.0))
        assert store.get("hojo-40m") == []
        assert store.render_model("hojo-40m", "builtin") is None

    def test_a_request_that_produced_nothing_is_refused(
        self, store: StatsStore
    ) -> None:
        for _ in range(4):
            store.record("hojo-40m", "builtin", RenderSample(0.0, 1.0, 4, 0))
        assert store.get("hojo-40m") == []


class TestTheFigureItself:
    def test_it_is_the_slope_not_the_average_cost(self, store: StatsStore) -> None:
        """The fixed cost every request pays is held out, not averaged in.

        A request producing 2 s of audio here costs 0.3 + 0.5 x 2 = 1.3 s, an
        average cost of 0.65 per audio second. The factor is 0.5, and it is
        the factor that says whether the model keeps up on long text.
        """
        _measure(store, "hojo-40m", "builtin")
        fit = store.get("hojo-40m")[0].render
        assert abs(fit.per_audio - 0.5) < 0.02
        assert abs(fit.fixed_s - 0.3) < 0.05

    def test_a_reply_split_into_many_requests_does_not_skew_it(
        self, store: StatsStore
    ) -> None:
        """The reason the card is fitted at all.

        A paced reply arrives as a burst of short requests of one length.
        Averaged, each would carry the whole fixed cost and read high; fitted,
        they add a point at one end of a line the longer ones already define.
        """
        _measure(store, "moss-nano", "builtin")
        for _ in range(8):
            store.record("moss-nano", "builtin", _sample(2.0))
        fit = store.get("moss-nano")[0].render
        assert abs(fit.per_audio - 0.5) < 0.05

    def test_it_follows_a_host_that_changed(self, store: StatsStore) -> None:
        """Only the last `MAX_RENDERS` count, so a moved model is re-measured."""
        for _ in range(MAX_RENDERS):
            _measure(store, "hojo-40m", "builtin", factor=2.0)
        for _ in range(MAX_RENDERS):
            _measure(store, "hojo-40m", "builtin", factor=0.4)
        assert store.get("hojo-40m")[0].render.per_audio < 0.6

    def test_only_the_most_recent_are_kept(self, store: StatsStore, tmp_path: Path):
        for i in range(MAX_RENDERS + 10):
            store.record("hojo-40m", "builtin", _sample(1.0 + i * 0.1))
        stored = json.loads((tmp_path / "stats.json").read_text())
        assert len(stored["hojo-40m"]["builtin"]["renders"]) == MAX_RENDERS


class TestAcrossRestarts:
    def test_measurements_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "stats.json"
        _measure(StatsStore(path), "omnivoice", "designed", factor=3.4)
        assert abs(StatsStore(path).get("omnivoice")[0].render.per_audio - 3.4) < 0.05

    def test_an_unreadable_file_is_not_fatal(self, tmp_path: Path) -> None:
        """Losing measurements is recoverable; refusing to start is not."""
        path = tmp_path / "stats.json"
        path.write_text("{not json", encoding="utf-8")
        assert StatsStore(path).get("omnivoice") == []

    def test_a_file_written_by_something_else_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "stats.json"
        path.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
        assert StatsStore(path).get("omnivoice") == []

    def test_a_file_that_only_kept_ratios_is_dropped(self, tmp_path: Path) -> None:
        """Both earlier layouts stored what a division left behind.

        The audio and wall seconds they came from are not in the file, so
        there is no line to fit and nothing worth carrying forward.
        """
        path = tmp_path / "stats.json"
        path.write_text(
            json.dumps(
                {
                    "omnivoice": [3.4, 7.2],
                    "hojo-40m": {"builtin": {"rtf": [0.6, 0.7]}},
                }
            ),
            encoding="utf-8",
        )
        store = StatsStore(path)
        assert store.get("omnivoice") == []
        assert store.get("hojo-40m") == []


class TestDeletingAModel:
    def test_it_forgets_what_was_measured(self, store: StatsStore) -> None:
        _measure(store, "omnivoice", "designed")
        store.forget("omnivoice")
        assert store.get("omnivoice") == []
        assert store.render_model("omnivoice", "designed") is None

    def test_forgetting_one_leaves_the_others(self, store: StatsStore) -> None:
        _measure(store, "omnivoice", "designed")
        _measure(store, "hojo-40m", "builtin")
        store.forget("omnivoice")
        assert store.get("hojo-40m") != []


class TestResettingEverything:
    def test_clear_forgets_every_model(self, store: StatsStore) -> None:
        _measure(store, "omnivoice", "designed")
        _measure(store, "hojo-40m", "builtin")
        store.clear()
        assert store.get("omnivoice") == []
        assert store.get("hojo-40m") == []

    def test_a_cleared_store_stays_cleared_across_a_restart(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "stats.json"
        first = StatsStore(path)
        _measure(first, "omnivoice", "designed")
        first.clear()
        assert StatsStore(path).get("omnivoice") == []


class TestOneFigurePerKind:
    """A clone and a designed voice are different work on the same model.

    Measured on OmniVoice, one host: designed 3.46, a clone of a ten-second
    recording 7.17. Averaged, the card would describe neither.
    """

    def test_kinds_are_kept_apart(self, store: StatsStore) -> None:
        _measure(store, "omnivoice", "designed", factor=3.46)
        _measure(store, "omnivoice", "reference", factor=7.17)
        measured = {m.kind: m.render.per_audio for m in store.get("omnivoice")}
        assert abs(measured["designed"] - 3.46) < 0.05
        assert abs(measured["reference"] - 7.17) < 0.05

    def test_a_kind_never_served_has_no_fit(self, store: StatsStore) -> None:
        _measure(store, "moss-nano", "reference", factor=1.2)
        assert store.render_model("moss-nano", "builtin") is None
        assert store.render_model("moss-nano", "reference") is not None

    def test_they_are_returned_in_a_stable_order(self, store: StatsStore) -> None:
        """A card that reorders itself between refreshes is unreadable."""
        _measure(store, "moss-nano", "reference", factor=1.4)
        _measure(store, "moss-nano", "builtin", factor=1.07)
        assert [m.kind for m in store.get("moss-nano")] == ["builtin", "reference"]

    def test_forgetting_a_model_drops_every_kind(self, store: StatsStore) -> None:
        _measure(store, "omnivoice", "designed")
        _measure(store, "omnivoice", "reference")
        store.forget("omnivoice")
        assert store.get("omnivoice") == []
