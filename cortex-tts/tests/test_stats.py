"""The measurements a model card and a live reply both rest on.

Three things matter: a figure is this host's or absent — never a
plausible-looking number from somewhere else; it is one voice's own, never
pooled with another's or borrowed from one; and it is read on the execution
provider now in use, because a sample from the other provider describes
another machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex_speech import RenderSample
from cortex_tts.stats import MAX_RENDERS, MIN_SAMPLES, WINDOW, StatsStore


def _sample(rtf: float, audio: float = 4.0, provider: str = "cpu") -> RenderSample:
    return RenderSample(audio_s=audio, wall_s=rtf * audio, provider=provider)


def _measure(
    store: StatsStore,
    model: str,
    kind: str,
    voice: str = "v1",
    rtf: float = 0.5,
    provider: str = "cpu",
    count: int = MIN_SAMPLES,
) -> None:
    for _ in range(count):
        store.record(model, kind, voice, _sample(rtf, provider=provider))


@pytest.fixture
def store(tmp_path: Path) -> StatsStore:
    return StatsStore(tmp_path / "stats.json")


class TestNothingMeasuredIsNothingShown:
    def test_an_unmeasured_model_has_no_figure(self, store: StatsStore) -> None:
        assert store.get("hojo-40m") == []
        assert store.rtf("hojo-40m", "builtin", "v1", "cpu") is None

    def test_fewer_than_min_samples_is_shown_but_decides_nothing(
        self, store: StatsStore
    ) -> None:
        """A voice spoken once has a cost worth showing; a verdict needs three."""
        _measure(store, "hojo-40m", "builtin", count=MIN_SAMPLES - 1)
        [shown] = store.get("hojo-40m")
        assert (shown.samples, shown.settled) == (MIN_SAMPLES - 1, False)
        assert store.rtf("hojo-40m", "builtin", "v1", "cpu") is None

    def test_min_samples_is(self, store: StatsStore) -> None:
        _measure(store, "hojo-40m", "builtin", count=MIN_SAMPLES)
        measured = store.rtf("hojo-40m", "builtin", "v1", "cpu")
        assert measured is not None
        assert measured.samples == MIN_SAMPLES

    def test_a_request_that_produced_nothing_is_refused(
        self, store: StatsStore
    ) -> None:
        for _ in range(4):
            store.record("hojo-40m", "builtin", "v1", RenderSample(0.0, 1.0, "cpu"))
        assert store.get("hojo-40m") == []


class TestTheFigureItself:
    def test_it_is_the_median_of_the_window(self, store: StatsStore) -> None:
        """One odd request cannot define it."""
        for rtf in (0.5, 0.5, 0.5, 4.0, 0.5):
            store.record("hojo-40m", "builtin", "v1", _sample(rtf))
        measured = store.rtf("hojo-40m", "builtin", "v1", "cpu")
        assert measured is not None
        assert measured.rtf == 0.5

    def test_the_fixed_cost_is_not_subtracted(self, store: StatsStore) -> None:
        """A short sentence pays it, and a streaming reply is short sentences."""
        for _ in range(MIN_SAMPLES):
            store.record(
                "omnivoice",
                "reference",
                "c",
                RenderSample(3.0, 1.5 + 0.6 * 3.0, "cuda"),
            )
        measured = store.rtf("omnivoice", "reference", "c", "cuda")
        assert measured is not None
        assert measured.rtf == pytest.approx(1.1)

    def test_only_the_newest_window_counts(self, store: StatsStore) -> None:
        """A host that changed is followed within `WINDOW` replies."""
        _measure(store, "hojo-40m", "builtin", rtf=2.0, count=MAX_RENDERS)
        _measure(store, "hojo-40m", "builtin", rtf=0.4, count=WINDOW)
        measured = store.rtf("hojo-40m", "builtin", "v1", "cpu")
        assert measured is not None
        assert measured.rtf == 0.4
        assert measured.samples == WINDOW

    def test_only_the_most_recent_are_kept(self, store: StatsStore, tmp_path: Path):
        for i in range(MAX_RENDERS + 10):
            store.record("hojo-40m", "builtin", "v1", _sample(0.5, audio=1.0 + i * 0.1))
        stored = json.loads((tmp_path / "stats.json").read_text())
        assert len(stored["models"]["hojo-40m"]["builtin:v1"]["renders"]) == MAX_RENDERS


class TestAFigureBelongsToAProvider:
    def test_samples_from_another_provider_do_not_count(
        self, store: StatsStore
    ) -> None:
        """The card fell out: what it measured describes another machine."""
        _measure(store, "hojo-40m", "builtin", rtf=0.3, provider="cuda")
        assert store.rtf("hojo-40m", "builtin", "v1", "cpu") is None
        measured = store.rtf("hojo-40m", "builtin", "v1", "cuda")
        assert measured is not None
        assert measured.provider == "cuda"

    def test_the_provider_now_in_use_wins(self, store: StatsStore) -> None:
        _measure(store, "hojo-40m", "builtin", rtf=0.3, provider="cuda")
        _measure(store, "hojo-40m", "builtin", rtf=0.65, provider="cpu")
        assert store.rtf("hojo-40m", "builtin", "v1", "cpu").rtf == 0.65  # type: ignore[union-attr]
        assert store.rtf("hojo-40m", "builtin", "v1", "cuda").rtf == 0.3  # type: ignore[union-attr]

    def test_a_model_not_resident_is_read_on_its_last_provider(
        self, store: StatsStore
    ) -> None:
        _measure(store, "hojo-40m", "builtin", rtf=0.3, provider="cuda")
        _measure(store, "hojo-40m", "builtin", rtf=0.65, provider="cpu")
        [measured] = store.get("hojo-40m")
        assert measured.provider == "cpu"
        assert measured.rtf == 0.65


class TestOneFigurePerVoice:
    """A clone and a built-in voice are different work on the same model.

    Measured on OmniVoice, one host: designed 3.46, a clone of a ten-second
    recording 7.17. Pooled, a cheap voice could carry a dear one across the
    threshold.
    """

    def test_voices_are_kept_apart(self, store: StatsStore) -> None:
        _measure(store, "omnivoice", "designed", "female-young", rtf=0.72)
        _measure(store, "omnivoice", "reference", "hsiao-chen", rtf=0.85)
        by_voice = {m.voice: m.rtf for m in store.get("omnivoice")}
        assert by_voice == {"female-young": 0.72, "hsiao-chen": 0.85}

    def test_a_voice_never_served_has_nothing(self, store: StatsStore) -> None:
        """Nothing is borrowed: three requests of its own, or buffered."""
        _measure(store, "omnivoice", "designed", "female-young", rtf=0.72)
        assert store.rtf("omnivoice", "reference", "just-uploaded", "cpu") is None

    def test_built_in_voices_of_one_model_are_apart_too(
        self, store: StatsStore
    ) -> None:
        store.record("moss-nano", "builtin", "weiguo", _sample(0.5))
        store.record("moss-nano", "builtin", "yuewen", _sample(0.5))
        store.record("moss-nano", "builtin", "junhao", _sample(0.5))
        assert store.rtf("moss-nano", "builtin", "weiguo", "cpu") is None

    def test_they_are_returned_in_a_stable_order(self, store: StatsStore) -> None:
        """A card that reorders itself between refreshes is unreadable."""
        _measure(store, "moss-nano", "reference", "wanwan", rtf=0.37)
        _measure(store, "moss-nano", "builtin", "yuewen", rtf=0.47)
        assert [(m.kind, m.voice) for m in store.get("moss-nano")] == [
            ("builtin", "yuewen"),
            ("reference", "wanwan"),
        ]


class TestAcrossRestarts:
    def test_measurements_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "stats.json"
        _measure(StatsStore(path), "omnivoice", "designed", rtf=3.4)
        [measured] = StatsStore(path).get("omnivoice")
        assert measured.rtf == 3.4

    def test_an_unreadable_file_is_not_fatal(self, tmp_path: Path) -> None:
        """Losing measurements is recoverable; refusing to start is not."""
        path = tmp_path / "stats.json"
        path.write_text("{not json", encoding="utf-8")
        assert StatsStore(path).get("omnivoice") == []

    def test_a_file_written_by_something_else_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "stats.json"
        path.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
        assert StatsStore(path).get("omnivoice") == []

    def test_an_earlier_layout_is_dropped(self, tmp_path: Path) -> None:
        """Samples without a provider cannot be told from another machine's."""
        path = tmp_path / "stats.json"
        path.write_text(
            json.dumps(
                {
                    "hojo-40m": {
                        "builtin:v1": {"renders": [[2.0, 1.3, 8, 0], [4.0, 2.3, 16, 0]]}
                    }
                }
            ),
            encoding="utf-8",
        )
        store = StatsStore(path)
        assert store.get("hojo-40m") == []
        store.record("hojo-40m", "builtin", "v1", _sample(0.5))
        stored = json.loads(path.read_text())
        assert stored["format"] == 2
        assert len(stored["models"]["hojo-40m"]["builtin:v1"]["renders"]) == 1


class TestDeletingAModel:
    def test_it_forgets_what_was_measured(self, store: StatsStore) -> None:
        _measure(store, "omnivoice", "designed")
        store.forget("omnivoice")
        assert store.get("omnivoice") == []
        assert store.rtf("omnivoice", "designed", "v1", "cpu") is None

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
