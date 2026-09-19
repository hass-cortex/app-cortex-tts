"""Listing a model's voices must not load the model.

Measured on the live host: the integration asks for every downloaded model's
voices at startup, and `EngineRegistry.voices` used to `acquire()` the engine
to answer. With the default of one resident model each answer evicted the
previous one, so one startup produced a full load-evict cycle per model with
built-in voices — 6.4 s for the MOSS bundle, to read a list of names out of a
JSON file that was already on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex_speech import BY_ID
from cortex_speech.catalog import model_dir
from cortex_speech.engine.base import Voice, reference_voices
from cortex_speech.engine.registry import EngineRegistry, ModelNotReadyError
from cortex_speech.references import ReferenceStore

FAKE = [Voice(id="v1", name="V One", language="zh", gender="female", source="builtin")]


@pytest.fixture
def registry(tmp_path: Path) -> EngineRegistry:
    return EngineRegistry(
        tmp_path, ReferenceStore(tmp_path / "references"), max_loaded=1
    )


def _pretend_downloaded(data_dir: Path, model_id: str) -> None:
    """Lay down every file the bundle declares, empty."""
    spec = BY_ID[model_id]
    root = model_dir(data_dir, spec.id)
    for name in spec.files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


class TestBuiltinVoices:
    async def test_they_are_read_without_acquiring_an_engine(
        self, registry: EngineRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point of the change: no load, so nothing is evicted."""

        async def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError("listing voices loaded an engine")

        monkeypatch.setattr(EngineRegistry, "acquire", refuse)
        monkeypatch.setattr(
            "cortex_speech.engine.preset.own_voices", lambda _directory: FAKE
        )
        _pretend_downloaded(tmp_path, "hojo-40m")

        assert await registry.voices("hojo-40m") == FAKE
        assert not registry.is_loaded("hojo-40m")

    async def test_the_reader_is_handed_the_bundle_directory(
        self, registry: EngineRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reader pointed at the data root would find no manifest."""
        seen: list[Path] = []
        monkeypatch.setattr(
            "cortex_speech.engine.moss.own_voices",
            lambda directory: (seen.append(directory), FAKE)[1],
        )
        _pretend_downloaded(tmp_path, "moss-nano")

        await registry.voices("moss-nano")
        assert seen == [model_dir(tmp_path, "moss-nano")]

    async def test_a_model_that_is_not_downloaded_is_refused(
        self, registry: EngineRegistry
    ) -> None:
        """`acquire` used to raise this; reading files directly must too."""
        with pytest.raises(ModelNotReadyError):
            await registry.voices("hojo-40m")


class TestCloningVoices:
    async def test_a_cloning_model_needs_no_bundle_on_disk(
        self, registry: EngineRegistry
    ) -> None:
        """Its voices are the reference store, which is never in the bundle."""
        assert await registry.voices("hojo-80m-clone") == []

    async def test_a_model_with_both_kinds_concatenates_them(
        self, registry: EngineRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MOSS is the reason `voices` adds rather than chooses."""
        monkeypatch.setattr(
            "cortex_speech.engine.moss.own_voices", lambda _directory: FAKE
        )
        _pretend_downloaded(tmp_path, "moss-nano")

        voices = await registry.voices("moss-nano")
        assert [v.source for v in voices] == ["builtin"]


class TestRenamedReferences:
    """A reference's name is display only, which is what makes it editable."""

    def test_a_rename_reaches_the_voice_and_leaves_the_id_alone(
        self, tmp_path: Path, reference_wav: bytes
    ) -> None:
        """The id is what a synthesis request names; only the label moves."""
        store = ReferenceStore(tmp_path / "references")
        added = store.add(name="Anna Su", transcript="你好。", audio=reference_wav)
        assert added.id == "anna-su"

        store.update(added.id, name="蘇小姐")

        assert reference_voices(store) == [
            Voice(
                id="anna-su",
                name="蘇小姐",
                language="zh",
                gender="unknown",
                source="reference",
            )
        ]

    def test_a_rename_survives_a_reopen(
        self, tmp_path: Path, reference_wav: bytes
    ) -> None:
        """The index is what a restart reads back, so the rename is in it."""
        root = tmp_path / "references"
        added = ReferenceStore(root).add(
            name="Anna Su", transcript="你好。", audio=reference_wav
        )
        ReferenceStore(root).update(added.id, name="蘇小姐")

        reopened = ReferenceStore(root).get(added.id)
        assert reopened is not None
        assert (reopened.id, reopened.name) == ("anna-su", "蘇小姐")
