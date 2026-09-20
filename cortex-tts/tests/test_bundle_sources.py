"""A model's bundle can span repositories.

MOSS-TTS-Nano publishes its weights and its audio codec separately and needs
both before it can speak. `ModelSpec` used to carry one `repo_id`, which is the
same shape of "too narrow" that `EngineKind` was.
"""

from __future__ import annotations

from pathlib import Path

from cortex_speech import BY_ID, CATALOG, inspect
from cortex_speech.catalog import BundleSource


class TestBundleSource:
    def test_a_source_without_a_subdir_lands_in_the_model_directory(self) -> None:
        source = BundleSource(repo_id="acme/m", files=("a.onnx", "b.json"))
        assert source.paths() == ("a.onnx", "b.json")

    def test_a_subdir_prefixes_every_file(self) -> None:
        """Two repos in one bundle must not collide on a shared file name."""
        source = BundleSource(repo_id="acme/m", files=("meta.json",), subdir="codec")
        assert source.paths() == ("codec/meta.json",)

    def test_spec_files_flattens_every_source_in_order(self) -> None:
        spec = BY_ID["moss-nano"]
        assert len(spec.sources) == 2
        assert len(spec.files) == sum(len(s.files) for s in spec.sources)
        assert spec.files[0].startswith("MOSS-TTS-Nano-100M-ONNX/")
        assert any(f.startswith("MOSS-Audio-Tokenizer-Nano-ONNX/") for f in spec.files)

    def test_single_repo_models_are_unchanged_by_the_generalisation(self) -> None:
        """The common case stays flat: no subdir, no path prefixes."""
        spec = BY_ID["hojo-40m"]
        assert len(spec.sources) == 1
        assert all("/" not in name for name in spec.files)


class TestInspect:
    def test_a_bundle_missing_its_second_repo_is_not_downloaded(
        self, tmp_path: Path
    ) -> None:
        """Half a multi-repo bundle loads into a failure deep inside ORT."""
        spec = BY_ID["moss-nano"]
        first = spec.sources[0]
        directory = tmp_path / "models" / spec.id / first.subdir
        directory.mkdir(parents=True)
        for name in first.files:
            (directory / name).write_bytes(b"x")

        state = inspect(tmp_path, spec)
        assert not state.downloaded
        assert all(
            name.startswith("MOSS-Audio-Tokenizer-Nano-ONNX/") for name in state.missing
        )

    def test_every_catalog_entry_declares_at_least_one_source(self) -> None:
        empty = [s.id for s in CATALOG if not s.sources]
        assert not empty, f"models with nothing to download: {empty}"
