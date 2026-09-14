"""Settings come from the environment, and only the environment."""

from pathlib import Path

import pytest

from cortex_tts import config


def test_references_default_under_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing set means one directory tree; the HA app is what sets /share."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("REFERENCES_DIR", raising=False)

    assert config.load().references_dir == tmp_path / "references"


def test_references_dir_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REFERENCES_DIR", str(tmp_path / "refs"))

    assert config.load().references_dir == tmp_path / "refs"
