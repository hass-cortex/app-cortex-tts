"""Fixtures shared by the API tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cortex_tts.app import create_app
from cortex_tts.preferences import FILE_NAME


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client against an empty data directory with auth enabled."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
    monkeypatch.setenv("API_KEY", "test-key")
    # Preloading downloads a bundle. It is a stored setting rather than an
    # environment variable, so writing the file is the only way to say no —
    # and a fixture that gets this wrong hangs on the network rather than
    # failing, which is how this one was found.
    (tmp_path / FILE_NAME).write_text(json.dumps({"preload": False}))
    with TestClient(create_app()) as test_client:
        yield test_client
