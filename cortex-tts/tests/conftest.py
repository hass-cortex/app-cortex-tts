"""Fixtures shared by the API tests."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from cortex_tts.api.deps import INGRESS_PEER
from cortex_tts.app import create_app
from cortex_tts.preferences import FILE_NAME


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty data directory with auth enabled."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
    monkeypatch.setenv("API_KEY", "test-key")
    # Preloading downloads a bundle. It is a stored setting rather than an
    # environment variable, so writing the file is the only way to say no —
    # and a fixture that gets this wrong hangs on the network rather than
    # failing, which is how this one was found.
    (tmp_path / FILE_NAME).write_text(json.dumps({"preload": False}))


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client on the published port: some LAN peer, key required."""
    _configure(tmp_path, monkeypatch)
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def ingress_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A client connecting from the Supervisor's address, as ingress does."""
    _configure(tmp_path, monkeypatch)
    with TestClient(create_app(), client=(INGRESS_PEER, 40000)) as test_client:
        yield test_client


@pytest.fixture
def reference_wav() -> bytes:
    """A quiet three-second tone: long enough to pass the reference validator."""
    rate = 16000
    t = np.arange(3 * rate) / rate
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, rate, format="WAV")
    return buffer.getvalue()
