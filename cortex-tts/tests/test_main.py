"""How the server is launched."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cortex_tts import __main__ as entry
from cortex_tts.preferences import Preferences


class TestShutdown:
    """A synthesis is a worker thread that cannot be interrupted, and uvicorn
    waits for in-flight requests before shutting down. Seen as a stop that
    took seven minutes behind one long CPU render; past the cap the request
    is cancelled and the lifespan closes the engines."""

    def test_the_server_stops_waiting_for_a_render_after_the_cap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}
        monkeypatch.setattr(entry.uvicorn, "run", lambda app, **kw: seen.update(kw))
        monkeypatch.setattr(
            entry.config,
            "load",
            lambda: SimpleNamespace(host="127.0.0.1", port=1, data_dir=tmp_path),
        )
        monkeypatch.setattr(entry.preferences, "load", lambda _dir: Preferences())

        entry.main()

        assert seen["timeout_graceful_shutdown"] == entry.GRACEFUL_SHUTDOWN_SECONDS
        assert 0 < entry.GRACEFUL_SHUTDOWN_SECONDS <= 30
