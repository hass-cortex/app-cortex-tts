"""`/api/events`: one socket says which of the UI's reads went stale."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cortex_speech.download import DownloadProgress
from cortex_tts.app import create_app
from cortex_tts.preferences import FILE_NAME
from cortex_tts.updates import Hub

AUTH = {"Authorization": "Bearer test-key"}


class TestTheHub:
    async def test_every_listener_hears_every_kind(self) -> None:
        hub = Hub()
        a, b = hub.subscribe(), hub.subscribe()
        hub.publish("models", "voices")
        assert [a.get_nowait(), a.get_nowait()] == ["models", "voices"]
        assert [b.get_nowait(), b.get_nowait()] == ["models", "voices"]

    async def test_a_listener_that_left_hears_nothing(self) -> None:
        hub = Hub()
        queue = hub.subscribe()
        hub.unsubscribe(queue)
        hub.publish("models")
        assert queue.empty()

    async def test_a_moving_download_ticks_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("cortex_tts.updates.DOWNLOAD_TICK_S", 0.01)
        hub = Hub()
        queue = hub.subscribe()
        progress = DownloadProgress("hojo-40m", total_files=4, state="running")

        class _Downloads:
            def all(self) -> list[DownloadProgress]:
                return [progress]

        hub.watch(_Downloads())  # type: ignore[arg-type]
        try:
            assert await asyncio.wait_for(queue.get(), 1) == "models"
            # Nothing moved: nothing said.
            await asyncio.sleep(0.05)
            assert queue.empty()
            progress.completed_files = 1
            assert await asyncio.wait_for(queue.get(), 1) == "models"
        finally:
            await hub.close()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
    monkeypatch.setenv("API_KEY", "test-key")
    (tmp_path / FILE_NAME).write_text(json.dumps({"preload": False}))
    return TestClient(create_app())


class TestTheSocket:
    def test_a_route_that_changes_something_is_heard(self, client: TestClient) -> None:
        with client, client.websocket_connect("/api/events", headers=AUTH) as ws:
            assert client.delete("/api/stats", headers=AUTH).status_code == 200
            assert ws.receive_json() == {"type": "models"}

    def test_settings_saved_is_heard_as_settings_and_models(
        self, client: TestClient
    ) -> None:
        with client, client.websocket_connect("/api/events", headers=AUTH) as ws:
            saved = client.put("/api/settings", headers=AUTH, json={"temperature": 0.7})
            assert saved.status_code == 200
            assert {ws.receive_json()["type"], ws.receive_json()["type"]} == {
                "settings",
                "models",
            }

    def test_the_key_is_required(self, client: TestClient) -> None:
        from starlette.websockets import WebSocketDisconnect

        with (
            client,
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(
                "/api/events", headers={"Authorization": "Bearer nope"}
            ) as ws,
        ):
            ws.receive_json()
