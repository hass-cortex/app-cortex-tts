"""`/api/speak/live` over a fake engine: the frames, the pacing, the leaving.

The engine is a stand-in that renders silence at a chosen speed so the
session's behaviour — what it sends, in what order, and when it stops
rendering — is observable without a bundle.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cortex_speech import BY_ID, RenderSample
from cortex_speech.catalog import model_dir
from cortex_speech.engine.base import (
    Delivery,
    StopCheck,
    Synthesis,
    Voice,
    check_stop,
)
from cortex_tts.app import create_app
from cortex_tts.preferences import FILE_NAME
from cortex_tts.stats import FILE_NAME as STATS_FILE
from cortex_tts.stats import StatsStore

AUTH = {"Authorization": "Bearer test-key"}
MODEL = "hojo-40m"
VOICE = Voice(id="v1", name="V1", language="zh", gender="female", source="builtin")


class _FakeEngine:
    """Renders 0.25 s of silence per character, in units it can stop between."""

    sample_rate = 24000
    provider = "cpu"

    def __init__(self, unit_seconds: float = 0.0) -> None:
        self.unit_seconds = unit_seconds
        self.units = 0
        self.calls: list[list[str]] = []
        self.entered = threading.Event()

    def synthesize(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Synthesis:
        del voice, delivery
        self.calls.append(segments)
        self.entered.set()
        waves = []
        for segment in segments:
            for _ in range(4):
                check_stop(stop)
                time.sleep(self.unit_seconds)
                self.units += 1
            waves.append(
                np.zeros(int(0.25 * len(segment) * self.sample_rate), np.float32)
            )
        return Synthesis(
            audio=np.concatenate(waves),
            sample_rate=self.sample_rate,
            segments=len(segments),
            inference_ms=1.0,
        )

    def forget(self, reference_id: str) -> None:
        del reference_id

    def close(self) -> None:
        pass


def _pretend_downloaded(data_dir: Path, model_id: str) -> None:
    spec = BY_ID[model_id]
    root = model_dir(data_dir, spec.id)
    for name in spec.files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _measured(data_dir: Path) -> None:
    """Pretend this host has seen the model: fast, with a small fixed cost."""
    store = StatsStore(data_dir / STATS_FILE)
    for audio in (2.0, 4.0, 8.0):
        store.record(
            MODEL, "builtin", RenderSample(audio, 0.2 + 0.3 * audio, int(audio * 4), 0)
        )


@pytest.fixture
def engine() -> _FakeEngine:
    return _FakeEngine()


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: _FakeEngine
) -> TestClient:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
    monkeypatch.setenv("API_KEY", "test-key")
    (tmp_path / FILE_NAME).write_text(json.dumps({"preload": False}))
    _pretend_downloaded(tmp_path, MODEL)
    monkeypatch.setattr(
        "cortex_speech.engine.registry.build", lambda backend, context: engine
    )
    monkeypatch.setattr(
        "cortex_speech.engine.registry.own_voices", lambda backend, path: [VOICE]
    )
    return TestClient(create_app())


def _speak(client: TestClient, text: str, **start) -> tuple[dict, bytes, dict]:
    """One whole reply over the socket: the ready frame, the audio, the done frame."""
    with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
        ws.send_json({"type": "start", "model": MODEL, "format": "wav", **start})
        ready = ws.receive_json()
        ws.send_json({"type": "text", "text": text})
        ws.send_json({"type": "end"})
        audio = b""
        batches: list[dict] = []
        while True:
            frame = ws.receive()
            if "bytes" in frame and frame["bytes"] is not None:
                audio += frame["bytes"]
            elif "text" in frame and frame["text"] is not None:
                message = json.loads(frame["text"])
                if message["type"] == "batch":
                    batches.append(message)
                    continue
                return ready, audio, message
            else:
                raise AssertionError(f"unexpected frame {frame!r}")


class TestBatchFrames:
    def test_each_request_is_announced_before_its_audio(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        _measured(tmp_path)
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
            ws.send_json({"type": "start", "model": MODEL, "format": "wav"})
            ws.receive_json()
            ws.send_json({"type": "text", "text": text})
            ws.send_json({"type": "end"})
            kinds: list[str] = []
            while True:
                frame = ws.receive()
                if frame.get("bytes") is not None:
                    kinds.append("audio")
                    continue
                message = json.loads(frame["text"])
                kinds.append(message["type"])
                if message["type"] == "batch":
                    assert message["mode"] in ("streaming", "paced", "buffered")
                    assert message["index"] == kinds.count("batch")
                if message["type"] == "done":
                    break
        assert kinds[0] == "batch"
        assert kinds.count("batch") == json.loads(frame["text"])["batches"]


class TestOpening:
    def test_the_first_frame_must_be_a_start_frame(self, client: TestClient) -> None:
        with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
            ws.send_json({"type": "text", "text": "hi"})
            error = ws.receive_json()
            assert error["type"] == "error"

    def test_a_wrong_key_is_refused_before_anything(self, client: TestClient) -> None:
        with (
            client,
            pytest.raises(WebSocketDisconnect) as refused,
            client.websocket_connect(
                "/api/speak/live", headers={"Authorization": "Bearer nope"}
            ) as ws,
        ):
            ws.send_json({"type": "start"})
            ws.receive_json()
        assert refused.value.code == 1008

    def test_an_unknown_voice_is_an_error_frame(self, client: TestClient) -> None:
        with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
            ws.send_json({"type": "start", "model": MODEL, "voice": "nobody"})
            error = ws.receive_json()
            assert error == {
                "type": "error",
                "code": "UNKNOWN_VOICE",
                "message": error["message"],
            }

    def test_ready_names_what_was_settled(self, client: TestClient) -> None:
        ready, _, _ = _speak(client, "好了。")
        assert ready["type"] == "ready"
        assert ready["model"] == MODEL
        assert ready["voice"] == "v1"
        assert ready["bitrate"] == 24000 * 16
        assert ready["chunk_streaming"] is False


class TestUnmeasuredHostIsBuffered:
    def test_everything_arrives_after_the_end_in_one_request(
        self, client: TestClient, engine: _FakeEngine
    ) -> None:
        ready, audio, done = _speak(client, "從前有一座山。山上有一間小廟。")
        assert ready["mode"] == "buffered"
        assert done["mode"] == "whole"
        assert done["batches"] == 1
        assert len(engine.calls) == 1
        # 44-byte WAV header plus the audio
        assert len(audio) > 44
        assert done["audio_seconds"] > 0


class TestMeasuredHostStreams:
    def test_the_reply_goes_out_in_more_than_one_request(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        _measured(tmp_path)
        # Slow enough that the later requests land after the opening hold has
        # released, so the lead is measured rather than everything arriving
        # in one release.
        engine.unit_seconds = 0.2
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        ready, audio, done = _speak(client, text)
        assert ready["mode"] == "streaming"
        assert done["batches"] >= 2
        assert len(engine.calls) == done["batches"]
        assert done["first_audio_ms"] >= 0
        assert done["min_lead_s"] is not None
        # What the model was busy for, not what the listener waited.
        assert 0 < done["render_ms"] < done["wall_ms"]
        assert 0 <= done["writer_ms"] <= done["wall_ms"]
        assert 0 <= done["load_ms"] <= done["wall_ms"]
        assert done["rtf"] == pytest.approx(
            done["render_ms"] / 1000 / done["audio_seconds"], abs=1e-3
        )

    def test_every_request_is_recorded(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        """One sample per request, whatever cadence the planner chose.

        A paced reply's requests are short and of one length, which is exactly
        the end of the line a fit needs and exactly what an average of ratios
        could not survive.
        """
        _measured(tmp_path)
        engine.unit_seconds = 0.2
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        _, _, done = _speak(client, text)
        assert done["batches"] >= 2
        stored = json.loads((tmp_path / STATS_FILE).read_text())
        assert len(stored[MODEL]["builtin"]["renders"]) == 3 + done["batches"]

    def test_buffered_can_still_be_asked_for(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        _measured(tmp_path)
        ready, _, done = _speak(client, "從前有一座山。", mode="buffered")
        assert ready["mode"] == "buffered"
        assert done["mode"] == "whole"


class TestLeaving:
    def test_a_cancel_stops_the_render_at_its_next_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _FakeEngine(unit_seconds=0.05)
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
        monkeypatch.setenv("API_KEY", "test-key")
        (tmp_path / FILE_NAME).write_text(json.dumps({"preload": False}))
        _pretend_downloaded(tmp_path, MODEL)
        monkeypatch.setattr(
            "cortex_speech.engine.registry.build", lambda backend, context: engine
        )
        monkeypatch.setattr(
            "cortex_speech.engine.registry.own_voices", lambda backend, path: [VOICE]
        )
        client = TestClient(create_app())
        with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
            ws.send_json({"type": "start", "model": MODEL, "format": "wav"})
            ws.receive_json()
            ws.send_json({"type": "text", "text": "從前有一座山。" * 20})
            ws.send_json({"type": "end"})
            assert engine.entered.wait(timeout=5)
            ws.send_json({"type": "cancel"})
        # Twenty segments of four units each would be eighty; the cancel
        # landed within the first few.
        deadline = time.monotonic() + 5
        while engine.units < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        settled = engine.units
        time.sleep(0.5)
        assert engine.units == settled, "the engine kept rendering after the cancel"
        assert engine.units < 80
