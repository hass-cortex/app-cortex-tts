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
from cortex_speech.audio import TARGET_PEAK
from cortex_speech.catalog import model_dir
from cortex_speech.engine.base import (
    Delivery,
    StopCheck,
    Synthesis,
    Voice,
    check_stop,
)
from cortex_tts.api.live import MAX_FRAME_BYTES
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


class _QuietEngine(_FakeEngine):
    """Renders a quiet tone rather than silence, so level can be measured.

    The models sit around -17 dBFS peak, which is what the level control
    exists to lift; silence cannot show whether it ran.
    """

    peak = 0.05

    def synthesize(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Synthesis:
        whole = super().synthesize(segments, voice, delivery=delivery, stop=stop)
        steps = np.arange(len(whole.audio), dtype=np.float32)
        tone = np.sin(2 * np.pi * 220.0 * steps / self.sample_rate) * self.peak
        return Synthesis(
            audio=tone.astype(np.float32),
            sample_rate=whole.sample_rate,
            segments=whole.segments,
            inference_ms=whole.inference_ms,
        )


def _peak_of(audio: bytes) -> float:
    """The loudest sample of a WAV reply, as a fraction of full scale.

    The 44-byte RIFF header goes first and would read as a very loud sample.
    """
    pcm = audio[44:] if audio[:4] == b"RIFF" else audio
    pcm = pcm[: len(pcm) - len(pcm) % 2]
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
    return float(np.max(np.abs(samples))) if samples.size else 0.0


def _pretend_downloaded(data_dir: Path, model_id: str) -> None:
    spec = BY_ID[model_id]
    root = model_dir(data_dir, spec.id)
    for name in spec.files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _measured(data_dir: Path, rtf: float = 0.3) -> None:
    """Pretend this host has seen the voice, at the given real-time factor."""
    store = StatsStore(data_dir / STATS_FILE)
    for audio in (2.0, 4.0, 8.0):
        store.record(
            MODEL, "builtin", VOICE.id, RenderSample(audio, rtf * audio, "cpu")
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


def _frame_sizes(client: TestClient, text: str, **start) -> list[int]:
    """Every binary frame's length, in order."""
    sizes: list[int] = []
    with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
        ws.send_json({"type": "start", "model": MODEL, "format": "wav", **start})
        ws.receive_json()
        ws.send_json({"type": "text", "text": text})
        ws.send_json({"type": "end"})
        while True:
            frame = ws.receive()
            if frame.get("bytes") is not None:
                sizes.append(len(frame["bytes"]))
            elif frame.get("text") is not None and json.loads(frame["text"])[
                "type"
            ] in {"done", "error"}:
                return sizes


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
                if message["type"] in ("batch", "rendered"):
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
                    assert message["mode"] in ("streaming", "buffered")
                    assert message["index"] == kinds.count("batch")
                if message["type"] == "done":
                    break
        assert kinds[0] == "batch"
        assert kinds.count("batch") == json.loads(frame["text"])["batches"]


class TestTheDeliveryIsLegibleFromOutside:
    """`batch` says how the reply was cut, `rendered` what the cut cost.

    Neither is in the audio, and during the opening hold there is no audio.
    The admin UI's live panel draws its timeline from these two alone.
    """

    @staticmethod
    def _frames(client: TestClient, text: str) -> list[dict]:
        with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
            ws.send_json({"type": "start", "model": MODEL, "format": "wav"})
            ws.receive_json()
            ws.send_json({"type": "text", "text": text})
            ws.send_json({"type": "end"})
            frames: list[dict] = []
            while True:
                frame = ws.receive()
                if frame.get("bytes") is not None:
                    continue
                message = json.loads(frame["text"])
                frames.append(message)
                if message["type"] == "done":
                    return frames

    def test_a_batch_says_what_it_is_about_to_say(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        _measured(tmp_path)
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        batches = [f for f in self._frames(client, text) if f["type"] == "batch"]
        assert batches, "the reply was cut into nothing"
        joined = "".join(b["text"] for b in batches)
        assert joined.replace(" ", "") == text.replace(" ", "")

    def test_every_request_reports_what_it_cost(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        _measured(tmp_path)
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        frames = self._frames(client, text)
        sent = [f for f in frames if f["type"] == "batch"]
        cost = [f for f in frames if f["type"] == "rendered"]
        done = frames[-1]
        assert [f["index"] for f in cost] == [f["index"] for f in sent]
        assert len(cost) == done["batches"]
        assert sum(f["audio_s"] for f in cost) == pytest.approx(
            done["audio_seconds"], abs=0.05
        )
        assert all(f["render_ms"] >= 0 for f in cost)

    def test_the_cost_is_told_even_while_the_audio_is_held(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        """Buffered releases nothing until the end, and still says what it did."""
        with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
            ws.send_json(
                {"type": "start", "model": MODEL, "format": "wav", "mode": "buffered"}
            )
            ws.receive_json()
            ws.send_json({"type": "text", "text": "從前有一座山。山上有一間小廟。"})
            ws.send_json({"type": "end"})
            seen: list[str] = []
            while True:
                frame = ws.receive()
                if frame.get("bytes") is not None:
                    seen.append("audio")
                    continue
                message = json.loads(frame["text"])
                seen.append(message["type"])
                if message["type"] == "done":
                    break
        assert seen.index("rendered") < seen.index("audio")


class TestACallerMayInsistOnADelivery:
    """`auto` takes the verdict; `streaming` and `buffered` overrule it."""

    @staticmethod
    def _spoken(client: TestClient, mode: str, text: str) -> dict:
        _, _, done = _speak(client, text, mode=mode)
        return done

    def test_auto_takes_the_verdict(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        _measured(tmp_path, rtf=0.3)
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        assert self._spoken(client, "auto", text)["mode"] == "streaming"

    def test_a_slow_voice_is_buffered_under_auto(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        _measured(tmp_path, rtf=1.1)
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        assert self._spoken(client, "auto", text)["mode"] == "buffered"

    def test_the_hosts_threshold_decides_auto(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        """The same 1.1 voice streams once this host's line is above it, and
        the `ready` frame names the line the verdict was made against."""
        _measured(tmp_path, rtf=1.1)
        # Stored before the app opens, as a host that set it earlier would have.
        (tmp_path / FILE_NAME).write_text(
            json.dumps({"preload": False, "stream_rtf": 1.3})
        )
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        ready, _, done = _speak(client, text, mode="auto")
        assert done["mode"] == "streaming"
        assert ready["threshold"] == 1.3

    def test_buffered_renders_as_written_and_releases_at_the_end(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        _measured(tmp_path)
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        done = self._spoken(client, "buffered", text)
        # Written in one go, so one request; buffered is about releasing.
        assert done["batches"] == 1
        assert done["mode"] == "buffered"

    def test_streaming_can_be_insisted_on_by_an_unmeasured_voice(
        self, client: TestClient, engine: _FakeEngine
    ) -> None:
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        done = self._spoken(client, "streaming", text)
        assert done["mode"] == "streaming"
        assert done["batches"] == 3


class TestOpening:
    def test_the_first_frame_must_be_a_start_frame(self, client: TestClient) -> None:
        with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
            ws.send_json({"type": "text", "text": "hi"})
            error = ws.receive_json()
            assert error["type"] == "error"

    def test_a_browser_may_carry_the_key_as_a_subprotocol(
        self, client: TestClient
    ) -> None:
        """A `WebSocket` constructor cannot set a header; this is what it has."""
        with (
            client,
            client.websocket_connect(
                "/api/speak/live",
                subprotocols=["cortex-tts", AUTH["Authorization"][7:]],
            ) as ws,
        ):
            ws.send_json({"type": "start", "model": MODEL, "format": "wav"})
            assert ws.receive_json()["type"] == "ready"

    def test_a_wrong_key_in_the_subprotocol_is_refused(
        self, client: TestClient
    ) -> None:
        with (
            client,
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(
                "/api/speak/live", subprotocols=["cortex-tts", "nope"]
            ) as ws,
        ):
            ws.send_json({"type": "start"})
            ws.receive_json()

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
        assert ready["mode"] == "buffered"
        assert ready["rtf"] is None
        assert ready["samples"] == 0

    def test_ready_carries_what_the_voice_measured(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        _measured(tmp_path, rtf=0.3)
        ready, _, _ = _speak(client, "好了。")
        assert ready["mode"] == "streaming"
        assert ready["rtf"] == pytest.approx(0.3)
        assert ready["samples"] == 3


class TestUnmeasuredVoiceIsBuffered:
    """Nothing measured is the one case that can never run dry."""

    def test_it_is_buffered(self, client: TestClient, engine: _FakeEngine) -> None:
        ready, audio, done = _speak(client, "從前有一座山。山上有一間小廟。")
        assert ready["mode"] == "buffered"
        assert done["mode"] == "buffered"
        assert done["batches"] == 1
        assert len(engine.calls) == 1
        # 44-byte WAV header plus the audio
        assert len(audio) > 44
        assert done["audio_seconds"] > 0

    def test_three_requests_make_it_measured(
        self, tmp_path: Path, client: TestClient, engine: _FakeEngine
    ) -> None:
        for _ in range(3):
            _speak(client, "從前有一座山。")
        ready, _, _ = _speak(client, "從前有一座山。")
        assert ready["samples"] == 3
        assert ready["rtf"] is not None


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
        assert done["batches"] == 3, "one sentence per request"
        assert len(engine.calls) == done["batches"]
        assert done["first_audio_ms"] >= 0
        assert done["bank_wait_ms"] is not None
        assert done["min_lead_s"] is not None
        assert done["gap_at"] in (2, 3)
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
        """One sample per request: a sentence is what a streaming reply is made of."""
        _measured(tmp_path)
        engine.unit_seconds = 0.2
        text = "從前有一座山，山上有一間小廟。廟裡住著一位老和尚和一位小和尚。每天早上他們都到溪邊打水。"
        _, _, done = _speak(client, text)
        assert done["batches"] == 3
        stored = json.loads((tmp_path / STATS_FILE).read_text())
        renders = stored["models"][MODEL][f"builtin:{VOICE.id}"]["renders"]
        assert len(renders) == 3 + done["batches"]
        assert all(r[2] == "cpu" for r in renders)

    def test_a_held_reply_is_not_one_enormous_frame(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        """A receiver buffers a frame whole, so every client caps one.

        Buffered holds the entire reply and releases it at once; unsliced that
        is a single frame of the whole answer, and aiohttp — which the
        integration uses — refuses one over 4 MB by default. The listener then
        gets a closed socket instead of audio.
        """
        _measured(tmp_path)
        # The fake engine renders 0.25 s per character at 24 kHz, 16-bit.
        text = "從前有一座山，山上有一間小廟。" * 6
        sizes = _frame_sizes(client, text, mode="buffered")
        assert sum(sizes) > MAX_FRAME_BYTES, "the reply was too small to test"
        assert max(sizes) <= MAX_FRAME_BYTES
        assert len(sizes) > 1

    def test_buffered_can_still_be_asked_for(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        _measured(tmp_path)
        ready, _, done = _speak(client, "從前有一座山。", mode="buffered")
        assert ready["mode"] == "buffered"
        assert done["mode"] == "buffered"


class TestBufferedArrivesAtTheLevelAFileWouldHave:
    """A held reply has a finished waveform, so it is levelled like one.

    `StreamGain` only attenuates, because a stream cannot know its own peak
    before it has ended. Buffered has ended by definition — and without this
    a listener who changes the setting hears the same words arrive far
    quieter, which is a difference the setting never promised.
    """

    @pytest.fixture
    def engine(self) -> _QuietEngine:
        return _QuietEngine()

    def test_it_is_lifted_to_the_target(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        _measured(tmp_path)
        _, audio, _ = _speak(client, "從前有一座山。", mode="buffered")
        assert _peak_of(audio) == pytest.approx(TARGET_PEAK, abs=0.02)

    def test_a_streamed_reply_is_left_where_the_model_put_it(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        """The contrast, so the test above is not passing for some other reason."""
        _measured(tmp_path)
        _, audio, _ = _speak(client, "從前有一座山。", mode="streaming")
        assert _peak_of(audio) == pytest.approx(_QuietEngine.peak, abs=0.02)


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


class _PacedEngine(_FakeEngine):
    """Renders at a chosen real-time factor: 0.25 s of audio per character."""

    def __init__(self, rtf: float) -> None:
        super().__init__()
        self.rtf = rtf

    def synthesize(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Synthesis:
        for segment in segments:
            check_stop(stop)
            time.sleep(self.rtf * 0.25 * len(segment))
        return super().synthesize(segments, voice, delivery=delivery, stop=stop)


class _BreakingEngine(_FakeEngine):
    """Fails after the reply is under way with an error nothing gave a shape to."""

    def synthesize(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Synthesis:
        del segments, voice, delivery, stop
        raise RuntimeError("Non-zero status code returned while running Add")


class TestARuntimeFailureAfterReadyIsSaidOnTheSocket:
    """A listener whose reply simply stops cannot tell a crash from a slow
    render, so even a failure no one gave a wire shape to ends in an `error`
    frame — not an exception left to the server."""

    @pytest.fixture
    def engine(self) -> _BreakingEngine:
        return _BreakingEngine()

    def test_the_error_frame_arrives(self, client: TestClient) -> None:
        with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
            ws.send_json({"type": "start", "model": MODEL, "format": "wav"})
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "text", "text": "好的。"})
            ws.send_json({"type": "end"})
            frames = []
            while True:
                frame = ws.receive()
                if frame.get("bytes") is not None:
                    continue
                if frame.get("type") == "websocket.close":
                    break
                message = json.loads(frame["text"])
                frames.append(message)
                if message["type"] in {"done", "error"}:
                    break
        assert frames[-1]["type"] == "error"
        assert "Non-zero status code" in frames[-1]["message"]


class TestAShortReplyInHandIsNotHeldForSixSeconds:
    """Two sentences, 3.25 s and 1.25 s of audio, at 1.05x with the whole
    text already in hand: the rest needs about 1.3 s of lead, the first
    sentence banks 3.25, so it plays as soon as it is rendered — before the
    second is. Held to six seconds it would have waited for both."""

    @pytest.fixture
    def engine(self) -> _PacedEngine:
        return _PacedEngine(rtf=1.05)

    def test_the_first_sentence_plays_before_the_second_is_rendered(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        _measured(tmp_path, rtf=1.05)
        with client, client.websocket_connect("/api/speak/live", headers=AUTH) as ws:
            # 1.05x is over the default threshold; the hold is what is under test.
            ws.send_json(
                {"type": "start", "model": MODEL, "format": "wav", "mode": "streaming"}
            )
            assert ws.receive_json()["rtf"] == pytest.approx(1.05)
            ws.send_json(
                {"type": "text", "text": "好的，客廳的燈已經打開了。現在幾點。"}
            )
            ws.send_json({"type": "end"})
            seen: list[str] = []
            while True:
                frame = ws.receive()
                if frame.get("bytes") is not None:
                    seen.append("audio")
                    continue
                message = json.loads(frame["text"])
                seen.append(message["type"])
                if message["type"] == "done":
                    break
        rendered = [i for i, kind in enumerate(seen) if kind == "rendered"]
        assert len(rendered) == 2
        assert seen.index("audio") < rendered[1], seen
        assert message["min_lead_s"] >= 0


class TestTheBankOutlastsAShortSentenceBeforeALongOne:
    """The shape that breaks releasing each sentence as it lands.

    A one-second sentence followed by a seven-second one at 0.6x: released
    unheld, the second lands 3.2 s after the first has finished playing.
    Banked to six seconds, both go out together and the third sentence lands
    with seconds of lead to spare. Replayed over production replies this is
    what the bank buys — see `scripts/replay_pacing.py`.
    """

    @pytest.fixture
    def engine(self) -> _PacedEngine:
        return _PacedEngine(rtf=0.6)

    def test_it_never_runs_dry(self, tmp_path: Path, client: TestClient) -> None:
        _measured(tmp_path, rtf=0.6)
        short, long, medium = (
            "好了。",
            "從前有一座山，山上有一間小廟，廟裡住著一位老和尚。",
            "每天早上他們都打水。",
        )
        ready, _, done = _speak(client, short + long + medium)
        assert ready["mode"] == "streaming"
        assert done["batches"] == 3
        assert done["bank_wait_ms"] > 0, "the first sentence waited in the bank"
        assert done["min_lead_s"] is not None and done["min_lead_s"] >= 0
        assert done["gap_at"] == 3
