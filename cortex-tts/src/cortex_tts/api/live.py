"""`/api/speak/live`: a reply spoken while it is still being written.

One WebSocket per reply. The client sends a `start` frame, then `text` frames
as the writer produces them, then `end`; the server answers with a `ready`
frame, binary audio frames, and a `done` frame carrying what the reply cost.
The server decides when to render what — see `cortex_speech.pacing` — from
what this host has measured about the model, because it is the one holding
the measurements, the engine queue and the clock.

What the transport owns, and the planner does not: releasing audio (the
opening hold), measuring the listener's lead, and noticing that the listener
has gone — a closed socket, a `cancel` frame, or a client that has stopped
reading — so the engine stops at its next checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field, replace

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from cortex_speech import (
    STREAM_ENCODERS,
    AbandonedError,
    EngineError,
    Finished,
    ModelSpec,
    Planner,
    Send,
    StreamGain,
    Wait,
)

from .deps import AppState, get_state_ws, require_api_key_ws
from .routes import (
    check_delivery,
    engine_errors,
    pick_voice,
    prepare_segments,
    render_sample,
    spec_or_404,
    told_language,
    voice_kind,
)
from .schemas import LiveStart, LiveText

_LOGGER = logging.getLogger(__name__)

live = APIRouter(prefix="/api", dependencies=[Depends(require_api_key_ws)])

# A listener that has not taken a frame for this long is gone, whatever the
# socket says. Home Assistant drains a stream without back-pressure, so a
# healthy client never comes near it.
STALL_S = 15.0

# How long the speaker waits for text before asking the planner again. The
# reader wakes it on every frame, so this only bounds the idle case.
IDLE_POLL_S = 0.5

# The most audio one WebSocket frame carries. A receiver buffers a frame whole
# before it sees any of it, so every client sets a ceiling — aiohttp's is 4 MB
# — and a reply held whole is one frame of the entire reply: a three-minute
# buffered answer is past it, and the listener gets a closed socket rather
# than audio. Sliced here so no client has to be configured for this one; the
# bytes are a stream and the receiver concatenates them, so a slice may fall
# anywhere.
MAX_FRAME_BYTES = 512 * 1024

# Close codes: policy violation for a bad opening frame, internal error for an
# engine failure after audio has started.
_BAD_REQUEST = 1008
_INTERNAL = 1011


@dataclass
class _Hold:
    """What the first batch said about releasing audio, applied by the session."""

    all: bool = False
    audio_s: float = 0.0
    wall_s: float = 0.0
    bank: bool = False
    released: bool = False
    banked: list[bytes] = field(default_factory=list)
    banked_s: float = 0.0


class _GoneError(Exception):
    """The listener left: socket closed, cancel sent, or reads stalled."""


class _Session:
    def __init__(self, websocket: WebSocket, state: AppState) -> None:
        self._ws = websocket
        self._state = state
        self._gone = asyncio.Event()
        self._wake = asyncio.Event()
        self._started = time.perf_counter()
        self._first_release: float | None = None
        self._sent_audio_s = 0.0
        self._min_lead: float | None = None
        self._first_audio_ms: float | None = None
        self._requests = 0
        self._render_s = 0.0
        self._planner: Planner | None = None
        self._batch_produced_s = 0.0
        self._batch_started: float | None = None
        self._ready_at: float | None = None
        self._ended_at: float | None = None
        self._hold = _Hold()
        # Kept so the reply's end can cancel it: an uncancelled timer holds
        # this session, its socket and its banked audio in the loop's heap
        # for the rest of the hold, which the listener has already left.
        self._hold_timer: asyncio.TimerHandle | None = None
        # The loop keeps only a weak reference to a bare task, so one left
        # unheld can be collected before it has released anything.
        self._timer_release: asyncio.Task | None = None
        # A container header goes out with the first audio, not before it:
        # sending it alone would start the listener's clock on silence.
        self._header = b""

    # -- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        """Serve one reply, start to done."""
        start = await self._opening_frame()
        if start is None:
            return
        state = self._state
        try:
            spec = spec_or_404(state, start.model or state.preferences.default_model)
            check_delivery(spec, start.delivery())
            voice = await pick_voice(state, spec, start.voice)
            with engine_errors():
                await state.registry.acquire(spec.id)
        except Exception as err:  # noqa: BLE001 - reported on the socket
            await self._refuse(err)
            return

        # From here the writer's clock runs; what came before was the model
        # being made resident, which is reported on its own.
        self._ready_at = time.perf_counter()
        kind = voice_kind(state, spec, voice.id)
        planner = Planner(
            state.stats.render_model(spec.id, kind, voice.id),
            chunk_streaming=spec.chunk_streaming,
            mode=start.mode,
        )
        self._planner = planner
        encoder = STREAM_ENCODERS[start.format]()
        await self._send_json(
            {
                "type": "ready",
                "model": spec.id,
                "voice": voice.id,
                "bitrate": encoder.bitrate or spec.sample_rate * 16,
                "sample_rate": spec.sample_rate,
                "chunk_streaming": spec.chunk_streaming,
                "mode": planner.mode,
            }
        )

        reader = asyncio.create_task(self._read(planner))
        language = start.language or voice.language
        delivery = start.delivery()
        try:
            self._header = encoder.open(spec.sample_rate)
            while not self._gone.is_set():
                # Cleared before planning so a frame that lands while the
                # planner runs is not a wake-up lost until the next poll.
                self._wake.clear()
                decision = planner.plan(self._lead())
                if isinstance(decision, Finished):
                    break
                if isinstance(decision, Wait):
                    await self._sleep_until_woken()
                    continue
                assert isinstance(decision, Send)
                self._apply_hold(decision)
                segments = prepare_segments(state, spec, decision.text, language, start)
                if not segments:
                    continue
                # Said before the render, so a listener can show how the
                # reply is being delivered while its first audio is still
                # being made; `done` has the last word once the count is known.
                await self._send_json(
                    {"type": "batch", "index": self._requests + 1, "mode": planner.mode}
                )
                await self._render(
                    spec,
                    kind,
                    voice.id,
                    segments,
                    decision.text,
                    replace(
                        delivery,
                        language=told_language(spec, decision.text, language),
                    ),
                    encoder,
                    StreamGain(),
                )
            if self._gone.is_set():
                raise _GoneError
            await self._emit(encoder.close(), 0.0)
            await self._release()
            done = self._done(planner.mode)
            _LOGGER.info(
                "spoke live as %s/%s -> %.2fs audio, %s in %d request(s), "
                "first audio %.0fms, min lead %s",
                spec.id,
                voice.id,
                done["audio_seconds"],
                done["mode"],
                done["batches"],
                done["first_audio_ms"],
                done["min_lead_s"],
            )
            await self._send_json(done)
            await self._ws.close()
        except (_GoneError, AbandonedError, WebSocketDisconnect):
            _LOGGER.info(
                "%s: listener left after %.1fs of audio", spec.id, self._sent_audio_s
            )
        except EngineError as err:
            await self._fail(err, code=_INTERNAL)
        except HTTPException as err:
            # An engine failure after `ready`, already given its wire shape
            # by `engine_errors`; the socket is the only place left to say it,
            # and it may itself be gone by now.
            with contextlib.suppress(_GoneError):
                await self._refuse(err, code=_INTERNAL)
        except asyncio.CancelledError:
            # The server is shutting down under the reply; the engine hears
            # it through `stop` in `finally`.
            _LOGGER.info("%s: reply cancelled by the server", spec.id)
            raise
        finally:
            self._gone.set()
            reader.cancel()
            if self._hold_timer is not None:
                self._hold_timer.cancel()

    async def _opening_frame(self) -> LiveStart | None:
        try:
            raw = await self._ws.receive_json()
            return LiveStart.model_validate(raw)
        except WebSocketDisconnect:
            return None
        except (ValidationError, ValueError) as err:
            await self._fail(
                ValueError(f"the first frame must be a start frame: {err}"),
                code=_BAD_REQUEST,
            )
            return None

    async def _refuse(self, err: Exception, *, code: int = _BAD_REQUEST) -> None:
        """Report a failure on the socket, then close it."""
        detail = getattr(err, "detail", None)
        if isinstance(detail, dict):
            await self._send_json({"type": "error", **detail})
        else:
            await self._send_json(
                {"type": "error", "code": "ERROR", "message": str(err)}
            )
        await self._ws.close(code=code)

    async def _fail(self, err: Exception, *, code: int) -> None:
        await self._send_json({"type": "error", "code": "ERROR", "message": str(err)})
        await self._ws.close(code=code)

    # -- the reader -------------------------------------------------------

    async def _read(self, planner: Planner) -> None:
        """Feed the planner from the socket until `end`, `cancel` or a close."""
        try:
            while not planner.ended:
                frame = await self._ws.receive_json()
                kind = frame.get("type") if isinstance(frame, dict) else None
                if kind == "text":
                    planner.feed(LiveText.model_validate(frame).text)
                elif kind == "end":
                    planner.end()
                    self._ended_at = time.perf_counter()
                elif kind == "cancel":
                    self._gone.set()
                    return
                self._wake.set()
            # After `end` the only thing left to hear is a close or a cancel.
            while True:
                frame = await self._ws.receive_json()
                if isinstance(frame, dict) and frame.get("type") == "cancel":
                    break
        except (WebSocketDisconnect, ValidationError, ValueError, RuntimeError):
            pass
        self._gone.set()
        self._wake.set()

    async def _sleep_until_woken(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), IDLE_POLL_S)

    # -- rendering --------------------------------------------------------

    async def _render(
        self,
        spec: ModelSpec,
        kind: str,
        voice_id: str,
        segments: list[str],
        text: str,
        delivery,
        encoder,
        gain: StreamGain,
    ) -> None:
        """Render one batch, emitting audio as the engine produces it."""
        self._requests += 1
        self._batch_produced_s = 0.0
        started = self._batch_started = time.perf_counter()
        produced = 0
        with engine_errors():
            async for chunk in self._state.registry.synthesize_stream(
                spec.id, segments, voice_id, delivery, stop=self._gone.is_set
            ):
                produced += len(chunk)
                self._batch_produced_s = produced / spec.sample_rate
                await self._emit(
                    encoder.encode(gain.frames(chunk)), len(chunk) / spec.sample_rate
                )
        wall = time.perf_counter() - started
        self._render_s += wall
        seconds = produced / spec.sample_rate
        if self._planner is not None:
            # What it really cost, so the opening hold for what is left is
            # sized by this reply and not only by the host's average day.
            self._planner.rendered(seconds, wall)
        if seconds:
            # Every request teaches the render model; a listener who left
            # never reaches here, so an abandoned render teaches it nothing.
            await asyncio.to_thread(
                self._state.stats.record,
                spec.id,
                kind,
                voice_id,
                render_sample(text, seconds, wall),
            )

    def _rtf(self) -> float | None:
        """What the model was busy for over the audio that came out."""
        if not self._sent_audio_s:
            return None
        return round(self._render_s / self._sent_audio_s, 3)

    # -- releasing audio --------------------------------------------------

    def _apply_hold(self, decision: Send) -> None:
        if self._requests:
            return
        self._hold = _Hold(
            all=decision.hold_all,
            audio_s=decision.hold_audio_s,
            wall_s=decision.hold_wall_s,
            bank=decision.hold_bank,
        )

    async def _emit(self, frames: bytes, seconds: float) -> None:
        """Send audio, or bank it while the opening hold says to."""
        if not frames and not seconds:
            return
        hold = self._hold
        if hold.released:
            await self._send_audio(frames, seconds)
            return
        first_audio = not hold.banked and seconds > 0
        hold.banked.append(frames)
        hold.banked_s += seconds
        if hold.all:
            return
        if hold.bank:
            # Enough banked to outlast what the rest is still predicted to
            # lose: released now, and sooner if the render runs ahead.
            assert self._planner is not None
            elapsed = (
                time.perf_counter() - self._batch_started
                if self._batch_started is not None
                else 0.0
            )
            if hold.banked_s >= self._planner.bank_needed(
                self._batch_produced_s, elapsed
            ):
                await self._release()
            return
        if hold.wall_s:
            if first_audio:
                # The first audio has arrived; the hold is counted from here.
                self._hold_timer = asyncio.get_running_loop().call_later(
                    hold.wall_s, self._release_later
                )
            return
        if not hold.wall_s and hold.banked_s >= hold.audio_s:
            await self._release()

    def _release_later(self) -> None:
        """Release from the opening hold's timer, which cannot await."""
        self._timer_release = asyncio.ensure_future(self._release())

    async def _release(self) -> None:
        hold = self._hold
        if hold.released or self._gone.is_set():
            return
        hold.released = True
        frames, seconds = b"".join(hold.banked), hold.banked_s
        hold.banked.clear()
        hold.banked_s = 0.0
        await self._send_audio(frames, seconds)

    async def _send_audio(self, frames: bytes, seconds: float) -> None:
        if self._first_release is None:
            self._first_release = time.perf_counter()
            self._first_audio_ms = (self._first_release - self._started) * 1000
        # The lead as this audio lands: what the listener still held. The
        # first release holds nothing by definition and is not a measurement.
        lead = self._lead()
        if lead is not None and self._sent_audio_s > 0:
            self._min_lead = (
                lead if self._min_lead is None else min(self._min_lead, lead)
            )
        self._sent_audio_s += seconds
        frames, self._header = self._header + frames, b""
        if not frames:
            return
        try:
            for start in range(0, len(frames), MAX_FRAME_BYTES):
                await asyncio.wait_for(
                    self._ws.send_bytes(frames[start : start + MAX_FRAME_BYTES]),
                    STALL_S,
                )
        except Exception as err:  # noqa: BLE001 - any failure to send is the same news
            self._gone.set()
            raise _GoneError from err

    def _lead(self) -> float | None:
        """Audio handed over minus wall time since the first byte left."""
        if self._first_release is None:
            return None
        return self._sent_audio_s - (time.perf_counter() - self._first_release)

    async def _send_json(self, frame: dict) -> None:
        try:
            await self._ws.send_json(frame)
        except Exception as err:  # noqa: BLE001 - any failure to send is the same news
            self._gone.set()
            raise _GoneError from err

    def _done(self, mode: str) -> dict:
        return {
            "type": "done",
            # What actually happened, not what was planned: a reply that fit
            # one request was spoken whole, whichever way it got there.
            "mode": mode if self._requests > 1 else "whole",
            "batches": self._requests,
            "audio_seconds": round(self._sent_audio_s, 3),
            # What the model was busy for, as opposed to what the listener
            # waited: the sum of the requests' render time, and that over
            # the audio it produced — the figure the stats card keeps.
            "render_ms": round(self._render_s * 1000, 1),
            # Where the wait went. The model being made resident is paid
            # before `ready`; the writer's time runs from `ready` to `end`
            # and is what a paced reply waits for before it can render.
            "load_ms": round(
                ((self._ready_at or self._started) - self._started) * 1000, 1
            ),
            "writer_ms": (
                round((self._ended_at - (self._ready_at or self._started)) * 1000, 1)
                if self._ended_at is not None
                else None
            ),
            "rtf": self._rtf(),
            "first_audio_ms": round(self._first_audio_ms or 0.0, 1),
            "min_lead_s": round(self._min_lead, 3)
            if self._min_lead is not None
            else None,
            "wall_ms": round((time.perf_counter() - self._started) * 1000, 1),
        }


@live.websocket("/speak/live")
async def speak_live(
    websocket: WebSocket, state: AppState = Depends(get_state_ws)
) -> None:
    """Speak a reply as it is written; see the module docstring for the frames."""
    await websocket.accept()
    session = _Session(websocket, state)
    with contextlib.suppress(WebSocketDisconnect):
        await session.run()
