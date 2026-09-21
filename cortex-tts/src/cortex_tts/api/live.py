"""`/api/speak/live`: a reply spoken while it is still being written.

One WebSocket per reply. The client sends a `start` frame, then `text` frames
as the writer produces them, then `end`; the server answers with a `ready`
frame, a `batch` frame before each request and a `rendered` frame after it,
binary audio frames, and a `done` frame carrying what the reply cost.

`batch` and `rendered` are what make the delivery legible from outside: the
first says what a request carries, the second what it actually cost. Neither
can be worked out from the audio, and while the opening is banked there is no
audio to work anything out from.

How the reply is spoken is settled at `ready` — see `cortex_speech.pacing`:
the voice's measured real-time factor on this host says whether it streams
(one sentence per request, played from a bank of `BANK_S`, or of less once
the writer has finished and `bank_needed` can be asked) or is buffered
(rendered as written, released once it is all rendered), unless the caller
insisted on one of the two.

What the transport owns: releasing audio (the bank), measuring the listener's
lead, and noticing that the listener has gone — a closed socket, a `cancel`
frame, or a client that has stopped reading — so the engine stops at its next
checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from cortex_speech import (
    BANK_S,
    BUFFERED,
    STREAM_ENCODERS,
    STREAMING,
    AbandonedError,
    Delivery,
    EngineError,
    ModelSpec,
    Pacer,
    RenderSample,
    StreamEncoder,
    StreamGain,
    bank_needed,
    levelled_frames,
    spoken_seconds,
    verdict,
)

from .deps import WS_SUBPROTOCOL, AppState, get_state_ws, require_api_key_ws
from .routes import (
    check_delivery,
    engine_errors,
    pick_voice,
    prepare_segments,
    provider_of,
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

# How long the speaker waits for text before asking the pacer again. The
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
class _Bank:
    """Audio kept back before the first sound.

    A streaming reply releases once `_Session._hold` seconds of audio are
    banked or the reply is rendered whole; a buffered one only when it is
    rendered whole.
    """

    released: bool = False
    frames: list[bytes] = field(default_factory=list)
    seconds: float = 0.0
    first_audio_at: float | None = None


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
        self._gap_at: int | None = None
        self._first_audio_ms: float | None = None
        self._bank_wait_ms: float | None = None
        self._requests = 0
        self._render_s = 0.0
        self._ready_at: float | None = None
        self._ended_at: float | None = None
        self._mode: str = BUFFERED
        self._bank = _Bank()
        self._pacer: Pacer | None = None
        self._measured_rtf: float | None = None
        # The request being rendered: what it was expected to produce, and
        # what it has produced so far — the bank's view of what is still owed.
        self._inflight_est = 0.0
        self._inflight_done = 0.0
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
        provider = provider_of(state, spec)
        measured = (
            state.stats.rtf(spec.id, kind, voice.id, provider) if provider else None
        )
        rtf = measured.rtf if measured else None
        # Every request is a measurement of this host in this voice — on the
        # provider it is actually running on, or it is no measurement at all.
        record: Callable[[float, float], None] | None = (
            (
                lambda seconds, wall: state.stats.record(
                    spec.id, kind, voice.id, RenderSample(seconds, wall, provider)
                )
            )
            if provider
            else None
        )
        threshold = state.preferences.stream_rtf
        self._mode = verdict(rtf, threshold) if start.mode == "auto" else start.mode
        self._measured_rtf = rtf
        pacer = self._pacer = Pacer(self._mode)
        encoder = STREAM_ENCODERS[start.format]()
        await self._send_json(
            {
                "type": "ready",
                "model": spec.id,
                "voice": voice.id,
                "bitrate": encoder.bitrate or spec.sample_rate * 16,
                "sample_rate": spec.sample_rate,
                "chunk_streaming": spec.chunk_streaming,
                # Settled here and never revised: what the voice measured on
                # this host, and the way the reply is spoken because of it.
                "mode": self._mode,
                "rtf": rtf,
                "samples": measured.samples if measured else 0,
                "threshold": threshold,
            }
        )

        reader = asyncio.create_task(self._read(pacer))
        language = start.language or voice.language
        delivery = start.delivery()
        try:
            self._header = encoder.open(spec.sample_rate)
            while not self._gone.is_set():
                # Cleared before asking so a frame that lands while the pacer
                # answers is not a wake-up lost until the next poll.
                self._wake.clear()
                text = pacer.next_request()
                if text is None:
                    if pacer.finished():
                        break
                    await self._sleep_until_woken()
                    continue
                segments = prepare_segments(state, spec, text, language, start)
                if not segments:
                    continue
                # Said before the render, so a listener can show how the
                # reply is being delivered while its first audio is still
                # being made.
                await self._send_json(
                    {
                        "type": "batch",
                        "index": self._requests + 1,
                        "mode": self._mode,
                        # What this request carries, as the writer wrote it.
                        # Where a reply was cut is the one thing a caller
                        # cannot work out from the audio it gets back.
                        "text": text,
                    }
                )
                await self._render(
                    spec,
                    voice.id,
                    segments,
                    replace(
                        delivery,
                        language=told_language(spec, text, language),
                    ),
                    encoder,
                    StreamGain(),
                    record,
                )
            if self._gone.is_set():
                raise _GoneError
            await self._emit(encoder.close(), 0.0)
            await self._release()
            done = self._done()
            _LOGGER.log(
                logging.WARNING
                if done["min_lead_s"] is not None and done["min_lead_s"] < 0
                else logging.INFO,
                "spoke live model=%s voice=%s setting=%s verdict=%s rtf=%s "
                "threshold=%.2f samples=%d provider=%s outcome=%s requests=%d "
                "audio_s=%.1f first_audio_ms=%.0f bank_wait_ms=%s min_lead_s=%s "
                "gap_at=%s",
                spec.id,
                voice.id,
                start.mode,
                verdict(rtf, threshold),
                f"{rtf:.2f}" if rtf is not None else "-",
                threshold,
                measured.samples if measured else 0,
                provider or "-",
                done["mode"],
                done["batches"],
                done["audio_seconds"],
                done["first_audio_ms"],
                _field(done["bank_wait_ms"]),
                _field(done["min_lead_s"]),
                _field(done["gap_at"]),
            )
            await self._send_json(done)
            # A measurement landed and a model may have been made resident.
            state.updates.publish("models")
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
        except Exception as err:  # noqa: BLE001 - the socket must still hear it
            # A runtime failure no one gave a shape to. Said on the socket
            # rather than left to the server: a listener whose reply simply
            # stops cannot tell a crash from a slow render.
            _LOGGER.exception("%s: reply failed", spec.id)
            with contextlib.suppress(_GoneError):
                await self._fail(err, code=_INTERNAL)
        except asyncio.CancelledError:
            # The server is shutting down under the reply; the engine hears
            # it through `stop` in `finally`.
            _LOGGER.info("%s: reply cancelled by the server", spec.id)
            raise
        finally:
            self._gone.set()
            reader.cancel()

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

    async def _read(self, pacer: Pacer) -> None:
        """Feed the pacer from the socket until `end`, `cancel` or a close."""
        try:
            while not pacer.ended:
                frame = await self._ws.receive_json()
                kind = frame.get("type") if isinstance(frame, dict) else None
                if kind == "text":
                    pacer.feed(LiveText.model_validate(frame).text)
                elif kind == "end":
                    pacer.end()
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
        voice_id: str,
        segments: list[str],
        delivery: Delivery,
        encoder: StreamEncoder,
        gain: StreamGain,
        record: Callable[[float, float], None] | None,
    ) -> None:
        """Render one request, emitting audio as the engine produces it.

        `record` is told what it cost, audio seconds and wall seconds, once
        known; `None` when this host cannot say which provider ran it.
        """
        self._requests += 1
        started = time.perf_counter()
        produced = 0
        self._inflight_est = sum(spoken_seconds(s) for s in segments)
        self._inflight_done = 0.0
        # Buffered releases nothing until the reply is over, so unlike a
        # streamed reply it ends up holding a finished waveform — and a
        # finished waveform can be levelled rather than merely kept under the
        # ceiling. Without this a held reply arrives about 16 dB under a file
        # of the same text, which is audible the moment a listener changes
        # **Speaking mode**.
        held: list[np.ndarray] = []
        with engine_errors():
            if self._mode == BUFFERED:
                # Nothing is released until the reply is over, so the render
                # can be awaited whole — which is what lets the engine notice
                # a generation that stopped early and try it at another seed.
                # The streamed path cannot: its chunks are already on the wire.
                synthesis = await self._state.registry.synthesize(
                    spec.id,
                    segments,
                    voice_id,
                    delivery=delivery,
                    stop=self._gone.is_set,
                )
                produced += len(synthesis.audio)
                self._inflight_done += len(synthesis.audio) / spec.sample_rate
                held.append(synthesis.audio)
            else:
                async for chunk in self._state.registry.synthesize_stream(
                    spec.id, segments, voice_id, delivery, stop=self._gone.is_set
                ):
                    produced += len(chunk)
                    self._inflight_done += len(chunk) / spec.sample_rate
                    await self._emit(
                        encoder.encode(gain.frames(chunk)),
                        len(chunk) / spec.sample_rate,
                    )
        if held:
            await self._emit(
                encoder.encode(levelled_frames(held)), produced / spec.sample_rate
            )
        wall = time.perf_counter() - started
        self._render_s += wall
        self._inflight_est = self._inflight_done = 0.0
        seconds = produced / spec.sample_rate
        # What the request actually cost, said once it is known. `batch` can
        # only carry the text; while audio is still banked a listener sees no
        # bytes at all, so this is the only account of where the time went.
        await self._send_json(
            {
                "type": "rendered",
                "index": self._requests,
                "audio_s": round(seconds, 3),
                "render_ms": round(wall * 1000, 1),
            }
        )
        if seconds and record:
            # A listener who left never reaches here, so an abandoned render
            # teaches the host nothing.
            await asyncio.to_thread(
                record,
                seconds,
                wall,
            )

    def _rtf(self) -> float | None:
        """What the model was busy for over the audio that came out."""
        if not self._sent_audio_s:
            return None
        return round(self._render_s / self._sent_audio_s, 3)

    # -- releasing audio --------------------------------------------------

    async def _emit(self, frames: bytes, seconds: float) -> None:
        """Send audio, or bank it while the opening is still being held."""
        if not frames and not seconds:
            return
        bank = self._bank
        if bank.released:
            await self._send_audio(frames, seconds)
            return
        if bank.first_audio_at is None and seconds > 0:
            bank.first_audio_at = time.perf_counter()
        bank.frames.append(frames)
        bank.seconds += seconds
        if self._mode == STREAMING and bank.seconds >= self._hold():
            await self._release()

    def _hold(self) -> float:
        """Audio to bank before the first sound.

        `BANK_S` while the writer is still writing, because how much reply
        is still to come is unknown. Once the writer has finished the rest is
        in hand, and holding more than it needs at this voice's measured pace
        is silence for nothing: a two-sentence reply at 1.05x waited three
        seconds it did not need to. The estimate uses the slow-side priors,
        so it errs toward holding.
        """
        pacer = self._pacer
        if pacer is None or not pacer.ended or self._measured_rtf is None:
            return BANK_S
        rest = [max(self._inflight_est - self._inflight_done, 0.0)]
        rest += [spoken_seconds(s) for s in pacer.remaining()]
        return min(BANK_S, bank_needed(self._measured_rtf, rest))

    async def _release(self) -> None:
        bank = self._bank
        if bank.released or self._gone.is_set():
            return
        bank.released = True
        if bank.first_audio_at is not None:
            self._bank_wait_ms = (time.perf_counter() - bank.first_audio_at) * 1000
        frames, seconds = b"".join(bank.frames), bank.seconds
        bank.frames.clear()
        bank.seconds = 0.0
        await self._send_audio(frames, seconds)

    async def _send_audio(self, frames: bytes, seconds: float) -> None:
        if self._first_release is None:
            self._first_release = time.perf_counter()
            self._first_audio_ms = (self._first_release - self._started) * 1000
        # The lead as this audio lands: what the listener still held. The
        # first release holds nothing by definition and is not a measurement.
        lead = self._lead()
        if (
            lead is not None
            and self._sent_audio_s > 0
            and (self._min_lead is None or lead < self._min_lead)
        ):
            self._min_lead = lead
            self._gap_at = self._requests
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

    def _done(self) -> dict:
        return {
            "type": "done",
            # How the reply was spoken. Buffered is about releasing, not
            # cutting, so a one-request reply held to its end is buffered too.
            "mode": self._mode,
            "batches": self._requests,
            "audio_seconds": round(self._sent_audio_s, 3),
            # What the model was busy for, as opposed to what the listener
            # waited: the sum of the requests' render time, and that over
            # the audio it produced. One reply's aggregate, not the card's
            # figure, which is a median of per-request factors.
            "render_ms": round(self._render_s * 1000, 1),
            # Where the wait went. The model being made resident is paid
            # before `ready`; the writer's time runs from `ready` to `end`;
            # the bank is what the first audio waited between being rendered
            # and being released.
            "load_ms": round(
                ((self._ready_at or self._started) - self._started) * 1000, 1
            ),
            "writer_ms": (
                round((self._ended_at - (self._ready_at or self._started)) * 1000, 1)
                if self._ended_at is not None
                else None
            ),
            "bank_wait_ms": (
                round(self._bank_wait_ms, 1) if self._bank_wait_ms is not None else None
            ),
            "rtf": self._rtf(),
            "first_audio_ms": round(self._first_audio_ms or 0.0, 1),
            "min_lead_s": round(self._min_lead, 3)
            if self._min_lead is not None
            else None,
            # The request whose audio landed at the lowest lead — where a
            # reply that ran dry went quiet.
            "gap_at": self._gap_at,
            "wall_ms": round((time.perf_counter() - self._started) * 1000, 1),
        }


def _field(value: object) -> str:
    """A log field: the value, or `-` for none."""
    return "-" if value is None else str(value)


@live.websocket("/speak/live")
async def speak_live(
    websocket: WebSocket, state: AppState = Depends(get_state_ws)
) -> None:
    """Speak a reply as it is written; see the module docstring for the frames."""
    # A client that offered subprotocols is dropped by the browser unless one
    # is selected, and the admin UI offers them to carry its key.
    offered = websocket.scope.get("subprotocols") or []
    await websocket.accept(
        subprotocol=WS_SUBPROTOCOL if WS_SUBPROTOCOL in offered else None
    )
    session = _Session(websocket, state)
    with contextlib.suppress(WebSocketDisconnect):
        await session.run()
