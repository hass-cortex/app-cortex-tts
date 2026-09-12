"""HTTP routes.

Synthesis is the hot path and everything else exists to support it: the model
routes get weights onto disk, the reference routes define cloned voices, and
the preview route lets an operator see what the text pipeline will actually
hand the model before spending a synthesis on it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from typing import NamedTuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi import status as http_status
from fastapi.responses import StreamingResponse

from cortex_speech import (
    CATALOG,
    CONTENT_TYPES,
    MAX_REFERENCE_SECONDS,
    STREAM_ENCODERS,
    AudioFormat,
    EngineError,
    ModelNotReadyError,
    ModelSpec,
    NoAudioError,
    ProviderUnavailableError,
    Reference,
    ReferenceError,
    StreamGain,
    TextOptions,
    UnknownModelError,
    UnknownVoiceError,
    encode,
    prepare,
    prepared_text,
)

from ..events import fire_models_changed
from ..preferences import save as save_preferences
from .deps import AppState, get_state, require_api_key
from .schemas import (
    DefaultsResponse,
    ErrorResponse,
    HealthResponse,
    ModelOut,
    OpenAISpeechRequest,
    PreviewRequest,
    PreviewResponse,
    ReferenceOut,
    ReferenceUpdate,
    SettingsOut,
    SettingsSaved,
    SettingsUpdate,
    SpeakRequest,
    SpeakStats,
    VoiceOut,
)

_LOGGER = logging.getLogger(__name__)

# A 20 s reference at 48 kHz 24-bit stereo is under 6 MB; this leaves room
# for a lossless container without spooling an arbitrary upload to disk.
MAX_REFERENCE_UPLOAD_BYTES = 32 * 1_000_000


public = APIRouter()
api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])
compat = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@public.get("/health", response_model=HealthResponse)
async def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Liveness probe. Deliberately unauthenticated, like the STT sibling."""
    return HealthResponse(
        version=state.version,
        loaded_models=len(state.registry.loaded_ids),
        execution_provider=state.preferences.execution_provider,
        providers_in_use=state.registry.providers_in_use,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@api.get("/defaults", response_model=DefaultsResponse)
async def defaults(state: AppState = Depends(get_state)) -> DefaultsResponse:
    """Report the configured default model and voice.

    The web UI has no other way to know them, and without it the picker
    silently opens on the first voice in the list rather than the one the
    addon is configured to speak with.
    """
    return DefaultsResponse(
        model=state.preferences.default_model, voice=state.preferences.default_voice
    )


@api.get("/settings", response_model=SettingsOut)
async def read_settings(state: AppState = Depends(get_state)) -> SettingsOut:
    """Report how the app is currently configured."""
    return SettingsOut(**asdict(state.preferences))


@api.put("/settings", response_model=SettingsSaved)
async def write_settings(
    body: SettingsUpdate, state: AppState = Depends(get_state)
) -> SettingsSaved:
    """Change settings, keeping what was not sent.

    These used to be addon options, where every change cost a restart and a
    change to their schema cost a rebuild. Nothing here needs that: most are
    read afresh on the next request, and the two that are bound when a
    session is created are adopted by dropping what is resident — the next
    reply pays a rebuild instead of the user paying a restart.

    A value that fails validation is ignored with a warning rather than
    rejecting the whole form, so one bad number cannot discard the model
    someone chose in another box. What comes back is what is now in force.
    """
    updated, ignored = state.preferences.validated(body.model_dump(exclude_none=True))
    try:
        save_preferences(state.preferences_path, updated)
    except OSError as err:
        raise _http(
            http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            "SETTINGS_NOT_WRITTEN",
            f"could not store the settings: {err}",
        ) from err

    # The registry knows which of these a resident engine can adopt and
    # drops only for the ones it cannot.
    reloaded = await state.registry.reconfigure(
        num_threads=updated.num_threads,
        max_loaded=updated.max_loaded_models,
        temperature=updated.temperature,
        execution_provider=updated.execution_provider,
    )

    state.preferences = updated
    _LOGGER.info("settings changed%s", " (models unloaded)" if reloaded else "")
    return SettingsSaved(
        settings=SettingsOut(**asdict(updated)), reloaded=reloaded, ignored=ignored
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _model_out(state: AppState, spec: ModelSpec) -> ModelOut:
    model_state = state.speech.model(spec)
    progress = state.downloads.status(spec.id)
    return ModelOut(
        id=spec.id,
        name=spec.name,
        description=spec.description,
        builtin_voices=spec.builtin_voices,
        cloning=spec.cloning,
        chunk_streaming=spec.chunk_streaming,
        temperature=spec.temperature,
        languages=list(spec.languages),
        sample_rate=spec.sample_rate,
        size_mb=spec.size_mb,
        rtf_hint=spec.rtf_hint,
        rss_hint_mb=spec.rss_hint_mb,
        recommended=spec.recommended,
        downloaded=model_state.downloaded,
        loaded=model_state.loaded,
        provider=state.registry.providers_in_use.get(spec.id),
        missing_files=model_state.missing,
        disk_bytes=model_state.disk_bytes,
        download_state=progress.state if progress else None,
        download_percent=progress.percent if progress else None,
        download_error=progress.error or None if progress else None,
    )


@api.get("/models", response_model=list[ModelOut])
async def list_models(state: AppState = Depends(get_state)) -> list[ModelOut]:
    """List every catalog model with its download and load state."""
    return [_model_out(state, spec) for spec in CATALOG]


@api.post("/models/{model_id}/download", response_model=ModelOut)
async def download_model(
    model_id: str, state: AppState = Depends(get_state)
) -> ModelOut:
    """Start (or rejoin) a background download of a model bundle."""
    spec = _spec_or_404(state, model_id)
    state.downloads.start(spec)
    return _model_out(state, spec)


@api.delete("/models/{model_id}", response_model=ModelOut)
async def delete_model(model_id: str, state: AppState = Depends(get_state)) -> ModelOut:
    """Unload a model and delete its bundle from disk."""
    spec = _spec_or_404(state, model_id)
    if state.downloads.is_running(model_id):
        # The worker would keep writing into the directory being removed and
        # a half bundle would reappear behind the delete.
        raise _http(
            http_status.HTTP_409_CONFLICT,
            "DOWNLOAD_RUNNING",
            f"{spec.name} is still downloading; wait for it to finish",
        )
    await state.registry.unload(model_id)
    # Up to two gigabytes of files; not on the event loop.
    await asyncio.to_thread(state.speech.delete_model, spec)
    await fire_models_changed(f"deleted:{model_id}")
    return _model_out(state, spec)


@api.post("/models/{model_id}/load", response_model=ModelOut)
async def load_model(model_id: str, state: AppState = Depends(get_state)) -> ModelOut:
    """Load a model into memory ahead of the first synthesis."""
    spec = _spec_or_404(state, model_id)
    with _engine_errors():
        await state.registry.acquire(model_id)
    return _model_out(state, spec)


@api.post("/models/{model_id}/unload", response_model=ModelOut)
async def unload_model(model_id: str, state: AppState = Depends(get_state)) -> ModelOut:
    """Drop a model from memory, leaving its bundle on disk."""
    spec = _spec_or_404(state, model_id)
    await state.registry.unload(model_id)
    return _model_out(state, spec)


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------


@api.get("/voices", response_model=list[VoiceOut])
async def list_voices(
    model: str | None = None, state: AppState = Depends(get_state)
) -> list[VoiceOut]:
    """List voices across downloaded models, or for one model.

    Only downloaded models are consulted: enumerating a model's voices means
    loading it, and a request for the voice list should not silently pull a
    400 MB download.
    """
    wanted = [s for s in CATALOG if model is None or s.id == model]
    if model is not None and not wanted:
        raise _http(
            http_status.HTTP_404_NOT_FOUND, "UNKNOWN_MODEL", f"no model {model!r}"
        )

    out: list[VoiceOut] = []
    for spec in wanted:
        if not state.speech.model(spec).downloaded:
            continue
        try:
            voices = await state.registry.voices(spec.id)
        except Exception as err:  # noqa: BLE001 - one broken bundle must not empty the list
            # A truncated bundle raises whatever its reader trips over; the
            # other models' voices are still worth answering with.
            _LOGGER.warning("could not list voices for %s: %s", spec.id, err)
            continue
        out.extend(
            VoiceOut(
                id=v.id,
                name=v.name,
                language=v.language,
                gender=v.gender,
                source=v.source,
                model_id=spec.id,
            )
            for v in voices
        )
    return out


# ---------------------------------------------------------------------------
# Text preview
# ---------------------------------------------------------------------------


@api.post("/preview", response_model=PreviewResponse)
async def preview_text(body: PreviewRequest) -> PreviewResponse:
    """Show what the text pipeline would hand the model, without synthesising."""
    options = TextOptions(
        normalize_text=body.normalize_text, convert_script=body.convert_script
    )
    segments = prepare(body.text, options)
    return PreviewResponse(
        original=body.text, prepared=prepared_text(segments), segments=segments
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class _Resolved(NamedTuple):
    """What both synthesis endpoints need before they can differ."""

    spec: ModelSpec
    segments: list[str]
    voice_id: str

    @property
    def model_id(self) -> str:
        """The id the request resolved to; `ModelSpec` already carries it."""
        return self.spec.id


async def _resolve(
    state: AppState,
    *,
    text: str,
    model: str | None,
    voice: str | None,
    normalize_text: bool,
    convert_script: bool,
    temperature: float | None = None,
) -> _Resolved:
    """Turn a request into everything a synthesis needs, or raise.

    Shared by both endpoints so they cannot disagree about which model a
    request meant, what the text prepares to, or which voice answers — and so
    the streaming one raises every one of these before its first byte, after
    which the status is already 200 and a failure can only truncate the audio.
    """
    model_id = model or state.preferences.default_model
    spec = _spec_or_404(state, model_id)

    # Answered from the catalog, so it does not wait behind a model that has
    # not been downloaded — a caller asking for a knob this model does not
    # have should hear that, not "not downloaded".
    if temperature is not None and not spec.temperature:
        raise _http(
            http_status.HTTP_400_BAD_REQUEST,
            "NO_TEMPERATURE",
            f"{spec.name} has no sampling temperature to set",
        )

    segments = prepare(
        text,
        TextOptions(normalize_text=normalize_text, convert_script=convert_script),
    )
    if not segments:
        raise _http(
            http_status.HTTP_400_BAD_REQUEST,
            "EMPTY_TEXT",
            "nothing to say once punctuation was stripped",
        )

    # Resolving the default voice reads the bundle off disk, so a model that
    # is not downloaded fails here, the same way a synthesis would.
    with _engine_errors():
        voice_id = voice or await _default_voice(state, model_id)

    if voice_id is None:
        raise _http(
            http_status.HTTP_409_CONFLICT,
            "NO_VOICE",
            f"{spec.name} has no voices — upload a reference recording first",
        )

    # A named voice is checked here, not left to the engine. The engine raises
    # when it resolves the voice, which on the streaming path happens after the
    # first frame has gone out — the caller then gets a 200, a header with
    # nothing behind it and no error at all. Voice ids are case-sensitive, so `yuewen`
    # for `Yuewen` is the easy way to hit that.
    if voice:
        with _engine_errors():
            known = {v.id for v in await state.registry.voices(model_id)}
        if voice not in known:
            match = next((v for v in known if v.lower() == voice.lower()), None)
            hint = f"; did you mean {match!r}?" if match else ""
            raise _http(
                http_status.HTTP_404_NOT_FOUND,
                "UNKNOWN_VOICE",
                f"{spec.name} has no voice {voice!r}{hint}",
            )
    return _Resolved(spec, segments, voice_id)


async def _synthesize(
    state: AppState,
    *,
    text: str,
    model: str | None,
    voice: str | None,
    fmt: AudioFormat,
    normalize_text: bool = True,
    convert_script: bool = True,
    normalize_level: bool = True,
    temperature: float | None = None,
) -> tuple[bytes, SpeakStats]:
    spec, segments, voice_id = await _resolve(
        state,
        text=text,
        model=model,
        voice=voice,
        normalize_text=normalize_text,
        convert_script=convert_script,
        temperature=temperature,
    )

    with _engine_errors():
        result = await state.registry.synthesize(
            spec.id, segments, voice_id, temperature=temperature
        )

    audio = encode(result.audio, result.sample_rate, fmt, normalize=normalize_level)
    seconds = len(result.audio) / result.sample_rate
    stats = SpeakStats(
        model_id=spec.id,
        voice=voice_id,
        segments=result.segments,
        characters=len(text),
        audio_seconds=round(seconds, 3),
        inference_ms=round(result.inference_ms, 1),
        rtf=round(result.inference_ms / 1000 / seconds, 3) if seconds else 0.0,
        prepared_text=prepared_text(segments),
    )
    _LOGGER.info(
        "spoke %d chars as %s/%s -> %.2fs audio in %.0fms (RTF %.2f)",
        stats.characters,
        spec.id,
        voice_id,
        stats.audio_seconds,
        stats.inference_ms,
        stats.rtf,
    )
    return audio, stats


def _audio_response(audio: bytes, fmt: AudioFormat, stats: SpeakStats) -> Response:
    return Response(
        content=audio,
        media_type=CONTENT_TYPES.get(fmt, "application/octet-stream"),
        headers={
            "X-Cortex-Model": stats.model_id,
            "X-Cortex-Voice": stats.voice,
            "X-Cortex-Inference-Ms": str(stats.inference_ms),
            "X-Cortex-Audio-Seconds": str(stats.audio_seconds),
            "X-Cortex-Rtf": str(stats.rtf),
            "X-Cortex-Segments": str(stats.segments),
            "Cache-Control": "no-store",
        },
    )


@api.post("/speak", responses={200: {"content": {"audio/wav": {}}}})
async def speak(body: SpeakRequest, state: AppState = Depends(get_state)) -> Response:
    """Synthesise text and return the audio file."""
    audio, stats = await _synthesize(
        state,
        text=body.text,
        model=body.model,
        voice=body.voice,
        fmt=body.format or "wav",
        normalize_text=body.normalize_text,
        convert_script=body.convert_script,
        normalize_level=body.normalize_level,
        temperature=body.temperature,
    )
    return _audio_response(audio, body.format or "wav", stats)


@api.post(
    "/speak/stream",
    responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}}}},
    response_class=StreamingResponse,
)
async def speak_stream(
    body: SpeakRequest, state: AppState = Depends(get_state)
) -> StreamingResponse:
    """Synthesise and send audio as it is produced, not when it is finished.

    Default `mp3`, because a streamed format must be writable without knowing
    how long the audio will be. MP3 is a bare sequence of self-describing
    frames and needs nothing declared; WAV has to declare a length it cannot
    know, and a general-purpose player given the maximal one waits for a file
    it believes is six hours long. `wav` is still available for a consumer
    that wants raw PCM. Measured on MOSS-TTS-Nano: 143-178
    ms to the first chunk against 1.8 s for the whole utterance.

    Every model works here. One that cannot emit mid-utterance sends its whole
    waveform as a single chunk, which is no worse than `/api/speak` and keeps
    callers from having to ask which kind of model they picked.

    `flac` and `ogg` are refused rather than quietly answered in another
    format: both need a size or an index in a header that would have to be
    written before the audio exists. `temperature` is refused for a model that
    has none. `normalize_level` is accepted and ignored: it asks for the
    finished waveform to be scaled to a target peak, which needs a finished
    waveform. What a stream gets instead is `StreamGain`, which keeps chunks
    under the same ceiling without ever raising them — see its docstring for
    why the two are not the same thing.

    `X-Cortex-Bitrate` is the one measurement that can be sent, because it is
    known before the first sample: a constant bitrate is what lets a consumer
    turn a byte count into a duration without decoding. The rest of what
    `/api/speak` returns describes a finished synthesis, and the headers are
    gone before the first sample exists.
    """
    requested = body.format or "mp3"
    build = STREAM_ENCODERS.get(requested)
    if build is None:
        raise _http(
            http_status.HTTP_400_BAD_REQUEST,
            "UNSUPPORTED_FORMAT",
            f"a stream cannot be {requested}: it would have to declare a size "
            f"or an index before the audio exists. Available: "
            f"{', '.join(sorted(STREAM_ENCODERS))}",
        )
    encoder = build()

    spec, segments, voice_id = await _resolve(
        state,
        text=body.text,
        model=body.model,
        voice=body.voice,
        normalize_text=body.normalize_text,
        convert_script=body.convert_script,
        temperature=body.temperature,
    )

    # Loading is the last thing that can fail, and it must fail here: once the
    # generator below has yielded the first frame the status is 200. Resolving
    # the voice read the bundle off disk but built nothing.
    with _engine_errors():
        await state.registry.acquire(spec.id)

    async def frames() -> AsyncIterator[bytes]:
        gain = StreamGain()
        yield encoder.open(spec.sample_rate)
        async for chunk in state.registry.synthesize_stream(
            spec.id, segments, voice_id
        ):
            yield encoder.encode(gain.frames(chunk))
        yield encoder.close()

    return StreamingResponse(
        frames(),
        media_type=encoder.content_type,
        headers={
            "X-Cortex-Model": spec.id,
            "X-Cortex-Voice": voice_id,
            "X-Cortex-Chunk-Streaming": "1" if spec.chunk_streaming else "0",
            # The one measurement that exists before the first sample, and the
            # only way a consumer can tell how much audio it is holding
            # without decoding it.
            "X-Cortex-Bitrate": str(encoder.bitrate or spec.sample_rate * 16),
            # Nothing downstream may buffer this; the point is the first frame.
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@compat.post("/audio/speech", responses={200: {"content": {"audio/wav": {}}}})
async def openai_speech(
    body: OpenAISpeechRequest, state: AppState = Depends(get_state)
) -> Response:
    """OpenAI-compatible synthesis, so existing clients work unchanged."""
    audio, stats = await _synthesize(
        state,
        text=body.input,
        model=body.model,
        voice=body.voice,
        fmt=body.response_format,
    )
    return _audio_response(audio, body.response_format, stats)


# ---------------------------------------------------------------------------
# Reference recordings (cloned voices)
# ---------------------------------------------------------------------------


def _reference_out(ref: Reference) -> ReferenceOut:
    return ReferenceOut(
        id=ref.id,
        name=ref.name,
        transcript=ref.transcript,
        raw_transcript=ref.raw_transcript,
        language=ref.language,
        gender=ref.gender,
        seconds=ref.seconds,
        created=ref.created,
    )


@api.get("/references", response_model=list[ReferenceOut])
async def list_references(state: AppState = Depends(get_state)) -> list[ReferenceOut]:
    """List stored reference recordings."""
    return [_reference_out(r) for r in state.references.list()]


@api.post(
    "/references",
    response_model=ReferenceOut,
    status_code=http_status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}},
)
async def add_reference(
    name: str = Form(...),
    transcript: str = Form(...),
    language: str = Form("zh"),
    gender: str = Form("unknown"),
    audio: UploadFile = File(...),
    state: AppState = Depends(get_state),
) -> ReferenceOut:
    """Store a reference recording and make it available as a cloned voice."""
    if audio.size is not None and audio.size > MAX_REFERENCE_UPLOAD_BYTES:
        raise _http(
            http_status.HTTP_413_CONTENT_TOO_LARGE,
            "BAD_REFERENCE",
            f"upload is {audio.size / 1_000_000:.0f} MB; a reference is at most "
            f"{MAX_REFERENCE_SECONDS:.0f} s of audio, well under "
            f"{MAX_REFERENCE_UPLOAD_BYTES // 1_000_000} MB",
        )
    payload = await audio.read()
    try:
        # Decoding, writing and hashing the recording is real work; a stream
        # on another connection must not stall behind it.
        reference = await asyncio.to_thread(
            partial(
                state.references.add,
                name=name,
                transcript=transcript,
                audio=payload,
                language=language,
                gender=gender,
            )
        )
    except ReferenceError as err:
        raise _http(
            http_status.HTTP_400_BAD_REQUEST, "BAD_REFERENCE", str(err)
        ) from err
    state.registry.forget_reference(reference.id)
    await fire_models_changed(f"reference-added:{reference.id}")
    return _reference_out(reference)


@api.patch(
    "/references/{reference_id}",
    response_model=ReferenceOut,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def update_reference(
    reference_id: str,
    body: ReferenceUpdate,
    state: AppState = Depends(get_state),
) -> ReferenceOut:
    """Correct a reference's transcript.

    A transcript that does not match the recording degrades the clone without
    any error, so this is the fix for a mistyped or mis-transcribed upload —
    without asking for the audio again.
    """
    try:
        reference = state.references.update(reference_id, transcript=body.transcript)
    except KeyError as err:
        raise _no_reference(reference_id) from err
    except ReferenceError as err:
        raise _http(
            http_status.HTTP_400_BAD_REQUEST, "BAD_REFERENCE", str(err)
        ) from err
    return _reference_out(reference)


@api.delete("/references/{reference_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_reference(
    reference_id: str, state: AppState = Depends(get_state)
) -> Response:
    """Delete a reference recording and the cloned voice it backed."""
    if not state.references.remove(reference_id):
        raise _no_reference(reference_id)
    state.registry.forget_reference(reference_id)
    await fire_models_changed(f"reference-removed:{reference_id}")
    return Response(status_code=http_status.HTTP_204_NO_CONTENT)


@api.get("/references/{reference_id}/audio")
async def reference_audio(
    reference_id: str, state: AppState = Depends(get_state)
) -> Response:
    """Return the stored recording, so the UI can play back what was uploaded."""
    reference = state.references.get(reference_id)
    if reference is None:
        raise _no_reference(reference_id)
    return Response(
        content=await asyncio.to_thread(reference.audio_path.read_bytes),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# One place decides what an engine failure looks like on the wire. Subclasses
# come first: NoAudioError and UnknownVoiceError are both EngineError.
_WIRE_ERRORS: tuple[tuple[type[Exception], int, str], ...] = (
    (ModelNotReadyError, http_status.HTTP_409_CONFLICT, "MODEL_NOT_READY"),
    (UnknownVoiceError, http_status.HTTP_404_NOT_FOUND, "UNKNOWN_VOICE"),
    (NoAudioError, http_status.HTTP_422_UNPROCESSABLE_CONTENT, "NO_AUDIO"),
    (
        ProviderUnavailableError,
        http_status.HTTP_503_SERVICE_UNAVAILABLE,
        "PROVIDER_UNAVAILABLE",
    ),
    (EngineError, http_status.HTTP_500_INTERNAL_SERVER_ERROR, "ENGINE_ERROR"),
)


@contextmanager
def _engine_errors() -> Iterator[None]:
    """Translate engine failures into the HTTP shape the API promises."""
    try:
        yield
    except tuple(kind for kind, _, _ in _WIRE_ERRORS) as err:
        for kind, status_code, code in _WIRE_ERRORS:
            if isinstance(err, kind):
                raise _http(status_code, code, str(err)) from err
        raise


def _no_reference(reference_id: str) -> HTTPException:
    """Return the 404 every reference route raises for an unknown id."""
    return _http(
        http_status.HTTP_404_NOT_FOUND,
        "UNKNOWN_REFERENCE",
        f"no reference {reference_id!r}",
    )


def _http(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code, detail={"code": code, "message": message}
    )


def _spec_or_404(state: AppState, model_id: str):
    try:
        return state.registry.spec(model_id)
    except UnknownModelError as err:
        raise _http(
            http_status.HTTP_404_NOT_FOUND, "UNKNOWN_MODEL", f"no model {model_id!r}"
        ) from err


async def _default_voice(state: AppState, model_id: str) -> str | None:
    """Pick a voice when the request named none.

    The configured default wins when the model actually offers it; otherwise
    the first available voice is used, which is what makes a freshly-uploaded
    cloning reference work without also configuring it as the default.
    """
    voices = await state.registry.voices(model_id)
    if not voices:
        return None
    ids = {v.id for v in voices}
    if state.preferences.default_voice in ids:
        return state.preferences.default_voice
    return voices[0].id
