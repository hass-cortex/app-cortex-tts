"""HTTP routes.

Synthesis is the hot path and everything else exists to support it: the model
routes get weights onto disk, the reference routes define cloned voices, and
the preview route lets an operator see what the text pipeline will actually
hand the model before spending a synthesis on it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi import status as http_status

from ..audio import CONTENT_TYPES, AudioFormat, encode
from ..catalog import CATALOG, ModelSpec, inspect
from ..download import remove_bundle
from ..engine.base import EngineError, NoAudioError, UnknownVoiceError
from ..engine.registry import ModelNotReadyError, UnknownModelError
from ..events import fire_models_changed
from ..refs import Reference, ReferenceError
from ..text.pipeline import TextOptions, prepare
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
    SpeakRequest,
    SpeakStats,
    VoiceOut,
)

_LOGGER = logging.getLogger(__name__)

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
        version=state.version, loaded_models=len(state.registry.loaded_ids)
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
        model=state.settings.default_model, voice=state.settings.default_voice
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _model_out(state: AppState, spec: ModelSpec) -> ModelOut:
    model_state = inspect(
        state.settings.data_dir, spec, loaded=state.registry.is_loaded(spec.id)
    )
    progress = state.downloads.status(spec.id)
    return ModelOut(
        id=spec.id,
        name=spec.name,
        description=spec.description,
        kind=str(spec.kind),
        languages=list(spec.languages),
        sample_rate=spec.sample_rate,
        size_mb=spec.size_mb,
        rtf_hint=spec.rtf_hint,
        rss_hint_mb=spec.rss_hint_mb,
        recommended=spec.recommended,
        downloaded=model_state.downloaded,
        loaded=model_state.loaded,
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
    await state.registry.unload(model_id)
    remove_bundle(state.settings.data_dir, spec)
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
        if not inspect(state.settings.data_dir, spec).downloaded:
            continue
        try:
            voices = await state.registry.voices(spec.id)
        except EngineError as err:
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
        original=body.text, prepared="".join(segments), segments=segments
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


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
    model_id = model or state.settings.default_model
    spec = _spec_or_404(state, model_id)

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

    # Resolving the default voice loads the model, so it fails the same way a
    # synthesis would and has to be guarded the same way.
    with _engine_errors():
        voice_id = voice or await _default_voice(state, model_id)

    if voice_id is None:
        raise _http(
            http_status.HTTP_409_CONFLICT,
            "NO_VOICE",
            f"{spec.name} has no voices — upload a reference recording first",
        )

    with _engine_errors():
        result = await state.registry.synthesize(
            model_id, segments, voice_id, temperature=temperature
        )

    audio = encode(result.audio, result.sample_rate, fmt, normalize=normalize_level)
    seconds = len(result.audio) / result.sample_rate
    stats = SpeakStats(
        model_id=model_id,
        voice=voice_id,
        segments=result.segments,
        characters=len(text),
        audio_seconds=round(seconds, 3),
        inference_ms=round(result.inference_ms, 1),
        rtf=round(result.inference_ms / 1000 / seconds, 3) if seconds else 0.0,
        prepared_text="".join(segments),
    )
    _LOGGER.info(
        "spoke %d chars as %s/%s -> %.2fs audio in %.0fms (RTF %.2f)",
        stats.characters,
        model_id,
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
            "X-Hojo-Model": stats.model_id,
            "X-Hojo-Voice": stats.voice,
            "X-Hojo-Inference-Ms": str(stats.inference_ms),
            "X-Hojo-Audio-Seconds": str(stats.audio_seconds),
            "X-Hojo-Rtf": str(stats.rtf),
            "X-Hojo-Segments": str(stats.segments),
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
        fmt=body.format,
        normalize_text=body.normalize_text,
        convert_script=body.convert_script,
        normalize_level=body.normalize_level,
        temperature=body.temperature,
    )
    return _audio_response(audio, body.format, stats)


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
    payload = await audio.read()
    try:
        reference = state.references.add(
            name=name,
            transcript=transcript,
            audio=payload,
            language=language,
            gender=gender,
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
        content=reference.audio_path.read_bytes(),
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
    if state.settings.default_voice in ids:
        return state.settings.default_voice
    return voices[0].id
