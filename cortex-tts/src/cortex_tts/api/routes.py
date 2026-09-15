"""HTTP routes.

Synthesis is the hot path and everything else exists to support it: the model
routes get weights onto disk, the reference routes define cloned voices, and
the preview route lets an operator see what the text pipeline will actually
hand the model before spending a synthesis on it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import NamedTuple, Protocol

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi import status as http_status

from cortex_speech import (
    CATALOG,
    CONTENT_TYPES,
    MAX_REFERENCE_SECONDS,
    AbandonedError,
    AudioFormat,
    Delivery,
    EngineError,
    ModelNotReadyError,
    ModelSpec,
    NoAudioError,
    OutOfMemoryError,
    ProviderUnavailableError,
    Reference,
    ReferenceError,
    RenderSample,
    TextOptions,
    UnknownModelError,
    UnknownVoiceError,
    UnsupportedLanguageError,
    Voice,
    count_scripts,
    encode,
    plan,
    prepare,
    prepared_text,
    resolve_language,
    run,
    segment,
    taiwan_readings,
)

from ..events import fire_models_changed
from ..preferences import save as save_preferences
from .deps import AppState, get_state, require_api_key
from .schemas import (
    DefaultsResponse,
    ErrorResponse,
    Gender,
    HealthResponse,
    MeasuredRtf,
    ModelOut,
    OpenAISpeechRequest,
    PreviewRequest,
    PreviewResponse,
    Reading,
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

# How often a whole render asks whether its caller is still there.
_DISCONNECT_POLL_S = 0.25

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
        loading_models=sorted(state.registry.loading_ids),
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

    Nothing here costs a restart: most are read afresh on the next request,
    and the two that are bound when a session is created are adopted by
    dropping what is resident — the next reply pays a rebuild instead of the
    user paying a restart.

    A value that fails validation is ignored with a warning rather than
    rejecting the whole form, so one bad number cannot discard the model
    someone chose in another box. What comes back is what is now in force.
    """
    # One writer at a time, and the new settings are published before the
    # registry is told: this is a read-modify-write over `state.preferences`,
    # and `reconfigure` waits on a synthesis in flight. A second request
    # arriving inside that wait would otherwise read the settings this one
    # replaced and write them back over it, having answered 200.
    async with state.settings_lock:
        updated, ignored = state.preferences.validated(
            body.model_dump(exclude_none=True)
        )
        try:
            await asyncio.to_thread(save_preferences, state.preferences_path, updated)
        except OSError as err:
            raise http_error(
                http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                "SETTINGS_NOT_WRITTEN",
                f"could not store the settings: {err}",
            ) from err
        state.preferences = updated

        # The registry knows which of these a resident engine can adopt and
        # drops only for the ones it cannot.
        reloaded = await state.registry.reconfigure(
            num_threads=updated.num_threads,
            max_loaded=updated.max_loaded_models,
            idle_seconds=updated.idle_unload_seconds,
            temperature=updated.temperature,
            execution_provider=updated.execution_provider,
        )

    _LOGGER.info("settings changed%s", " (models unloaded)" if reloaded else "")
    return SettingsSaved(
        settings=SettingsOut(**asdict(updated)), reloaded=reloaded, ignored=ignored
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def voice_kind(state: AppState, spec: ModelSpec, voice_id: str) -> str:
    """Which `Voice.source` a rendered voice belongs to.

    A stored recording is a clone whichever model spoke it; anything else is
    the model's own, and the catalog says whether those are designed or
    bundled. No model has both, so the spec settles it without a lookup.
    """
    if state.references.get(voice_id) is not None:
        return "reference"
    return "designed" if spec.designed_voices else "builtin"


def _model_out(state: AppState, spec: ModelSpec) -> ModelOut:
    model_state = state.speech.model(spec)
    progress = state.downloads.status(spec.id)
    measured = state.stats.get(spec.id)
    return ModelOut(
        id=spec.id,
        name=spec.name,
        description=spec.description,
        builtin_voices=spec.builtin_voices,
        designed_voices=spec.designed_voices,
        cloning=spec.cloning,
        chunk_streaming=spec.chunk_streaming,
        temperature=spec.temperature,
        language_choice=spec.language_choice,
        style_instruction=spec.style_instruction,
        reads_numerals=spec.reads_numerals,
        needs_number_words=spec.needs_number_words,
        languages=list(spec.languages),
        sample_rate=spec.sample_rate,
        size_mb=spec.size_mb,
        rss_hint_mb=spec.rss_hint_mb,
        rtf=[
            MeasuredRtf(
                kind=m.kind,
                per_audio=m.render.per_audio,
                fixed_s=m.render.fixed_s,
                spread_s=m.render.spread_s,
                cjk_per_s=m.render.cjk_per_s,
                latin_per_s=m.render.latin_per_s,
                requests=m.render.samples,
            )
            for m in measured
        ],
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
    spec = spec_or_404(state, model_id)
    state.downloads.start(spec)
    return _model_out(state, spec)


@api.delete("/models/{model_id}", response_model=ModelOut)
async def delete_model(model_id: str, state: AppState = Depends(get_state)) -> ModelOut:
    """Unload a model and delete its bundle from disk."""
    spec = spec_or_404(state, model_id)
    if state.downloads.is_running(model_id):
        # The worker would keep writing into the directory being removed and
        # a half bundle would reappear behind the delete.
        raise http_error(
            http_status.HTTP_409_CONFLICT,
            "DOWNLOAD_RUNNING",
            f"{spec.name} is still downloading; wait for it to finish",
        )
    await state.registry.unload(model_id)
    # Up to two gigabytes of files; not on the event loop.
    await asyncio.to_thread(state.speech.delete_model, spec)
    await asyncio.to_thread(state.stats.forget, model_id)
    await fire_models_changed(f"deleted:{model_id}")
    return _model_out(state, spec)


@api.post("/models/{model_id}/load", response_model=ModelOut)
async def load_model(model_id: str, state: AppState = Depends(get_state)) -> ModelOut:
    """Load a model into memory ahead of the first synthesis."""
    spec = spec_or_404(state, model_id)
    with engine_errors():
        await state.registry.acquire(model_id)
    return _model_out(state, spec)


@api.post("/models/{model_id}/unload", response_model=ModelOut)
async def unload_model(model_id: str, state: AppState = Depends(get_state)) -> ModelOut:
    """Drop a model from memory, leaving its bundle on disk."""
    spec = spec_or_404(state, model_id)
    await state.registry.unload(model_id)
    return _model_out(state, spec)


@api.delete("/models/{model_id}/stats", response_model=ModelOut)
async def reset_model_stats(
    model_id: str, state: AppState = Depends(get_state)
) -> ModelOut:
    """Forget this model's measured real-time factors.

    A model's speed belongs to the host, and the host changes — a model moved
    onto the GPU, a thread count raised. The stored figures then misjudge it
    until enough new replies push them out of the window. This drops them so
    the next reply measures fresh.
    """
    spec = spec_or_404(state, model_id)
    await asyncio.to_thread(state.stats.forget, model_id)
    return _model_out(state, spec)


@api.delete("/stats", response_model=list[ModelOut])
async def reset_all_stats(state: AppState = Depends(get_state)) -> list[ModelOut]:
    """Forget every model's measured real-time factors at once."""
    await asyncio.to_thread(state.stats.clear)
    return [_model_out(state, spec) for spec in CATALOG]


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
        raise http_error(
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
async def preview_text(
    body: PreviewRequest, state: AppState = Depends(get_state)
) -> PreviewResponse:
    """Show what the text pipeline would hand the model, without synthesising."""
    spec = spec_or_404(state, body.model or state.preferences.default_model)
    text = body.text.strip()
    options = text_options(state, spec, text, body.language, body)
    decided = plan(
        text,
        options,
        body.language,
        reads_numerals=spec.reads_numerals,
        needs_number_words=spec.needs_number_words,
    )
    prepared = run(text, decided, options.normalize_options)
    segments = segment(prepared, stop=decided.locale.stop)
    readings: list[Reading] = []
    if decided.rewrites.get("taiwan_readings"):
        # The rewrites are read off the text the pass saw, not diffed back out
        # of the result — a stand-in is one glyph for one glyph, so a diff
        # could not tell two adjacent rewrites apart.
        before = run(
            text,
            replace(decided, rewrites={**decided.rewrites, "taiwan_readings": False}),
            options.normalize_options,
        )
        readings = [Reading(word=w, standin=s) for w, s in taiwan_readings(before)]
    return PreviewResponse(
        original=body.text,
        prepared=prepared_text(segments),
        segments=segments,
        language=decided.language,
        passes=decided.passes,
        readings=readings,
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class Switches(Protocol):
    """What a request said about the text switches; ``None`` is nothing."""

    @property
    def normalize_text(self) -> bool | None: ...
    @property
    def expand_numbers(self) -> bool | None: ...
    @property
    def convert_script(self) -> bool | None: ...
    @property
    def taiwan_readings(self) -> bool | None: ...


@dataclass(frozen=True)
class _Unasked:
    """A request that said nothing about the text switches."""

    normalize_text: bool | None = None
    expand_numbers: bool | None = None
    convert_script: bool | None = None
    taiwan_readings: bool | None = None


def text_options(
    state: AppState,
    spec: ModelSpec,
    text: str,
    language: str | None,
    asked: Switches,
) -> TextOptions:
    """The request's text switches, with the settings' rules behind them.

    A switch the request leaves out is answered by the rule for this model
    and the language the text resolves to; one no rule answers stays
    ``None`` for the pipeline to decide — except `normalize_text`, which
    the pipeline takes as a plain flag and is on unless something says no.
    """
    rule = state.preferences.text_defaults(spec.id, resolve_language(text, language))

    def settle(name: str) -> bool | None:
        value = getattr(asked, name)
        return getattr(rule, name) if value is None else value

    normalize = settle("normalize_text")
    return TextOptions(
        normalize_text=True if normalize is None else normalize,
        expand_numbers=settle("expand_numbers"),
        convert_script=settle("convert_script"),
        taiwan_readings=settle("taiwan_readings"),
    )


class _Resolved(NamedTuple):
    """What both synthesis endpoints need before they can differ."""

    spec: ModelSpec
    segments: list[str]
    voice_id: str
    delivery: Delivery
    """What the engine is told — the request's, with the language it may not
    take removed and the one the text was read as filled in where it may."""

    @property
    def model_id(self) -> str:
        """The id the request resolved to; `ModelSpec` already carries it."""
        return self.spec.id


def check_delivery(spec: ModelSpec, delivery: Delivery) -> None:
    """Refuse a knob the catalog says this model does not have.

    Answered from the catalog, so it does not wait behind a model that has
    not been downloaded — a caller asking for a knob this model does not
    have should hear that, not "not downloaded".
    """
    if delivery.temperature is not None and not spec.temperature:
        raise http_error(
            http_status.HTTP_400_BAD_REQUEST,
            "NO_TEMPERATURE",
            f"{spec.name} has no sampling temperature to set",
        )
    if delivery.instruct and not spec.style_instruction:
        raise http_error(
            http_status.HTTP_400_BAD_REQUEST,
            "NO_STYLE_INSTRUCTION",
            f"{spec.name} takes no style instruction",
        )


async def pick_voice(state: AppState, spec: ModelSpec, voice: str | None) -> Voice:
    """The voice a request meant, or raise.

    Listing the voices reads the bundle off disk, so a model that is not
    downloaded fails here, the same way a synthesis would. A named voice is
    checked here, not left to the engine, which would not raise until it
    resolves the voice — on a live reply that is after `ready` has gone out,
    and the caller is then waiting on audio that will never come. Voice ids
    are case-sensitive, so `yuewen` for `Yuewen` is the easy way to hit it.
    """
    with engine_errors():
        voices = await state.registry.voices(spec.id)
    voice_id = voice or _default_voice(state, voices)
    if voice_id is None:
        raise http_error(
            http_status.HTTP_409_CONFLICT,
            "NO_VOICE",
            f"{spec.name} has no voices — upload a reference recording first",
        )
    chosen = next((v for v in voices if v.id == voice_id), None)
    if chosen is None:
        match = next((v.id for v in voices if v.id.lower() == voice_id.lower()), None)
        hint = f"; did you mean {match!r}?" if match else ""
        raise http_error(
            http_status.HTTP_404_NOT_FOUND,
            "UNKNOWN_VOICE",
            f"{spec.name} has no voice {voice_id!r}{hint}",
        )
    return chosen


def prepare_segments(
    state: AppState,
    spec: ModelSpec,
    text: str,
    language: str | None,
    asked: Switches,
) -> list[str]:
    """Run the text path for one request; empty when there is nothing to say."""
    return prepare(
        text,
        text_options(state, spec, text, language, asked),
        language,
        reads_numerals=spec.reads_numerals,
        needs_number_words=spec.needs_number_words,
    )


def told_language(spec: ModelSpec, text: str, language: str | None) -> str | None:
    """What the engine is told to read the text as, where it takes a language.

    Only where the catalog says the model takes one; elsewhere the voice
    decides, and the tag has done its work in the pipeline. A sniffed tag is
    passed on too: what the pipeline read the text as is what the model
    should read it as.
    """
    return resolve_language(text, language) if spec.language_choice else None


async def _resolve(
    state: AppState,
    *,
    text: str,
    model: str | None,
    voice: str | None,
    asked: Switches,
    delivery: Delivery = Delivery(),
) -> _Resolved:
    """Turn a request into everything a synthesis needs, or raise.

    Every refusal happens here, before anything is rendered: which model a
    request meant, what the text prepares to, and which voice answers.
    """
    spec = spec_or_404(state, model or state.preferences.default_model)
    check_delivery(spec, delivery)

    # Whether there is anything to say does not depend on the voice, so it is
    # answered first — before a model with no voices, or one not downloaded,
    # gets to answer instead.
    segments = prepare_segments(state, spec, text, delivery.language, asked)
    if not segments:
        raise http_error(
            http_status.HTTP_400_BAD_REQUEST,
            "EMPTY_TEXT",
            "nothing to say once punctuation was stripped",
        )

    # The language the text is read in falls back to the voice's when the
    # request names none, so the text is prepared again once the voice is
    # known — only when that changes anything.
    chosen = await pick_voice(state, spec, voice)
    language = delivery.language or chosen.language
    if language != delivery.language:
        segments = prepare_segments(state, spec, text, language, asked)

    told = told_language(spec, text, language)
    return _Resolved(spec, segments, chosen.id, replace(delivery, language=told))


async def _synthesize(
    state: AppState,
    *,
    text: str,
    model: str | None,
    voice: str | None,
    fmt: AudioFormat,
    asked: Switches = _Unasked(),
    normalize_level: bool = True,
    delivery: Delivery = Delivery(),
    request: Request | None = None,
) -> tuple[bytes, SpeakStats]:
    spec, segments, voice_id, delivery = await _resolve(
        state, text=text, model=model, voice=voice, asked=asked, delivery=delivery
    )

    # A whole render sends nothing until it is done, so nothing about the
    # socket is learned by writing to it; the request is asked instead, and
    # the engine stops at its next checkpoint once the caller has gone.
    gone = asyncio.Event()

    async def watch() -> None:
        while request is not None and not gone.is_set():
            if await request.is_disconnected():
                gone.set()
                return
            await asyncio.sleep(_DISCONNECT_POLL_S)

    watcher = asyncio.create_task(watch())
    started = time.perf_counter()
    try:
        with engine_errors():
            result = await state.registry.synthesize(
                spec.id, segments, voice_id, delivery=delivery, stop=gone.is_set
            )
    except AbandonedError:
        _LOGGER.info("%s: caller left before the render finished", spec.id)
        raise http_error(499, "ABANDONED", "the caller went away") from None
    finally:
        gone.set()
        watcher.cancel()
    wall = time.perf_counter() - started

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
    # Split by the kind of voice this was — a clone costs about twice what a
    # designed voice does on the same model. The store is asked rather than
    # the voice list, because that is a dict lookup and the list is a disk
    # read on the hot path. The wall clock, not `inference_ms`: what the
    # caller waited is what the fit has to predict.
    kind = voice_kind(state, spec, voice_id)
    await asyncio.to_thread(
        state.stats.record, spec.id, kind, render_sample(text, seconds, wall)
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


def render_sample(text: str, audio_s: float, wall_s: float) -> RenderSample:
    """One request as the render model learns it, from the caller's own text."""
    cjk, latin = count_scripts(text)
    return RenderSample(audio_s=audio_s, wall_s=wall_s, cjk=cjk, latin=latin)


@api.post("/speak", responses={200: {"content": {"audio/wav": {}}}})
async def speak(
    body: SpeakRequest, request: Request, state: AppState = Depends(get_state)
) -> Response:
    """Synthesise text and return the audio file."""
    audio, stats = await _synthesize(
        state,
        text=body.text,
        model=body.model,
        voice=body.voice,
        fmt=body.format or "wav",
        asked=body,
        normalize_level=body.normalize_level,
        delivery=body.delivery(),
        request=request,
    )
    return _audio_response(audio, body.format or "wav", stats)


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
    gender: Gender = Form("unknown"),
    audio: UploadFile = File(...),
    state: AppState = Depends(get_state),
) -> ReferenceOut:
    """Store a reference recording and make it available as a cloned voice."""
    if audio.size is not None and audio.size > MAX_REFERENCE_UPLOAD_BYTES:
        raise http_error(
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
        raise http_error(
            http_status.HTTP_400_BAD_REQUEST, "BAD_REFERENCE", str(err)
        ) from err
    await state.registry.forget_reference(reference.id)
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
    """Correct a reference's transcript, gender label or language.

    A transcript that does not match the recording degrades the clone without
    any error, so this is the fix for a mistyped or mis-transcribed upload —
    without asking for the audio again.
    """
    try:
        reference = state.references.update(
            reference_id,
            transcript=body.transcript,
            gender=body.gender,
            language=body.language,
        )
    except KeyError as err:
        raise _no_reference(reference_id) from err
    except ReferenceError as err:
        raise http_error(
            http_status.HTTP_400_BAD_REQUEST, "BAD_REFERENCE", str(err)
        ) from err
    if body.gender is not None or body.language is not None:
        # Both are part of what the voice picker shows.
        await fire_models_changed(f"reference-updated:{reference_id}")
    return _reference_out(reference)


@api.delete("/references/{reference_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_reference(
    reference_id: str, state: AppState = Depends(get_state)
) -> Response:
    """Delete a reference recording and the cloned voice it backed."""
    if not state.references.remove(reference_id):
        raise _no_reference(reference_id)
    await state.registry.forget_reference(reference_id)
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
    (
        UnsupportedLanguageError,
        http_status.HTTP_400_BAD_REQUEST,
        "UNSUPPORTED_LANGUAGE",
    ),
    (NoAudioError, http_status.HTTP_422_UNPROCESSABLE_CONTENT, "NO_AUDIO"),
    (
        ProviderUnavailableError,
        http_status.HTTP_503_SERVICE_UNAVAILABLE,
        "PROVIDER_UNAVAILABLE",
    ),
    # 503 rather than 500: the engine has been dropped, so the same request a
    # moment later may well work. It is a condition, not a defect in the call.
    (
        OutOfMemoryError,
        http_status.HTTP_503_SERVICE_UNAVAILABLE,
        "OUT_OF_MEMORY",
    ),
    (EngineError, http_status.HTTP_500_INTERNAL_SERVER_ERROR, "ENGINE_ERROR"),
)


@contextmanager
def engine_errors() -> Iterator[None]:
    """Translate engine failures into the HTTP shape the API promises."""
    try:
        yield
    except AbandonedError:
        # Not a failure and not for the wire: the caller's own news coming
        # back, which each transport answers in its own way.
        raise
    except tuple(kind for kind, _, _ in _WIRE_ERRORS) as err:
        for kind, status_code, code in _WIRE_ERRORS:
            if isinstance(err, kind):
                raise http_error(status_code, code, str(err)) from err
        raise


def _no_reference(reference_id: str) -> HTTPException:
    """Return the 404 every reference route raises for an unknown id."""
    return http_error(
        http_status.HTTP_404_NOT_FOUND,
        "UNKNOWN_REFERENCE",
        f"no reference {reference_id!r}",
    )


def http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code, detail={"code": code, "message": message}
    )


def spec_or_404(state: AppState, model_id: str):
    try:
        return state.registry.spec(model_id)
    except UnknownModelError as err:
        raise http_error(
            http_status.HTTP_404_NOT_FOUND, "UNKNOWN_MODEL", f"no model {model_id!r}"
        ) from err


def _default_voice(state: AppState, voices: list[Voice]) -> str | None:
    """Pick a voice when the request named none.

    The configured default wins when the model actually offers it; otherwise
    the first available voice is used, which is what makes a freshly-uploaded
    cloning reference work without also configuring it as the default.
    """
    if not voices:
        return None
    ids = {v.id for v in voices}
    if state.preferences.default_voice in ids:
        return state.preferences.default_voice
    return voices[0].id
