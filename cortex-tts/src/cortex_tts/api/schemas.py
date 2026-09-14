"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cortex_speech import AudioFormat, Delivery

API_VERSION = 1
"""Bumped when a route, field or header the integration reads changes shape.

The app's release version says nothing about the wire; this does, and it is
what a client compares before trusting anything else it reads."""


class HealthResponse(BaseModel):
    """Unauthenticated liveness probe."""

    status: Literal["ok"] = "ok"
    version: str
    server: str = "cortex-tts"
    api_version: int = API_VERSION
    loaded_models: int
    loading_models: list[str] = []
    """Ids of the models being built right now, at most one.

    A bundle takes seconds to become a session, and `loaded_models` counts
    zero for the whole of that — which reads exactly like idle. This is how a
    caller tells "nothing is happening" from "wait, it is coming"."""
    execution_provider: str
    """What was asked for — `auto`, `cpu` or `cuda`."""
    providers_in_use: dict[str, str] = {}
    """What each resident model actually got, keyed by model id.

    Empty until something is loaded, because nothing is known before a session
    exists. `auto` on a host whose CUDA libraries do not load reports `cpu`
    here while still reporting `auto` above, and that difference is the whole
    point of sending both."""


class DefaultsResponse(BaseModel):
    """What the app picks when a request names no model or voice.

    These are the configured names, not a resolved choice: naming a voice the
    installed models do not offer is allowed, and the caller falls back the
    same way synthesis does.
    """

    model: str
    voice: str


class VoiceOut(BaseModel):
    """A selectable voice."""

    id: str
    name: str
    language: str | None
    gender: str
    source: str
    model_id: str


class MeasuredRtf(BaseModel):
    """A real-time factor this host measured, for one kind of voice."""

    kind: str
    """`builtin`, `designed` or `reference` — see `Voice.source`."""
    rtf: float
    samples: int
    """How many syntheses `rtf` is the median of."""


class ModelOut(BaseModel):
    """A catalog entry with its runtime state."""

    id: str
    name: str
    description: str
    builtin_voices: bool
    designed_voices: bool
    cloning: bool
    chunk_streaming: bool
    temperature: bool
    language_choice: bool
    """Whether a request may name the language. False where the voice decides."""
    style_instruction: bool
    """Whether a request may carry a free-text instruction beside the voice."""
    languages: list[str]
    sample_rate: int
    size_mb: int
    rss_hint_mb: int
    rtf: list[MeasuredRtf] = []
    """What this host measured, one entry per kind of voice; empty until it has.

    Never a figure from anywhere else — see `cortex_tts.stats`."""
    downloaded: bool
    loaded: bool
    provider: str | None = None
    """What this model's sessions are actually running on, once loaded.

    Read from the sessions rather than from the setting: a runtime that lists
    a CUDA provider it cannot load falls back with only a warning, so the two
    can disagree and only this one is evidence."""
    missing_files: list[str]
    disk_bytes: int
    download_state: str | None = None
    download_percent: float | None = None
    download_error: str | None = None


class SpeakRequest(BaseModel):
    """A synthesis request.

    The text switches exist because the pipeline they control is the
    difference between an intelligible voice and noise on this model; they are
    exposed so a caller that has already normalised its text can say so, not
    so they can be turned off casually.
    """

    text: str = Field(min_length=1, max_length=4000)
    model: str | None = None
    voice: str | None = None
    format: AudioFormat | None = None
    """Container to answer in. `/api/speak` defaults to wav and
    `/api/speak/stream` to mp3, because a stream has to be writable
    without knowing how long the audio will be."""
    normalize_text: bool = True
    convert_script: bool = True
    normalize_level: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    language: str | None = Field(default=None, max_length=32)
    """Which language to read the text as: a whole tag, `zh-TW` or `zh`.

    Sent whole rather than reduced by the caller, because how much of it means
    anything is the model's to say — one of these names two Chinese dialects,
    another names 646 languages including several narrower than `zh`. The
    engine tries the tag, then its shorter forms.

    Only models declaring `language_choice`; the rest are told the voice
    decides, rather than accepting it and doing nothing with it."""
    instruct: str | None = Field(default=None, max_length=200)
    """A free-text instruction beside the voice: "speak slowly, in a warm tone".

    Only models declaring `style_instruction`."""

    def delivery(self) -> Delivery:
        """The three fields as the one object the engines read.

        Built here, once, rather than carried as three parameters down to
        wherever an engine is finally called: three is where a widening
        signature starts costing every function in between, which is the
        reason `Delivery` exists at all."""
        return Delivery(
            temperature=self.temperature,
            language=self.language,
            instruct=self.instruct,
        )


class SpeakStats(BaseModel):
    """What a synthesis cost, returned as response headers and in previews."""

    model_id: str
    voice: str
    segments: int
    characters: int
    audio_seconds: float
    inference_ms: float
    rtf: float
    prepared_text: str


class ReferenceOut(BaseModel):
    """A stored reference recording."""

    id: str
    name: str
    transcript: str
    raw_transcript: str
    language: str
    gender: str
    seconds: float
    created: float


class ReferenceUpdate(BaseModel):
    """Correct the transcript of a stored reference.

    The audio is not resent — only the text the model is told it contains.
    """

    transcript: str = Field(min_length=1, max_length=2000)


class PreviewRequest(BaseModel):
    """Dry-run of the text path, with no synthesis."""

    text: str = Field(min_length=1, max_length=4000)
    normalize_text: bool = True
    convert_script: bool = True


class PreviewResponse(BaseModel):
    """What the model would actually be asked to say."""

    original: str
    prepared: str
    segments: list[str]


class ErrorResponse(BaseModel):
    """Uniform error body."""

    code: str
    message: str


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech request.

    Present so existing clients and scripts work unchanged; ``model`` carries
    a Hojo model id and ``voice`` a voice id from this server.
    """

    input: str = Field(min_length=1, max_length=4000)
    model: str | None = None
    voice: str | None = None
    response_format: AudioFormat = "wav"


class SettingsOut(BaseModel):
    """How the app behaves, as the user last set it."""

    num_threads: int
    execution_provider: str
    max_loaded_models: int
    idle_unload_seconds: int
    max_synthesis_seconds: int
    default_model: str
    default_voice: str
    temperature: float
    preload: bool


class SettingsUpdate(BaseModel):
    """A partial change. Anything omitted keeps its current value.

    Every field is optional so the UI can send one box rather than the whole
    form — and so a field this app has not heard of, from a newer UI, is
    ignored rather than resetting the rest.
    """

    num_threads: int | None = None
    execution_provider: str | None = None
    max_loaded_models: int | None = None
    idle_unload_seconds: int | None = None
    max_synthesis_seconds: int | None = None
    default_model: str | None = None
    default_voice: str | None = None
    temperature: float | None = None
    preload: bool | None = None


class SettingsSaved(BaseModel):
    """What was stored, and whether it is speaking yet."""

    settings: SettingsOut
    reloaded: bool
    """Whether resident models were dropped to adopt this.

    Thread count and execution provider are bound when ONNX Runtime creates a
    session, so they cannot be adopted by an engine already loaded. True means
    the next reply pays a rebuild; the rest of the settings are read afresh on
    every request and are already in force.
    """
    ignored: list[str] = []
    """Fields whose value was not usable and kept what they had.

    One bad number must not discard the model someone chose in another box,
    so the save goes through — but a caller told "saved" with nothing to say
    which field was refused has no way to notice. This is that list."""
