"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..audio import AudioFormat


class HealthResponse(BaseModel):
    """Unauthenticated liveness probe."""

    status: Literal["ok"] = "ok"
    version: str
    server: str = "hojo-tts"
    loaded_models: int


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


class ModelOut(BaseModel):
    """A catalog entry with its runtime state."""

    id: str
    name: str
    description: str
    kind: str
    languages: list[str]
    sample_rate: int
    size_mb: int
    rtf_hint: float
    rss_hint_mb: int
    recommended: bool
    downloaded: bool
    loaded: bool
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
    format: AudioFormat = "wav"
    normalize_text: bool = True
    convert_script: bool = True
    normalize_level: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)


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
