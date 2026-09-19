"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from cortex_speech import AudioFormat, Delivery

API_VERSION = 4
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
    """What this host measured for one kind of voice.

    The fit a live reply is planned from — `fixed_s + per_audio × audio` — which
    is also where a card's figure comes from. `per_audio` is the slope, and a
    real-time factor is `fixed_s / audio + per_audio`, so `audio_ref_s` says
    at what length to quote it — the mean request this host served.
    """

    kind: str
    """`builtin`, `designed` or `reference` — see `Voice.source`."""
    voice: str | None = None
    """Which voice, when the kind is `reference`: a clone's cost follows the
    length of its own recording, so clones are measured one voice at a time."""
    per_audio: float
    """Render seconds per audio second."""
    fixed_s: float
    """Render seconds a request costs before any audio."""
    audio_ref_s: float
    """The mean audio of the requests fitted: where to quote the factor."""
    spread_s: float
    """One standard deviation of what the line failed to explain."""
    cjk_per_s: float
    latin_per_s: float
    requests: int
    """How many requests the fit rests on."""


class ModelOut(BaseModel):
    """A catalog entry with its runtime state."""

    id: str
    name: str
    description: str
    builtin_voices: bool
    designed_voices: bool
    cloning: bool
    reads_reference_transcript: bool
    """Whether a clone's transcript reaches this model. Only meaningful with
    `cloning`: every cloning engine takes the recording, not every one is
    also told what it says."""
    chunk_streaming: bool
    temperature: bool
    language_choice: bool
    """Whether a request may name the language. False where the voice decides."""
    style_instruction: bool
    """Whether a request may carry a free-text instruction beside the voice."""
    reads_numerals: bool
    """Whether the model reads digits itself where the pipeline has no locale
    written for the language; the generic number expansion then stands aside."""
    needs_number_words: bool
    """Whether the model cannot say a digit at all, so a bare number is read
    as a quantity for it unless the request says otherwise."""
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


class SpeakCommon(BaseModel):
    """What both ways of asking for speech settle.

    A request for a file and the opening frame of a live reply differ in how
    the words arrive and in what container can carry them; everything else —
    which model and voice, how the text is prepared, how it is delivered — is
    the same question, asked once here so the two cannot drift into answering
    it differently.

    The text switches exist because the pipeline they control is the
    difference between an intelligible voice and noise on this model; they are
    exposed so a caller that has already normalised its text can say so, not
    so they can be turned off casually.
    """

    model: str | None = None
    voice: str | None = None
    normalize_text: bool | None = None
    """Expand units, clock literals and dates into words. Left out, a
    settings rule for the model and language may answer; otherwise on."""
    expand_numbers: bool | None = None
    """Read a bare number — no unit, clock or date around it — as a quantity.
    Left out, a settings rule may answer; otherwise the model decides: on for
    one that cannot say a digit at all (Hojo), off for the rest, because a
    bare number is as often a phone number, a room or a model as a count,
    and a wrong reading misleads."""
    convert_script: bool | None = None
    """Chinese only. Left out, a settings rule may answer; otherwise on."""
    taiwan_readings: bool | None = None
    """Chinese only. Left out, a settings rule may answer; otherwise on for
    `zh-TW` and `zh-Hant`, off elsewhere."""
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    language: str | None = Field(default=None, max_length=32)
    """Which language to read the text as: a whole tag, `zh-TW` or `zh`.

    It picks the text pipeline's locale on every model — which words numbers
    become, and which rewrites run. On a model declaring `language_choice` it
    is also what the model is told to read the text as, sent whole rather than
    reduced, because how much of a tag means anything is the model's to say:
    one names two Chinese dialects, another 646 languages. Left out, the
    voice's language is used, and failing that the text is sniffed."""
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


class LiveStart(SpeakCommon):
    """The opening frame of `/api/speak/live`: everything but the words.

    The text follows in `text` frames as the writer produces it; this frame
    settles what cannot change once audio has started.
    """

    type: Literal["start"] = "start"
    format: Literal["mp3", "wav"] = "mp3"
    """MP3 unless raw PCM is wanted: a stream cannot declare a length."""
    mode: Literal["auto", "buffered", "planned", "unheld", "streaming"] = "auto"
    """`auto` lets the server choose from what it has measured, and is what
    anything serving a listener should send.

    The other three insist, for a caller comparing one delivery against
    another on the same reply: `buffered` releases nothing until the whole
    reply is rendered, `planned` waits for the whole reply and then plans it,
    and `streaming` renders as the text arrives even where the server would
    have judged the model unable to keep ahead. Insisting is honoured as far
    as the reply allows — streaming needs a cost line before the first byte,
    and a host that has none still paces — so the `done` frame, not this
    field, is what says how the reply went."""


class LiveText(BaseModel):
    """A piece of the reply, as the writer produced it."""

    type: Literal["text"] = "text"
    text: str = Field(max_length=4000)


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
    """What the model is told the recording says, stored as it was typed.
    References written before that rule may carry a pipeline-prepared form
    here that differs from `raw_transcript`."""
    raw_transcript: str
    language: str
    gender: str
    seconds: float
    created: float


Gender = Literal["female", "male", "unknown"]


class ReferenceUpdate(BaseModel):
    """Correct the name, transcript, gender label or language of a reference.

    The audio is not resent — only what the voice is called, the text the
    model is told it contains, how the voice is labelled in the picker, or
    which language it speaks.
    """

    name: str | None = Field(None, max_length=200)
    """What the voice is called in the picker. The id is not re-derived from
    it: that is what a synthesis request names, so renaming must not move it.
    Sent empty, the reference falls back to being known by its id."""
    transcript: str | None = Field(None, min_length=1, max_length=2000)
    """Exactly what the recording says. Stored as sent: the text pipeline
    runs only to refuse a transcript with nothing pronounceable in it, never
    to rewrite one. A transcript that does not match the audio degrades the
    clone and reports nothing, so the caller's wording wins."""
    gender: Gender | None = None
    language: str | None = Field(None, min_length=1, max_length=32)
    """A whole tag: `zh-TW` makes the voice Taiwanese, so text read in it
    gets Taiwan readings by default; `zh` says only Chinese."""

    @model_validator(mode="after")
    def _something_to_change(self) -> ReferenceUpdate:
        if (
            self.name is None
            and self.transcript is None
            and self.gender is None
            and self.language is None
        ):
            raise ValueError(
                "send a name, a transcript, a gender, a language, or several"
            )
        return self


class PreviewRequest(BaseModel):
    """Dry-run of the text path, with no synthesis."""

    text: str = Field(min_length=1, max_length=4000)
    model: str | None = None
    """The model the text is meant for, as on a reply: one that reads
    digits itself changes what is prepared for a language without a locale.
    Left out, the default model's answer."""
    language: str | None = Field(default=None, max_length=32)
    """The language of the text, as on a reply; sniffed when left out."""
    normalize_text: bool | None = None
    expand_numbers: bool | None = None
    convert_script: bool | None = None
    taiwan_readings: bool | None = None


class Reading(BaseModel):
    """One word rewritten for its Taiwan reading."""

    word: str
    standin: str


class PreviewResponse(BaseModel):
    """What the model would actually be asked to say, and why."""

    original: str
    prepared: str
    segments: list[str]
    language: str
    """The tag the text was read as — the request's, or sniffed."""
    passes: dict[str, bool]
    """Every switch this request had, and whether it ran: `normalize_text` and
    `expand_numbers` always, `convert_script` and `taiwan_readings` on Chinese
    only. `expand_numbers` is the one the model decides rather than the
    language, from `needs_number_words`."""
    readings: list[Reading]
    """The Taiwan-reading rewrites, in text order; empty when the pass is off."""


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


class TextRule(BaseModel):
    """What the text switches default to for a model, a language, or both.

    A request that leaves a switch out gets the rule's answer; a switch left
    out of the rule (or null) stays the pipeline's call. Rules cascade per
    switch, the most specific one that says something winning: `model` and
    `language` both set beats either alone, which beats neither. `language`
    matches a tag it equals or prefixes (`zh` covers `zh-TW`)."""

    model: str | None = None
    language: str | None = None
    normalize_text: bool | None = None
    expand_numbers: bool | None = None
    convert_script: bool | None = None
    taiwan_readings: bool | None = None


class SettingsOut(BaseModel):
    """How the app behaves, as the user last set it."""

    num_threads: int
    execution_provider: str
    max_loaded_models: int
    idle_unload_seconds: int
    default_model: str
    default_voice: str
    temperature: float
    preload: bool
    max_sentence_pause: float
    text_rules: list[TextRule]


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
    default_model: str | None = None
    default_voice: str | None = None
    temperature: float | None = None
    preload: bool | None = None
    max_sentence_pause: float | None = Field(default=None, ge=0.0, le=10.0)
    text_rules: list[TextRule] | None = None
    """The whole list; sending one replaces what was stored."""


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
