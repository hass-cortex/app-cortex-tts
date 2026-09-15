"""The contract every synthesis backend satisfies."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ..references import ReferenceStore


@dataclass(frozen=True)
class Voice:
    """A selectable voice.

    Attributes:
        id: Identifier passed back in a synthesis request.
        name: Human-readable label.
        language: Base language code, or ``None`` when the voice reads
            whatever it is given — OmniVoice's designed voices. A cloned
            voice carries the language of its recording, which says what was
            said in it rather than what the voice may be asked to say.
        gender: ``female``, ``male`` or ``unknown``.
        source: ``builtin`` for voices shipped in the bundle, ``designed``
            for ones the model builds from an attribute vocabulary, and
            ``reference`` for a cloned recording. Three rather than two
            because they cost different amounts to render — measured on
            OmniVoice, a designed voice against a clone of a ten-second
            recording: RTF 3.46 against 7.17 on the same host — so a caller
            comparing them has to know which it is looking at.
    """

    id: str
    name: str
    language: str | None
    gender: str
    source: str


@dataclass(frozen=True)
class Delivery:
    """How to say it — everything in a request that is neither words nor voice.

    One object rather than three keyword arguments, for the reason
    `BuildContext` is one: the third of them is where a widening signature
    starts costing every engine that does not care. An engine reads what it
    supports and ignores the rest; the API refuses what a model's catalog
    entry does not declare, so an ignored field is a caller's mistake rather
    than a silent one.

    Attributes:
        temperature: Sampling temperature, or ``None`` for the engine's own.
        language: Which language to read the text as, as a base code
            (``zh``, ``ja``). ``None`` lets the engine decide — from the
            voice, or from the script the text reads as. Only models
            declaring `language_choice` take one.
        instruct: A free-text instruction beside the voice ("speak slowly,
            in a warm tone"). Only models declaring `style_instruction` take
            one.
    """

    temperature: float | None = None
    language: str | None = None
    instruct: str | None = None


def narrowing(code: str) -> list[str]:
    """A language tag and the shorter forms to fall back to, longest first.

    `zh-Hant-TW` before `zh-Hant` before `zh`: the most specific thing the
    model might know, then the next. The caller sends the tag whole, because
    how much of it means anything is the model's to decide — one of these
    names two Chinese dialects, another names Cantonese and Min Nan apart
    from Chinese — so the narrowing happens here, against each model's own
    list, and not in whoever wrote the request.

    One rule for every engine: a tag cannot mean different amounts depending
    on which model reads it.
    """
    parts = code.strip().lower().replace("_", "-").split("-")
    return ["-".join(parts[: n + 1]) for n in reversed(range(len(parts)))]


@dataclass(frozen=True)
class Synthesis:
    """One rendered utterance.

    Attributes:
        audio: Mono float32 waveform in [-1, 1].
        sample_rate: Samples per second.
        segments: How many text segments were joined to produce it.
        inference_ms: Wall-clock spent inside the model.
    """

    audio: np.ndarray
    sample_rate: int
    segments: int
    inference_ms: float


class Engine(Protocol):
    """A loaded model that can turn prepared text into audio.

    Implementations are synchronous and CPU-bound; the API layer runs them in
    a worker thread and serialises calls per engine.
    """

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        ...

    def synthesize(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Synthesis:
        """Render prepared text segments as one continuous waveform.

        Args:
            segments: Already normalised and script-converted text.
            stop: Consulted between units of work; see `StopCheck`.
            voice: A voice id the registry listed for this model.
            delivery: How to say it. An engine reads the fields it supports.

        Returns:
            The rendered utterance.

        Raises:
            UnknownVoiceError: The voice id is not available.
            UnsupportedLanguageError: The language is not one this model reads.
        """
        ...

    def forget(self, reference_id: str) -> None:
        """Drop whatever was cached for a reference recording.

        Called when a reference is edited or deleted. An engine with no
        reference-derived voices has nothing to drop.
        """
        ...

    def close(self) -> None:
        """Release the sessions, now.

        A card's memory comes back when the sessions die, and the registry
        needs that to happen at the line it drops the engine — not when the
        collector next reaches whatever else held it. Idempotent; any call
        after this raises `SessionClosedError`.
        """
        ...


@runtime_checkable
class StreamingEngine(Protocol):
    """An engine that can emit audio before a segment is finished.

    Deliberately separate from `Engine` rather than an optional method on it.
    Two of the three engines cannot do this, and a protocol they would have to
    decline is a protocol that lies about them — the registry asks with
    `isinstance` and falls back, so an engine that stays silent about streaming
    is simply not asked.
    """

    def synthesize_stream(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """Yield mono float32 chunks in playback order.

        A generator rather than a plain iterator: the registry closes it when
        the consumer stops early, and the engine must stop rendering then.

        Chunks are not padded or aligned to segment boundaries; a consumer that
        needs whole segments should use `Engine.synthesize` instead. The caller
        is responsible for any gap between segments, because the engine no
        longer knows where one ended.

        Raises:
            UnknownVoiceError: The voice id is not available.
            NoAudioError: The model produced nothing for the text.
            AbandonedError: `stop` said the listener is gone.
        """
        ...


class EngineError(RuntimeError):
    """Base class for engine failures surfaced to the API."""


class AbandonedError(EngineError):
    """The listener went away, so the render stopped at its next checkpoint.

    Not a failure: nothing was wrong with the text or the model. Raised so the
    engine lock is released and the caller can tell an abandoned render from
    one that produced nothing.
    """


StopCheck = Callable[[], bool]
"""Asked at every checkpoint of a render; ``True`` means nobody is listening.

Passed in rather than raised at the engine, because only the transport knows
the connection is gone. An engine checks between the units it produces — a
decode step, a diffusion step, a codec chunk, a segment — since an ONNX
`run()` cannot be interrupted, so that unit is the most a lost listener costs.
"""


def check_stop(stop: StopCheck | None) -> None:
    """Raise `AbandonedError` when the stop check says to. No check, no cost."""
    if stop is not None and stop():
        raise AbandonedError("the listener went away")


class UnknownVoiceError(EngineError):
    """The requested voice does not exist on this engine."""


class NoAudioError(EngineError):
    """The model produced no audio tokens for the given text."""


class UnsupportedLanguageError(EngineError):
    """The model does not read the language the request named."""


def reference_voices(references: ReferenceStore) -> list[Voice]:
    """Return one voice per stored reference recording.

    Cloning models have no voices of their own — the uploaded recordings are
    the voices. Keeping this outside the engine means the list can be read
    without loading a multi-gigabyte bundle just to enumerate it.
    """
    return [
        Voice(
            id=ref.id,
            name=ref.name,
            language=ref.language,
            gender=ref.gender,
            source="reference",
        )
        for ref in references.list()
    ]
