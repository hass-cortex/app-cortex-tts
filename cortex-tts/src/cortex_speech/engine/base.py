"""The contract every synthesis backend satisfies."""

from __future__ import annotations

from collections.abc import Generator
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
        language: Base language code, or ``None`` when the voice is
            language-agnostic (a cloned reference recording).
        gender: ``female``, ``male`` or ``unknown``.
        source: ``builtin`` for bundled voices, ``reference`` for cloned ones.
    """

    id: str
    name: str
    language: str | None
    gender: str
    source: str


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
        self, segments: list[str], voice: str, *, temperature: float | None = None
    ) -> Synthesis:
        """Render prepared text segments as one continuous waveform.

        Args:
            segments: Already normalised and script-converted text.
            voice: A voice id the registry listed for this model.
            temperature: Overrides the engine's configured sampling
                temperature for this call. ``None`` uses the configured one.

        Returns:
            The rendered utterance.

        Raises:
            UnknownVoiceError: The voice id is not available.
        """
        ...

    def forget(self, reference_id: str) -> None:
        """Drop whatever was cached for a reference recording.

        Called when a reference is edited or deleted. An engine with no
        reference-derived voices has nothing to drop.
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
        self, segments: list[str], voice: str
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
        """
        ...


class EngineError(RuntimeError):
    """Base class for engine failures surfaced to the API."""


class UnknownVoiceError(EngineError):
    """The requested voice does not exist on this engine."""


class NoAudioError(EngineError):
    """The model produced no audio tokens for the given text."""


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
