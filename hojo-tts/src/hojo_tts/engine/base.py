"""The contract every synthesis backend satisfies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..refs import ReferenceStore


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

    def voices(self) -> list[Voice]:
        """Return the voices this engine can currently produce."""
        ...

    def synthesize(
        self, segments: list[str], voice: str, *, temperature: float | None = None
    ) -> Synthesis:
        """Render prepared text segments as one continuous waveform.

        Args:
            segments: Already normalised and script-converted text.
            voice: A voice id returned by :meth:`voices`.
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
