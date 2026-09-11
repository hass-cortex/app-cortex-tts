"""The 40M engine: fixed voices, no reference audio."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np

from ..vendor.hojo40 import HojoTTSLightOnnx
from .base import (
    NoAudioError,
    Synthesis,
    UnknownVoiceError,
    Voice,
)
from .join import join_segments
from .overrun import render_with_retries

_LOGGER = logging.getLogger(__name__)

# Voice ids are opaque upstream (`hojo_{lang}_{sex}_{nn}`); the shape is stable
# enough to label the picker without shipping a hand-maintained table that
# would silently go stale when the bundle gains a voice.
_VOICE_ID = re.compile(r"^hojo_(?P<lang>[a-z]{2})_(?P<sex>[fmu])_(?P<index>\d+)$")
_SEX = {"f": "female", "m": "male", "u": "unknown"}
_LANG_LABEL = {"zh": "Chinese", "en": "English"}


def _describe(voice_id: str) -> Voice:
    match = _VOICE_ID.match(voice_id)
    if not match:
        return Voice(
            id=voice_id,
            name=voice_id,
            language=None,
            gender="unknown",
            source="builtin",
        )
    lang = match["lang"]
    gender = _SEX.get(match["sex"], "unknown")
    label = _LANG_LABEL.get(lang, lang.upper())
    return Voice(
        id=voice_id,
        name=f"{label} {gender} {int(match['index'])}",
        language=lang,
        gender=gender,
        source="builtin",
    )


class PresetEngine:
    """Wraps the vendored 40M runtime with voice metadata and segment joining."""

    def __init__(
        self, models_dir: Path, *, num_threads: int = 0, temperature: float = 0.8
    ) -> None:
        """Load the 40M bundle.

        Args:
            models_dir: Directory holding the downloaded ONNX bundle.
            num_threads: ONNX Runtime thread count; 0 lets ORT decide.
            temperature: Default sampling temperature.
        """
        self._temperature = temperature
        started = time.perf_counter()
        self._model = HojoTTSLightOnnx(str(models_dir), num_threads=num_threads)
        self._voices = [_describe(v) for v in self._model.available_voices]
        _LOGGER.info(
            "loaded 40M bundle from %s in %.2fs (%d voices)",
            models_dir,
            time.perf_counter() - started,
            len(self._voices),
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return int(self._model.sample_rate)

    def forget(self, reference_id: str) -> None:
        """Nothing to drop: this engine's voices are baked into the bundle."""

    def voices(self) -> list[Voice]:
        """Return the bundled voices."""
        return list(self._voices)

    def _render(self, text: str, voice: str, temperature: float) -> np.ndarray:
        """Render one segment, seeding each attempt for the shared retry."""

        def generate(seed: int) -> np.ndarray:
            try:
                return self._model.generate(
                    text, voice=voice, temperature=temperature, seed=seed
                )
            except RuntimeError as err:
                # No audio tokens at all: punctuation-only text, or a segment
                # left empty upstream. Nothing downstream can recover.
                raise NoAudioError(f"model produced no audio for {text!r}") from err

        return render_with_retries(text, self.sample_rate, generate)

    def synthesize(
        self, segments: list[str], voice: str, *, temperature: float | None = None
    ) -> Synthesis:
        """Render segments with a bundled voice."""
        if voice not in {v.id for v in self._voices}:
            raise UnknownVoiceError(f"unknown voice {voice!r}")

        started = time.perf_counter()
        waves: list[np.ndarray] = []
        for text in segments:
            waves.append(
                self._render(
                    text,
                    voice,
                    self._temperature if temperature is None else temperature,
                )
            )

        if not waves:
            raise NoAudioError("no segments to synthesize")

        return Synthesis(
            audio=join_segments(waves, self.sample_rate),
            sample_rate=self.sample_rate,
            segments=len(waves),
            inference_ms=(time.perf_counter() - started) * 1000,
        )
