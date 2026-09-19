"""The 40M engine: fixed voices, no reference audio."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np

from ..providers import (
    ExecutionProvider,
    in_use,
    release_sessions,
    requested,
    sessions_of,
    verify,
)
from ..vendor.hojo40 import VOICES_NPZ_NAME, HojoTTSLightOnnx
from .base import (
    AbandonedError,
    Delivery,
    NoAudioError,
    StopCheck,
    Synthesis,
    UnknownVoiceError,
    Voice,
    check_stop,
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

# The seed this runtime samples with unless told otherwise, matching the
# default in `vendor/hojo40.py`'s `generate`. Tried first, so an ordinary
# request gets what the model produces upstream; `overrun.RETRY_SEEDS` only
# follow when a generation truncates.
_UPSTREAM_SEED = 42


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


def own_voices(directory: Path) -> list[Voice]:
    """Read the bundle's voice ids off disk, opening no ONNX session.

    Named separately from the engine because listing voices must not load a
    model — see `EngineRegistry.voices`. The vendored `SpeakerVoices` reader
    would materialise the speaker embeddings and the shared token-embedding
    table on the way to the ids; the npz is a zip, so naming the one member
    reads only that member.
    """
    with np.load(directory / VOICES_NPZ_NAME, allow_pickle=True) as bundle:
        # `sorted`, because that is the order the engine reports.
        ids = sorted(str(voice) for voice in bundle["voice_ids"])
    return [_describe(voice_id) for voice_id in ids]


class PresetEngine:
    """Wraps the vendored 40M runtime with voice metadata and segment joining."""

    def __init__(
        self,
        models_dir: Path,
        *,
        num_threads: int = 0,
        temperature: float = 0.8,
        execution_provider: ExecutionProvider = "auto",
    ) -> None:
        """Load the 40M bundle.

        Args:
            models_dir: Directory holding the downloaded ONNX bundle.
            num_threads: ONNX Runtime thread count; 0 lets ORT decide.
            temperature: Default sampling temperature.
            execution_provider: Which provider to ask ONNX Runtime for.

        Raises:
            ProviderUnavailableError: `cuda` was required and CPU is what the
                sessions came back on.
        """
        self._temperature = temperature
        started = time.perf_counter()
        self._model = HojoTTSLightOnnx(
            str(models_dir),
            num_threads=num_threads,
            provider=requested(execution_provider),
        )
        self.provider = verify(execution_provider, in_use(sessions_of(self._model)))
        self._voices = [_describe(v) for v in self._model.available_voices]
        _LOGGER.info(
            "loaded 40M bundle from %s in %.2fs on %s (%d voices)",
            models_dir,
            time.perf_counter() - started,
            self.provider,
            len(self._voices),
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return int(self._model.sample_rate)

    def forget(self, reference_id: str) -> None:
        """Nothing to drop: this engine's voices are baked into the bundle."""

    def close(self) -> None:
        """Release the sessions; see `Engine.close`."""
        release_sessions(self._model)

    def _render(
        self, text: str, voice: str, temperature: float, stop: StopCheck | None
    ) -> np.ndarray:
        """Render one segment, seeding each attempt for the shared retry."""

        def generate(seed: int) -> np.ndarray:
            try:
                return self._model.generate(
                    text,
                    voice=voice,
                    temperature=temperature,
                    seed=seed,
                    on_step=lambda: check_stop(stop),
                )
            except AbandonedError:
                # An EngineError is a RuntimeError; the guard below must not
                # relabel a listener leaving as the model producing nothing.
                raise
            except RuntimeError as err:
                # No audio tokens at all: punctuation-only text, or a segment
                # left empty upstream. Nothing downstream can recover.
                raise NoAudioError(f"model produced no audio for {text!r}") from err

        return render_with_retries(
            text, self.sample_rate, generate, seed=_UPSTREAM_SEED
        )

    def synthesize(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Synthesis:
        """Render segments with a bundled voice."""
        if voice not in {v.id for v in self._voices}:
            raise UnknownVoiceError(f"unknown voice {voice!r}")

        started = time.perf_counter()
        temperature = (
            self._temperature if delivery.temperature is None else delivery.temperature
        )
        waves = [self._render(text, voice, temperature, stop) for text in segments]

        if not waves:
            raise NoAudioError("no segments to synthesize")

        return Synthesis(
            audio=join_segments(waves, self.sample_rate),
            sample_rate=self.sample_rate,
            segments=len(waves),
            inference_ms=(time.perf_counter() - started) * 1000,
        )
