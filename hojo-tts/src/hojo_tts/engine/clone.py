"""The 80M engine: every voice is a cloned reference recording.

The vendored ``generate`` re-encodes the reference on every call — once
through the codec encoder and once through the speaker encoder — even when a
hundred consecutive utterances use the same voice. This engine hoists both
encodes into a per-reference cache and composes the remaining steps itself,
which is the difference between paying that cost once and paying it per
sentence of a multi-segment response.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ..refs import Reference, ReferenceStore
from ..vendor.hojo80 import (
    HojoTTSLightOnnx,
    _build_ref_codec_prompt,
    _wav_from_mag_phase,
)
from .base import (
    NoAudioError,
    Synthesis,
    UnknownVoiceError,
    Voice,
    reference_voices,
)
from .join import join_segments
from .overrun import render_with_retries

_LOGGER = logging.getLogger(__name__)


class _PromptCache:
    """Codec and speaker encodings for one reference recording."""

    __slots__ = ("codes", "speaker_vec", "fingerprint")

    def __init__(
        self, codes: np.ndarray, speaker_vec: np.ndarray, fingerprint: str
    ) -> None:
        self.codes = codes
        self.speaker_vec = speaker_vec
        self.fingerprint = fingerprint


class CloneEngine:
    """Wraps the vendored 80M runtime, caching per-reference encodings."""

    def __init__(
        self,
        models_dir: Path,
        references: ReferenceStore,
        *,
        num_threads: int = 0,
        temperature: float = 0.8,
    ) -> None:
        """Load the 80M bundle.

        Args:
            models_dir: Directory holding the downloaded ONNX bundle.
            references: Store of uploaded reference recordings.
            num_threads: ONNX Runtime thread count; 0 lets ORT decide.
            temperature: Default sampling temperature.
        """
        self._temperature = temperature
        started = time.perf_counter()
        self._model = HojoTTSLightOnnx(str(models_dir), num_threads=num_threads)
        self._references = references
        self._prompts: dict[str, _PromptCache] = {}
        _LOGGER.info(
            "loaded 80M bundle from %s in %.2fs",
            models_dir,
            time.perf_counter() - started,
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return int(self._model.sample_rate)

    def voices(self) -> list[Voice]:
        """Return one voice per stored reference recording."""
        return reference_voices(self._references)

    def forget(self, reference_id: str) -> None:
        """Drop cached encodings for a reference that changed or was deleted."""
        self._prompts.pop(reference_id, None)

    def _prompt_for(self, ref: Reference) -> _PromptCache:
        """Return cached encodings for a reference, encoding on first use.

        The fingerprint check means replacing a recording under the same id
        invalidates the cache instead of silently synthesising the old voice.
        """
        cached = self._prompts.get(ref.id)
        if cached is not None and cached.fingerprint == ref.fingerprint:
            return cached

        started = time.perf_counter()
        audio_path = str(ref.audio_path)
        prompt = _PromptCache(
            codes=self._model._encode_ref_codes(audio_path),
            speaker_vec=self._model._encode_speaker(audio_path),
            fingerprint=ref.fingerprint,
        )
        self._prompts[ref.id] = prompt
        _LOGGER.info(
            "encoded reference %s (%d codec tokens) in %.2fs",
            ref.id,
            prompt.codes.size,
            time.perf_counter() - started,
        )
        return prompt

    def _render(
        self, text: str, ref_text: str, prompt: _PromptCache, temperature: float
    ) -> np.ndarray:
        """Render one segment, retrying a generation that stopped early.

        Sampling is seeded so the same text yields the same audio, matching
        the runtime this wraps. That also means a seed which truncates would
        truncate for good, so a short result is retried with the next seed
        rather than returned as a sentence that stops mid-way.
        """

        def generate(seed: int) -> np.ndarray:
            np.random.seed(seed)
            input_ids = _build_ref_codec_prompt(
                self._model.tokenizer, ref_text, text, prompt.codes
            )
            generated, last_hidden = self._model._generate_coarse_tokens(
                input_ids,
                max_new_tokens=2048,
                min_new_tokens=10,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.1,
            )
            try:
                mag, phase = self._model._decode_from_coarse(
                    input_ids, generated, last_hidden, prompt.speaker_vec
                )
            except RuntimeError as err:
                raise NoAudioError(f"model produced no audio for {text!r}") from err
            return _wav_from_mag_phase(mag, phase, self._model.istft)

        return render_with_retries(text, self.sample_rate, generate)

    def synthesize(
        self, segments: list[str], voice: str, *, temperature: float | None = None
    ) -> Synthesis:
        """Render segments in a cloned voice."""
        ref = self._references.get(voice)
        if ref is None:
            raise UnknownVoiceError(f"unknown reference voice {voice!r}")

        prompt = self._prompt_for(ref)
        started = time.perf_counter()
        waves: list[np.ndarray] = []

        for text in segments:
            waves.append(
                self._render(
                    text,
                    ref.transcript,
                    prompt,
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
