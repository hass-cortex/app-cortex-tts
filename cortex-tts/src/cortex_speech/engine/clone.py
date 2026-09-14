"""The 80M engine: every voice is a cloned reference recording.

The vendored ``generate`` re-encodes the reference on every call — once
through the codec encoder and once through the speaker encoder — even when a
hundred consecutive utterances use the same voice. This engine hoists both
encodes into the shared conditioning cache and composes the remaining steps
itself, which is the difference between paying that cost once and paying it
per sentence of a multi-segment response.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..providers import (
    ExecutionProvider,
    in_use,
    release_sessions,
    requested,
    sessions_of,
    verify,
)
from ..references import Reference, ReferenceStore
from ..vendor.hojo80 import (
    HojoTTSLightOnnx,
    _build_ref_codec_prompt,
    _wav_from_mag_phase,
)
from .base import (
    Delivery,
    NoAudioError,
    Synthesis,
    UnknownVoiceError,
)
from .conditioning import ConditioningCache
from .join import join_segments
from .overrun import render_with_retries

_LOGGER = logging.getLogger(__name__)


class _Prompt(NamedTuple):
    """Codec and speaker encodings for one reference recording."""

    codes: np.ndarray
    speaker_vec: np.ndarray


class _PromptSidecar:
    suffix = ".hojo80.npz"

    def dump(self, value: _Prompt, path: Path) -> None:
        with path.open("wb") as handle:
            np.savez(handle, codes=value.codes, speaker_vec=value.speaker_vec)

    def load(self, path: Path) -> _Prompt:
        with np.load(path) as data:
            return _Prompt(codes=data["codes"], speaker_vec=data["speaker_vec"])


class CloneEngine:
    """Wraps the vendored 80M runtime, caching per-reference encodings."""

    def __init__(
        self,
        models_dir: Path,
        references: ReferenceStore,
        *,
        num_threads: int = 0,
        temperature: float = 0.8,
        execution_provider: ExecutionProvider = "auto",
    ) -> None:
        """Load the 80M bundle.

        Args:
            models_dir: Directory holding the downloaded ONNX bundle.
            references: Store of uploaded reference recordings.
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
        self._references = references
        self._prompts: ConditioningCache[_Prompt] = ConditioningCache(
            _PromptSidecar(), references.root
        )
        _LOGGER.info(
            "loaded 80M bundle from %s in %.2fs on %s",
            models_dir,
            time.perf_counter() - started,
            self.provider,
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return int(self._model.sample_rate)

    def forget(self, reference_id: str) -> None:
        """Drop cached encodings for a reference that changed or was deleted."""
        self._prompts.forget(reference_id)

    def close(self) -> None:
        """Release the sessions; see `Engine.close`."""
        release_sessions(self._model)

    def _encode(self, ref: Reference) -> _Prompt:
        """Run both of the runtime's encoders over a reference recording."""
        started = time.perf_counter()
        audio_path = str(ref.audio_path)
        prompt = _Prompt(
            codes=self._model._encode_ref_codes(audio_path),
            speaker_vec=self._model._encode_speaker(audio_path),
        )
        _LOGGER.info(
            "encoded reference %s (%d codec tokens) in %.2fs",
            ref.id,
            prompt.codes.size,
            time.perf_counter() - started,
        )
        return prompt

    def _render(
        self, text: str, ref_text: str, prompt: _Prompt, temperature: float
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
        self, segments: list[str], voice: str, *, delivery: Delivery = Delivery()
    ) -> Synthesis:
        """Render segments in a cloned voice."""
        ref = self._references.get(voice)
        if ref is None:
            raise UnknownVoiceError(f"unknown reference voice {voice!r}")

        prompt = self._prompts.get(ref, self._encode)
        started = time.perf_counter()
        temperature = (
            self._temperature if delivery.temperature is None else delivery.temperature
        )
        waves = [
            self._render(text, ref.transcript, prompt, temperature) for text in segments
        ]

        if not waves:
            raise NoAudioError("no segments to synthesize")

        return Synthesis(
            audio=join_segments(waves, self.sample_rate),
            sample_rate=self.sample_rate,
            segments=len(waves),
            inference_ms=(time.perf_counter() - started) * 1000,
        )
