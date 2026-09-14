"""Qwen3-TTS: nine bundled speakers on one checkpoint, cloning on the other.

The two bundles run the same code. They differ only in what the checkpoint
lets the talker be conditioned on — a speaker id the model was trained with,
or a recording encoded at request time — so the engine picks between them by
looking at what the bundle actually carries rather than at its catalog id.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator, Iterator
from pathlib import Path

import numpy as np

from ..audio import decode_reference
from ..providers import (
    ExecutionProvider,
    in_use,
    release_sessions,
    requested,
    sessions_of,
    verify,
)
from ..references import Reference, ReferenceStore
from ..text.pipeline import is_chinese
from ..vendor.qwen3_tts_ort import (
    DECODER_FRAMES,
    SAMPLE_RATE,
    Qwen3TtsError,
    Qwen3TtsOnnx,
    ReferenceConditioning,
)
from .base import (
    Delivery,
    NoAudioError,
    Synthesis,
    UnknownVoiceError,
    UnsupportedLanguageError,
    Voice,
    narrowing,
)
from .conditioning import ConditioningCache
from .join import fade_in, join_segments, segment_gap
from .overrun import expected_seconds, render_with_retries

_LOGGER = logging.getLogger(__name__)

# The codec runs at 12.5 frames a second: the decoder is exported at 25 frames
# and returns two seconds of audio.
FRAMES_PER_SECOND = DECODER_FRAMES / 2.0

# What the bundle's speaker ids mean. Upstream publishes this as prose in the
# model card and nothing in the checkpoint carries it, so it is a table here —
# `tests/test_qwen3_voices.py` fails if the checkpoint grows a speaker this
# does not describe.
_SPEAKERS: dict[str, tuple[str, str, str, str]] = {
    # id: (language, codec language tag, gender, label)
    "vivian": ("zh", "chinese", "female", "Bright young female"),
    "serena": ("zh", "chinese", "female", "Warm, gentle young female"),
    "uncle_fu": ("zh", "chinese", "male", "Seasoned male, mellow timbre"),
    "dylan": ("zh", "beijing_dialect", "male", "Youthful Beijing male"),
    "eric": ("zh", "sichuan_dialect", "male", "Lively Chengdu male"),
    "ryan": ("en", "english", "male", "Dynamic male with rhythm"),
    "aiden": ("en", "english", "male", "Sunny American male"),
    "ono_anna": ("ja", "japanese", "female", "Playful Japanese female"),
    "sohee": ("ko", "korean", "female", "Warm Korean female"),
}

# Which language tag a cloned voice reads in. The request carries no language,
# so the reference's own is used, and the text's script decides when the
# reference does not say.
_TAG_FOR_LANGUAGE = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}

# Ceiling on a single segment's generation, as a multiple of the duration the
# text needs. The talker stops by sampling end-of-speech and can fail to; at
# temperature 0 it reliably does, and without a ceiling one 120-character
# segment runs to the model's 2048-frame limit — measured at 2.7 minutes of
# invented audio for ten characters of text.
_FRAME_BUDGET_RATIO = 2.5
_FRAME_BUDGET_FLOOR_SECONDS = 4.0

# Sampling the model was tuned with. Only the temperature is a caller's to
# change; the rest travel with it.
_TOP_K = 50
_TOP_P = 1.0
_REPETITION_PENALTY = 1.05


def _describe(speaker: str) -> Voice:
    language, _, gender, label = _SPEAKERS.get(speaker, ("", "", "unknown", speaker))
    return Voice(
        id=speaker,
        name=f"{speaker.replace('_', ' ').title()} ({label})" if label else speaker,
        language=language or None,
        gender=gender,
        source="builtin",
    )


def own_voices(directory: Path) -> list[Voice]:
    """Read the checkpoint's speaker list off disk, opening no session.

    Named separately from the engine because listing voices must not load a
    model — see `EngineRegistry.voices`. `config.json` is where the talker
    itself reads the ids from, so the two cannot disagree.
    """
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    speakers = config.get("talker_config", {}).get("spk_id") or {}
    return [_describe(name) for name in speakers]


def _frame_budget(text: str) -> int:
    """How many frames one segment may produce before it is cut off."""
    seconds = max(expected_seconds(text), 0.0) * _FRAME_BUDGET_RATIO
    return int((seconds + _FRAME_BUDGET_FLOOR_SECONDS) * FRAMES_PER_SECOND)


class Qwen3TtsEngine:
    """Wraps the Qwen3-TTS ONNX bundle, caching per-reference conditioning."""

    def __init__(
        self,
        models_dir: Path,
        references: ReferenceStore,
        *,
        num_threads: int = 0,
        temperature: float = 0.9,
        execution_provider: ExecutionProvider = "auto",
    ) -> None:
        """Load the bundle.

        Args:
            models_dir: Directory holding the downloaded bundle.
            references: Store of uploaded reference recordings.
            num_threads: ONNX Runtime thread count; 0 lets ORT decide.
            temperature: Default sampling temperature.
            execution_provider: Which provider to ask ONNX Runtime for.

        Raises:
            ProviderUnavailableError: `cuda` was required and CPU is what the
                sessions came back on.
        """
        from .qwen_tokenizer import build

        self._references = references
        self._temperature = temperature
        started = time.perf_counter()
        self._runtime = Qwen3TtsOnnx(
            models_dir,
            build(models_dir),
            execution_provider=requested(execution_provider),
            num_threads=num_threads,
        )
        self.provider = verify(execution_provider, in_use(sessions_of(self._runtime)))
        self._voices = [_describe(name) for name in self._runtime.speakers]
        self._builtin = {voice.id for voice in self._voices}
        self._conditioning: ConditioningCache[ReferenceConditioning] = (
            ConditioningCache()
        )
        _LOGGER.info(
            "loaded Qwen3-TTS bundle from %s in %.2fs on %s "
            "(%d built-in voices, cloning %s)",
            models_dir,
            time.perf_counter() - started,
            self.provider,
            len(self._voices),
            "available" if self._runtime.can_clone else "unavailable",
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return SAMPLE_RATE

    def forget(self, reference_id: str) -> None:
        """Drop cached conditioning for a reference that changed or went away."""
        self._conditioning.forget(reference_id)

    def close(self) -> None:
        """Release the sessions; see `Engine.close`."""
        release_sessions(self._runtime)

    # -- voices ------------------------------------------------------------

    def _encode(self, reference: Reference) -> ReferenceConditioning:
        """Encode a reference recording once, for reuse across replies."""
        started = time.perf_counter()
        wave = decode_reference(str(reference.audio_path), sample_rate=SAMPLE_RATE)
        conditioning = self._runtime.condition(wave, reference.transcript)
        _LOGGER.info(
            "encoded reference %s into %d prompt frames in %.0fms",
            reference.id,
            len(conditioning.codes),
            (time.perf_counter() - started) * 1000,
        )
        return conditioning

    def _resolve(
        self, voice: str
    ) -> tuple[str | None, ReferenceConditioning | None, str | None]:
        """Turn a voice id into (speaker, conditioning, language tag)."""
        if voice in self._builtin:
            # A checkpoint may ship a speaker this table has not heard of; it
            # is still selectable, and the text's script picks the language.
            described = _SPEAKERS.get(voice)
            return voice, None, described[1] if described else None
        reference = self._references.get(voice)
        if reference is None:
            raise UnknownVoiceError(f"unknown voice {voice!r}")
        if not self._runtime.can_clone:
            raise UnknownVoiceError(
                f"{voice!r} is a reference recording and this checkpoint has no "
                "speaker encoder; use the cloning bundle for it"
            )
        try:
            conditioning = self._conditioning.get(reference, self._encode)
        except Qwen3TtsError as err:
            raise NoAudioError(str(err)) from err
        return None, conditioning, _TAG_FOR_LANGUAGE.get(reference.language)

    # -- rendering ---------------------------------------------------------

    def _language(self, code: str | None) -> str | None:
        """Turn a requested language tag into the one the talker reads.

        The caller's tag arrives whole — `zh-TW`, not `zh` — because only the
        model knows whether the distinction means anything to it. This one
        also names two Chinese dialects, so the full tag is tried before its
        base: a model that can tell them apart gets the chance to.

        Raises:
            UnsupportedLanguageError: Not one this checkpoint was trained on.
        """
        if not code:
            return None
        for candidate in narrowing(code):
            tag = _TAG_FOR_LANGUAGE.get(candidate, candidate)
            if tag in self._runtime.languages:
                return tag
        spoken = ", ".join(sorted(self._runtime.languages))
        raise UnsupportedLanguageError(
            f"Qwen3-TTS does not read {code!r}; it takes one of: {spoken}"
        )

    def _frames(
        self,
        text: str,
        *,
        speaker: str | None,
        conditioning: ReferenceConditioning | None,
        language: str | None,
        instruct: str | None,
        temperature: float,
        seed: int,
    ) -> Iterator[np.ndarray]:
        return self._runtime.frames(
            text,
            speaker=speaker,
            reference=conditioning,
            instruct=instruct,
            language=language or ("chinese" if is_chinese(text) else "english"),
            max_frames=_frame_budget(text),
            temperature=temperature,
            top_k=_TOP_K,
            top_p=_TOP_P,
            repetition_penalty=_REPETITION_PENALTY,
            seed=seed,
        )

    def _blocks(self, frames: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """Decode frames in the fixed-size blocks the codec decoder takes.

        A block decoded on its own is the same audio as the same block inside a
        longer decode, so this is both how a whole utterance is rendered and
        how a streamed one leaves early.
        """
        block: list[np.ndarray] = []
        for frame in frames:
            block.append(frame)
            if len(block) == DECODER_FRAMES:
                yield self._runtime.decode(np.stack(block))
                block = []
        if block:
            yield self._runtime.decode(np.stack(block))

    def _render(
        self,
        text: str,
        *,
        speaker: str | None,
        conditioning: ReferenceConditioning | None,
        language: str | None,
        instruct: str | None,
        temperature: float,
    ) -> np.ndarray:
        def generate(seed: int) -> np.ndarray:
            blocks = list(
                self._blocks(
                    self._frames(
                        text,
                        speaker=speaker,
                        conditioning=conditioning,
                        language=language,
                        instruct=instruct,
                        temperature=temperature,
                        seed=seed,
                    )
                )
            )
            if not blocks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(blocks)

        return render_with_retries(text, self.sample_rate, generate)

    def synthesize(
        self, segments: list[str], voice: str, *, delivery: Delivery = Delivery()
    ) -> Synthesis:
        """Render segments with a bundled speaker or a cloned voice.

        Raises:
            UnsupportedLanguageError: `delivery.language` is not one of the ten.
        """
        if not segments:
            raise NoAudioError("no segments to synthesize")
        speaker, conditioning, voice_language = self._resolve(voice)
        # A request outranks the voice's own language, which is only
        # upstream's recommendation for it.
        language = self._language(delivery.language) or voice_language
        temperature = (
            self._temperature if delivery.temperature is None else delivery.temperature
        )

        started = time.perf_counter()
        waves = [
            self._render(
                text,
                speaker=speaker,
                conditioning=conditioning,
                language=language,
                instruct=delivery.instruct,
                temperature=temperature,
            )
            for text in segments
        ]
        if not any(wave.size for wave in waves):
            raise NoAudioError(f"model produced no audio for {segments!r}")

        return Synthesis(
            audio=join_segments(waves, self.sample_rate),
            sample_rate=self.sample_rate,
            segments=len(waves),
            inference_ms=(time.perf_counter() - started) * 1000,
        )

    def synthesize_stream(
        self, segments: list[str], voice: str, *, delivery: Delivery = Delivery()
    ) -> Generator[np.ndarray, None, None]:
        """Yield two seconds of audio at a time, as the codec fills a block.

        No retry here, unlike `synthesize`: by the time a short generation is
        recognisable as short, its audio has already been sent.
        """
        if not segments:
            raise NoAudioError("no segments to synthesize")
        speaker, conditioning, voice_language = self._resolve(voice)
        language = self._language(delivery.language) or voice_language
        gap = segment_gap(self.sample_rate)

        produced = False
        for index, text in enumerate(segments):
            if index:
                yield gap
            for block in self._blocks(
                self._frames(
                    text,
                    speaker=speaker,
                    conditioning=conditioning,
                    language=language,
                    instruct=delivery.instruct,
                    temperature=(
                        self._temperature
                        if delivery.temperature is None
                        else delivery.temperature
                    ),
                    seed=0,
                )
            ):
                if not block.size:
                    continue
                if not produced:
                    block = fade_in(block, self.sample_rate)
                produced = True
                yield block

        if not produced:
            raise NoAudioError(f"model produced no audio for {segments!r}")
