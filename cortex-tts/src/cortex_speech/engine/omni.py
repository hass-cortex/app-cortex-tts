"""OmniVoice: a designed voice or a cloned one, with the transformer in ONNX.

Two things separate this engine from the others.

Its language model runs twice over: upstream's PyTorch module defines the
shape, and an int4 ONNX graph does the arithmetic. `forward` is replaced
outright, so the 2.45 GB of transformer weights the checkpoint ships are never
read — the module is built on the meta device and only the audio tokenizer,
which ONNX does not cover, is real. Measured peak resident memory 1.1 GB
against 4.7 GB for the same pipeline loaded whole.

That split is also what `cuda` means here, and it means less than it does
elsewhere. The execution provider reaches the ONNX graph and nothing else: the
audio tokenizer, the prompt handling and the iterative decoder stay in torch on
the CPU whatever it is set to. `provider` is therefore honest about the
sessions and silent about the rest — `providers.in_use` is defined over ONNX
Runtime sessions, and this engine has exactly one. It is still the biggest win
a card gives any model in the catalog: RTF 3.83 to 0.80 on a GTX 1650, because
the graph is where the time goes. Moving the torch half across was left alone
deliberately — the loop hands ONNX small integer tensors every step, so it
would buy GPU work at the price of a copy in each direction.

And it has no bundled voices in the usual sense. A voice is a handful of
attributes — sex, age, pitch, whisper — drawn from a closed vocabulary the
model was trained on, so the voices below are a chosen set rather than a
list read off disk. The reference recordings are the other half, as usual.
"""

from __future__ import annotations

import logging
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
from ..references import Reference, ReferenceStore
from ..text.pipeline import is_chinese
from .base import (
    Delivery,
    NoAudioError,
    StopCheck,
    Synthesis,
    UnknownVoiceError,
    UnsupportedLanguageError,
    Voice,
    check_stop,
    narrowing,
)
from .conditioning import ConditioningCache
from .join import join_segments

_LOGGER = logging.getLogger(__name__)

# Where each half of the bundle lands. `catalog.py` declares these paths; the
# engine is the only thing that has to agree with them.
ONNX_MODEL = "omnivoice.int4.onnx"
AUDIO_TOKENIZER_DIR = "audio_tokenizer"

# Decoding steps. Upstream defaults to 32; this is the iterative unmasking
# loop, so the cost is linear in it. Ten is what the project measured as the
# point where fewer steps start to be audible.
DECODE_STEPS = 10

# Upstream's own default, and left alone. This decoder has two temperatures:
# one picks which positions to unmask next (5.0, where the variation between
# renders comes from) and this one picks the token at a chosen position, where
# 0 is greedy. It is not the quantity the app's single "temperature" setting
# names on the other models, so the catalog entry declares no temperature
# rather than quietly mapping 0.8 onto something that was tuned at 0.
CLASS_TEMPERATURE = 0.0

# The voices, as attribute sets from the model's own instruct vocabulary
# (`vendor/omnivoice/voice_design.py`). English is what the model is told; it
# translates the attributes itself when the text reads as Chinese, so one
# definition covers both.
# What `generate` is told to read the text as when the caller says nothing.
# Upstream matches these case-insensitively against its 646 language ids.
_LANGUAGE_NAMES = {True: "chinese", False: "english"}

_DESIGNED_VOICES: dict[str, tuple[str, str, str]] = {
    # id: (instruct, gender, label)
    "female-young": ("female, young adult", "female", "Young female"),
    "female-young-high": (
        "female, young adult, high pitch",
        "female",
        "Young female, high pitch",
    ),
    "female-middle": ("female, middle-aged", "female", "Middle-aged female"),
    "female-elderly": ("female, elderly", "female", "Elderly female"),
    "female-whisper": ("female, young adult, whisper", "female", "Whispering female"),
    "male-young": ("male, young adult", "male", "Young male"),
    "male-young-low": (
        "male, young adult, low pitch",
        "male",
        "Young male, low pitch",
    ),
    "male-middle": ("male, middle-aged", "male", "Middle-aged male"),
    "male-elderly-low": (
        "male, elderly, low pitch",
        "male",
        "Elderly male, low pitch",
    ),
}


def _describe(voice_id: str) -> Voice:
    _, gender, label = _DESIGNED_VOICES[voice_id]
    return Voice(
        id=voice_id,
        # No language: the attributes say nothing about one, and the model
        # reads whatever script the text is in.
        language=None,
        name=label,
        gender=gender,
        source="designed",
    )


def own_voices(directory: Path) -> list[Voice]:
    """Return the designed voices.

    Unlike the other backends' readers this opens nothing at all: the set is
    a choice made here, not a table in the bundle. It keeps the reader's
    signature so the registry needs no special case.
    """
    del directory
    return [_describe(voice_id) for voice_id in _DESIGNED_VOICES]


def retranscribed(prompt: object, transcript: str) -> object:
    """Put the reference's current transcript on a cached clone prompt.

    The cache is keyed by the recording alone, and the transcript can be
    corrected without re-uploading it, so the prompt's copy is replaced on
    every use rather than trusted.
    """
    prompt.ref_text = transcript  # type: ignore[attr-defined]
    return prompt


class _PromptSidecar:
    """The clone prompt in the vendored runtime's own portable format."""

    suffix = ".omni.pt"

    def dump(self, value: object, path: Path) -> None:
        value.save(str(path))  # type: ignore[attr-defined]

    def load(self, path: Path) -> object:
        from ..vendor.omnivoice.modeling import VoiceClonePrompt

        return VoiceClonePrompt.load(str(path))


class OmniVoiceEngine:
    """Wraps the vendored OmniVoice pipeline with an ONNX language model."""

    def __init__(
        self,
        models_dir: Path,
        references: ReferenceStore,
        *,
        num_threads: int = 0,
        temperature: float = 0.0,
        execution_provider: ExecutionProvider = "auto",
    ) -> None:
        """Load the bundle.

        Args:
            models_dir: Directory holding the downloaded bundle.
            references: Store of uploaded reference recordings.
            num_threads: ONNX Runtime thread count; 0 lets ORT decide.
            temperature: Ignored — see `CLASS_TEMPERATURE`. The catalog entry
                declares `temperature=False`, so the API refuses one.
            execution_provider: Which provider to ask ONNX Runtime for.

        Raises:
            ProviderUnavailableError: `cuda` was required and CPU is what the
                sessions came back on.
        """
        self._references = references
        self._temperature = temperature
        from ..vendor.omnivoice_ort import load

        started = time.perf_counter()
        # The session is held here, not only inside the model, so
        # `sessions_of` can read what it actually came back on.
        self._model, self._session = load(
            models_dir,
            onnx_model=ONNX_MODEL,
            audio_tokenizer_dir=AUDIO_TOKENIZER_DIR,
            execution_provider=requested(execution_provider),
            num_threads=num_threads,
        )
        self.provider = verify(execution_provider, in_use(sessions_of(self)))
        self._prompts: ConditioningCache[object] = ConditioningCache(
            _PromptSidecar(), references.root
        )
        _LOGGER.info(
            "loaded OmniVoice bundle from %s in %.2fs on %s (%d designed voices)",
            models_dir,
            time.perf_counter() - started,
            self.provider,
            len(_DESIGNED_VOICES),
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return int(self._model.sampling_rate)

    def forget(self, reference_id: str) -> None:
        """Drop cached conditioning for a reference that changed or went away."""
        self._prompts.forget(reference_id)

    def close(self) -> None:
        """Release the sessions; see `Engine.close`."""
        release_sessions(self)

    def _encode(self, reference: Reference) -> object:
        """Encode a reference recording into a reusable clone prompt."""
        started = time.perf_counter()
        prompt = self._model.create_voice_clone_prompt(
            str(reference.audio_path), ref_text=reference.transcript
        )
        _LOGGER.info(
            "encoded reference %s into %d prompt frames in %.0fms",
            reference.id,
            prompt.ref_audio_tokens.shape[-1],
            (time.perf_counter() - started) * 1000,
        )
        return prompt

    def _language(self, code: str | None) -> str | None:
        """Check a requested language against the model's own list.

        Upstream takes either one of its 646 ids or a language name, and
        answers an unknown one by falling back silently — which reads as the
        request having worked. The list is checked here so it does not.
        """
        if not code:
            return None
        from ..vendor.omnivoice.lang_map import LANG_IDS, LANG_NAME_TO_ID

        # The tag arrives whole and is narrowed here, not by the caller: this
        # model names 646 languages, several of which are narrower than the
        # base code a caller would otherwise have thrown away.
        for candidate in narrowing(code):
            if candidate in LANG_IDS or candidate in LANG_NAME_TO_ID:
                return candidate
        raise UnsupportedLanguageError(
            f"OmniVoice does not read {code!r}; it takes one of 646 language "
            "ids like 'zh' or 'yue', or a name like 'chinese'"
        )

    def _resolve(self, voice: str) -> tuple[str | None, object | None]:
        """Turn a voice id into (instruct, clone prompt); exactly one is set."""
        designed = _DESIGNED_VOICES.get(voice)
        if designed is not None:
            return designed[0], None
        reference = self._references.get(voice)
        if reference is None:
            raise UnknownVoiceError(f"unknown voice {voice!r}")
        return None, retranscribed(
            self._prompts.get(reference, self._encode), reference.transcript
        )

    def synthesize(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Synthesis:
        """Render segments with a designed voice or a cloned one.

        Raises:
            UnsupportedLanguageError: `delivery.language` is not one of the
                646 the model reads.
        """
        if not segments:
            raise NoAudioError("no segments to synthesize")
        instruct, prompt = self._resolve(voice)

        started = time.perf_counter()
        # One call, not one per segment: the pipeline batches, and a reference
        # prompt is encoded into the batch once rather than per sentence.
        #
        # The language is told per segment. A request carries none, so the
        # script the text reads as is the only thing that can answer, and
        # upstream is measurably better when told than when left to guess.
        language = self._language(delivery.language)
        rendered = self._model.generate(
            text=list(segments),
            language=[
                language or _LANGUAGE_NAMES[is_chinese(text)] for text in segments
            ],
            instruct=instruct,
            voice_clone_prompt=prompt,
            num_step=DECODE_STEPS,
            class_temperature=CLASS_TEMPERATURE,
            on_step=lambda: check_stop(stop),
        )
        waves = [np.asarray(wave, dtype=np.float32).reshape(-1) for wave in rendered]
        if not any(wave.size for wave in waves):
            raise NoAudioError(f"model produced no audio for {segments!r}")

        return Synthesis(
            audio=join_segments(waves, self.sample_rate),
            sample_rate=self.sample_rate,
            segments=len(waves),
            inference_ms=(time.perf_counter() - started) * 1000,
        )
