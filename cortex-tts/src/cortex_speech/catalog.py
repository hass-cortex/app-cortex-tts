"""The models this app can run, and where their weights come from.

Weights are not baked into the image — 670 MB of ONNX would triple it for
users who only ever want one model. The catalog names what is available; the
files arrive on first download and live under ``/data`` so Home Assistant's
"remove with data" sweeps them up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .text.scripts import chars_within


@dataclass(frozen=True)
class BundleSource:
    """One Hugging Face repository contributing files to a model's bundle.

    Most models are one repo. MOSS-TTS-Nano is not: its weights and its audio
    codec are published separately, and both are needed before it can speak.
    Modelling that as a list rather than a special case keeps the next
    multi-repo model from being another one.

    Attributes:
        repo_id: The Hugging Face repository.
        files: Members to fetch. Anything else in the repo is skipped.
        subdir: Where under the model's directory they land. Empty means the
            directory itself; a name keeps two repos' files from colliding.
    """

    repo_id: str
    files: tuple[str, ...]
    subdir: str = ""

    def paths(self) -> tuple[str, ...]:
        """Return each file's path relative to the model's directory."""
        if not self.subdir:
            return self.files
        return tuple(f"{self.subdir}/{name}" for name in self.files)


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable model and everything needed to run it.

    Attributes:
        id: Stable identifier used in the API and in entity unique ids.
        name: Human-readable name.
        description: One line shown in the UI.
        sources: The repositories the bundle is assembled from, in fetch
            order. Usually one.
        backend: Which engine implementation runs it. The registry looks this
            up in a table each engine module registers itself in, so a new
            model is a new module and a new key, never a new branch.
        builtin_voices: Whether the bundle ships selectable voices.
        designed_voices: Whether the model's own voices are built from an
            attribute vocabulary instead of shipped in the bundle. Separate
            from `builtin_voices` because the two cost different amounts to
            render and a caller has to be able to tell them apart — see
            `Voice.source`. No model has both.
        reads_reference_transcript: Whether a clone's transcript reaches the
            model. Only meaningful with `cloning`. Every cloning engine here
            takes the recording, but only some are also told what it says:
            MOSS conditions on the codec frames alone and its synthesis call
            has no prompt text, so a transcript typed for it changes nothing.
            The field is still validated and still required, because one
            recording is a voice on every cloning model at once — the
            transcript MOSS ignores is the one OmniVoice learns from.
        cloning: Whether a reference recording can condition it. Independent
            of ``builtin_voices``: a model may have both, one, or neither.
        chunk_streaming: Whether the engine can emit audio before the whole
            utterance is rendered, which a live reply carries per request.
        temperature: Whether a sampling temperature means anything to it.
            False on MOSS, which fuses its sampling into a dedicated ONNX
            graph, and on OmniVoice, whose own temperatures are not this
            quantity. Silently ignoring the parameter is indistinguishable
            from honouring it, so a caller is told instead.
        language_choice: Whether the caller may say which language the text is
            read as. False where the voice decides it and nothing else can —
            the Hojo models and MOSS take no language at all, so a request
            naming one would be quietly ignored.
        style_instruction: Whether it takes a free-text instruction beside the
            voice ("speak slowly, in a warm tone"). Distinct from
            `language_choice` because
            a model can take one and not the other, and distinct from
            OmniVoice's designed voices, which are a closed vocabulary the
            model validates rather than free text.
        reads_numerals: Whether the model reads Arabic digits and unit
            symbols itself, in the languages the text pipeline has no locale
            written for. Measured, never assumed: a model with a language
            model behind it may read "14:35" more naturally than num2words
            can, and another with the same claim reads German digits as
            noise. Where it is true the generic locale stands aside; the
            Chinese and English locales never do, having been measured
            against every model and won.
        needs_number_words: Whether the model cannot pronounce an Arabic
            digit at all, so a bare number is expanded for it by default —
            a quantity reading of a room number misleads, but digits it
            cannot say are noise. Measured: Hojo read 110 as 十億億安 and
            an English sentence with three numbers as nonsense; MOSS and
            OmniVoice read digits themselves.
        misreads: Words this model reads with the wrong character, measured,
            which the locale respells with a stand-in it reads right
            (`text/zh/standins.tsv`). The defect is the model's to declare and
            the fix is the language's, so neither is guessed for the other.
        size_mb: Approximate on-disk size once downloaded.
        languages: Base language codes the model was trained on.
        sample_rate: Output sample rate in Hz.
        rss_hint_mb: Approximate resident memory once loaded.
    """

    id: str
    name: str
    description: str
    sources: tuple[BundleSource, ...]
    backend: str
    size_mb: int
    languages: tuple[str, ...]
    builtin_voices: bool = False
    designed_voices: bool = False
    cloning: bool = False
    reads_reference_transcript: bool = False
    chunk_streaming: bool = False
    temperature: bool = False
    language_choice: bool = False
    style_instruction: bool = False
    reads_numerals: bool = False
    needs_number_words: bool = False
    misreads: tuple[str, ...] = ()
    max_chars_per_segment: int | None = None
    max_audio_s: float | None = None
    # The third bound, and the only one `segment_limit` cannot settle: what
    # one call into the model may carry, counted in the model's own text
    # tokens. Applied by the engine, which is the only thing here holding a
    # tokenizer; see `ModelSpec.segment_limit` and the entry that declares it.
    max_text_tokens: int | None = None
    sample_rate: int = 24000
    rss_hint_mb: int = 0

    def segment_limit(self, text: str) -> int | None:
        """Characters one synthesis may carry, for this text; `None` for no bound.

        Two bounds meet here and they are not the same kind. A model's
        character budget is `max_chars_per_segment`. A model's generation
        budget is counted in audio, and a segment over it is not slow but
        truncated — so `max_audio_s` is turned into characters by
        `text.scripts.chars_within`, which owns the speech rates. What is
        settled here is only which of the two bounds a text meets first.

        A third bound, `max_text_tokens`, is deliberately absent: counting it
        needs the model's own tokenizer, and characters per token is a
        property of the script rather than of the text (measured on MOSS:
        3.8-4.0 for Latin, nothing like it for Han). Converting it here would
        make a segment depend on a guess about the writing system; the engine
        holds the tokenizer and applies it exactly.
        """
        if self.max_audio_s is None:
            return self.max_chars_per_segment
        fits = chars_within(self.max_audio_s, text)
        if not fits:
            return self.max_chars_per_segment
        if self.max_chars_per_segment is None:
            return fits
        return min(self.max_chars_per_segment, fits)

    @property
    def files(self) -> tuple[str, ...]:
        """Every bundle member, as a path relative to the model's directory."""
        return tuple(path for source in self.sources for path in source.paths())


_COMMON_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="hojo-40m",
        name="Hojo TTS Light 40M",
        description="15 built-in voices (2 Chinese). Fast enough for one CPU core.",
        sources=(
            BundleSource(
                repo_id="HojoAI/Hojo-TTS-Light-40M",
                files=(
                    "Hojo-TTS-Light-40M-llm.onnx",
                    "Hojo-TTS-Light-40M-fine_local.onnx",
                    "Hojo-TTS-Light-40M-decoder.onnx",
                    "Hojo-TTS-Light-40M-voice.npz",
                    *_COMMON_FILES,
                ),
            ),
        ),
        backend="hojo-preset",
        needs_number_words=True,
        # Measured 2026-09-21: 行程 came out as héng chéng in 「你有 3 個行程」;
        # respelled 形程 it reads xíng chéng.
        misreads=("行程",),
        # `render` asks the vendored loop for at most 2048 tokens and the
        # codec runs at 50 Hz. The default character bound reaches 29 s of
        # Chinese, so this never binds — declared so that stays checkable
        # rather than coincidental.
        max_audio_s=40.96,
        size_mb=241,
        languages=("zh", "en"),
        builtin_voices=True,
        temperature=True,
        rss_hint_mb=780,
    ),
    ModelSpec(
        id="moss-nano",
        name="MOSS-TTS-Nano",
        description="18 built-in voices (6 Chinese) and cloning from a "
        "recording. 48 kHz, and it starts speaking before a sentence is "
        "finished.",
        # Two repos: the weights and the audio codec are published apart, and
        # the runtime needs both. The subdirectories match the layout upstream's
        # own loader expects to find under the model directory.
        sources=(
            BundleSource(
                repo_id="OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
                subdir="MOSS-TTS-Nano-100M-ONNX",
                files=(
                    "moss_tts_prefill.onnx",
                    "moss_tts_decode_step.onnx",
                    "moss_tts_local_cached_step.onnx",
                    "moss_tts_local_decoder.onnx",
                    "moss_tts_local_fixed_sampled_frame.onnx",
                    "moss_tts_global_shared.data",
                    "moss_tts_local_shared.data",
                    "browser_poc_manifest.json",
                    "tts_browser_onnx_meta.json",
                    "tokenizer.model",
                ),
            ),
            BundleSource(
                repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
                subdir="MOSS-Audio-Tokenizer-Nano-ONNX",
                files=(
                    "moss_audio_tokenizer_encode.onnx",
                    "moss_audio_tokenizer_encode.data",
                    "moss_audio_tokenizer_decode_full.onnx",
                    "moss_audio_tokenizer_decode_step.onnx",
                    "moss_audio_tokenizer_decode_shared.data",
                    "codec_browser_onnx_meta.json",
                ),
            ),
        ),
        backend="moss",
        size_mb=729,  # 642 MB of weights plus the 87 MB codec
        languages=("zh", "en", "ja"),
        builtin_voices=True,
        cloning=True,
        chunk_streaming=True,
        # Two bounds, and they fail differently.
        #
        # The ceiling is a hard stop: the shipped manifest sets
        # `max_new_frames` to 375 and the codec runs at 12.5 Hz. Measured on
        # this host, seven inputs from 155 to 284 Chinese characters all came
        # back as exactly 30.0 s, the rest of each one missing.
        max_audio_s=30.0,
        # The budget is not a ceiling at all. This model stops when it samples
        # an end-of-audio token, so a chunk can end anywhere, and a longer one
        # is more chances to end early: the same 411-character English text at
        # the engine's pinned seed came back whole as 12.00 s of the 28 s it
        # needed, and cut to this budget as all of it. Measured at the chunk
        # rather than the request — the largest chunk that kept all its text
        # was 49 tokens, the smallest that lost some was 61 — which is why
        # this is well under upstream's own 75, a default belonging to a code
        # path this app does not call.
        max_text_tokens=50,
        sample_rate=48000,
        rss_hint_mb=1990,
    ),
    ModelSpec(
        id="omnivoice",
        name="OmniVoice 0.8B",
        description="Designs a voice from attributes — sex, age, pitch, whisper "
        "— or clones one from a recording. Reads the ten languages listed here, "
        "and takes any of the 646 ids its own table names. The heaviest "
        "model here: it needs torch, and it is well over real time on a CPU.",
        # Two repos, and one file deliberately absent from both: the
        # checkpoint's 2.45 GB `model.safetensors`. The int4 export below
        # replaces the transformer outright, so those weights would be
        # downloaded, held in memory and never read — see
        # `vendor/omnivoice_ort.py`.
        sources=(
            BundleSource(
                repo_id="k2-fsa/OmniVoice",
                files=(
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                    "audio_tokenizer/config.json",
                    "audio_tokenizer/model.safetensors",
                    "audio_tokenizer/preprocessor_config.json",
                ),
            ),
            BundleSource(
                repo_id="rhasspy/omnivoice-onnx",
                # The graph and its external weights, which onnxruntime expects
                # to find beside it under the name the graph records.
                files=("omnivoice.int4.onnx", "omnivoice.int4.onnx.data"),
            ),
        ),
        backend="omnivoice",
        size_mb=1384,
        # The model reads far more than these; this is what the project has
        # exercised, and `docs/models.md` says so.
        languages=("zh", "en", "ja", "ko", "de", "fr", "it", "pt", "ru", "es"),
        designed_voices=True,
        cloning=True,
        reads_reference_transcript=True,
        language_choice=True,
        rss_hint_mb=1140,
    ),
)

BY_ID: dict[str, ModelSpec] = {spec.id: spec for spec in CATALOG}


@dataclass
class ModelState:
    """Runtime view of one catalog entry.

    Attributes:
        spec: The static catalog entry.
        downloaded: Whether every bundle file is present on disk.
        loaded: Whether an engine is currently holding it in memory.
        missing: Bundle members not yet on disk.
        disk_bytes: Bytes the present members occupy.
    """

    spec: ModelSpec
    downloaded: bool
    loaded: bool
    missing: list[str] = field(default_factory=list)
    disk_bytes: int = 0


def model_dir(data_dir: Path, model_id: str) -> Path:
    """Return the directory holding one model's bundle."""
    return data_dir / "models" / model_id


def orphaned_bundles(data_dir: Path) -> dict[str, int]:
    """Return downloaded bundles no catalog entry claims, and their size.

    `inspect` only ever looks where a known id says to look, so a model that
    leaves the catalog leaves its weights behind — MOSS is 2 GB — and nothing
    ever mentions them again.

    Reported rather than deleted: an id can also disappear because a release
    renamed it, and a start-up that quietly removes a gigabyte the user waited
    on is a worse failure than one that keeps it.
    """
    root = data_dir / "models"
    if not root.is_dir():
        return {}
    return {
        directory.name: sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file()
        )
        for directory in sorted(root.iterdir())
        if directory.is_dir() and directory.name not in BY_ID
    }


def inspect(data_dir: Path, spec: ModelSpec, *, loaded: bool = False) -> ModelState:
    """Report whether a model's bundle is complete on disk.

    A partially-downloaded bundle counts as not downloaded: loading it would
    fail deep inside ONNX Runtime with a missing-file error that says nothing
    useful, so the gap is reported here instead.
    """
    directory = model_dir(data_dir, spec.id)
    missing: list[str] = []
    total = 0
    for name in spec.files:
        path = directory / name
        if path.is_file():
            total += path.stat().st_size
        else:
            missing.append(name)
    return ModelState(
        spec=spec,
        downloaded=not missing,
        loaded=loaded,
        missing=missing,
        disk_bytes=total,
    )
