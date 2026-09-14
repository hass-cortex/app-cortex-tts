"""The models this app can run, and where their weights come from.

Weights are not baked into the image — 670 MB of ONNX would triple it for
users who only ever want one model. The catalog names what is available; the
files arrive on first download and live under ``/data`` so Home Assistant's
"remove with data" sweeps them up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
        cloning: Whether a reference recording can condition it. Independent
            of ``builtin_voices``: a model may have both, one, or neither.
        chunk_streaming: Whether the engine can emit audio before the whole
            utterance is rendered, which `/api/speak/stream` carries over HTTP.
        temperature: Whether a sampling temperature means anything to it. MOSS
            fuses its sampling into a dedicated ONNX graph and cannot read one,
            and silently ignoring the parameter is indistinguishable from
            honouring it — so a caller is told instead.
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
    chunk_streaming: bool = False
    temperature: bool = False
    language_choice: bool = False
    style_instruction: bool = False
    sample_rate: int = 24000
    rss_hint_mb: int = 0

    @property
    def files(self) -> tuple[str, ...]:
        """Every bundle member, as a path relative to the model's directory."""
        return tuple(path for source in self.sources for path in source.paths())


_COMMON_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

# Qwen3-TTS is assembled from two repositories and one export directory.
# `cpu_int4` names a precision rather than a device: the export ships the same
# graphs again under `cuda_int4`, and which provider runs them is this app's
# decision, not the export's.
_QWEN3_EXPORT = "cpu_int4"
_QWEN3_ONNX = tuple(
    f"{_QWEN3_EXPORT}/{name}.onnx"
    for name in (
        "text_embed",
        "codec_embed",
        "residual_embed",
        "talker_cache",
        "code_predictor",
        "tok_decoder",
    )
)
# Only the cloning checkpoint exports these: the encoder that turns a
# recording into codec frames, and the one that turns it into an x-vector.
_QWEN3_CLONE_ONNX = tuple(
    f"{_QWEN3_EXPORT}/{name}.onnx" for name in ("tok_encoder", "speaker_encoder")
)
# The checkpoint ships no assembled `tokenizer.json`; `engine.qwen_tokenizer`
# builds one from these. `config.json` carries the talker's token ids and the
# speaker table beside them.
_QWEN3_TEXT = ("config.json", "vocab.json", "merges.txt", "tokenizer_config.json")

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
        size_mb=241,
        languages=("zh", "en"),
        builtin_voices=True,
        temperature=True,
        rss_hint_mb=780,
    ),
    ModelSpec(
        id="hojo-80m-clone",
        name="Hojo TTS Light 80M (voice cloning)",
        description="Clones a voice from a few seconds of reference audio. "
        "No built-in voices; ~4x the compute of the 40M.",
        sources=(
            BundleSource(
                repo_id="HojoAI/Hojo-TTS-Light",
                files=(
                    "Hojo-TTS-Light-llm.onnx",
                    "Hojo-TTS-Light-decoder.onnx",
                    "Hojo-TTS-Light-encoder.onnx",
                    "Hojo-TTS-Light-speaker.onnx",
                    "Hojo-TTS-Light-voice.npz",
                    *_COMMON_FILES,
                ),
            ),
        ),
        backend="hojo-clone",
        size_mb=437,
        languages=("zh", "en"),
        cloning=True,
        temperature=True,
        rss_hint_mb=2050,
    ),
    ModelSpec(
        id="moss-nano",
        name="MOSS-TTS-Nano",
        description="18 built-in voices (6 Chinese) and cloning from a "
        "recording. 48 kHz, and the only model here that can start speaking "
        "before a sentence is finished.",
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
        sample_rate=48000,
        rss_hint_mb=1990,
    ),
    ModelSpec(
        id="qwen3-tts-0.6b",
        name="Qwen3-TTS 0.6B (built-in voices)",
        description="Nine built-in speakers across five languages. The widest "
        "language coverage here, and the slowest model by a wide margin.",
        # Two repos: Qwen publishes the checkpoint, onnx-community the export.
        # Only the int4 graphs are fetched — the export's `cpu_*` and `cuda_*`
        # directories hold byte-identical files and differ only in a manifest
        # naming an execution provider this app chooses for itself.
        sources=(
            BundleSource(
                repo_id="onnx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                files=_QWEN3_ONNX,
            ),
            BundleSource(
                repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                files=_QWEN3_TEXT,
            ),
        ),
        backend="qwen3-tts",
        size_mb=1021,
        languages=("zh", "en", "ja", "ko", "de", "fr", "it", "pt", "ru", "es"),
        builtin_voices=True,
        chunk_streaming=True,
        temperature=True,
        language_choice=True,
        style_instruction=True,
        rss_hint_mb=1580,
    ),
    ModelSpec(
        id="qwen3-tts-0.6b-clone",
        name="Qwen3-TTS 0.6B (voice cloning)",
        description="The same model conditioned on a reference recording "
        "instead of a bundled speaker. No built-in voices.",
        sources=(
            BundleSource(
                repo_id="onnx-community/Qwen3-TTS-12Hz-0.6B-Base",
                files=(*_QWEN3_ONNX, *_QWEN3_CLONE_ONNX),
            ),
            BundleSource(
                repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                files=_QWEN3_TEXT,
            ),
        ),
        backend="qwen3-tts",
        size_mb=1271,
        languages=("zh", "en", "ja", "ko", "de", "fr", "it", "pt", "ru", "es"),
        cloning=True,
        chunk_streaming=True,
        temperature=True,
        language_choice=True,
        rss_hint_mb=2120,
    ),
    ModelSpec(
        id="omnivoice",
        name="OmniVoice 0.8B",
        description="Designs a voice from attributes — sex, age, pitch, whisper "
        "— or clones one from a recording. Reads 800+ languages. The heaviest "
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
