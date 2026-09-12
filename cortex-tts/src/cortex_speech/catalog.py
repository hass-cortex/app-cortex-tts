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
        cloning: Whether a reference recording can condition it. Independent
            of ``builtin_voices``: a model may have both, one, or neither.
        chunk_streaming: Whether the engine can emit audio before the whole
            utterance is rendered, which `/api/speak/stream` carries over HTTP.
        temperature: Whether a sampling temperature means anything to it. MOSS
            fuses its sampling into a dedicated ONNX graph and cannot read one,
            and silently ignoring the parameter is indistinguishable from
            honouring it — so a caller is told instead.
        size_mb: Approximate on-disk size once downloaded.
        languages: Base language codes the model was trained on.
        sample_rate: Output sample rate in Hz.
        rtf_hint: Real-time factor measured on the project's reference host
            — a 4-core Home Assistant OS VM (KVM), two threads, CPU — with the
            one Chinese text set in `scripts/bench_rtf.py`, so the figures are
            comparable with each other. The median over four sentence lengths.
            Shown in the UI so the cost difference between models is visible
            before downloading; a user's own host is read from the
            integration's sensor, never from this.
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
    cloning: bool = False
    chunk_streaming: bool = False
    temperature: bool = False
    sample_rate: int = 24000
    rtf_hint: float = 0.0
    rss_hint_mb: int = 0
    recommended: bool = False

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
        size_mb=241,
        languages=("zh", "en"),
        builtin_voices=True,
        temperature=True,
        rtf_hint=0.67,
        rss_hint_mb=780,
        recommended=True,
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
        rtf_hint=1.42,
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
        rtf_hint=1.06,
        rss_hint_mb=1990,
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
