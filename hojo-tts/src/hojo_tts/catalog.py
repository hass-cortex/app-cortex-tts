"""The models this app can run, and where their weights come from.

Weights are not baked into the image — 670 MB of ONNX would triple it for
users who only ever want one model. The catalog names what is available; the
files arrive on first download and live under ``/data`` so Home Assistant's
"remove with data" sweeps them up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class EngineKind(StrEnum):
    """How a model is conditioned on a speaker."""

    PRESET = "preset"
    """Fixed voices shipped in the bundle; selected by voice id."""

    CLONE = "clone"
    """Zero-shot cloning; every request needs a reference recording."""


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable model and everything needed to run it.

    Attributes:
        id: Stable identifier used in the API and in entity unique ids.
        name: Human-readable name.
        description: One line shown in the UI.
        repo_id: Hugging Face repository holding the ONNX bundle.
        files: Bundle members to fetch. Anything else in the repo is skipped.
        kind: Preset-voice or cloning.
        size_mb: Approximate on-disk size once downloaded.
        languages: Base language codes the model was trained on.
        sample_rate: Output sample rate in Hz.
        rtf_hint: Measured real-time factor on a modern desktop core, shown
            in the UI so the cost difference between models is visible before
            downloading rather than after.
        rss_hint_mb: Approximate resident memory once loaded.
    """

    id: str
    name: str
    description: str
    repo_id: str
    files: tuple[str, ...]
    kind: EngineKind
    size_mb: int
    languages: tuple[str, ...]
    sample_rate: int = 24000
    rtf_hint: float = 0.0
    rss_hint_mb: int = 0

    @property
    def voices_from_references(self) -> bool:
        """Whether this model's voices come from the reference store.

        When they do, listing them needs no engine — which is what keeps a
        voice list from loading a 2 GB bundle.
        """
        return self.kind is EngineKind.CLONE

    recommended: bool = False


_COMMON_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="hojo-40m",
        name="Hojo TTS Light 40M",
        description="15 built-in voices (2 Chinese). Fast enough for one CPU core.",
        repo_id="HojoAI/Hojo-TTS-Light-40M",
        files=(
            "Hojo-TTS-Light-40M-llm.onnx",
            "Hojo-TTS-Light-40M-fine_local.onnx",
            "Hojo-TTS-Light-40M-decoder.onnx",
            "Hojo-TTS-Light-40M-voice.npz",
            *_COMMON_FILES,
        ),
        kind=EngineKind.PRESET,
        size_mb=241,
        languages=("zh", "en"),
        rtf_hint=0.21,
        rss_hint_mb=780,
        recommended=True,
    ),
    ModelSpec(
        id="hojo-80m-clone",
        name="Hojo TTS Light 80M (voice cloning)",
        description="Clones a voice from a few seconds of reference audio. "
        "No built-in voices; ~4x the compute of the 40M.",
        repo_id="HojoAI/Hojo-TTS-Light",
        files=(
            "Hojo-TTS-Light-llm.onnx",
            "Hojo-TTS-Light-decoder.onnx",
            "Hojo-TTS-Light-encoder.onnx",
            "Hojo-TTS-Light-speaker.onnx",
            "Hojo-TTS-Light-voice.npz",
            *_COMMON_FILES,
        ),
        kind=EngineKind.CLONE,
        size_mb=437,
        languages=("zh", "en"),
        rtf_hint=0.79,
        rss_hint_mb=2050,
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
