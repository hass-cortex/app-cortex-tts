"""On-device speech synthesis: models, engines, text preparation, references.

This package is the whole library, and this module is its whole public surface.
Nothing here knows about Home Assistant, the Supervisor, HTTP or FastAPI; the
app that does lives in ``cortex_tts`` and depends on this package one way.

Import from ``cortex_speech`` — never from ``cortex_speech.engine.registry`` or
any other submodule. ``tests/test_architecture.py`` enforces that, so an
internal reshuffle stays internal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio import (
    CONTENT_TYPES,
    MP3_BITRATE,
    STREAM_ENCODERS,
    AudioFormat,
    StreamGain,
    decode_reference,
    encode,
    level,
    wav_header,
)
from .catalog import (
    BY_ID,
    CATALOG,
    ModelSpec,
    ModelState,
    inspect,
    orphaned_bundles,
)
from .download import DownloadManager, DownloadProgress, remove_bundle
from .engine.base import (
    Delivery,
    EngineError,
    NoAudioError,
    StreamingEngine,
    Synthesis,
    UnknownVoiceError,
    UnsupportedLanguageError,
    Voice,
)
from .engine.registry import EngineRegistry, ModelNotReadyError, UnknownModelError
from .notifications import notify_models_changed
from .notifications import subscribe as subscribe_models_changed
from .providers import (
    EXECUTION_PROVIDERS,
    ExecutionProvider,
    ProviderUnavailableError,
)
from .references import (
    MAX_SECONDS as MAX_REFERENCE_SECONDS,
)
from .references import (
    Reference,
    ReferenceError,
    ReferenceStore,
)
from .text.options import NormalizeOptions
from .text.pipeline import TextOptions, prepare, prepared_text

__all__ = [
    "SpeechConfig",
    "SpeechService",
    # catalog
    "CATALOG",
    "BY_ID",
    "ModelSpec",
    "ModelState",
    # synthesis
    "Voice",
    "Synthesis",
    "Delivery",
    "UnsupportedLanguageError",
    "EngineRegistry",
    "StreamingEngine",
    # references
    "Reference",
    "ReferenceStore",
    "MAX_REFERENCE_SECONDS",
    # downloads
    "DownloadManager",
    "DownloadProgress",
    # audio encoding
    "AudioFormat",
    "CONTENT_TYPES",
    "encode",
    "decode_reference",
    "level",
    "STREAM_ENCODERS",
    "MP3_BITRATE",
    "StreamGain",
    "wav_header",
    # text
    "TextOptions",
    "NormalizeOptions",
    "prepare",
    "prepared_text",
    # notifications
    "EXECUTION_PROVIDERS",
    "ExecutionProvider",
    "ProviderUnavailableError",
    "notify_models_changed",
    "subscribe_models_changed",
    # errors
    "EngineError",
    "NoAudioError",
    "UnknownVoiceError",
    "UnknownModelError",
    "ModelNotReadyError",
    "ReferenceError",
]


@dataclass(frozen=True)
class SpeechConfig:
    """Everything the library needs to run, in its own vocabulary.

    The app translates its own options into this once, at the boundary. Nothing
    in here is named after a Home Assistant concept, which is what keeps the
    library's configuration from drifting with the addon's.

    Attributes:
        data_dir: Root holding ``models/`` and ``references/``.
        num_threads: ONNX Runtime thread count; 0 lets the runtime decide.
        max_loaded_models: How many engines may stay resident at once.
        temperature: Default sampling temperature for engines that have one.
        execution_provider: Which ONNX Runtime provider to ask for. `auto`
            takes a GPU when one answers; `cuda` refuses to fall back, because
            a machine with a GPU quietly running on its CPU is the failure
            nobody notices.
    """

    data_dir: Path
    num_threads: int = 0
    max_loaded_models: int = 1
    temperature: float = 0.8
    execution_provider: ExecutionProvider = "auto"


class SpeechService:
    """The library assembled and ready to use.

    Owns the reference store, the engine registry and the download manager, so
    the app constructs one object instead of wiring four together and having to
    know the order.
    """

    def __init__(self, config: SpeechConfig) -> None:
        """Assemble the library against ``config``'s directories."""
        self._config = config
        self._references = ReferenceStore(config.data_dir / "references")
        self._downloads = DownloadManager(config.data_dir)
        self._registry = EngineRegistry(
            config.data_dir,
            self._references,
            num_threads=config.num_threads,
            max_loaded=config.max_loaded_models,
            temperature=config.temperature,
            execution_provider=config.execution_provider,
        )

    @property
    def config(self) -> SpeechConfig:
        """The configuration this service was built with."""
        return self._config

    @property
    def references(self) -> ReferenceStore:
        """The store of uploaded reference recordings."""
        return self._references

    @property
    def registry(self) -> EngineRegistry:
        """The engine registry, which loads and bounds resident models."""
        return self._registry

    @property
    def downloads(self) -> DownloadManager:
        """The model download manager."""
        return self._downloads

    def models(self) -> list[ModelState]:
        """Return every catalog entry with its on-disk and in-memory state."""
        loaded = self._registry.loaded_ids
        return [
            inspect(self._config.data_dir, spec, loaded=spec.id in loaded)
            for spec in CATALOG
        ]

    def model(self, spec: ModelSpec) -> ModelState:
        """Return one catalog entry's on-disk and in-memory state.

        The app asks the service rather than calling `inspect` with a directory
        of its own: where a bundle lives is the library's business, and a second
        copy of that path in the app is a second thing to keep in step.
        """
        return inspect(
            self._config.data_dir, spec, loaded=self._registry.is_loaded(spec.id)
        )

    def orphans(self) -> dict[str, int]:
        """Return downloaded bundles no catalog entry claims, and their size.

        Asked of the service for the same reason as `model`: the app does not
        know where a bundle lives, and should not learn.
        """
        return orphaned_bundles(self._config.data_dir)

    def delete_model(self, spec: ModelSpec) -> int:
        """Remove a downloaded bundle. Returns how many files went."""
        return remove_bundle(self._config.data_dir, spec)
