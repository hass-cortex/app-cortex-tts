"""Which implementation runs a model, looked up rather than branched on.

`ModelSpec.backend` is a key into this table. A new engine registers a builder
here and the registry needs no change — which is the point: the version of this
that used an enum and an `if` could not take a third engine without an edit,
and a fourth would have made it a ladder.

Builders are called lazily, inside `EngineRegistry._build`, so a backend whose
imports are expensive or fragile costs nothing until a model that uses it is
actually loaded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..providers import ExecutionProvider
from ..references import ReferenceStore
from .base import Engine, EngineError, Voice


class UnknownBackendError(EngineError):
    """A catalog entry names a backend nothing registered."""


@dataclass(frozen=True)
class BuildContext:
    """Everything a backend needs to construct an engine.

    Passing a context rather than a widening argument list means adding a
    setting one backend needs does not touch the signature of the others.

    Attributes:
        directory: Where this model's bundle was downloaded.
        references: The shared reference store; a backend that cannot clone
            simply ignores it.
        num_threads: ONNX Runtime thread count; 0 lets the runtime decide.
        temperature: Default sampling temperature, for backends that have one.
        execution_provider: Which ONNX Runtime provider to ask for. Every
            engine honours it, and every engine reports back what it actually
            got — see `cortex_speech.providers`.
    """

    directory: Path
    references: ReferenceStore
    num_threads: int
    temperature: float
    execution_provider: ExecutionProvider = "auto"


Builder = Callable[[BuildContext], Engine]
VoiceReader = Callable[[Path], list[Voice]]

_BUILDERS: dict[str, Builder] = {}
_VOICE_READERS: dict[str, VoiceReader] = {}


def register(
    backend: str, builder: Builder, *, voices: VoiceReader | None = None
) -> None:
    """Make ``backend`` constructible.

    Args:
        backend: The key catalog entries use.
        builder: Called with a `BuildContext` to construct the engine.
        voices: Reads the bundle's built-in voice list straight from disk.
            Required of any backend whose models bring their own voices, and
            omitted by one whose voices are all reference recordings.

    Raises:
        ValueError: The key is already registered, which means two modules
            claim the same backend and one of them would silently lose.
    """
    if backend in _BUILDERS:
        raise ValueError(f"backend {backend!r} is already registered")
    _BUILDERS[backend] = builder
    if voices is not None:
        _VOICE_READERS[backend] = voices


def build(backend: str, context: BuildContext) -> Engine:
    """Construct the engine for ``backend``.

    Raises:
        UnknownBackendError: Nothing registered that key.
    """
    builder = _BUILDERS.get(backend)
    if builder is None:
        known = ", ".join(sorted(_BUILDERS)) or "none"
        raise UnknownBackendError(f"no backend {backend!r} (registered: {known})")
    return builder(context)


def own_voices(backend: str, directory: Path) -> list[Voice]:
    """Return the voices a model brings itself, without constructing it.

    "Own" rather than "built-in": the bundled kind and OmniVoice's designed
    kind are both the model's own, read the same way and listed side by side,
    and a reader named for one of them lies about the other.

    Raises:
        UnknownBackendError: Nothing registered that key, or what registered
            it cannot answer without loading.
    """
    reader = _VOICE_READERS.get(backend)
    if reader is None:
        raise UnknownBackendError(
            f"backend {backend!r} registered no way to list its own voices"
        )
    return reader(directory)


def registered() -> frozenset[str]:
    """Return every backend key currently registered."""
    return frozenset(_BUILDERS)


def reads_voices() -> frozenset[str]:
    """Return every backend that can list built-in voices off disk."""
    return frozenset(_VOICE_READERS)


def _register_builtin_backends() -> None:
    """Register the backends that ship with the library.

    Each builder imports its engine module inside the call, not at module load.
    The cloning engine drags in torch and librosa, and a problem in either must
    not stop a preset model — or the whole app — from starting. The voice
    readers are deferred the same way, and for the same reason.
    """

    def hojo_preset(context: BuildContext) -> Engine:
        from .preset import PresetEngine

        return PresetEngine(
            context.directory,
            num_threads=context.num_threads,
            temperature=context.temperature,
            execution_provider=context.execution_provider,
        )

    def hojo_clone(context: BuildContext) -> Engine:
        from .clone import CloneEngine

        return CloneEngine(
            context.directory,
            context.references,
            num_threads=context.num_threads,
            temperature=context.temperature,
            execution_provider=context.execution_provider,
        )

    def moss(context: BuildContext) -> Engine:
        from .moss import MossEngine

        return MossEngine(
            context.directory,
            context.references,
            num_threads=context.num_threads,
            temperature=context.temperature,
            execution_provider=context.execution_provider,
        )

    def qwen3_tts(context: BuildContext) -> Engine:
        from .qwen3 import Qwen3TtsEngine

        return Qwen3TtsEngine(
            context.directory,
            context.references,
            num_threads=context.num_threads,
            temperature=context.temperature,
            execution_provider=context.execution_provider,
        )

    def omnivoice(context: BuildContext) -> Engine:
        from .omni import OmniVoiceEngine

        return OmniVoiceEngine(
            context.directory,
            context.references,
            num_threads=context.num_threads,
            temperature=context.temperature,
            execution_provider=context.execution_provider,
        )

    def hojo_preset_voices(directory: Path) -> list[Voice]:
        from .preset import own_voices

        return own_voices(directory)

    def moss_voices(directory: Path) -> list[Voice]:
        from .moss import own_voices

        return own_voices(directory)

    def qwen3_tts_voices(directory: Path) -> list[Voice]:
        from .qwen3 import own_voices

        return own_voices(directory)

    def omnivoice_voices(directory: Path) -> list[Voice]:
        from .omni import own_voices

        return own_voices(directory)

    register("hojo-preset", hojo_preset, voices=hojo_preset_voices)
    # No reader: every 80M voice is a reference recording, which the registry
    # lists from the store without asking any backend.
    register("hojo-clone", hojo_clone)
    register("moss", moss, voices=moss_voices)
    # One backend, two catalog entries: the cloning checkpoint lists no
    # built-in voices, but the reader is harmless there and the registry only
    # calls it for a model that declares them.
    register("qwen3-tts", qwen3_tts, voices=qwen3_tts_voices)
    register("omnivoice", omnivoice, voices=omnivoice_voices)


_register_builtin_backends()
