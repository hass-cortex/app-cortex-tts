"""Which ONNX Runtime execution provider the engines actually got.

Two facts shape this module, and the second is why it exists at all.

`onnxruntime.get_available_providers()` lists what the build was compiled
with, not what will work. Measured on a host with a GTX 1650: it listed
`CUDAExecutionProvider`, and creating a session then fell back to the CPU with
nothing but a warning, because the CUDA libraries the wheel wanted were a major
version ahead of the driver. A provider list is a claim; a session is evidence.

So nothing here trusts the claim. The choice is turned into a request, the
sessions that come back are read to see what was honoured, and asking for
`cuda` and silently getting CPU is made into an error — a machine with a GPU
quietly running on its CPU is the failure that costs an afternoon.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

ExecutionProvider = Literal["auto", "cpu", "cuda"]
"""What a caller may ask for.

`auto` uses the GPU when one answers and the CPU when none does. `cuda`
insists: it fails rather than fall back. `cpu` never looks.
"""

EXECUTION_PROVIDERS: tuple[ExecutionProvider, ...] = ("auto", "cpu", "cuda")

_CUDA = "CUDAExecutionProvider"

# What to hand CUDA beside the provider name. `kNextPowerOfTwo` is ONNX
# Runtime's default: it rounds every allocation up, so the arena ends up
# holding roughly twice what the sessions asked for and keeps it for the life
# of the process. Measured on a 4 GB GTX 1650, Qwen3-TTS cloning fell from
# 3222 MiB to 1626 and a model's residue after unloading from ~750 MiB to
# ~100 — which is the difference between "cannot load this model after any
# other" and "switch freely". No measurable cost: RTF 2.82 against 2.78.
CUDA_OPTIONS = {"arena_extend_strategy": "kSameAsRequested"}
_CUDA_PRELOADED = False


class _Session(Protocol):
    """The one thing this module needs from an ONNX Runtime session."""

    def get_providers(self) -> list[str]: ...


class ProviderUnavailableError(RuntimeError):
    """`cuda` was required and the sessions came back on the CPU."""


def requested(choice: ExecutionProvider) -> str:
    """Return what to hand the vendored runtimes, which know `cpu` and `cuda`.

    `auto` asks for CUDA only when the runtime claims to have it — and that
    claim is not trusted afterwards, only used to avoid asking for something
    the build plainly does not contain.

    Asking is also where the CUDA libraries are made loadable, because it is
    the one point every backend passes through before it creates a session.
    One vendored runtime does this for itself and the others do not; leaving it
    to them means a backend that works and a backend that silently lands on the
    CPU, for a reason that is nowhere near either.
    """
    if choice == "cpu" or (choice == "auto" and not _cuda_is_offered()):
        return "cpu"
    _preload_cuda()
    return "cuda"


def sessions_of(runtime: Any) -> list[Any]:
    """Return every ONNX Runtime session an object holds.

    Found, not listed. The vendored runtimes each build a different set of
    graphs and gain one now and then; a hand-written list of attribute names
    would go on reporting `cuda` after the new graph quietly landed on the CPU,
    which is precisely the failure this module exists to make loud.
    """
    found: list[Any] = []
    for value in getattr(runtime, "__dict__", {}).values():
        held = value.values() if isinstance(value, dict) else [value]
        found.extend(session for session in held if hasattr(session, "get_providers"))
    return found


def in_use(sessions: list[Any]) -> str:
    """Return the provider these sessions are actually running on.

    `cuda` only when every session got it. A graph that fell back while its
    neighbours did not is not a GPU deployment; it is a slow one wearing the
    label, and the number that matters — how fast a reply renders — will say
    so while the label does not.
    """
    if not sessions:
        return "cpu"
    return "cuda" if all(_CUDA in s.get_providers() for s in sessions) else "cpu"


def verify(choice: ExecutionProvider, actual: str) -> str:
    """Return the provider in use, refusing a silent downgrade.

    Raises:
        ProviderUnavailableError: `cuda` was required and CPU is what arrived.
    """
    if choice == "cuda" and actual != "cuda":
        raise ProviderUnavailableError(
            "execution_provider is set to cuda but the sessions came back on "
            "the CPU. The runtime lists a CUDA provider it cannot load — check "
            "that onnxruntime-gpu matches the installed CUDA and driver."
        )
    return actual


def _cuda_is_offered() -> bool:
    """Whether this build claims a CUDA provider. A claim, not a promise."""
    import onnxruntime as ort

    return _CUDA in ort.get_available_providers()


def _preload_cuda() -> None:
    """Load the CUDA and cuDNN libraries the wheel ships, once.

    Without it ONNX Runtime looks only where the system puts them, and a wheel
    carrying its own falls back to the CPU with a warning nobody reads.
    """
    global _CUDA_PRELOADED
    if _CUDA_PRELOADED:
        return
    import onnxruntime as ort

    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload(cuda=True, cudnn=True)
    _CUDA_PRELOADED = True
