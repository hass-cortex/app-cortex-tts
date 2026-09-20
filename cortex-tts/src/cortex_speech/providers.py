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

from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    import onnxruntime as ort

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

# The Python wrapper's references into the C++ session. ONNX Runtime has no
# `close()`; its own `_reset_session` nulls exactly these before `_sess`, and
# that is what returns a CUDA arena to the card. Pinned to the wrapper's
# source by a test, because these names are not API.
_SESSION_REFERENCES = (
    "_sess_options",
    "_inputs_meta",
    "_outputs_meta",
    "_overridable_initializers",
    "_input_meminfos",
    "_output_meminfos",
    "_input_epdevices",
    "_model_meta",
    "_providers",
    "_provider_options",
    "_profiling_start_time_ns",
)


class SessionClosedError(RuntimeError):
    """The engine holding this session has been closed."""


class _ClosedSession:
    """What a released wrapper points at, so a late call says why it failed."""

    def __getattr__(self, name: str) -> Any:
        raise SessionClosedError(f"session is closed ({name} was called)")


def run_options(session: Any) -> ort.RunOptions | None:
    """What this session's `run` should carry; `None` where there is nothing.

    Each session owns a BFC arena that only grows, and a runtime with many
    sessions keeps every one's high-water mark. Shrinking after each run
    returns the extensions. Measured on a 4 GB GTX 1650 with MOSS-TTS-Nano's
    nine sessions: 3694 MiB standing without it, 2784 with, at RTF 0.38
    against 0.36.

    Asked of the session rather than of the provider the caller wanted, for
    the same reason `in_use` reads the sessions: a graph that fell back to the
    CPU has no `gpu:0` arena, and asking to shrink one it does not have is an
    invalid argument that fails the run rather than doing nothing. A runtime
    may build several graphs and not every one of them takes CUDA.
    """
    if _CUDA not in session.get_providers():
        return None
    import onnxruntime as ort

    options = ort.RunOptions()
    options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    return options


def release_sessions(runtime: Any) -> int:
    """Destroy every session an object holds, now, and say how many.

    This is what returns a card's memory: the C++ session dies with its last
    Python reference, and the wrapper holds several. Dropping them here makes
    release a line in the code rather than whatever the collector gets to.
    Idempotent — a wrapper already released has nothing left to drop.
    """
    released = 0
    for session in sessions_of(runtime):
        if isinstance(getattr(session, "_sess", None), _ClosedSession):
            continue
        for name in _SESSION_REFERENCES:
            if hasattr(session, name):
                setattr(session, name, None)
        session._sess = _ClosedSession()  # noqa: SLF001 - the wrapper has no close
        released += 1
    return released


class _Session(Protocol):
    """The one thing this module needs from an ONNX Runtime session."""

    def get_providers(self) -> list[str]: ...


class ProviderUnavailableError(RuntimeError):
    """`cuda` was required and the sessions came back on the CPU."""


def installed_builds() -> list[str]:
    """Which ONNX Runtime distributions this environment holds.

    The CPU and the GPU wheel unpack into the same package directory, so an
    environment holding both runs whichever installed last — and says so
    nowhere. One build is the only sane state; `pyproject.toml` keeps them in
    conflicting dependency groups so a sync cannot produce two.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    found = []
    for name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            distribution(name)
        except PackageNotFoundError:
            continue
        found.append(name)
    return found


def check_installation() -> None:
    """Refuse to start on an environment holding both ONNX Runtime builds.

    Raises:
        RuntimeError: Both are installed; the message says how to sync one.
    """
    builds = installed_builds()
    if len(builds) > 1:
        raise RuntimeError(
            "both onnxruntime and onnxruntime-gpu are installed and one shadows "
            "the other; sync exactly one group — `uv sync --frozen` for the "
            "CPU build, `uv sync --frozen --no-default-groups --group dev "
            "--group cuda` (or `cuda12`) for a card"
        )


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
