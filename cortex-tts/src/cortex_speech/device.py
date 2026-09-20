"""The device: how much memory it has left, and when it has none.

Every diagnosis of this app's memory behaviour so far has meant reading
`nvidia-smi` in one window beside the log in another, and then lining the two
up by timestamp. The number belongs in the log: an arena that grew, an engine
that did not give its memory back and a card another process filled are three
different problems that produce the same symptom, and the figure at each
lifecycle transition is what separates them.

`nvidia-smi` rather than a binding. `pynvml` would be a dependency for one
integer, and `torch.cuda` is only present when one of the two model extras is
installed — this has to work on a plain ONNX Runtime install too. The cost is
a subprocess, so it is read once per lifecycle transition and never per
request; loading a model takes seconds and this takes milliseconds.

Device-wide, not per-process, and deliberately: the card is shared. On the
host this was written for it also runs Ollama, and "our arena grew" versus
"something else took the card" is exactly the distinction a reader needs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import NamedTuple

_LOGGER = logging.getLogger(__name__)

_QUERY = "--query-gpu=memory.used,memory.total"
_TIMEOUT_SECONDS = 2.0

_missing_reported = False


class Memory(NamedTuple):
    """Used and total device memory, in MiB."""

    used: int
    total: int

    def __str__(self) -> str:
        return f"{self.used}/{self.total} MiB"


def memory() -> Memory | None:
    """Read the first GPU's memory use, or `None` when there is no reading.

    Never raises and never blocks for long: no driver, no binary, a timeout or
    output in a shape this does not recognise all mean the same thing to a
    caller, which is that this line of the log has no figure on it. A log line
    is not worth an exception.
    """
    global _missing_reported
    binary = shutil.which("nvidia-smi")
    if binary is None:
        if not _missing_reported:
            _missing_reported = True
            _LOGGER.debug("no nvidia-smi; device memory will not be logged")
        return None
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [binary, _QUERY, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as err:
        _LOGGER.debug("could not read device memory: %s", err)
        return None
    first = done.stdout.strip().splitlines()
    if not first:
        return None
    try:
        used, total = (int(field.strip()) for field in first[0].split(",", 1))
    except ValueError:
        return None
    return Memory(used, total)


# What ONNX Runtime says when the device has no memory left. Matched on the
# message, not the type: the arena raises through a pybind class that only
# exists once onnxruntime is imported, and nothing else here needs that import.
# The first two are the BFC arena, inside a kernel or while extending. The
# rest are the libraries a kernel calls failing to get their workspace once
# the arena has the card: measured on a 4 GB GTX 1650 with three models
# resident (3666/4096 MiB), Hojo failed every request with cuBLAS "resource
# allocation failed" and cuDNN "INTERNAL_ERROR" and was kept, so nothing
# recovered until a restart.
_EXHAUSTED = (
    "Failed to allocate memory for requested buffer",
    "out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "the resource allocation failed",
    "CUDNN_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_INTERNAL_ERROR",
)


def exhausted(error: BaseException) -> bool:
    """Whether a failure is the device running out of memory.

    A GPU arena never gives memory back short of losing the session, so this
    is not a transient error that retrying fixes: it is the signal to drop the
    engine holding the arena. It reads the message of every exception in the
    chain, because the runtime wraps its own.
    """
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        text = str(error)
        if any(mark in text for mark in _EXHAUSTED):
            return True
        error = error.__cause__ or error.__context__  # type: ignore[assignment]
    return False
