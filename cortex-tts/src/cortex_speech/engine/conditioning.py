"""Per-reference conditioning, cached once for every engine that needs it.

Encoding a reference recording is the dominant cost of a cloned utterance —
960 ms for a 125-frame MOSS prompt, paid again for every sentence by a runtime
that takes a file path rather than the codes it derives from it.

The cache lives here rather than inside each engine so that invalidation has
one definition. Two engines had two rules: one compared the recording's
fingerprint, the other trusted `forget()` alone.

An engine that hands over a `Sidecar` also keeps the encoding on disk, next to
the recording, so an engine that was unloaded and comes back does not pay the
encode again. The fingerprint is part of the file name: a replaced recording
misses by construction, and the stale file is swept when the new one lands.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ..references import Reference

_LOGGER = logging.getLogger(__name__)


class Sidecar[T](Protocol):
    """How one engine's conditioning is written beside the recording."""

    @property
    def suffix(self) -> str:
        """File suffix naming the engine, e.g. ``.omni.pt``."""
        ...

    def dump(self, value: T, path: Path) -> None:
        """Write ``value`` to ``path``."""
        ...

    def load(self, path: Path) -> T:
        """Read what ``dump`` wrote."""
        ...


class ConditioningCache[T]:
    """Whatever an engine derives from a reference recording, kept by id."""

    __slots__ = ("_directory", "_entries", "_sidecar")

    def __init__(
        self, sidecar: Sidecar[T] | None = None, directory: Path | None = None
    ) -> None:
        """Cache in memory, and on disk under ``directory`` when given both."""
        self._entries: dict[str, tuple[str, T]] = {}
        self._sidecar = sidecar if directory is not None else None
        self._directory = directory

    def get(self, reference: Reference, encode: Callable[[Reference], T]) -> T:
        """Return a reference's conditioning, encoding it on first use.

        The fingerprint covers the stored audio only, so correcting a
        transcript keeps the encoding while replacing the recording under a
        reused id drops it.
        """
        cached = self._entries.get(reference.id)
        if cached is not None and cached[0] == reference.fingerprint:
            return cached[1]
        value = self._read(reference)
        if value is None:
            value = encode(reference)
            self._write(reference, value)
        self._entries[reference.id] = (reference.fingerprint, value)
        return value

    def forget(self, reference_id: str) -> None:
        """Drop a reference's conditioning after it changed or was deleted."""
        self._entries.pop(reference_id, None)
        for stale in self._sidecars(reference_id):
            stale.unlink(missing_ok=True)

    def _path(self, reference: Reference) -> Path:
        assert self._sidecar is not None and self._directory is not None
        return self._directory / (
            f"{reference.id}.{reference.fingerprint}{self._sidecar.suffix}"
        )

    def _sidecars(self, reference_id: str) -> list[Path]:
        if self._sidecar is None or self._directory is None:
            return []
        return list(self._directory.glob(f"{reference_id}.*{self._sidecar.suffix}"))

    def _read(self, reference: Reference) -> T | None:
        if self._sidecar is None:
            return None
        path = self._path(reference)
        if not path.is_file():
            return None
        try:
            value = self._sidecar.load(path)
        except Exception as err:  # noqa: BLE001 - any bad file means re-encode
            _LOGGER.warning("ignoring unreadable %s (%s); re-encoding", path, err)
            path.unlink(missing_ok=True)
            return None
        _LOGGER.debug("loaded conditioning for %s from %s", reference.id, path)
        return value

    def _write(self, reference: Reference, value: T) -> None:
        if self._sidecar is None:
            return
        path = self._path(reference)
        for stale in self._sidecars(reference.id):
            stale.unlink(missing_ok=True)
        partial = path.with_name(path.name + ".partial")
        try:
            self._sidecar.dump(value, partial)
            os.replace(partial, path)
        except OSError as err:
            _LOGGER.warning("could not keep conditioning at %s (%s)", path, err)
            partial.unlink(missing_ok=True)
