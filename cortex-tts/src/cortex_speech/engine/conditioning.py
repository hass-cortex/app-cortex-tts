"""Per-reference conditioning, cached once for every engine that needs it.

Encoding a reference recording is the dominant cost of a cloned utterance —
960 ms for a 125-frame MOSS prompt, paid again for every sentence by a runtime
that takes a file path rather than the codes it derives from it. The
measurements are in docs/adr/0002.

The cache lives here rather than inside each engine so that invalidation has
one definition. Two engines had two rules: one compared the recording's
fingerprint, the other trusted `forget()` alone.
"""

from __future__ import annotations

from collections.abc import Callable

from ..references import Reference


class ConditioningCache[T]:
    """Whatever an engine derives from a reference recording, kept by id."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, T]] = {}

    def get(self, reference: Reference, encode: Callable[[Reference], T]) -> T:
        """Return a reference's conditioning, encoding it on first use.

        The fingerprint covers the stored audio only, so correcting a
        transcript keeps the encoding while replacing the recording under a
        reused id drops it.
        """
        cached = self._entries.get(reference.id)
        if cached is not None and cached[0] == reference.fingerprint:
            return cached[1]
        value = encode(reference)
        self._entries[reference.id] = (reference.fingerprint, value)
        return value

    def forget(self, reference_id: str) -> None:
        """Drop a reference's conditioning after it changed or was deleted."""
        self._entries.pop(reference_id, None)
