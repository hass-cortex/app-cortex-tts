"""Sentences out of text that arrives in pieces."""

from __future__ import annotations

# The live form of the batch splitter: the same terminators, so a sentence
# the pacer hands out is the sentence `segment` would have cut. The ASCII
# stop is only a break when followed by whitespace (a decimal guard), so a
# buffer ending in "." is still open: the next character decides.
from ..text.scripts import ENDS_SENTENCE, SENTENCE_BREAK


class SentenceBuffer:
    """Accumulates text deltas and hands out the sentences that are complete.

    The last sentence is complete only once its stop has arrived; until then
    it is the `tail` — unless the writer has finished, when the tail is all
    there will ever be.
    """

    def __init__(self) -> None:
        self._text = ""
        self._ended = False

    @property
    def ended(self) -> bool:
        """Whether the writer said there is no more."""
        return self._ended

    def feed(self, delta: str) -> None:
        """Append what the writer produced."""
        self._text += delta

    def end(self) -> None:
        """The writer has finished; whatever is buffered is the whole reply."""
        self._ended = True

    def _split(self) -> tuple[list[str], str]:
        parts = [p for p in SENTENCE_BREAK.split(self._text) if p.strip()]
        if not parts:
            return [], ""
        if self._ended or ENDS_SENTENCE.search(self._text):
            return parts, ""
        return parts[:-1], parts[-1]

    @property
    def sentences(self) -> list[str]:
        """Complete sentences, in order, still in the buffer."""
        return self._split()[0]

    @property
    def tail(self) -> str:
        """Text after the last complete sentence; empty once the writer ended."""
        return self._split()[1]

    def take_sentences(self, count: int) -> list[str]:
        """Remove and return the first `count` complete sentences."""
        sentences, tail = self._split()
        taken, kept = sentences[:count], sentences[count:]
        self._text = "".join(kept) + tail
        return taken

    def take_all(self) -> str:
        """Remove and return everything, sentences and tail alike."""
        text, self._text = self._text, ""
        return text

    def __bool__(self) -> bool:
        return bool(self._text.strip())
