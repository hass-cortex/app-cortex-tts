"""Which way a live reply is spoken, and what each request carries.

Two outcomes. **Streaming**: one sentence per request, rendered as sentences
arrive, and nothing plays until `BANK_S` of audio has been rendered or the
reply has been rendered whole — or, once the writer has finished and the rest
of the reply is in hand, until `bank_needed` says the rest cannot run dry;
from then on each sentence goes out as it finishes. **Buffered**: whenever the engine is free it takes every complete
sentence waiting as one request, and nothing plays until everything has been
rendered.

Which one a reply gets is `verdict`: the voice's median real-time factor on
this host against a threshold. `STREAM_RTF` is the default; a host may move
it (`stream_rtf` in the app's settings), trading the wait before the first
word against the chance of a gap. The bank is what left at most 2% of 182
multi-sentence production replies a gap under two seconds on a host at
0.6–0.7 and none on a GPU host, where releasing each sentence unheld broke a
third of them. Both constants are argued from `scripts/replay_pacing.py`,
and a change to either comes with its output.
"""

from __future__ import annotations

from typing import Literal

from .sentences import SentenceBuffer

STREAMING = "streaming"
BUFFERED = "buffered"
Mode = Literal["streaming", "buffered"]

# A voice whose median real-time factor is under this streams — the default;
# the app lets a host set its own.
STREAM_RTF = 0.8

# Audio seconds rendered before a streaming reply's first sound, while how
# much reply is still to come is unknown.
BANK_S = 6.0


def bank_needed(rtf: float, remaining: list[float]) -> float:
    """Audio to hold before the first sound so the rest never runs dry.

    `remaining` is the audio, in seconds, still to be rendered, in order —
    the unrendered part of the request in flight first. Rendering runs back
    to back at `rtf` while playback runs at one, so each sentence must be
    rendered by the time playback reaches it: it needs ``rtf × a`` of lead,
    less what the sentences before it gave back (or plus what they ate).
    The bound is exact for an engine that hands a sentence over whole and
    conservative for one that releases audio within it.
    """
    needed, before = 0.0, 0.0
    for audio in remaining:
        needed = max(needed, rtf * audio + (rtf - 1.0) * before)
        before += audio
    return needed


def verdict(rtf: float | None, threshold: float = STREAM_RTF) -> Mode:
    """Stream if this voice measured under the threshold; buffered otherwise.

    `None` — nothing measured yet, or too little — is buffered: a reply that
    releases nothing until it is rendered cannot run dry.
    """
    if rtf is None:
        return BUFFERED
    return STREAMING if rtf < threshold else BUFFERED


class Pacer:
    """Hands out the next request's text for a reply arriving in pieces.

    Pure: no clock, no socket. The caller feeds text, says when the writer
    has finished, and asks for the next request whenever the engine is free.
    """

    def __init__(self, mode: Mode) -> None:
        self.mode: Mode = mode
        self._buffer = SentenceBuffer()

    def feed(self, delta: str) -> None:
        """Append what the writer produced."""
        self._buffer.feed(delta)

    def end(self) -> None:
        """The writer has finished; whatever is buffered is the whole reply."""
        self._buffer.end()

    @property
    def ended(self) -> bool:
        """Whether the writer said there is no more."""
        return self._buffer.ended

    def finished(self) -> bool:
        """Nothing left to render: the writer ended and the buffer is empty."""
        return self._buffer.ended and not self._buffer

    def remaining(self) -> list[str]:
        """What has not been handed out yet, sentence by sentence, the
        unfinished tail last; left in place."""
        tail = self._buffer.tail
        return [*self._buffer.sentences, *([tail] if tail.strip() else [])]

    def next_request(self) -> str | None:
        """Text for the next request, or `None` when nothing is ready.

        Streaming takes one sentence, so playback can start on the first;
        buffered takes everything complete, so the engine renders as few
        requests as the writer's pace allows. After `end` the unfinished tail
        is a sentence too.
        """
        sentences = self._buffer.sentences
        if not sentences:
            if self._buffer.ended and self._buffer:
                return self._buffer.take_all()
            return None
        if self.mode == STREAMING:
            [sentence] = self._buffer.take_sentences(1)
            return sentence
        if self._buffer.ended:
            return self._buffer.take_all()
        return "".join(self._buffer.take_sentences(len(sentences)))
