"""What to render next, and how long to hold the opening.

The planner sees text as the writer produces it and the listener's lead as the
transport measures it, and answers one question at a time: send this much
now, wait for more text, or there is nothing left. It never touches a clock
or a socket, so every decision here is reproducible from its inputs.

Three ways a reply can be spoken, chosen per reply rather than per model:

- **streaming**: the model gains lead on every request, so the first batch
  goes out as soon as it has enough content and the rest are sized to fit
  inside the lead the listener holds. The opening bank is small.
- **paced**: the whole reply was written before anything had to be sent (or
  the model cannot gain lead, so nothing could be sent safely before it was).
  Every batch is known, so the opening hold is computed exactly: the least
  wait after which playback never catches the renderer.
- **buffered**: nothing is released until everything is rendered. What a
  model gets while this host has not measured it, and what a caller can ask
  for outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .model import RenderModel
from .sentences import SentenceBuffer, clause_pieces

STREAMING: Final = "streaming"
PACED: Final = "paced"
BUFFERED: Final = "buffered"

# The first request needs enough content to be worth a generation of its own:
# a bare "好了，" is a wasted prefill, a sentence-shaped pause and another
# boundary a moment later. Measured over 408 production replies, the first
# clause is under a second of speech in 42% of them and the median first
# sentence is 3.7 s, so a floor of three seconds rarely waits.
FIRST_FLOOR_S = 3.0

# Past this much speech with no sentence end in sight, a run-on first sentence
# is cut at a clause mark rather than waited for — but only where the rest is
# predicted to land in time, or the cut buys nothing but a pause mid-sentence.
FIRST_CEILING_S = 6.0

# How much audio a reply must have produced before its own pace is believed
# over the fit's. A second is enough to be a measurement and short enough to
# arrive before the opening hold is released.
SLIP_EVIDENCE_S = 1.0

# What the lead must keep in hand beyond the prediction. Renders vary, the
# host is briefly busy, the estimate of how fast the voice speaks is rounded.
MARGIN_S = 0.5

# A streaming opening bank above this is not a stream worth having: the model
# is not gaining lead, so the bank would grow with the reply, whose length is
# unknown. Such a reply is paced instead — sent once it has been written,
# with an exact hold.
HOLD_CAP_S = 4.0

# When the whole reply is known, sentences are grouped to about this much
# speech per request: fewer boundaries than one per sentence, and short of the
# lengths at which an autoregressive model's cost turns quadratic.
PACED_BATCH_S = 12.0


@dataclass(frozen=True)
class Send:
    """Render this text now.

    Attributes:
        text: What to render, as the writer wrote it.
        hold_audio_s: Bank this much audio before releasing any. Set on the
            first batch of a streaming reply to a chunk-streaming engine.
        hold_wall_s: Release this many seconds after the first audio byte
            arrives. Set on the first batch of a streaming reply to an engine
            that hands requests over whole.
        hold_bank: Bank audio until it covers what the rest of the reply is
            still predicted to lose — ask `Planner.bank_needed` as audio
            arrives. Set on the first batch of a paced reply: the plan is
            known, so the bank follows how the render actually goes rather
            than a figure fixed before it started.
        hold_all: Release nothing until the whole reply is rendered.
    """

    text: str
    hold_audio_s: float = 0.0
    hold_wall_s: float = 0.0
    hold_bank: bool = False
    hold_all: bool = False


class Wait:
    """Nothing to send yet; more text is expected."""


class Finished:
    """Everything the writer produced has been sent."""


Decision = Send | Wait | Finished


class Planner:
    """Turns arriving text and the listener's lead into render decisions.

    Args:
        model: What this host measured for the model and voice kind, or
            ``None`` when it has not — in which case the reply is buffered.
        chunk_streaming: Whether the engine emits audio while a request is
            still rendering.
        mode: ``"auto"`` to choose per reply, or ``"buffered"`` to hold
            everything regardless.
    """

    def __init__(
        self,
        model: RenderModel | None,
        *,
        chunk_streaming: bool,
        mode: str = "auto",
        floor_s: float = FIRST_FLOOR_S,
        ceiling_s: float = FIRST_CEILING_S,
        margin_s: float = MARGIN_S,
    ) -> None:
        self._model = model
        self._chunk = chunk_streaming
        self._floor = floor_s
        self._ceiling = ceiling_s
        self._margin = margin_s
        self._buffer = SentenceBuffer()
        self._sent = 0
        self._queued: list[str] = []
        self._rendering: str | None = None
        # What this reply's finished requests actually cost, against what the
        # fit said they would. The fit describes the host on its average day.
        self._spent_audio_s = 0.0
        self._spent_wall_s = 0.0
        self.mode: str = BUFFERED if mode == BUFFERED or model is None else STREAMING

    def feed(self, text: str) -> None:
        """More of the reply arrived."""
        self._buffer.feed(text)

    def rendered(self, audio_s: float, wall_s: float) -> None:
        """Note what one finished request of this reply actually cost."""
        if audio_s > 0 and wall_s > 0:
            self._spent_audio_s += audio_s
            self._spent_wall_s += wall_s

    def _slip(self, produced_s: float = 0.0, elapsed_s: float = 0.0) -> float:
        """How much dearer this reply is running than the fit predicted.

        A reply that meets a busy host has to notice before it releases audio
        it cannot sustain: the host's measured average cannot see a burst
        coming, but this reply is already inside one.

        The request still rendering counts, not only the finished ones. A
        paced reply normally releases part-way through its first request —
        measured on a host made 1.4x slower than its fit, the release came at
        12.4 s and the first request did not finish until 12.2 s — so waiting
        for a request to end is waiting until after the decision.

        Never below 1. A hold that is too long costs a wait the listener can
        sit through; one that is too short costs a gap they cannot un-hear,
        and by then there is no taking the audio back.
        """
        if self._model is None:
            return 1.0
        audio = self._spent_audio_s + max(0.0, produced_s)
        wall = self._spent_wall_s + max(0.0, elapsed_s)
        if audio < SLIP_EVIDENCE_S or wall <= 0:
            # A request that has produced almost nothing has spent almost no
            # time predictably: at the top of a batch the prediction tends to
            # zero while the wall does not, and the ratio of the two is
            # arithmetic, not evidence. Measured before this guard, a reply on
            # a host with no fixed cost held its whole nine seconds rather
            # than the 1.6 s the line asked for.
            return 1.0
        predicted = self._model.render_seconds(audio)
        if predicted <= 0:
            return 1.0
        return max(1.0, wall / predicted)

    def end(self) -> None:
        """The reply is complete."""
        self._buffer.end()

    @property
    def ended(self) -> bool:
        """Whether the writer has finished."""
        return self._buffer.ended

    def plan(self, lead_s: float | None) -> Decision:
        """Decide what to do now.

        Args:
            lead_s: Audio handed to the listener minus wall time since the
                first byte, or ``None`` before anything has been sent.
        """
        if self.mode == BUFFERED:
            return self._buffered()
        if self._sent == 0:
            return self._first()
        if self.mode == PACED:
            return self._paced_next()
        return self._streaming_next(lead_s or 0.0)

    # -- buffered ---------------------------------------------------------

    def _buffered(self) -> Decision:
        if not self._buffer.ended:
            return Wait()
        if self._buffer:
            self._sent += 1
            return Send(self._buffer.take_all(), hold_all=True)
        return Finished()

    # -- the first request --------------------------------------------------

    def _first(self) -> Decision:
        assert self._model is not None
        if self._buffer.ended:
            return self._pace_whole_reply()

        sentences = self._buffer.sentences
        head = "".join(sentences)
        if sentences and self._model.audio_seconds(head) >= self._floor:
            count = self._sentences_within(sentences, self._ceiling)
            return self._open("".join(self._buffer.take_sentences(count)))

        whole = head + self._buffer.tail
        if self._model.audio_seconds(whole) >= self._ceiling:
            cut = self._clause_cut(whole)
            if cut is not None:
                self._buffer.take_prefix(cut)
                return self._open(cut)
        return Wait()

    def _sentences_within(self, sentences: list[str], limit_s: float) -> int:
        """How many leading sentences fit in `limit_s` of speech; at least one."""
        assert self._model is not None
        count, text = 0, ""
        for sentence in sentences:
            if count and self._model.audio_seconds(text + sentence) > limit_s:
                break
            text += sentence
            count += 1
        return count

    def _clause_cut(self, text: str) -> str | None:
        """The longest opening cut, within the ceiling, whose rest lands in time.

        The rest is what has arrived so far; more may follow, so this is the
        optimistic side of the estimate, which is why the margin exists.
        """
        assert self._model is not None
        pieces = clause_pieces(text)
        best: str | None = None
        prefix = ""
        for piece in pieces[:-1]:
            prefix += piece
            spoken = self._model.audio_seconds(prefix)
            if spoken < self._floor:
                continue
            if spoken > self._ceiling:
                break
            rest = self._model.audio_seconds(text[len(prefix) :])
            deficit = self._model.deficit(rest, chunk_streaming=self._chunk)
            if deficit <= spoken - self._margin:
                best = prefix
        return best

    def _open(self, text: str) -> Decision:
        """Send the first batch of a streaming reply, or pace it instead.

        Whether the model can stream at all is judged on a typical batch —
        about the ceiling's worth of speech — not on this first piece, which
        a clause cut may have made short. The opening hold covers the first
        boundary: the next request is assumed to be a typical one, and
        whatever of its deficit this batch's playback does not cover is held
        back first. A model that is not gaining lead per request, or one
        whose opening hold would pass the cap, cannot stream a reply of
        unknown length safely — that reply is paced once written.
        """
        assert self._model is not None
        audio = self._model.audio_seconds(text)
        typical = max(audio, self._ceiling)
        next_deficit = self._model.deficit(typical, chunk_streaming=self._chunk)
        if self._chunk:
            hold = next_deficit + self._margin + self._model.spread_s
        else:
            hold = max(0.0, next_deficit - audio) + self._margin + self._model.spread_s
        gains_lead = typical - self._model.render_seconds(typical) > self._margin
        if hold > HOLD_CAP_S or not gains_lead:
            # Put the text back where it was: the paced plan takes everything.
            self._buffer = _prepend(self._buffer, text)
            self.mode = PACED
            return Wait() if not self._buffer.ended else self._pace_whole_reply()

        self.mode = STREAMING
        self._sent += 1
        if self._chunk:
            return Send(text, hold_audio_s=round(hold, 3))
        return Send(text, hold_wall_s=round(hold, 3))

    # -- streaming --------------------------------------------------------

    def _streaming_next(self, lead_s: float) -> Decision:
        assert self._model is not None
        sentences = self._buffer.sentences
        if self._buffer.ended and self._buffer.tail:
            sentences = [*sentences, self._buffer.tail]
        if not sentences:
            return Finished() if self._buffer.ended else Wait()

        # The largest prefix whose deficit the lead covers; one sentence when
        # none does, because waiting only shrinks the lead further.
        room = lead_s - self._margin
        count, text = 0, ""
        for sentence in sentences:
            candidate = text + sentence
            deficit = self._model.deficit(
                self._model.audio_seconds(candidate), chunk_streaming=self._chunk
            )
            if count and deficit > room:
                break
            text, count = candidate, count + 1
        self._sent += 1
        if count == len(sentences) and self._buffer.ended:
            return Send(self._buffer.take_all())
        return Send("".join(self._buffer.take_sentences(count)))

    # -- paced ----------------------------------------------------------------

    def _pace_whole_reply(self) -> Decision:
        """The reply is known: group it, compute the exact hold, send the first."""
        assert self._model is not None
        self.mode = PACED
        text = self._buffer.take_all()
        if not text.strip():
            return Finished()
        sentences = SentenceBuffer()
        sentences.feed(text)
        sentences.end()
        self._queued = self._group(sentences.sentences, PACED_BATCH_S)
        self._sent += 1
        self._rendering = self._queued.pop(0)
        return Send(self._rendering, hold_bank=True)

    def _paced_next(self) -> Decision:
        if not self._queued:
            self._rendering = None
            return Finished()
        self._sent += 1
        self._rendering = self._queued.pop(0)
        return Send(self._rendering)

    def bank_needed(self, produced_s: float, elapsed_s: float = 0.0) -> float:
        """Audio the listener must hold now so the rest never runs dry.

        Requests are dispatched back to back. A chunk-streaming engine loses
        `fixed + (b − 1)·audio` over each request; the one rendering has
        paid its fixed part and `produced_s` of its audio already. A
        whole-render engine hands each request over at its end, so what is
        needed is the deepest point of the same accounting `_exact_hold`
        does, taken over what is still to come with the current request
        counted whole. The margin and the fit's spread are on top, once.
        """
        assert self._model is not None
        # Dearer by whatever this reply has cost so far, so every deficit below
        # is taken from the line it is actually running on.
        model = self._model.scaled(self._slip(produced_s, elapsed_s))
        remaining = ([self._rendering] if self._rendering else []) + list(self._queued)
        if not remaining:
            return 0.0
        if self._chunk:
            needed = 0.0
            for index, batch in enumerate(remaining):
                audio = model.audio_seconds(batch)
                if index == 0:
                    needed += max(0.0, model.per_audio - 1.0) * max(
                        0.0, audio - produced_s
                    )
                else:
                    needed += model.deficit(audio, chunk_streaming=True)
        else:
            needed = self._exact_hold(remaining, from_dispatch=True)
        # Scaled by what this reply has cost so far, not only by the fit: the
        # margin and the spread are fixed allowances and stay outside it.
        return needed + self._margin + model.spread_s

    def _group(self, sentences: list[str], limit_s: float) -> list[str]:
        assert self._model is not None
        batches: list[str] = []
        current = ""
        for sentence in sentences:
            if current and self._model.audio_seconds(current + sentence) > limit_s:
                batches.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            batches.append(current)
        return batches

    def _exact_hold(self, batches: list[str], *, from_dispatch: bool = False) -> float:
        """Seconds to hold after the first audio so playback never catches up.

        With `from_dispatch`, the bare figure from the moment the first of
        these batches was dispatched — no margin, no spread — for a caller
        that adds its own.

        Requests are dispatched back to back. A whole-render engine makes a
        batch's audio available when its render ends; a chunk-streaming one
        makes it available from its fixed cost onwards, at the rate it renders.
        Playback starting at `H` needs the cumulative audio at every one of
        those points, and `H` is the smallest value that satisfies all of them.
        """
        assert self._model is not None
        model = self._model
        dispatched = 0.0
        played_before = 0.0
        needed = 0.0
        first_audio_at: float | None = None
        for batch in batches:
            audio = model.audio_seconds(batch)
            render = model.render_seconds(audio)
            if self._chunk:
                starts = dispatched + model.fixed_s
                ends = dispatched + render
                needed = max(
                    needed, starts - played_before, ends - played_before - audio
                )
                first_audio_at = starts if first_audio_at is None else first_audio_at
            else:
                ends = dispatched + render
                needed = max(needed, ends - played_before)
                first_audio_at = ends if first_audio_at is None else first_audio_at
            dispatched += render
            played_before += audio
        if from_dispatch:
            return max(0.0, needed)
        return (
            max(0.0, needed - (first_audio_at or 0.0)) + self._margin + model.spread_s
        )


def _prepend(buffer: SentenceBuffer, text: str) -> SentenceBuffer:
    fresh = SentenceBuffer()
    fresh.feed(text)
    fresh.feed(buffer.take_all())
    if buffer.ended:
        fresh.end()
    return fresh
