"""What to render next, and how long to hold the opening.

The planner sees text as the writer produces it and the listener's lead as the
transport measures it, and answers one question at a time: send this much
now, wait for more text, or there is nothing left. It never touches a clock
or a socket, so every decision here is reproducible from its inputs.

Four ways a reply can be spoken, chosen per reply rather than per model:

- **streaming**: the model gains lead on every request, so the first batch
  goes out as soon as it has enough content and the rest are sized to fit
  inside the lead the listener holds. The opening bank is small.
- **planned**: the whole reply was written before anything had to be sent (or
  the model cannot gain lead, so nothing could be sent safely before it was).
  Every batch is known, so the opening hold is computed exactly: the least
  wait after which playback never catches the renderer.
- **unheld**: the same words in hand, grouped for the soonest first word
  instead of the fewest boundaries, and sent with no hold at all. It runs dry
  on a model that cannot keep ahead, which is the trade it exists to offer: a
  reply that starts at once against one that never stalls. Asked for, never
  concluded — a strategy nothing here switches to on its own.
- **buffered**: nothing is released until everything is rendered. What a
  caller can ask for outright, and never a conclusion this planner draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .model import (
    RenderModel,
    RenderSample,
    count_scripts,
    speech_rates,
)
from .schedule import Schedule
from .sentences import SentenceBuffer

STREAMING: Final = "streaming"
PLANNED: Final = "planned"
UNHELD: Final = "unheld"
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

# How much silence a sentence end may absorb before it stops being a pause.
# Playback that catches the renderer at a sentence end is heard as a longer
# gap between sentences, which is where a gap belongs; the hold only has to
# cover what a boundary cannot swallow. Measured on one 313-character reply,
# OmniVoice with a clone: holding until nothing could ever run dry put the
# first word at 6.29 s, and letting each sentence end carry its own pause put
# it at 4.87 s with one gap of 0.55 s, after the first sentence.
#
# Only a sentence end earns it. A batch cut at a clause mark by `cut_to_cap`
# ends mid-sentence, and silence there is a fault rather than a pause — and a
# chunk-streaming engine earns nothing either, because its audio arrives
# continuously, so running dry falls wherever it ran out.
SENTENCE_PAUSE_S = 1.5

# The most speech one request may carry. It is where a fitted line stops
# describing the model, so past it the planner would be choosing between plans
# on arithmetic it knows to be wrong — and the search below only ever proposes
# a longer request because the line says it is cheaper.
#
# Measured on OmniVoice with a clone, GTX 1650, one request at a time against a
# line fitted from requests under 8 s:
#
#     audio    3.9   6.3   9.2  10.7  12.7  15.2  19.1  23.6
#     over by    0%    0%    1%   19%   18%   33%   60%  100%
#
# So the line holds to 9.2 s and has broken by 10.7 s. Nine is the longest
# request measured still on it. The floor of the total-cost curve is near the
# same place — the same 20.9 s reply cost 27.9 s whole, 17.0 s in halves and
# 16.5 s in thirds — so a cap here is not paid for in extra boundaries.
#
# Where the break falls is the model's, not the host's, and it is not the same
# for every model: MOSS-TTS-Nano stayed within 6% of its own line out to 28 s.
# So this is the default — the earliest break measured — and a model that
# breaks later says so in `ModelSpec.batch_cap_s`. Holding one to this figure
# is not free, which is what the extra-boundary reasoning above missed: it
# counted render time (MOSS pays 0.05 s a boundary against OmniVoice's 1.2)
# and not the cuts. A sentence longer than the cap is cut at its own clause
# marks and every segment is terminated, so those land in the audio as full
# stops — measured on one 63 s reply, held to nine MOSS took ten requests
# instead of three, 11% more render, and two cuts inside sentences, for 0.09 s
# of opening a chunk-streaming engine does not gain: it is audible from its
# fixed cost whatever the cut.
BATCH_CAP_S = 9.0

# How much further than the longest request that stayed on the line the
# planner may reach, when the line held all the way to the top of what this
# host has tried. That is the only way the cap can ever grow: every sample is
# itself capped, so a store left alone can confirm a break or find an earlier
# one and never find a later one — the cap would bound the evidence that would
# move it. A step rather than a leap because the step is what is being risked:
# a request past the line costs a hold computed too short, and the drift just
# past a length that held is a few per cent on every model measured. It stops
# on its own — the first request that misses shortens `holds_to_s`, and the
# search stops proposing that length.
PROBE = 1.25


@dataclass(frozen=True)
class Send:
    """Render this text now.

    Attributes:
        text: What to render, as the writer wrote it.
        hold_audio_s: Bank this much audio before releasing any. Set on the
            first batch of a streaming reply to a chunk-streaming engine.
        hold_wall_s: Release this many seconds after the first audio byte
            arrives. Set on the first batch of a streaming reply to an engine
            that hands requests over whole, and on a planned reply, whose plan
            says exactly when playback may start.
        hold_bank: Bank audio until it covers what the rest of the reply is
            still predicted to lose — ask `Planner.bank_needed` as audio
            arrives. Set on the first batch of a planned reply beside the
            deadline, as the early release: it follows how the render actually
            goes, so a host running ahead of its fit speaks sooner.
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
            ``None`` when it has not — in which case the reply is planned
            from its own first request.
        chunk_streaming: Whether the engine emits audio while a request is
            still rendering.
        mode: ``"auto"`` to choose per reply, or one of ``"buffered"``,
            ``"planned"`` and ``"streaming"`` to insist. Insisting is for a
            caller who wants to hear one particular delivery rather than the
            best one — the admin UI's live panel offers all four, so the three
            can be compared on the same reply. `auto` remains what anything
            serving a listener should send.
        batch_cap_s: The most speech one request may carry, where this host
            has not measured its own — `ModelSpec.batch_cap_s`, the earliest
            break seen anywhere. `_cap_from` prefers the measurement.
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
        pause_s: float = SENTENCE_PAUSE_S,
        batch_cap_s: float = BATCH_CAP_S,
    ) -> None:
        self._model = model
        # The sums, kept apart from the choosing. It holds no reply state and
        # takes the cost line per call, because a reply replaces its own — and
        # it owns the tuning, which is why nothing here keeps a second copy.
        self._when = Schedule(
            chunk_streaming=chunk_streaming,
            pause_s=pause_s,
            margin_s=margin_s,
            floor_s=floor_s,
            ceiling_s=ceiling_s,
            # What the model declares is where to start, not what to believe:
            # the break belongs to the host as much as to the model, so this
            # host's own samples replace it as soon as they can say anything.
            batch_cap_s=_cap_from(model, batch_cap_s),
        )
        self._buffer = SentenceBuffer()
        self._sent = 0
        self._queued: list[str] = []
        self._rendering: str | None = None
        # What this reply's finished requests actually cost, against what the
        # fit said they would. The fit describes the host on its average day.
        self._spent_audio_s = 0.0
        self._spent_wall_s = 0.0
        # Buffered is asked for, never concluded: a host that has measured
        # nothing still knows how to cut a reply and how to learn from its
        # own first request, which is enough to pace it.
        self.mode: str = BUFFERED if mode == BUFFERED else STREAMING
        # What the caller insisted on, where they did. It can only be honoured
        # as far as the reply allows — forced streaming needs a cost line
        # before the first byte, and a host that has none still paces — so the
        # `done` frame, not this, is what says how the reply went.
        self._forced: str | None = (
            mode if mode in (PLANNED, UNHELD, STREAMING) else None
        )

    def feed(self, text: str) -> None:
        """More of the reply arrived."""
        self._buffer.feed(text)

    def rendered(self, audio_s: float, wall_s: float) -> None:
        """Note what one finished request of this reply actually cost."""
        if self._model is None and self._rendering:
            self._learn_from(self._rendering, audio_s, wall_s)
        if audio_s > 0 and wall_s > 0:
            self._spent_audio_s += audio_s
            self._spent_wall_s += wall_s

    def _slip(self, produced_s: float = 0.0, elapsed_s: float = 0.0) -> float:
        """How much dearer this reply is running than the fit predicted.

        A reply that meets a busy host has to notice before it releases audio
        it cannot sustain: the host's measured average cannot see a burst
        coming, but this reply is already inside one.

        The request still rendering counts, not only the finished ones. A
        planned reply normally releases part-way through its first request —
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
            if self._model is None or self._forced in (PLANNED, UNHELD):
                return self._plan_when_written()
            return self._first()
        if self.mode in (PLANNED, UNHELD):
            # Both took the whole reply and queued it; they differ in how it
            # was cut and whether the opening waits, not in what comes next.
            return self._planned_next()
        return self._streaming_next(lead_s or 0.0)

    # -- buffered ---------------------------------------------------------

    def _buffered(self) -> Decision:
        """One request, released when it is done. Nothing decided.

        Buffered means one render of the whole reply, and only ever because
        the caller asked for it: a host that has measured nothing still plans
        from its own first request. The cap does not reach here and neither
        does the clause cut: both exist to buy the listener an earlier first
        word, and a held reply has no earlier first word to buy.
        What a cut would buy instead is a shorter total render — measured on
        OmniVoice, 20.9 s of speech cost 27.9 s whole against 16.5 s in
        thirds — and the price is an unmeasured change in how the reply
        sounds, since every segment is terminated and so a clause mark cut at
        is spoken as a full stop. Buffered is the one mode that does not make
        that trade on anyone's behalf.
        """
        if not self._buffer.ended:
            return Wait()
        if self._buffer:
            self._sent += 1
            return Send(self._buffer.take_all(), hold_all=True)
        return Finished()

    # -- unmeasured -------------------------------------------------------

    def _plan_when_written(self) -> Decision:
        """Wait for the whole reply, then pace it.

        Two callers, one behaviour: a model this host has served nothing of,
        and a caller who asked for planned outright.

        Nothing can be promised before the first request, so nothing is
        released until it lands — but that request is itself the measurement,
        and from it the reply is planned like any other. What it cannot do is
        stream: sizing a batch to the listener's lead needs a cost line before
        the first byte, and there is none until one exists.
        """
        if not self._buffer.ended:
            return Wait()
        return self._plan_whole_reply()

    def _learn_from(self, text: str, audio_s: float, wall_s: float) -> None:
        """Take this reply's first finished request as its cost line.

        One point fixes no slope, so it is read as the whole cost: a request
        of the same size costs what this one did. That is what the cut makes
        them — `_group` fills every batch to one limit — and it is the only
        reading of one point that does not invent a second number. The pace is
        read the same way, from the characters this request actually spoke,
        which is worth far more than the prior it replaces.

        What this buys and what it costs, simulated against the OmniVoice
        clone as measured (1.5 s fixed, 0.61x, three requests): the first word
        at 6.8 s where holding the reply whole puts it at 16.5 s, with 1.72 s
        of lead in hand at the tightest moment. The tolerance that lead
        represents is 30%: a host that turns that much dearer after the first
        request still just holds, and one 60% dearer runs dry. Everything
        measurable here sits far inside it — 0.9% across back-to-back requests
        on an idle GPU, 1.8-7.8% for a fitted line's own spread against a
        typical render. What it does not cover is another process taking the
        card part-way through the reply, which is not measured. It is the
        first reply of a model on a host that has served nothing; by the
        fourth there is a fit, and `spread_s` with it.
        """
        if audio_s <= 0 or wall_s <= 0:
            return
        cjk, latin = count_scripts(text)
        sample = RenderSample(audio_s, wall_s, cjk, latin)
        rates = speech_rates([sample]) or (0.0, 0.0)
        self._model = RenderModel(
            fixed_s=0.0,
            per_audio=wall_s / audio_s,
            cjk_per_s=rates[0],
            latin_per_s=rates[1],
            samples=1,
        )

    # -- the first request --------------------------------------------------

    def _first(self) -> Decision:
        """Open a streaming reply, or decide there is nothing to stream.

        A reply already written out is planned rather than streamed: the
        length is known, so grouping the whole of it beats cutting it as it
        goes. Insisting on `streaming` is the exception, and it has to be.
        Whether `end` had arrived by the time the planner was first asked is
        a matter of which task the loop ran first — the transport creates the
        reader and plans before it has run — so honouring the request only
        while the words were still coming put the same reply on two different
        deliveries. Measured with the fake engine: ten runs on a fresh app
        planned every time, eight on a warm one streamed seven. A mode that
        depends on scheduling is one nobody can compare or reproduce.
        """
        assert self._model is not None
        if self._buffer.ended and self._forced != STREAMING:
            return self._plan_whole_reply()

        # What is in hand, the unterminated tail included once nothing more is
        # coming — the same reading `_streaming_next` takes of a finished
        # buffer, so the first batch sees the text the rest of the reply will.
        sentences = self._buffer.sentences
        if self._buffer.ended and self._buffer.tail:
            sentences = [*sentences, self._buffer.tail]
        sentences = self._when.cut_to_cap(self._model, sentences)
        head = "".join(sentences)
        if sentences and self._model.audio_seconds(head) >= self._when.floor:
            count = self._when.pieces_within(self._model, sentences, self._when.ceiling)
            text = "".join(sentences[:count])
            self._buffer.take_prefix(text)
            return self._open(text)

        whole = head if self._buffer.ended else head + self._buffer.tail
        if self._model.audio_seconds(whole) >= self._when.ceiling:
            cut = self._when.clause_cut(self._model, whole)
            if cut is not None:
                self._buffer.take_prefix(cut)
                return self._open(cut)
        # Too little to open with. A reply still being written may yet grow;
        # a finished one will not, so it is planned rather than waited for —
        # which is what insisted-on `streaming` gets when the whole reply is
        # shorter than an opening, and what the `done` frame then reports.
        return self._plan_whole_reply() if self._buffer.ended else Wait()

    def _open(self, text: str) -> Decision:
        """Send the first batch of a streaming reply, or pace it instead.

        Whether the model can stream at all is judged on a typical batch —
        about the ceiling's worth of speech — not on this first piece, which
        a clause cut may have made short. The opening hold covers the first
        boundary: the next request is assumed to be a typical one, and
        whatever of its deficit this batch's playback does not cover is held
        back first. A model that is not gaining lead per request cannot
        stream a reply of unknown length safely — the bank would grow with a
        length nobody knows yet — so that reply is planned once written. That
        is the whole of the test: a second one on the hold's own length was
        measured to decide nothing any line fitted here reaches, and to cut
        the openings it did reach short of what they were planned for.
        """
        assert self._model is not None
        audio = self._model.audio_seconds(text)
        typical = max(audio, self._when.ceiling)
        next_deficit = self._model.deficit(typical, chunk_streaming=self._when.chunk)
        spread = self._model.spread_for(self._model.render_seconds(typical))
        if self._when.chunk:
            hold = next_deficit + self._when.margin + spread
        else:
            hold = max(0.0, next_deficit - audio) + self._when.margin + spread
        gains_lead = typical - self._model.render_seconds(typical) > self._when.margin
        if self._forced != STREAMING and not gains_lead:
            # Put the text back where it was: the planned plan takes everything.
            self._buffer = _prepend(self._buffer, text)
            self.mode = PLANNED
            return Wait() if not self._buffer.ended else self._plan_whole_reply()

        self.mode = STREAMING
        self._sent += 1
        if self._when.chunk:
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

        # The largest prefix the lead covers and the cap allows, one piece
        # when neither does, because waiting only shrinks the lead further.
        # Both bounds are load-bearing: the lead says what playback can
        # survive, the cap says where this model's fitted line stops
        # describing it, and a lead large enough will otherwise join every
        # remaining sentence into one request. Measured on MOSS, which
        # declares fifteen seconds: a 269-character batch came back as 163
        # characters' worth of audio with the middle of it missing.
        # `cut_to_cap` has already bounded a single piece, so the one taken
        # when nothing fits is within the cap too.
        pieces = self._when.cut_to_cap(self._model, sentences)
        room = lead_s - self._when.margin
        count, text = 0, ""
        for piece in pieces:
            candidate = text + piece
            audio = self._model.audio_seconds(candidate)
            deficit = self._model.deficit(audio, chunk_streaming=self._when.chunk)
            if count and (deficit > room or audio > self._when.batch_cap):
                break
            text, count = candidate, count + 1
        self._sent += 1
        self._buffer.take_prefix(text)
        return Send(text)

    # -- planned ----------------------------------------------------------------

    def _plan_whole_reply(self) -> Decision:
        """The reply is known: group it for what was asked, and send the first.

        Two ways to speak a written-out reply, and which one is the caller's
        to say. **planned** groups for the fewest boundaries and holds the
        opening for exactly as long as never running dry takes — however long
        that is, because a hold that is cut short is a promise that is not
        kept. **unheld** groups for the soonest first word and holds nothing:
        it runs dry, audibly, and is for a model too slow to speak the reply
        any other way.

        Nothing here chooses between them. A reply that is slow enough to want
        `unheld` is slow enough that a person can hear it and say so, and one
        that switched strategy on its own — per reply, by a length nobody can
        see — would be a reply nobody could measure or explain.
        """
        text = self._buffer.take_all()
        if not text.strip():
            self.mode = PLANNED
            return Finished()
        self.mode = UNHELD if self._forced == UNHELD else PLANNED
        self._queued = self._when.best_batches(
            self._model, _sentences_of(text), held=self.mode == PLANNED
        )
        self._sent += 1
        self._rendering = self._queued.pop(0)
        # The bank is only re-asked when a request lands, and a whole-render
        # engine lands one batch at a time, so on its own it can only release
        # on a boundary — measured on OmniVoice, 8.73 s against the 5.62 s the
        # same plan's own arithmetic allows. The deadline is that arithmetic;
        # the bank stays as the early release for a render that runs ahead.
        #
        # A chunk-streaming engine gets one too. Its bank is re-asked on every
        # chunk, so for a model that keeps up the two agree to within the
        # sampling: measured across real-time factors 0.4 to 1.2, the bank
        # released between 0.43 s earlier and 0.06 s later than the deadline.
        # Past that the bank stops being enough on its own — at 1.6 the first
        # request banks 13.0 s against the 17.0 s the rest is predicted to
        # lose, so nothing releases until a later request lands and the
        # deadline is the only thing that speaks on the plan's own terms.
        # The timer is armed on the first audio byte, which a whole-render
        # engine delays when the host is slow and a chunk-streaming one emits
        # at its fixed cost regardless; `Schedule.of` already measures from
        # each engine's own first byte. No deadline for an unheld reply: not
        # holding is the whole of what it is.
        deadline = (
            round(self._when.hold_for(self._model, [self._rendering, *self._queued]), 3)
            if self.mode == PLANNED and self._model is not None
            else 0.0
        )
        return Send(
            self._rendering, hold_bank=self.mode == PLANNED, hold_wall_s=deadline
        )

    def _planned_next(self) -> Decision:
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
        needed is the deepest point of the same accounting `Schedule.of`
        does, taken over what is still to come with the current request
        counted whole. The margin and the fit's spread are on top, once.
        """
        if self._model is None:
            # The first request of an unmeasured reply has not landed yet, so
            # there is no line to promise anything against. Hold it all.
            return float("inf")
        # Dearer by whatever this reply has cost so far, so every deficit below
        # is taken from the line it is actually running on.
        model = self._model.scaled(self._slip(produced_s, elapsed_s))
        remaining = ([self._rendering] if self._rendering else []) + list(self._queued)
        if not remaining:
            return 0.0
        if self._when.chunk:
            needed = 0.0
            at_risk = 0.0
            for index, batch in enumerate(remaining):
                audio = model.audio_seconds(batch)
                at_risk = max(at_risk, model.render_seconds(audio))
                if index == 0:
                    needed += max(0.0, model.per_audio - 1.0) * max(
                        0.0, audio - produced_s
                    )
                else:
                    needed += model.deficit(audio, chunk_streaming=True)
        else:
            needed = self._when.hold_for(model, remaining, from_dispatch=True)
            at_risk = self._when.of(model, remaining)[2]
        # Scaled by what this reply has cost so far, not only by the fit: the
        # margin stays outside it, and the spread is charged against the
        # render whose over-running is what it insures against.
        return needed + self._when.margin + model.spread_for(at_risk)


def _cap_from(model: RenderModel | None, declared: float) -> float:
    """The most speech one request may carry, measured where it can be.

    `ModelSpec.batch_cap_s` is the earliest break seen anywhere, and stands
    until this host has served enough of this model and voice to find its own.
    Where the line held to the top of what was tried, one step past that is
    offered instead — see `PROBE`.
    """
    if model is None or model.holds_to_s <= 0.0:
        return declared
    if model.holds_to_s >= model.sampled_to_s:
        return model.holds_to_s * PROBE
    return model.holds_to_s


def _sentences_of(text: str) -> list[str]:
    """Text that is finished, as the sentences it is made of."""
    buffer = SentenceBuffer()
    buffer.feed(text)
    buffer.end()
    return buffer.sentences


def _prepend(buffer: SentenceBuffer, text: str) -> SentenceBuffer:
    fresh = SentenceBuffer()
    fresh.feed(text)
    fresh.feed(buffer.take_all())
    if buffer.ended:
        fresh.end()
    return fresh
