"""When a listener hears what, given a plan of batches and a cost line.

This is the arithmetic under every pacing decision, and it makes none of them.
It is handed a `RenderModel` and a list of batches and answers questions about
timing; what to do with the answer — stream or plan, honour a hold or cut it
short, which grouping to prefer — belongs to whoever asked.

Keeping it apart is what lets two strategies share one set of sums. The model
is a parameter rather than state because a reply replaces its own: the first
request of an unmeasured reply becomes its cost line, and a reply running dear
is charged against a line scaled to what it has actually cost.
"""

from __future__ import annotations

from .model import RenderModel, spoken_seconds
from .sentences import clause_pieces, ends_sentence


class Schedule:
    """The timing of a plan. No decisions, no state beyond its tuning."""

    __slots__ = ("_chunk", "_pause", "_margin", "_floor", "_ceiling", "_batch_cap")

    def __init__(
        self,
        *,
        chunk_streaming: bool,
        pause_s: float,
        margin_s: float,
        floor_s: float,
        ceiling_s: float,
        batch_cap_s: float,
    ) -> None:
        """Build one for a reply.

        Args:
            chunk_streaming: Whether the engine emits audio while a request is
                still rendering, which changes when a batch's audio exists.
            pause_s: Silence a sentence end may absorb before it stops being a
                pause; see `SENTENCE_PAUSE_S`.
            margin_s: The lead a plan must keep over running dry.
            floor_s: The least speech worth a request of its own.
            ceiling_s: The most speech to wait for before cutting at a clause.
            batch_cap_s: The most speech one request may carry on this model,
                already resolved against what this host has measured.
        """
        self._chunk = chunk_streaming
        self._pause = pause_s
        self._margin = margin_s
        self._floor = floor_s
        self._ceiling = ceiling_s
        self._batch_cap = batch_cap_s

    # The tuning is this object's, and the planner reads it here rather than
    # keeping a second copy: two objects holding one figure is how a seventh
    # one gets added to only one of them.
    @property
    def chunk(self) -> bool:
        """Whether the engine emits audio while a request is still rendering."""
        return self._chunk

    @property
    def floor(self) -> float:
        """The least speech a first request may carry."""
        return self._floor

    @property
    def ceiling(self) -> float:
        """The most speech a first request may carry."""
        return self._ceiling

    @property
    def margin(self) -> float:
        """What the lead keeps in hand beyond the prediction."""
        return self._margin

    @property
    def batch_cap(self) -> float:
        """The most speech one request may carry, after `_cap_from`."""
        return self._batch_cap

    def pieces_within(
        self, model: RenderModel, pieces: list[str], limit_s: float
    ) -> int:
        """How many leading pieces fit in `limit_s` of speech; at least one."""
        count, text = 0, ""
        for piece in pieces:
            if count and model.audio_seconds(text + piece) > limit_s:
                break
            text += piece
            count += 1
        return count

    def clause_cut(self, model: RenderModel, text: str) -> str | None:
        """The longest opening cut, within the ceiling, whose rest lands in time.

        The rest is what has arrived so far; more may follow, so this is the
        optimistic side of the estimate, which is why the margin exists.
        """
        pieces = clause_pieces(text)
        best: str | None = None
        prefix = ""
        for piece in pieces[:-1]:
            prefix += piece
            spoken = model.audio_seconds(prefix)
            if spoken < self._floor:
                continue
            if spoken > self._ceiling:
                break
            rest = model.audio_seconds(text[len(prefix) :])
            deficit = model.deficit(rest, chunk_streaming=self._chunk)
            if deficit <= spoken - self._margin:
                best = prefix
        return best

    def cut_to_cap(self, model: RenderModel | None, sentences: list[str]) -> list[str]:
        """The same text in pieces no longer than one request may carry.

        `group` can join sentences but never cut below one, so a reply
        written as a single long enumeration — no stop until the end, only
        clause marks — reached the renderer whole however it was planned.
        Where a sentence is over the cap its own clauses become the pieces,
        which `group` then joins back up to the cap like any others.
        """
        out: list[str] = []
        for sentence in sentences:
            if self.audio_seconds(model, sentence) <= self._batch_cap:
                out.append(sentence)
                continue
            pieces = clause_pieces(sentence)
            out.extend(pieces if len(pieces) > 1 else [sentence])
        return out

    def audio_seconds(self, model: RenderModel | None, text: str) -> float:
        """How long this text takes to say, fitted where there is a fit."""
        if model is None:
            return spoken_seconds(text)
        return model.audio_seconds(text)

    def group(
        self, model: RenderModel | None, sentences: list[str], limit_s: float
    ) -> list[str]:
        batches: list[str] = []
        current = ""
        for sentence in sentences:
            if current and self.audio_seconds(model, current + sentence) > limit_s:
                batches.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            batches.append(current)
        return batches

    def of(self, model: RenderModel, batches: list[str]) -> tuple[float, float, float]:
        """The deepest point of a plan, when its first audio exists, and the
        render that would cause it.

        Requests are dispatched back to back. A whole-render engine makes a
        batch's audio available when its render ends; a chunk-streaming one
        makes it available from its fixed cost onwards, at the rate it
        renders. Playback starting at `H` needs the cumulative audio at every
        one of those points; the deepest is the smallest `H` that satisfies
        all of them.

        This is the one piece of arithmetic under every decision here. What
        each engine does differently is in the branch below and nowhere else —
        it is physics, not policy.

        The third figure is the render of the batch that set the deepest
        point: the one whose over-running is what the hold is insuring
        against, and so the length the fit's spread is charged in proportion
        to.
        """
        dispatched = 0.0
        played_before = 0.0
        needed = 0.0
        # Silence the boundaries already crossed can carry, so the bank only
        # has to cover what is left over.
        absorbed = 0.0
        at_risk = 0.0
        first_audio_at: float | None = None
        for batch in batches:
            audio = model.audio_seconds(batch)
            render = model.render_seconds(audio)
            if self._chunk:
                starts = dispatched + model.fixed_s
                ends = dispatched + render
                deepest = max(starts - played_before, ends - played_before - audio)
                first_audio_at = starts if first_audio_at is None else first_audio_at
            else:
                ends = dispatched + render
                deepest = ends - played_before - absorbed
                first_audio_at = ends if first_audio_at is None else first_audio_at
                if ends_sentence(batch):
                    absorbed += self._pause
            if deepest >= needed:
                needed, at_risk = deepest, render
            dispatched += render
            played_before += audio
        return needed, first_audio_at or 0.0, at_risk

    def hold_for(
        self, model: RenderModel, batches: list[str], *, from_dispatch: bool = False
    ) -> float:
        """Seconds to hold after the first audio so playback never catches up.

        With `from_dispatch`, the bare figure from the moment the first of
        these batches was dispatched — no margin, no spread — for a caller
        that adds its own.

        The plan is costed on the line made dearer by its own `drift`, because
        this is the one figure here that sums the line's predictions over a
        whole reply. Scatter cancels over a sum; a line that is systematically
        low does not, and the opening is short by that fraction of the entire
        render. The bank keeps the undrifted line: it is re-asked as the reply
        runs and has the reply's own pace to go on, which is better evidence
        than a window measured before it started.
        """
        if from_dispatch:
            return max(0.0, self.of(model, batches)[0])
        insured = model.scaled(1.0 + model.drift)
        needed, first_audio_at, at_risk = self.of(insured, batches)
        return (
            max(0.0, needed - first_audio_at)
            + self._margin
            + insured.spread_for(at_risk)
        )

    def first_audio_at(self, model: RenderModel, batches: list[str]) -> float:
        """When the first batch exists, whether or not anything is held back.

        `speaks_at` is the same question for a reply that waits out its hold;
        this one is for a reply that does not, where the first batch landing
        is the moment a listener hears something.
        """
        return self.of(model, batches)[1]

    def speaks_at(self, model: RenderModel, batches: list[str]) -> float:
        """When the listener hears the first word, if the plan is these batches.

        The arithmetic only. The margin is the same for every plan, and the
        spread is charged against the render a plan puts at risk — so a finer
        cut carries a smaller one and would look sooner for it. That is an
        allowance shrinking, not a listener hearing anything earlier, and
        comparing plans by it made a chunk-streaming engine prefer thirteen
        requests over one where every plan is audible at the same moment.
        """
        needed, first_audio_at, _ = self.of(model, batches)
        return max(needed, first_audio_at)

    def best_batches(
        self, model: RenderModel | None, sentences: list[str], *, held: bool = True
    ) -> list[str]:
        """The cut points that get the listener speaking soonest.

        Every rule this replaces was a hand-worked case of one question — how
        soon can the first word go out with the lead never falling below the
        margin — and the answer differed per engine, per model and per host,
        so each case grew its own constant. `of` answers the question
        for any plan, so the plan is chosen by asking rather than by a
        constant, and what an engine does differently falls out of the
        arithmetic instead of a branch: a chunk-streaming one finds nothing to
        gain by splitting, a model slower than real time finds no split that
        keeps the lead, and both land on the plan they were special-cased to.

        The family searched is one limit applied to every request, and the
        limits worth trying are the ones that move a boundary. The model's own
        `batch_cap_s` is the only cap left: past it an autoregressive model's
        cost turns quadratic, which is a property of that architecture and not
        of this host.
        """
        sentences = self.cut_to_cap(model, sentences)
        if model is None:
            return self.group(model, sentences, self._batch_cap)
        running = 0.0
        limits: list[float] = []
        for sentence in sentences:
            running += model.audio_seconds(sentence)
            if running >= self._batch_cap:
                break
            limits.append(running)
        limits.append(self._batch_cap)

        # Largest first, and only a strictly sooner first word displaces it:
        # where several plans speak at the same moment — which is every plan on
        # a chunk-streaming engine, audible from its fixed cost whatever the
        # cut — the fewest boundaries wins. Measured on MOSS, the same story in
        # one-sentence requests fell 41% behind playback against 4.1% in nine-
        # second ones; all of that is boundaries.
        # Which plan is best depends on what is going to be done with it. A
        # held reply is audible when its hold ends, so a plan is judged on the
        # later of the hold and the first render; one that is not held is
        # audible when the first batch lands, and a smaller first batch wins
        # even though it costs more boundaries later.
        audible = self.speaks_at if held else self.first_audio_at
        # Seeded with the largest limit rather than with nothing: `limits`
        # always ends in the cap, so there is always a plan, and starting from
        # one says that in the code instead of asserting it afterwards.
        plan = self.group(model, sentences, limits[-1])
        best = audible(model, plan)
        for limit in reversed(limits[:-1]):
            batches = self.group(model, sentences, limit)
            at = audible(model, batches)
            if at < best - 1e-6:
                best, plan = at, batches
        return plan
