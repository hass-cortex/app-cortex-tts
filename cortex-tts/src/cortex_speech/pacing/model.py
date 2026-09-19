"""What one model costs on this host, learned from the requests it served.

Two straight lines. Rendering a request takes a fixed part plus a part
proportional to the audio it produces: the fixed part is the prompt and the
reference being encoded, the proportional part is the real-time factor.
Speech takes a per-character time that differs by script: one 漢字 is a
syllable, one Latin letter is a fraction of one. Both are fitted from what
this host actually did rather than carried from another machine, because
neither survives a change of CPU, execution provider or reference recording.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

# Fewer samples than this and the fit is the noise; the planner treats the
# model as unmeasured and holds the whole reply instead.
MIN_SAMPLES = 3

# How far over the short requests' real-time factor a long one may run before
# the line is no longer describing it. Applied to this host's own samples, one
# tolerance reproduces every break measured by hand: OmniVoice on a GTX 1650 is
# 1% over at 9.2 s and 19% over by 10.7 s, Hojo 40M 8% over at 9.4 s and 19% by
# 14.2 s, MOSS-TTS-Nano 4% over at 15.0 s and 12% by 21.8 s — so ten per cent
# lands the break at nine, nine and fifteen, which is where each was set.
LINE_TOLERANCE = 0.10

# Fewer samples than this and there is no long half to compare against a short
# one; the break is left unsaid rather than guessed from three points.
MIN_BREAK_SAMPLES = 6

# Fewer samples than this and there is no older half to fit and no newer half
# to test it on, which is what `drift` is.
MIN_DRIFT_SAMPLES = 8

# The most a line is believed to be wrong about what comes next. A hold is
# widened in proportion to `drift`, so one bad window would otherwise buy a
# reply minutes of silence. Measured across the eight lines one host had
# fitted, the newer half of a window ran between 10.0% cheaper and 7.9% dearer
# than the older half predicted, so a quarter is far outside anything seen and
# a line past it is not a line worth insuring.
MAX_DRIFT = 0.25

# Speech rates a script starts from until this host has spoken it: measured on
# the Hojo 40M in production, 108 characters of Chinese as 26.6 s and 309 of
# Latin as 21.1 s. Deliberately on the slow side — everything sized from an
# estimate wants to over-estimate the audio rather than under.
PRIOR_CJK_PER_S = 4.1
PRIOR_LATIN_PER_S = 14.7

_CJK = re.compile(r"[　-鿿豈-﫿＀-￯]")


def count_scripts(text: str) -> tuple[int, int]:
    """Return (CJK characters, other non-space characters) in the text."""
    cjk = len(_CJK.findall(text))
    other = sum(1 for ch in text if not ch.isspace()) - cjk
    return cjk, max(other, 0)


@dataclass(frozen=True)
class RenderSample:
    """One request as it actually went.

    Attributes:
        audio_s: Seconds of audio it produced.
        wall_s: Seconds from asking to the last byte, prefill included.
        cjk: CJK characters in the text as the caller wrote it.
        latin: Other non-space characters.
    """

    audio_s: float
    wall_s: float
    cjk: int
    latin: int


@dataclass(frozen=True)
class RenderModel:
    """The fitted lines.

    Attributes:
        fixed_s: Render seconds a request costs before any audio: `a`.
        per_audio: Render seconds per audio second: `b`. This is the slope,
            and it is the real-time factor only where `fixed_s` is zero — at
            any other length the factor is `render_seconds(a) / a`, which is
            `fixed_s / a + per_audio` and so falls as the request grows.
        spread_s: One standard deviation of what the line failed to explain,
            over requests of `spread_ref_s` render on average. A reply's hold
            is widened by it once — a host has bad moments, not a bad moment
            per request — and by `spread_for`, not by this figure directly.
        spread_ref_s: The mean render the spread was measured over. Zero
            leaves `spread_for` returning `spread_s` whatever it is asked.
        audio_ref_s: The mean audio of the requests fitted. A real-time factor
            has to be quoted at a length, and this is the one this host's own
            traffic actually asks for.
        drift: How much dearer than this line the requests after it run, as a
            fraction, and never less than zero — a line that over-predicts
            needs no insurance. `spread_s` is the scatter around the line and
            `drift` is the movement of the line itself; a hold has to survive
            the second, because it sums the line's predictions over a whole
            reply and a systematic offset accumulates where scatter cancels.
            Zero until there are samples enough to say.
        holds_to_s: The longest request, in seconds of audio, whose real-time
            factor is still within `LINE_TOLERANCE` of the shortest requests'
            — where this line stops describing this model *on this host*.
            Zero until there are samples enough to say.
        sampled_to_s: The longest request the fit has seen. Equal to
            `holds_to_s` when the line held all the way to the top of what was
            tried, which is the only evidence that a longer one is worth
            trying.
        cjk_per_s: CJK characters spoken per second.
        latin_per_s: Other characters spoken per second.
        samples: How many requests the fit rests on.
    """

    fixed_s: float
    per_audio: float
    cjk_per_s: float
    latin_per_s: float
    samples: int
    spread_s: float = 0.0
    spread_ref_s: float = 0.0
    audio_ref_s: float = 0.0
    drift: float = 0.0
    holds_to_s: float = 0.0
    sampled_to_s: float = 0.0

    def spread_for(self, render_s: float) -> float:
        """The allowance for a render this long going wrong.

        `spread_s` is one figure for a whole sample of requests, and charging
        it flat charges a two-second render what a ten-second one is worth.
        Measured on this host's own samples, that is the wrong shape: across
        four cost lines the residual grew with the render (MOSS r=+0.67,
        OmniVoice's clone r=+0.38), and on the line where it mattered most the
        *relative* error was flat instead — 21.9% of the render on the short
        half against 20.7% on the long. So the allowance is charged in
        proportion to the render it is protecting, which leaves a request of
        the sample's own average paying exactly what it paid before.

        Capped at twice the measured figure: past the lengths the samples
        cover, proportion is an extrapolation and a hold has to end.
        """
        if self.spread_ref_s <= 0.0:
            return self.spread_s
        return min(self.spread_s * render_s / self.spread_ref_s, self.spread_s * 2)

    def audio_seconds(self, text: str) -> float:
        """How long this text will take to say."""
        cjk, latin = count_scripts(text)
        return cjk / self.cjk_per_s + latin / self.latin_per_s

    def with_rates(self, rates: tuple[float, float] | None) -> RenderModel:
        """The same cost line, speaking at this voice's own pace."""
        if rates is None:
            return self
        return replace(self, cjk_per_s=rates[0], latin_per_s=rates[1])

    def real_time_factor(self, audio_s: float = 0.0) -> float:
        """Render seconds over audio seconds, which is what RTF means here.

        Not `per_audio`: that is the slope, and the two agree only where the
        line has no fixed cost. With one, a request pays `fixed_s / a` on top,
        so the factor falls as the request grows and quoting a single number
        without its length says nothing. Asked for no length, the answer is
        given at `audio_ref_s` — the mean request this host actually served.
        """
        at = audio_s or self.audio_ref_s
        if at <= 0:
            return self.per_audio
        return self.render_seconds(at) / at

    def scaled(self, factor: float) -> RenderModel:
        """The same line with every cost dearer by `factor`.

        Both parts move, and the deficit is taken from the moved line rather
        than scaled after the fact: a chunk-streaming engine loses
        `per_audio - 1`, and scaling that difference under-corrects. At a
        fitted 1.26 running 1.10x dear, the loss is 1.26x1.10 - 1 = 0.386 per
        audio second, not (1.26 - 1) x 1.10 = 0.286.

        The speech rates are untouched: a busy host renders slower, it does
        not make the voice speak fewer characters a second.
        """
        if factor <= 1.0:
            return self
        return replace(
            self,
            fixed_s=self.fixed_s * factor,
            per_audio=self.per_audio * factor,
            spread_s=self.spread_s * factor,
        )
        # `spread_ref_s` is a length, not a cost: what the samples were, not
        # what they took. A dearer host does not make them shorter.

    def render_seconds(self, audio_s: float) -> float:
        """How long a request producing this much audio takes to render."""
        return self.fixed_s + self.per_audio * audio_s

    def deficit(self, audio_s: float, *, chunk_streaming: bool) -> float:
        """Seconds of playback a request of this size cannot cover by itself.

        An engine that hands a request over whole has produced nothing until
        it is done, so everything it costs is a deficit. One that emits audio
        while rendering pays only its fixed part up front and then loses
        (or gains) the difference between rendering and playing.
        """
        if chunk_streaming:
            return self.fixed_s + max(0.0, self.per_audio - 1.0) * audio_s
        return self.render_seconds(audio_s)

    @staticmethod
    def dearest(models: Sequence[RenderModel]) -> RenderModel | None:
        """The safe-side envelope of several lines; `None` if given none.

        For a voice this host has not measured, standing in the lines it has
        measured for the same model. Measured on one host across every line it
        had fitted, the slope is a property of the model and the intercept is
        not: MOSS 0.401 built-in against 0.360 cloned, OmniVoice 0.717 designed
        against 0.718 and 0.608 for two clones — while the intercepts ran 0.308
        against 1.157 and 1.496.

        Every part is taken at its dearest, because what this is for is
        deciding whether audio is safe to release: too dear costs a wait the
        listener sits through, too cheap costs a gap they cannot un-hear. The
        reference recording's length would be the obvious way to predict an
        intercept and does not work — 1.496 came from a 3.5 s recording and
        1.157 from a 4.52 s one, the wrong way round.

        The rates are the priors, not any of these voices': a pace belongs to
        one voice and is the one thing a single render settles, so a caller
        with even one sample of the real voice overwrites them.

        Dearest across every line is still an estimate, and one case can come
        out under: the model's first clone, where only a designed voice has
        been measured and the intercept it lends is the small one. The
        planner's slip correction is what covers that — it compares this
        reply's own cost against the prediction before it releases anything.
        """
        lines = list(models)
        if not lines:
            return None
        return RenderModel(
            fixed_s=max(m.fixed_s for m in lines),
            per_audio=max(m.per_audio for m in lines),
            spread_s=max(m.spread_s for m in lines),
            # The shortest reference makes the borrowed spread dearest at any
            # given render, which is the safe side to be on.
            spread_ref_s=min(
                (m.spread_ref_s for m in lines if m.spread_ref_s > 0), default=0.0
            ),
            cjk_per_s=PRIOR_CJK_PER_S,
            latin_per_s=PRIOR_LATIN_PER_S,
            samples=0,
        )

    @staticmethod
    def fit(samples: Sequence[RenderSample]) -> RenderModel | None:
        """Fit both lines to the requests given; `None` until there are enough.

        How far back "recent" reaches is the caller's: whoever keeps the
        samples decides how long a host that changed goes on being believed.

        The cost line is `_fit_line`'s; what this adds is everything a reply
        needs beside it — the spread, the drift, the break, the speech rates.
        """
        recent = [s for s in samples if s.audio_s > 0 and s.wall_s > 0]
        if len(recent) < MIN_SAMPLES:
            return None

        audio = np.array([s.audio_s for s in recent])
        wall = np.array([s.wall_s for s in recent])
        fixed, per_audio = _fit_line(audio, wall)

        # A model near the streaming threshold is decided by its bad days,
        # not its average: a GPU that has just woken, a host briefly busy.
        # The spread is kept beside the line rather than folded into it —
        # folded into the fixed part it would be charged once per request,
        # and an eleven-request reply was held ten seconds it never needed.
        #
        # It is only as wide as the samples were varied, and a benchmark
        # under-reports it by an order of magnitude: 24 back-to-back requests
        # of one length on an idle GPU gave 0.017 s where the same model's
        # lines from real traffic carry 0.156 and 0.305. Every hold is widened
        # by this, so a figure tuned on a clean run holds too briefly in
        # production.
        predicted = fixed + per_audio * audio
        residual = wall - predicted
        spread = float(np.std(residual)) if len(recent) > 2 else 0.0
        # What the spread is one standard deviation *of*: charged against a
        # render this long, it is the figure measured here; against a shorter
        # or longer one, `spread_for` scales it.
        spread_ref = float(np.mean(predicted))

        holds_to, sampled_to = _break(audio, wall)

        cjk = np.array([s.cjk for s in recent], dtype=float)
        latin = np.array([s.latin for s in recent], dtype=float)
        cjk_rate, latin_rate = _speech_rates(cjk, latin, audio)
        return RenderModel(
            fixed_s=round(fixed, 3),
            spread_s=round(spread, 3),
            spread_ref_s=round(spread_ref, 3),
            audio_ref_s=round(float(np.mean(audio)), 3),
            per_audio=round(per_audio, 3),
            cjk_per_s=round(cjk_rate, 2),
            latin_per_s=round(latin_rate, 2),
            drift=_drift(recent),
            holds_to_s=round(holds_to, 2),
            sampled_to_s=round(sampled_to, 2),
            samples=len(recent),
        )


def _drift(samples: Sequence[RenderSample]) -> float:
    """How far this line misses the requests that come after it.

    The question a hold asks of a line is not "how noisy is it" but "will the
    next few requests cost what you say", and those are different: measured
    across eight lines, the within-window scatter was 2.0-9.1% of a render
    while the older half missed the newer one's total by up to 7.9% on lines
    whose scatter was 2.6%. So it is measured the way it is used — fit the
    older half, add up what it predicts for the newer half, and compare with
    what that half actually cost.

    Totals rather than each request, because a hold sums predictions: a run of
    requests that are each 4% dear is a reply whose opening is 4% short, while
    the same scatter with no offset costs it nothing.
    """
    if len(samples) < MIN_DRIFT_SAMPLES:
        return 0.0
    half = len(samples) // 2
    audio = np.array([s.audio_s for s in samples[:half]])
    wall = np.array([s.wall_s for s in samples[:half]])
    # Rounded as `fit` rounds, because the miss being measured is the miss of
    # the line as it will actually be used.
    fixed, per_audio = (round(value, 3) for value in _fit_line(audio, wall))
    predicted = sum(fixed + per_audio * s.audio_s for s in samples[half:])
    if predicted <= 0:
        return 0.0
    actual = sum(s.wall_s for s in samples[half:])
    return round(min(max((actual - predicted) / predicted, 0.0), MAX_DRIFT), 4)


def _fit_line(audio: np.ndarray, wall: np.ndarray) -> tuple[float, float]:
    """The cost line `fixed + per_audio × audio` through these requests.

    Least squares with the physical constraints a fit can violate on a handful
    of noisy points: no negative fixed cost, no non-positive factor. When every
    sample is about the same length there is no slope to fit, so the fixed part
    is taken as zero and the factor as the median ratio — the only honest
    answer to one length, and a lossy one: the per-request cost is folded into
    the factor and stops being visible at any other length. Feeding this a run
    of same-length requests is therefore how a two-parameter line is destroyed,
    which a benchmark does and a reply does not.
    """
    if np.ptp(audio) < 0.5:
        return 0.0, float(np.median(wall / audio))
    design = np.column_stack([np.ones_like(audio), audio])
    (fixed, per_audio), *_ = np.linalg.lstsq(design, wall, rcond=None)
    fixed = max(float(fixed), 0.0)
    per_audio = max(float(per_audio), 0.05)
    if fixed == 0.0:
        # The intercept was negative: refit the slope through the origin.
        per_audio = max(float(np.sum(audio * wall) / np.sum(audio * audio)), 0.05)
    return fixed, per_audio


def chars_within(seconds: float, text: str) -> int:
    """How many characters of this text's scripts fit in that much audio.

    The inverse of `spoken_seconds` for a mixed text: each script speaks at its
    own rate, so the answer depends on what the text is made of, not only on
    how long it is. The priors are used rather than a fitted line because the
    callers are cutting text, and a segment must not depend on which voice
    happens to say it — and they are the slow-side priors, so this
    under-fills rather than over-fills whatever it is bounding.
    """
    cjk, latin = count_scripts(text)
    total = cjk + latin
    if total <= 0 or seconds <= 0:
        return 0
    per_char = (cjk / PRIOR_CJK_PER_S + latin / PRIOR_LATIN_PER_S) / total
    return max(1, int(seconds / per_char))


def spoken_seconds(text: str) -> float:
    """How long text takes to say before this host has measured anything.

    The priors, applied the way `RenderModel.audio_seconds` applies a fit.
    Cutting a reply into requests needs a length and nothing else, and a host
    with no figures still has to do it — so this answers that one question and
    no other. Deliberately slow, which errs towards shorter requests.
    """
    cjk, latin = count_scripts(text)
    return cjk / PRIOR_CJK_PER_S + latin / PRIOR_LATIN_PER_S


def speech_rates(samples: Sequence[RenderSample]) -> tuple[float, float] | None:
    """How fast a voice speaks, from renders of it; `None` with nothing to fit.

    Separate from `RenderModel.fit` because it answers a different question
    about a different thing. What a request costs belongs to the model and the
    host — measured across MOSS's built-in voices, the cost of a second of
    audio varies 4%. How long the text takes to say belongs to the *voice*:
    the same 26 characters ran 6.64 s as Weiguo and 5.12 s as Yuewen, a 30%
    spread, and every batch the planner sizes is sized in seconds of speech.

    It is also far cheaper to learn: characters over seconds, from a single
    render, where a cost line needs several of different lengths.
    """
    usable = [s for s in samples if s.audio_s > 0]
    if not usable:
        return None
    cjk = np.array([s.cjk for s in usable], dtype=float)
    latin = np.array([s.latin for s in usable], dtype=float)
    audio = np.array([s.audio_s for s in usable])
    return _speech_rates(cjk, latin, audio)


def _break(audio: np.ndarray, wall: np.ndarray) -> tuple[float, float]:
    """Where this host's own samples say the cost line stops describing it.

    Returns the longest sampled length still on the line, and the longest
    sampled length. They are equal when the line held all the way to the top
    of what was tried, which is the only evidence there is that a longer
    request would also be on it.

    The baseline is the median real-time factor of the shortest third, not of
    every sample: a per-request fixed cost is dearest per audio second there,
    so comparing against it under-states the drift rather than flattering it.

    This is what makes the cap a measurement instead of a constant. The break
    belongs to the model *and* to the host — an autoregressive decode's
    quadratic term is attention over a growing cache, and where it surfaces
    depends on whether the machine is short of compute or of bandwidth. The
    three models measured on one GTX 1650 broke at 9, 9 and 15 seconds; the
    same three on a CPU would not break in the same places, and nothing here
    has to know that in advance.
    """
    order = np.argsort(audio)
    audio, wall = audio[order], wall[order]
    if len(audio) < MIN_BREAK_SAMPLES or float(np.ptp(audio)) < 1.0:
        return 0.0, 0.0
    rtf = wall / audio
    short = max(2, len(audio) // 3)
    baseline = float(np.median(rtf[:short]))
    ceiling = baseline * (1 + LINE_TOLERANCE)
    held = float(audio[0])
    for length, factor in zip(audio, rtf, strict=True):
        if factor <= ceiling:
            held = float(length)
    return held, float(audio[-1])


def _speech_rates(
    cjk: np.ndarray, latin: np.ndarray, audio: np.ndarray
) -> tuple[float, float]:
    """Characters per second for each script, from `audio = cjk/r1 + latin/r2`.

    A script this host has never spoken keeps its prior: there is nothing to
    fit it from, and inventing a rate from the other script would be a guess
    dressed as a measurement.
    """
    has_cjk, has_latin = cjk.sum() > 0, latin.sum() > 0
    if has_cjk and has_latin:
        design = np.column_stack([cjk, latin])
        (per_cjk, per_latin), *_ = np.linalg.lstsq(design, audio, rcond=None)
        if per_cjk > 0 and per_latin > 0:
            return 1.0 / float(per_cjk), 1.0 / float(per_latin)
    if has_cjk:
        # Latin is sparse (a unit, a device name): read it at the prior rate
        # and fit the rest.
        remaining = audio - latin / PRIOR_LATIN_PER_S
        per_cjk = float(np.sum(cjk * remaining) / np.sum(cjk * cjk))
        return (1.0 / per_cjk if per_cjk > 0 else PRIOR_CJK_PER_S), PRIOR_LATIN_PER_S
    if has_latin:
        remaining = audio - cjk / PRIOR_CJK_PER_S
        per_latin = float(np.sum(latin * remaining) / np.sum(latin * latin))
        return PRIOR_CJK_PER_S, (
            1.0 / per_latin if per_latin > 0 else PRIOR_LATIN_PER_S
        )
    return PRIOR_CJK_PER_S, PRIOR_LATIN_PER_S
