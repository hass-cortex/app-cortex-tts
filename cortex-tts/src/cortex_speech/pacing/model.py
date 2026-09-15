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
from dataclasses import dataclass

import numpy as np

# Fewer samples than this and the fit is the noise; the planner treats the
# model as unmeasured and holds the whole reply instead.
MIN_SAMPLES = 3

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
        per_audio: Render seconds per audio second: `b`, the real-time factor.
        spread_s: One standard deviation of what the line failed to explain.
            A reply's hold is widened by it once — a host has bad moments,
            not a bad moment per request.
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

    def audio_seconds(self, text: str) -> float:
        """How long this text will take to say."""
        cjk, latin = count_scripts(text)
        return cjk / self.cjk_per_s + latin / self.latin_per_s

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
    def fit(samples: Sequence[RenderSample]) -> RenderModel | None:
        """Fit both lines to the requests given; `None` until there are enough.

        How far back "recent" reaches is the caller's: whoever keeps the
        samples decides how long a host that changed goes on being believed.

        Least squares with the physical constraints a fit can violate on a
        handful of noisy points: no negative fixed cost, no non-positive
        factor. When every sample is about the same length there is no slope
        to fit, so the fixed part is taken as zero and the factor as the
        median ratio.
        """
        recent = [s for s in samples if s.audio_s > 0 and s.wall_s > 0]
        if len(recent) < MIN_SAMPLES:
            return None

        audio = np.array([s.audio_s for s in recent])
        wall = np.array([s.wall_s for s in recent])
        if np.ptp(audio) < 0.5:
            fixed, per_audio = 0.0, float(np.median(wall / audio))
        else:
            design = np.column_stack([np.ones_like(audio), audio])
            (fixed, per_audio), *_ = np.linalg.lstsq(design, wall, rcond=None)
            fixed = max(float(fixed), 0.0)
            per_audio = max(float(per_audio), 0.05)
            if fixed == 0.0:
                # The intercept was negative: refit the slope through the origin.
                per_audio = max(
                    float(np.sum(audio * wall) / np.sum(audio * audio)), 0.05
                )

        # A model near the streaming threshold is decided by its bad days,
        # not its average: a GPU that has just woken, a host briefly busy.
        # The spread is kept beside the line rather than folded into it —
        # folded into the fixed part it would be charged once per request,
        # and an eleven-request reply was held ten seconds it never needed.
        residual = wall - (fixed + per_audio * audio)
        spread = float(np.std(residual)) if len(recent) > 2 else 0.0

        cjk = np.array([s.cjk for s in recent], dtype=float)
        latin = np.array([s.latin for s in recent], dtype=float)
        cjk_rate, latin_rate = _speech_rates(cjk, latin, audio)
        return RenderModel(
            fixed_s=round(fixed, 3),
            spread_s=round(spread, 3),
            per_audio=round(per_audio, 3),
            cjk_per_s=round(cjk_rate, 2),
            latin_per_s=round(latin_rate, 2),
            samples=len(recent),
        )


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
