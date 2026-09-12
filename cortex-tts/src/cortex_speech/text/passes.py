"""The substitution machinery both normalisers are built from.

Order is not negotiable, and it is the same order in either script: a
construct that owns its number has to claim it before the bare-number pass
reads that digit as a quantity. Percent takes "68%" first; the clock takes
"14:35" before the colon is read as a pause.

What a construct *sounds like* is the script's business, so the patterns and
the running order live here and the readings live in the per-script modules.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .options import NormalizeOptions

# A sign only counts when nothing numeric precedes it; otherwise the dash in
# "25-30" would be read as minus thirty.
NUMBER = r"(?:(?<!\d)[+-])?\d+(?:\.\d+)?"

# `\b` treats every CJK character as a word character, so it is not a
# boundary between 現在是 and 14. These are: a digit or Latin letter on either
# side is what makes a run part of something else.
LEAD = r"(?<![0-9A-Za-z])"
TAIL = r"(?![0-9A-Za-z])"

DASH = r"\s*[-~～–—]\s*"

THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
DATE = re.compile(rf"{LEAD}(\d{{4}})-(\d{{1,2}})-(\d{{1,2}}){TAIL}")
CLOCK = re.compile(rf"{LEAD}(\d{{1,2}}):(\d{{2}})(?::(\d{{2}}))?{TAIL}")
VERSION = re.compile(rf"{LEAD}\d+(?:\.\d+){{2,}}{TAIL}")
RANGE = re.compile(rf"(?<=\d){DASH}(?=\d)")


def quantity(unit: str) -> re.Pattern[str]:
    """A number, or a range of two, followed by a unit.

    Groups: the first number, the second number of a range or None, and the
    unit. A range is claimed together with its unit so that "25-30°C" reads
    as one temperature span, not as a number and a negative temperature.
    """
    return re.compile(rf"({NUMBER})(?:{DASH}({NUMBER}))?\s*({unit})")


PERCENT = quantity(r"[%％]")
TEMPERATURE = quantity(r"°C|℃|°F|℉")
DEGREE = quantity(r"°")

Render = Callable[[re.Match[str], NormalizeOptions], str]


@dataclass(frozen=True)
class Pass:
    """One substitution in the normalisation order.

    Attributes:
        pattern: What this pass claims before later passes can see it.
        render: How the matched construct is spoken.
        gate: NormalizeOptions attribute that switches the pass off, if any.
    """

    pattern: re.Pattern[str]
    render: Render
    gate: str | None = None


def drop(match: re.Match[str], options: NormalizeOptions) -> str:
    """Remove the match; what the thousands separator pass does."""
    return ""


def _bind(render: Render, options: NormalizeOptions) -> Callable[[re.Match[str]], str]:
    """Adapt a render to the single-argument callable ``re.sub`` expects."""

    def substitute(match: re.Match[str]) -> str:
        return render(match, options)

    return substitute


def run(text: str, steps: Sequence[Pass], options: NormalizeOptions) -> str:
    """Apply every enabled pass, in the order given."""
    out = text
    for step in steps:
        if step.gate is not None and not getattr(options, step.gate):
            continue
        out = step.pattern.sub(_bind(step.render, options), out)
    return out
