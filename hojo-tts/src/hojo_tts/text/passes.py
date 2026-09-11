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

NUMBER = r"[+-]?\d+(?:\.\d+)?"

DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
PERCENT = re.compile(rf"({NUMBER})\s*[%％]")
TEMPERATURE = re.compile(rf"({NUMBER})\s*(°C|℃|°F|℉)")
DEGREE = re.compile(rf"({NUMBER})\s*°")
VERSION = re.compile(r"\b\d+(?:\.\d+){2,}\b")
RANGE = re.compile(r"(?<=\d)\s*[-~～–—]\s*(?=\d)")

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
