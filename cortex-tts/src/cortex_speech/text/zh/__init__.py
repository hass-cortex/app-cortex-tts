"""Chinese: the locale the pipeline was built for, and the deepest one.

Beyond numbers it has two rewrites of its own — script conversion, because
the models read Simplified, and Taiwan readings, because they read the
mainland way — and neither exists for any other language.
"""

from __future__ import annotations

from ..locales import Locale, Rewrite
from .normalize import normalize
from .readings import apply_taiwan_readings, taiwan_readings
from .script import is_traditional, to_simplified


def is_taiwanese(tag: str) -> bool:
    """Return whether a language tag names Taiwan's Chinese.

    The region says so outright; the Traditional script says so unless a
    region says otherwise — Hong Kong writes Traditional and reads Cantonese.
    """
    subtags = tag.lower().split("-")[1:]
    if "tw" in subtags:
        return True
    region = next((s for s in subtags if len(s) == 2), None)
    return "hant" in subtags and region is None


LOCALE = Locale(
    code="zh",
    normalize=normalize,
    stop="。",
    close_gaps=True,
    rewrites=(
        # Conversion is a no-op on Simplified text, so it is simply always on.
        Rewrite("convert_script", to_simplified, lambda _: True),
        Rewrite(
            "taiwan_readings",
            apply_taiwan_readings,
            is_taiwanese,
            requires=("convert_script",),
        ),
    ),
)

__all__ = [
    "LOCALE",
    "apply_taiwan_readings",
    "is_taiwanese",
    "is_traditional",
    "normalize",
    "taiwan_readings",
    "to_simplified",
]
