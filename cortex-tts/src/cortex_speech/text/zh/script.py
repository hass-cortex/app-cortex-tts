"""Traditional-to-Simplified glyph conversion.

The models are trained on Simplified text: Traditional glyphs are tokens they
have barely seen, and a Traditional sentence comes out as the wrong words
(measured: 32% character error rate, against 4% converted).
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _converter():
    """Return a cached OpenCC Traditional->Simplified converter.

    ``t2s`` is glyph-only. ``tw2sp`` would also rewrite vocabulary (設定 to
    设置), changing the words the model says.
    """
    from opencc import OpenCC

    return OpenCC("t2s")


def to_simplified(text: str) -> str:
    """Convert Traditional glyphs to Simplified, leaving Latin text alone."""
    return _converter().convert(text)


def is_traditional(text: str) -> bool:
    """Return whether the text carries any glyph the conversion would change."""
    return to_simplified(text) != text
