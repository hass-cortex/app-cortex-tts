"""Traditional-to-Simplified glyph conversion.

The models are trained on Simplified text: Traditional glyphs are tokens they
have barely seen, and a Traditional sentence comes out as the wrong words
(measured: 32% character error rate, against 4% converted).
"""

from __future__ import annotations

import re
from functools import lru_cache

# OpenCC leaves 著 alone: in Simplified the glyph survives only for the zhù
# sense (著作, 顯著), and the conversion to 着 happens just for the phrases its
# table lists — 看著, 住著, 拿著 are not among them. A model trained on
# Simplified text reads any 著 it meets as zhù, so 住著 comes out as 住住.
# Every reading but zhù is written 着, so every 著 outside a zhù word becomes
# 着. The Taiwan-readings table is generated through this same function, so
# its keys carry 着 too and nothing there depends on the glyph OpenCC kept.
_ZHU_BEFORE = "顯显巨名專专論论編编原土"
_ZHU_AFTER = "作名述稱称錄录者書书"
_ZHE = re.compile(rf"(?<![{_ZHU_BEFORE}])著(?![{_ZHU_AFTER}])")


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
    return _zhe(_converter().convert(text))


def _zhe(text: str) -> str:
    """Write every 著 that is not zhù as 着; see the note at the top."""
    return _ZHE.sub("着", text) if "著" in text else text


def is_traditional(text: str) -> bool:
    """Return whether the text carries any glyph the conversion would change."""
    return to_simplified(text) != text
