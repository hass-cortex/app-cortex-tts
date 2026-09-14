"""English: numbers, units, dates and clock literals read out in words."""

from __future__ import annotations

from ..locales import Locale
from .normalize import cardinal, clock, decimal, normalize, ordinal, year

LOCALE = Locale(code="en", normalize=normalize)

__all__ = ["LOCALE", "cardinal", "clock", "decimal", "normalize", "ordinal", "year"]
