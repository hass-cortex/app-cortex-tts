"""Switches shared by both normalisers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizeOptions:
    """Switches for the normalisation passes.

    Attributes:
        temperature_prefix: Prepend 攝氏/華氏 to bare degree readings.
        expand_units: Convert unit symbols to their spoken Chinese form.
        expand_time: Read clock literals as 點/分.
        expand_dates: Read ISO dates as 年月日.
        expand_numbers: Read a number that stands on its own — no unit,
            clock or date around it — as a quantity. Off unless asked: a
            bare number is as often a phone number, a room, a model or a
            score as a count, and a wrong reading misleads where digits
            left alone merely go unread.
    """

    temperature_prefix: bool = True
    expand_units: bool = True
    expand_time: bool = True
    expand_dates: bool = True
    expand_numbers: bool = False


DEFAULT_OPTIONS = NormalizeOptions()
