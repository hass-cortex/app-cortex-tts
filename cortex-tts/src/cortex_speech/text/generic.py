"""The locale a language gets when nobody has written it one.

The models pronounce no Arabic numeral in any language, so the least a
language needs is its numbers read out — ``num2words`` knows how to for some
forty of them — and the shapes Home Assistant puts numbers in read the way
that language reads them: unit symbols named as CLDR names them (``babel``,
through the table in `units`: 26.5°C is "sechsundzwanzig Komma fünf Grad
Celsius"), an ISO date laid out
as CLDR lays it out ("14. September 2026", "2026年9月14日") with its numbers
in words, a clock literal as hour words then minute words. Only those fixed
shapes: a number this locale is not sure of is left as digits, because a
number read wrong is worse than one left unread.

Anything more — an "Uhr" between the hour and the minute, a month read as
an ordinal, the language's own units — is a locale written beside `zh/` and
`en/`, not a rule added here.
"""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Callable
from functools import cache

from . import passes, units
from .locales import Locale
from .options import NormalizeOptions
from .passes import Pass

_LOGGER = logging.getLogger(__name__)

# Languages written without spaces between words, ended by the ideographic
# full stop. Korean writes spaces and a Latin stop, so it is not one.
_UNSPACED = frozenset({"ja"})

# Languages whose num2words year reading is an era name (令和八年), which is
# not how a date read off a sensor should sound.
_ERA_YEARS = frozenset({"ja", "ko"})


# A number as Home Assistant writes one (26.5, -3, 1,234) or as the language
# writes one (26,5 where the comma is the decimal mark). A comma before
# exactly three digits is always a thousands separator, because that is what
# a machine-formatted 1,234 is, whatever the locale.
_NUMBER = re.compile(
    rf"{passes.LEAD}(?:(?<!\d)[+-])?\d+(?:,\d{{3}}(?!\d))*(?:[.,]\d+)?{passes.TAIL}"
)
_QUANTITY = re.compile(
    rf"({_NUMBER.pattern})(?:{passes.DASH}({_NUMBER.pattern}))?\s*"
    + "("
    + units.alternation(set(units.HA_UNITS) | units.UNNAMED)
    + ")"
)
_DAY_DOT = re.compile(r"(?<!\d)(\d{1,2})\.(?=\s)")
_YEAR = re.compile(r"(?<!\d)\d{4}(?!\d)")
_DIGITS = re.compile(r"\d+")


@cache
def _speaks(code: str) -> bool:
    """Whether num2words has the language; asked once per language."""
    from num2words import num2words

    try:
        num2words(1, lang=code)
    except NotImplementedError:
        _LOGGER.info("no number words for %r; digits are left as written", code)
        return False
    return True


@cache
def _decimal_comma(code: str) -> bool:
    from babel.core import UnknownLocaleError
    from babel.numbers import get_decimal_symbol

    try:
        return get_decimal_symbol(code) == ","
    except UnknownLocaleError:
        return False


def _value(literal: str, code: str) -> float | int:
    """The number a literal means, in the language's own conventions."""
    literal = re.sub(r",(?=\d{3}(?!\d))", "", literal)
    if "," in literal and _decimal_comma(code):
        literal = literal.replace(",", ".")
    return float(literal) if "." in literal else int(literal)


def _words(value: float | int, code: str, to: str = "cardinal") -> str:
    from num2words import num2words

    try:
        return num2words(value, lang=code, to=to)
    except NotImplementedError:
        return num2words(value, lang=code)


def _named(value: float | int, words: str, unit: str, code: str) -> str:
    """The unit's CLDR name around the number words, or the symbol if none."""
    named = units.name(value, words, unit, code)
    if named is not None:
        return named
    # A symbol stays welded to the words (68%); a letter needs its space back.
    return f"{words} {unit}" if unit[0].isalpha() else f"{words}{unit}"


def _renders(code: str) -> tuple[Pass, ...]:
    """The pass table for one language, in the order the contract requires."""

    def number(match: re.Match[str], options: NormalizeOptions) -> str:
        return _words(_value(match.group(0), code), code)

    def quantity(match: re.Match[str], options: NormalizeOptions) -> str:
        first, second, unit = match.groups()
        words = _words(_value(first, code), code)
        if second:
            words = f"{words} - {_words(_value(second, code), code)}"
        if not options.expand_units:
            return f"{words} {unit}" if unit[0].isalpha() else f"{words}{unit}"
        # A range takes the second number's agreement, as it is read last.
        return _named(_value(second or first, code), words, unit, code)

    def date(match: re.Match[str], options: NormalizeOptions) -> str:
        from babel.core import UnknownLocaleError
        from babel.dates import format_date

        y, m, d = (int(g) for g in match.groups())
        try:
            laid_out = format_date(datetime.date(y, m, d), "long", locale=code)
        except (ValueError, UnknownLocaleError):
            return match.group(0)
        # CLDR marks an ordinal day with a full stop (German "14."); the
        # year is the only four-digit number; what is left is a cardinal.
        laid_out = _DAY_DOT.sub(
            lambda n: _words(int(n.group(1)), code, "ordinal"), laid_out
        )
        year = _YEAR.sub(
            lambda n: _words(
                int(n.group(0)), code, "cardinal" if code in _ERA_YEARS else "year"
            ),
            laid_out,
            count=1,
        )
        return _DIGITS.sub(lambda n: _words(int(n.group(0)), code), year)

    def clock(match: re.Match[str], options: NormalizeOptions) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23 or minute > 59:
            return match.group(0)
        words = [_words(hour, code), _words(minute, code)]
        if match.group(3):
            words.append(_words(int(match.group(3)), code))
        return " ".join(words)

    return (
        Pass(passes.DATE, date, "expand_dates"),
        Pass(passes.CLOCK, clock, "expand_time"),
        Pass(_QUANTITY, quantity),
        Pass(_NUMBER, number, "expand_numbers"),
    )


def _normalizer(code: str) -> Callable[[str, NormalizeOptions], str]:
    steps = _renders(code)

    def normalize(text: str, options: NormalizeOptions) -> str:
        if not _speaks(code):
            return text
        return passes.run(text, steps, options)

    return normalize


@cache
def locale(code: str) -> Locale:
    """Return the generic locale for a primary language subtag."""
    return Locale(
        code=code,
        normalize=_normalizer(code),
        stop="。" if code in _UNSPACED else ".",
        close_gaps=code in _UNSPACED,
        written=False,
    )
