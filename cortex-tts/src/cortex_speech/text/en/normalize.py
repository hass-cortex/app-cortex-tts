"""Number expansion for Latin-script text — the Latin mirror of ``normalize``.

The running order lives in `passes`; this module only says how each
construct sounds in English.
"""

from __future__ import annotations

import re

from .. import passes, units
from ..options import DEFAULT_OPTIONS, NormalizeOptions
from ..passes import Pass

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"))

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    5: "fifth",
    8: "eighth",
    9: "ninth",
    12: "twelfth",
}


def cardinal(value: int) -> str:
    """Return the spoken form of an integer."""
    if value < 0:
        return "minus " + cardinal(-value)
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, rest = divmod(value, 10)
        return _TENS[tens] + (f"-{_ONES[rest]}" if rest else "")
    if value < 1000:
        head, rest = divmod(value, 100)
        out = f"{_ONES[head]} hundred"
        return out + (f" and {cardinal(rest)}" if rest else "")
    for limit, name in _SCALES:
        if value >= limit:
            head, rest = divmod(value, limit)
            out = f"{cardinal(head)} {name}"
            if not rest:
                return out
            joiner = " and " if rest < 100 else " "
            return out + joiner + cardinal(rest)
    return str(value)


def ordinal(value: int) -> str:
    """Return the spoken ordinal, for dates."""
    if value in _ORDINALS:
        return _ORDINALS[value]
    word = cardinal(value)
    if value % 10 in _ORDINALS and value % 100 not in (11, 12, 13):
        head, _, last = word.rpartition("-")
        return f"{head}-{_ORDINALS[value % 10]}" if head else _ORDINALS[value % 10]
    return word[:-1] + "ieth" if word.endswith("y") else word + "th"


def decimal(text: str) -> str:
    """Return the spoken form of a decimal literal, fraction digit by digit."""
    sign = ""
    if text[0] in "+-":
        sign = "minus " if text[0] == "-" else ""
        text = text[1:]
    whole, _, frac = text.partition(".")
    out = sign + cardinal(int(whole or 0))
    if frac:
        out += " point " + " ".join(_ONES[int(d)] for d in frac)
    return out


def year(value: int) -> str:
    """Return a spoken year: 2026 is "twenty twenty-six", not a quantity."""
    if 2000 <= value <= 2009:
        rest = value - 2000
        return "two thousand" + (f" and {cardinal(rest)}" if rest else "")
    if 1100 <= value <= 9999 and value % 100:
        return f"{cardinal(value // 100)} {cardinal(value % 100)}"
    return cardinal(value)


def clock(hour: int, minute: int, second: int | None = None) -> str:
    """Return a spoken clock time."""
    out = cardinal(hour)
    if minute == 0:
        out += " o'clock" if second is None else " hundred"
    elif minute < 10:
        out += f" oh {_ONES[minute]}"
    else:
        out += f" {cardinal(minute)}"
    if second:
        out += f" and {cardinal(second)} seconds"
    return out


# A digit welded to a letter is an identifier ("P0"), not a quantity. The dot
# is rejected only before another digit, so a number can end a sentence.
_STANDALONE = re.compile(rf"(?<![A-Za-z0-9.])({passes.NUMBER})(?![A-Za-z0-9])(?!\.\d)")


def _date(match: re.Match[str], options: NormalizeOptions) -> str:
    y, m, d = (int(g) for g in match.groups())
    if not 1 <= m <= 12:
        return match.group(0)
    return f"the {ordinal(d)} of {_MONTHS[m - 1]} {year(y)}"


def _clock(match: re.Match[str], options: NormalizeOptions) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3)) if match.group(3) else None
    if hour > 23 or minute > 59:
        return match.group(0)
    return clock(hour, minute, second)


def _span(match: re.Match[str]) -> str:
    """Read the number, or both numbers of a range joined by "to"."""
    first = decimal(match.group(1))
    second = match.group(2)
    return first if second is None else f"{first} to {decimal(second)}"


def _percent(match: re.Match[str], options: NormalizeOptions) -> str:
    return f"{_span(match)} percent"


def _temperature(match: re.Match[str], options: NormalizeOptions) -> str:
    scale = "Celsius" if match.group(3) in ("°C", "℃") else "Fahrenheit"
    return f"{_span(match)} degrees {scale}"


def _degree(match: re.Match[str], options: NormalizeOptions) -> str:
    return f"{_span(match)} degrees"


# Symbols Home Assistant emits that CLDR has no English name for.
_OWN_UNITS: dict[str, tuple[str, str]] = {
    "mWh": ("milliwatt-hour", "milliwatt-hours"),
    "Wh": ("watt-hour", "watt-hours"),
    "MWh": ("megawatt-hour", "megawatt-hours"),
    "GWh": ("gigawatt-hour", "gigawatt-hours"),
    "μA": ("microampere", "microamperes"),
    "µA": ("microampere", "microamperes"),
    "μV": ("microvolt", "microvolts"),
    "µV": ("microvolt", "microvolts"),
    "mV": ("millivolt", "millivolts"),
    "kV": ("kilovolt", "kilovolts"),
    "VA": ("volt-ampere", "volt-amperes"),
    "kVA": ("kilovolt-ampere", "kilovolt-amperes"),
    "mHz": ("millihertz", "millihertz"),
    "mPa": ("millipascal", "millipascals"),
    "dB": ("decibel", "decibels"),
    "dBA": ("decibel", "decibels"),
    "ppm": ("part per million", "parts per million"),
    "ppb": ("part per billion", "parts per billion"),
}
_UNIT = passes.quantity(units.alternation(set(units.HA_UNITS) | units.UNNAMED))


def _unit(match: re.Match[str], options: NormalizeOptions) -> str:
    """Read a number and its unit as CLDR names it: "48 W" -> forty-eight watts."""
    words, symbol = _span(match), match.group(3)
    # A range agrees with the number read last.
    value = float(match.group(2) or match.group(1))
    named = units.name(value, words, symbol, "en")
    if named is not None:
        return named
    if symbol in _OWN_UNITS:
        return f"{words} {_OWN_UNITS[symbol][0 if value == 1 else 1]}"
    return f"{words} {symbol}"


def _version(match: re.Match[str], options: NormalizeOptions) -> str:
    return " point ".join(cardinal(int(part)) for part in match.group(0).split("."))


def _range(match: re.Match[str], options: NormalizeOptions) -> str:
    """Read a dash between numbers as a range, not a subtraction."""
    return " to "


def _number(match: re.Match[str], options: NormalizeOptions) -> str:
    return decimal(match.group(1))


# Whatever still holds Arabic numerals once every other pass has run is welded
# to letters — "P0", "v1.2", "24V". The model pronounces no digit at all, so
# leaving these alone does not keep them intact, it makes them silent.
_WELDED = re.compile(r"\d+(?:\.\d+)*")


def _welded(match: re.Match[str], options: NormalizeOptions) -> str:
    """Read a number welded to a letter.

    After a letter it is a label, read digit by digit ("P zero"); before one it
    carries a unit and stays a quantity ("twenty-four V"). A space is inserted
    only where a letter is in the way.
    """
    lead = match.string[match.start() - 1 : match.start()]
    tail = match.string[match.end() : match.end() + 1]
    if lead.isalpha():
        spoken = " ".join(
            "point" if c == "." else _ONES[int(c)] for c in match.group(0)
        )
    else:
        spoken = " point ".join(decimal(part) for part in match.group(0).split("."))
    return f"{' ' if lead.isalpha() else ''}{spoken}{' ' if tail.isalpha() else ''}"


_PASSES: tuple[Pass, ...] = (
    Pass(passes.THOUSANDS, passes.drop),
    Pass(passes.DATE, _date, "expand_dates"),
    Pass(passes.CLOCK, _clock, "expand_time"),
    Pass(passes.PERCENT, _percent),
    Pass(passes.TEMPERATURE, _temperature, "expand_units"),
    Pass(passes.DEGREE, _degree, "expand_units"),
    Pass(passes.VERSION, _version),
    Pass(_UNIT, _unit, "expand_units"),
    Pass(passes.RANGE, _range, "expand_numbers"),
    Pass(_STANDALONE, _number, "expand_numbers"),
    Pass(_WELDED, _welded, "expand_numbers"),
)


def normalize(text: str, options: NormalizeOptions = DEFAULT_OPTIONS) -> str:
    """Expand numerals in Latin-script text into words.

    Args:
        text: Raw text, typically a Home Assistant response.
        options: Which passes to apply.

    Returns:
        Text with every Arabic-numeral construct read out in words. Units
        keep their symbol: plural agreement is a separate problem from
        reading the number.
    """
    return passes.run(text, _PASSES, options)
