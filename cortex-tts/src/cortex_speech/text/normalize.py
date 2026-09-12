"""Turn Home Assistant text into something the TTS model can pronounce.

Measured on this model: unnormalised sensor text ("26.5°C", "68%", "14:35")
comes out at 36-50% character error rate, against 4% once the numbers and
units are spelled out. The model ships no text normalisation, so this module
is the difference between a usable voice and noise.

Latin words are left untouched: the model reads English natively, and
mangling "Home Assistant" into Chinese would make it worse. The running
order lives in `passes`.
"""

from __future__ import annotations

import re

from . import passes
from .numbers import cardinal, decimal, digit_string, hours, minutes
from .options import DEFAULT_OPTIONS, NormalizeOptions
from .passes import Pass

# Units whose spoken form simply follows the number.
SUFFIX_UNITS: dict[str, str] = {
    "°C": "度",
    "℃": "度",
    "°F": "華氏度",
    "℉": "華氏度",
    "°": "度",
    "kWh": "度電",
    "kwh": "度電",
    "Wh": "瓦時",
    "kW": "千瓦",
    "kw": "千瓦",
    "W": "瓦",
    "mA": "毫安培",
    "A": "安培",
    "mV": "毫伏特",
    "V": "伏特",
    "kB": "KB",
    "MB": "MB",
    "GB": "GB",
    "TB": "TB",
    "hPa": "百帕",
    "Pa": "帕",
    "km/h": "公里每小時",
    "m/s": "公尺每秒",
    "km": "公里",
    "cm": "公分",
    "mm": "毫米",
    "m²": "平方公尺",
    "m³": "立方公尺",
    "kg": "公斤",
    "mg": "毫克",
    "µg/m³": "微克每立方公尺",
    "μg/m³": "微克每立方公尺",
    "ug/m3": "微克每立方公尺",
    "lx": "勒克斯",
    "ppm": "ppm",
    "小時": "小時",
}

# Longest first so "km/h" is matched before "km", and "kWh" before "W".
_UNIT_ALTERNATION = "|".join(
    re.escape(unit) for unit in sorted(SUFFIX_UNITS, key=len, reverse=True)
)
# A unit followed by a Latin letter is part of a longer token ("kWhx"); a
# CJK character after it is the sentence continuing.
_UNIT = re.compile(
    rf"({passes.NUMBER})(?:{passes.DASH}({passes.NUMBER}))?"
    rf"\s*({_UNIT_ALTERNATION})(?![A-Za-z])"
)

# A version only reads as one when it has three segments or an explicit
# introduction; without that, "2026.9" is a decimal quantity.
_INTRODUCED_VERSION = re.compile(
    rf"(?<=[版本v])\s*\d+(?:\.\d+)+{passes.TAIL}", flags=re.IGNORECASE
)
_TRAILING_VERSION = re.compile(rf"{passes.LEAD}(\d+(?:\.\d+)+)(?=\s*版)")

_BARE_NUMBER = re.compile(passes.NUMBER)

# Words that make a following number an identifier rather than a quantity. A
# bare "v" counts only as its own word, not as the last letter of "TV".
_IDENTIFIER_LEAD = re.compile(r"(?:版本|version|(?<![A-Za-z])v)\s*$", re.IGNORECASE)


def _span(match: re.Match[str]) -> str:
    """Read the number, or both numbers of a range joined by 到."""
    first = decimal(match.group(1))
    second = match.group(2)
    return first if second is None else f"{first}到{decimal(second)}"


def _date(match: re.Match[str], options: NormalizeOptions) -> str:
    year, month, day = match.group(1), int(match.group(2)), int(match.group(3))
    return f"{digit_string(year)}年{cardinal(month)}月{cardinal(day)}日"


def _clock(match: re.Match[str], options: NormalizeOptions) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    second = match.group(3)
    if hour > 23 or minute > 59:
        return match.group(0)
    text = f"{hours(hour)}點"
    # "3:05分" already says 分; do not say it twice.
    follows_fen = match.string[match.end() : match.end() + 1] == "分"
    if minute == 0 and second is None:
        text += "" if follows_fen else "整"
    elif minute:
        text += minutes(minute) + ("" if follows_fen else "分")
    elif second is not None:
        # 十四點三十秒 is indistinguishable by ear from 14:30.
        text += "零分"
    if second is not None and int(second):
        text += f"{cardinal(int(second))}秒"
    return text


def _percent(match: re.Match[str], options: NormalizeOptions) -> str:
    return "百分之" + _span(match)


def _temperature(match: re.Match[str], options: NormalizeOptions) -> str:
    value = _span(match)
    if match.group(3) in ("°F", "℉"):
        return f"華氏{value}度"
    return f"攝氏{value}度" if options.temperature_prefix else f"{value}度"


def _numbered_unit(match: re.Match[str], options: NormalizeOptions) -> str:
    return _span(match) + SUFFIX_UNITS[match.group(3)]


def _degree(match: re.Match[str], options: NormalizeOptions) -> str:
    """Read a degree sign with no scale letter: "26.5°" is a plain 度."""
    return _span(match) + "度"


def _version(match: re.Match[str], options: NormalizeOptions) -> str:
    """Read a dotted version as digits joined by 點: 2026.9.1 -> 二零二六點九點一."""
    return "點".join(digit_string(part) for part in match.group(0).split("."))


def _range(match: re.Match[str], options: NormalizeOptions) -> str:
    """Read a dash between numbers as 到, so "25-30" is not a subtraction."""
    return "到"


def _bare_number(match: re.Match[str], options: NormalizeOptions) -> str:
    """Read a standalone number, choosing quantity vs identifier reading."""
    literal = match.group(0)
    text = match.string
    if _IDENTIFIER_LEAD.search(text[: match.start()]):
        return digit_string(literal)
    # A bare 4-digit run immediately followed by 年 is a year, not a count.
    tail = text[match.end() : match.end() + 1]
    if tail == "年" and len(literal) == 4 and literal.isdigit():
        return digit_string(literal)
    return decimal(literal)


_PASSES: tuple[Pass, ...] = (
    Pass(passes.THOUSANDS, passes.drop),
    Pass(passes.DATE, _date, "expand_dates"),
    Pass(passes.CLOCK, _clock, "expand_time"),
    Pass(passes.PERCENT, _percent),
    Pass(passes.TEMPERATURE, _temperature, "expand_units"),
    Pass(_UNIT, _numbered_unit, "expand_units"),
    Pass(passes.DEGREE, _degree, "expand_units"),
    Pass(passes.VERSION, _version),
    Pass(_INTRODUCED_VERSION, _version),
    Pass(_TRAILING_VERSION, _version),
    Pass(passes.RANGE, _range),
    Pass(_BARE_NUMBER, _bare_number),
)


def normalize(text: str, options: NormalizeOptions = DEFAULT_OPTIONS) -> str:
    """Expand numbers, units and symbols into pronounceable Chinese.

    Args:
        text: Raw text, typically a Home Assistant response.
        options: Which passes to apply.

    Returns:
        Text with every Arabic-numeral construct replaced by Chinese
        characters. Latin words and unmatched symbols are left alone.
    """
    return passes.run(text, _PASSES, options)
