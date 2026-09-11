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
_UNIT = re.compile(rf"({passes.NUMBER})\s*({_UNIT_ALTERNATION})\b")

# A version only reads as one when it has three segments or an explicit
# introduction; without that, "2026.9" is a decimal quantity.
_INTRODUCED_VERSION = re.compile(
    r"(?<=[版本本v])\s*\b\d+(?:\.\d+)+\b", flags=re.IGNORECASE
)
_TRAILING_VERSION = re.compile(r"\b(\d+(?:\.\d+)+)(?=\s*版)")

_BARE_NUMBER = re.compile(passes.NUMBER)

# Characters that make a following number an identifier rather than a quantity.
_IDENTIFIER_LEAD = re.compile(r"(?:版本|version|v)\s*$", re.IGNORECASE)


def _date(match: re.Match[str], options: NormalizeOptions) -> str:
    year, month, day = match.group(1), int(match.group(2)), int(match.group(3))
    return f"{digit_string(year)}年{cardinal(month)}月{cardinal(day)}日"


def _clock(match: re.Match[str], options: NormalizeOptions) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    second = match.group(3)
    text = f"{hours(hour)}點"
    if minute == 0 and second is None:
        text += "整"
    elif minute:
        text += f"{minutes(minute)}分"
    if second is not None and int(second):
        text += f"{cardinal(int(second))}秒"
    return text


def _percent(match: re.Match[str], options: NormalizeOptions) -> str:
    return "百分之" + decimal(match.group(1))


def _temperature(match: re.Match[str], options: NormalizeOptions) -> str:
    value = decimal(match.group(1))
    if match.group(2) in ("°F", "℉"):
        return f"華氏{value}度"
    return f"攝氏{value}度" if options.temperature_prefix else f"{value}度"


def _numbered_unit(match: re.Match[str], options: NormalizeOptions) -> str:
    return decimal(match.group(1)) + SUFFIX_UNITS[match.group(2)]


def _degree(match: re.Match[str], options: NormalizeOptions) -> str:
    """Read a degree sign with no scale letter: "26.5°" is a plain 度."""
    return decimal(match.group(1)) + "度"


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
