"""Turn Home Assistant text into something the TTS model can pronounce.

Measured on the Hojo 40M: unnormalised sensor text ("26.5°C", "68%", "14:35")
comes out at 36-50% character error rate, against 4% once the numbers and
units are spelled out. No model here ships text normalisation, so this module
is the difference between a usable voice and noise.

Latin words are left untouched: the model reads English natively, and
mangling "Home Assistant" into Chinese would make it worse. The running
order lives in `passes`.
"""

from __future__ import annotations

import re

from .. import passes
from ..options import DEFAULT_OPTIONS, NormalizeOptions
from ..passes import Pass
from .numbers import cardinal, decimal, digit_string, hours, minutes

# Home Assistant's unit symbols (`homeassistant/const.py`) and how each is
# said after the number. Absent on purpose: the symbols that are also words
# or labels (in, st, ac, ha, K, B, d, w, y).
SUFFIX_UNITS: dict[str, str] = {
    "°C": "度",
    "℃": "度",
    "°F": "華氏度",
    "℉": "華氏度",
    "°": "度",
    # power and energy
    "mW": "毫瓦",
    "W": "瓦",
    "kW": "千瓦",
    "MW": "百萬瓦",
    "GW": "十億瓦",
    "mWh": "毫瓦時",
    "Wh": "瓦時",
    "kWh": "度電",
    "kwh": "度電",
    "MWh": "百萬瓦時",
    "GWh": "十億瓦時",
    "J": "焦耳",
    "kJ": "千焦耳",
    "MJ": "百萬焦耳",
    "cal": "卡",
    "kcal": "大卡",
    "VA": "伏安",
    "kVA": "千伏安",
    "kWh/100km": "度電每百公里",
    "Wh/km": "瓦時每公里",
    "km/kWh": "公里每度電",
    # electricity
    "μA": "微安培",
    "µA": "微安培",
    "mA": "毫安培",
    "A": "安培",
    "μV": "微伏特",
    "µV": "微伏特",
    "mV": "毫伏特",
    "V": "伏特",
    "kV": "千伏特",
    "S/cm": "西門子每公分",
    "mS/cm": "毫西門子每公分",
    "μS/cm": "微西門子每公分",
    "µS/cm": "微西門子每公分",
    # time
    "μs": "微秒",
    "µs": "微秒",
    "ms": "毫秒",
    "s": "秒",
    "min": "分鐘",
    "h": "小時",
    # length, area, volume
    "mm": "毫米",
    "cm": "公分",
    "m": "公尺",
    "km": "公里",
    "ft": "英尺",
    "yd": "碼",
    "mi": "英里",
    "nmi": "海里",
    "mm²": "平方毫米",
    "cm²": "平方公分",
    "m²": "平方公尺",
    "km²": "平方公里",
    "ft²": "平方英尺",
    "mi²": "平方英里",
    "mL": "毫升",
    "L": "公升",
    "gal": "加侖",
    "m³": "立方公尺",
    "ft³": "立方英尺",
    # mass
    "μg": "微克",
    "µg": "微克",
    "mg": "毫克",
    "g": "公克",
    "kg": "公斤",
    "oz": "盎司",
    "lb": "磅",
    # frequency, pressure, sound, light
    "mHz": "毫赫茲",
    "Hz": "赫茲",
    "kHz": "千赫茲",
    "MHz": "兆赫",
    "GHz": "吉赫",
    "rpm": "轉每分鐘",
    "mPa": "毫帕",
    "Pa": "帕",
    "hPa": "百帕",
    "kPa": "千帕",
    "bar": "巴",
    "mbar": "毫巴",
    "mmHg": "毫米汞柱",
    "inHg": "英寸汞柱",
    "psi": "psi",
    "dB": "分貝",
    "dBA": "分貝",
    "dBm": "分貝毫瓦",
    "lx": "勒克斯",
    # rates and concentrations
    "km/h": "公里每小時",
    "mph": "英里每小時",
    "kn": "節",
    "m/s": "公尺每秒",
    "m/min": "公尺每分鐘",
    "mm/s": "毫米每秒",
    "ft/s": "英尺每秒",
    "mm/h": "毫米每小時",
    "mm/d": "毫米每天",
    "W/m²": "瓦每平方公尺",
    "L/h": "公升每小時",
    "L/min": "公升每分鐘",
    "L/s": "公升每秒",
    "mL/s": "毫升每秒",
    "gal/h": "加侖每小時",
    "gal/min": "加侖每分鐘",
    "m³/h": "立方公尺每小時",
    "m³/min": "立方公尺每分鐘",
    "m³/s": "立方公尺每秒",
    "ft³/min": "立方英尺每分鐘",
    "g/m³": "公克每立方公尺",
    "mg/m³": "毫克每立方公尺",
    "µg/m³": "微克每立方公尺",
    "μg/m³": "微克每立方公尺",
    "ug/m3": "微克每立方公尺",
    "mg/dL": "毫克每分升",
    "mmol/L": "毫莫耳每公升",
    "Bq/m³": "貝克每立方公尺",
    "ppm": "ppm",
    "ppb": "ppb",
    # information
    "bit": "位元",
    "kbit": "千位元",
    "Mbit": "百萬位元",
    "Gbit": "十億位元",
    "kB": "KB",
    "MB": "MB",
    "GB": "GB",
    "TB": "TB",
    "bit/s": "位元每秒",
    "kbit/s": "千位元每秒",
    "Mbit/s": "百萬位元每秒",
    "Gbit/s": "十億位元每秒",
    "kB/s": "KB每秒",
    "MB/s": "MB每秒",
    "GB/s": "GB每秒",
}

# Chinese unit and measure words, in both scripts: a number before one of
# these is a quantity, whatever the request says about bare numbers. 號 and
# 年 are not here — 302號 is a label, and a year reads digit by digit.
# fmt: off
WORD_UNITS: frozenset[str] = frozenset({
    # time
    "秒", "分", "分鐘", "分钟", "小時", "小时", "天", "週", "周", "個月", "个月", "月", "日", "點", "点",
    # length, mass, volume
    "毫米", "公分", "公尺", "米", "公里", "毫克", "微克", "克", "公克", "公斤", "毫升", "升", "公升",
    # physics and money
    "度", "瓦", "千瓦", "伏特", "安培", "毫安", "帕", "百帕", "勒克斯", "分貝", "分贝", "赫茲", "赫兹",
    "卡", "大卡", "元", "塊", "块",
    # measure words
    "倍", "歲", "岁", "次", "個", "个", "人", "台", "張", "张", "件", "條", "条", "杯", "顆", "颗",
    "步", "隻", "只", "位", "樓", "楼", "層", "层", "格", "段", "級", "级", "檔", "档",
})
# fmt: on

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
_WORD_UNIT = passes.quantity(
    "|".join(re.escape(unit) for unit in sorted(WORD_UNITS, key=len, reverse=True))
)
# A four-digit run before 年 is a year: 二零二六年, never 二千零二十六年.
_YEAR = re.compile(rf"{passes.LEAD}(\d{{4}})(?=\s*年)")

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


def _word_unit(match: re.Match[str], options: NormalizeOptions) -> str:
    """Read the number and keep the unit word as written: 25.9 度 -> 二十五點九度."""
    return _span(match) + match.group(3)


def _year(match: re.Match[str], options: NormalizeOptions) -> str:
    return digit_string(match.group(1))


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
    return decimal(literal)


_PASSES: tuple[Pass, ...] = (
    Pass(passes.THOUSANDS, passes.drop),
    Pass(passes.DATE, _date, "expand_dates"),
    Pass(_YEAR, _year, "expand_dates"),
    Pass(passes.CLOCK, _clock, "expand_time"),
    Pass(passes.PERCENT, _percent),
    Pass(passes.TEMPERATURE, _temperature, "expand_units"),
    Pass(_UNIT, _numbered_unit, "expand_units"),
    Pass(passes.DEGREE, _degree, "expand_units"),
    Pass(passes.VERSION, _version),
    Pass(_INTRODUCED_VERSION, _version),
    Pass(_TRAILING_VERSION, _version),
    Pass(_WORD_UNIT, _word_unit, "expand_units"),
    Pass(passes.RANGE, _range, "expand_numbers"),
    Pass(_BARE_NUMBER, _bare_number, "expand_numbers"),
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
