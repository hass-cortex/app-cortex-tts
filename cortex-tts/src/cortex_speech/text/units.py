"""The unit symbols Home Assistant emits, as CLDR names them.

Home Assistant writes a sensor's unit as one of the symbols in its
``homeassistant/const.py`` (``UnitOf*``): ``W``, ``kWh``, ``hPa``, ``L/min``.
CLDR names those units in every language it covers, with the number's
agreement (kilomètre, kilomètres), so the English and the generic locales
read them through this one table instead of each carrying its own. The
Chinese locale keeps its own readings: CLDR says 千瓦時 where Taiwan says 度.

A symbol with no CLDR unit (``ppm``, ``dBm``) is left as written — read as
letters, it still means what it meant. Absent on purpose: the symbols that
are also words or labels (``in``, ``st``, ``ac``, ``ha``, ``K``, ``B``,
``d``, ``w``, ``y``), and the prefixes past tera that no household sensor
reports.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Symbol -> CLDR unit id, or (numerator, denominator) for a rate CLDR
# composes with its "per" pattern. Both mu glyphs: Home Assistant writes
# U+03BC, keyboards write U+00B5.
HA_UNITS: dict[str, str | tuple[str, str]] = {
    "%": "concentr-percent",
    "％": "concentr-percent",
    "°C": "temperature-celsius",
    "℃": "temperature-celsius",
    "°F": "temperature-fahrenheit",
    "℉": "temperature-fahrenheit",
    "mW": "power-milliwatt",
    "W": "power-watt",
    "kW": "power-kilowatt",
    "MW": "power-megawatt",
    "GW": "power-gigawatt",
    "J": "energy-joule",
    "kJ": "energy-kilojoule",
    "cal": "energy-calorie",
    "kcal": "energy-kilocalorie",
    "kWh": "energy-kilowatt-hour",
    "mA": "electric-milliampere",
    "A": "electric-ampere",
    "V": "electric-volt",
    "μs": "duration-microsecond",
    "µs": "duration-microsecond",
    "ms": "duration-millisecond",
    "s": "duration-second",
    "min": "duration-minute",
    "h": "duration-hour",
    "mm": "length-millimeter",
    "cm": "length-centimeter",
    "m": "length-meter",
    "km": "length-kilometer",
    "ft": "length-foot",
    "yd": "length-yard",
    "mi": "length-mile",
    "nmi": "length-nautical-mile",
    "Hz": "frequency-hertz",
    "kHz": "frequency-kilohertz",
    "MHz": "frequency-megahertz",
    "GHz": "frequency-gigahertz",
    "Pa": "pressure-pascal",
    "hPa": "pressure-hectopascal",
    "kPa": "pressure-kilopascal",
    "bar": "pressure-bar",
    "mbar": "pressure-millibar",
    "mmHg": "pressure-millimeter-ofhg",
    "inHg": "pressure-inch-ofhg",
    "psi": "pressure-pound-force-per-square-inch",
    "ft³": "volume-cubic-foot",
    "m³": "volume-cubic-meter",
    "L": "volume-liter",
    "mL": "volume-milliliter",
    "gal": "volume-gallon",
    "m²": "area-square-meter",
    "cm²": "area-square-centimeter",
    "km²": "area-square-kilometer",
    "ft²": "area-square-foot",
    "mi²": "area-square-mile",
    "g": "mass-gram",
    "kg": "mass-kilogram",
    "mg": "mass-milligram",
    "μg": "mass-microgram",
    "µg": "mass-microgram",
    "oz": "mass-ounce",
    "lb": "mass-pound",
    "mg/dL": "concentr-milligram-ofglucose-per-deciliter",
    "mmol/L": "concentr-millimole-per-liter",
    "m/s": "speed-meter-per-second",
    "km/h": "speed-kilometer-per-hour",
    "mph": "speed-mile-per-hour",
    "kn": "speed-knot",
    "bit": "digital-bit",
    "kbit": "digital-kilobit",
    "Mbit": "digital-megabit",
    "Gbit": "digital-gigabit",
    "kB": "digital-kilobyte",
    "MB": "digital-megabyte",
    "GB": "digital-gigabyte",
    "TB": "digital-terabyte",
    "lx": "light-lux",
    "W/m²": ("power-watt", "area-square-meter"),
    "mm/h": ("length-millimeter", "duration-hour"),
    "mm/d": ("length-millimeter", "duration-day"),
    "m/min": ("length-meter", "duration-minute"),
    "mm/s": ("length-millimeter", "duration-second"),
    "ft/s": ("length-foot", "duration-second"),
    "L/h": ("volume-liter", "duration-hour"),
    "L/min": ("volume-liter", "duration-minute"),
    "L/s": ("volume-liter", "duration-second"),
    "mL/s": ("volume-milliliter", "duration-second"),
    "gal/min": ("volume-gallon", "duration-minute"),
    "gal/h": ("volume-gallon", "duration-hour"),
    "m³/h": ("volume-cubic-meter", "duration-hour"),
    "m³/min": ("volume-cubic-meter", "duration-minute"),
    "m³/s": ("volume-cubic-meter", "duration-second"),
    "ft³/min": ("volume-cubic-foot", "duration-minute"),
    "g/m³": ("mass-gram", "volume-cubic-meter"),
    "mg/m³": ("mass-milligram", "volume-cubic-meter"),
    "μg/m³": ("mass-microgram", "volume-cubic-meter"),
    "µg/m³": ("mass-microgram", "volume-cubic-meter"),
    "km/kWh": ("length-kilometer", "energy-kilowatt-hour"),
    "kB/s": ("digital-kilobyte", "duration-second"),
    "MB/s": ("digital-megabyte", "duration-second"),
    "GB/s": ("digital-gigabyte", "duration-second"),
    "bit/s": ("digital-bit", "duration-second"),
    "kbit/s": ("digital-kilobit", "duration-second"),
    "Mbit/s": ("digital-megabit", "duration-second"),
    "Gbit/s": ("digital-gigabit", "duration-second"),
}

# Symbols Home Assistant emits that CLDR has no unit for. Matched so that
# the number before them is still read; the symbol itself stays.
UNNAMED: frozenset[str] = frozenset(
    {
        "mWh", "Wh", "MWh", "GWh", "Wh/km", "kWh/100km",
        "μA", "µA", "μV", "µV", "mV", "kV", "VA", "kVA",
        "mHz", "mPa", "dB", "dBA", "dBm", "rpm",
        "ppm", "ppb", "Bq/m³", "S/cm", "mS/cm", "μS/cm", "µS/cm",
    }
)  # fmt: skip


def alternation(symbols: Iterable[str]) -> str:
    """A regex alternation of the symbols, longest first so `km/h` beats `km`."""
    return "|".join(re.escape(s) for s in sorted(symbols, key=len, reverse=True))


def name(value: float | int, words: str, symbol: str, code: str) -> str | None:
    """The unit's CLDR name around the number words, or None if it has none.

    The number goes in as a number, so the name agrees with it (kilomètre,
    kilomètres), and comes out replaced by its words. Asked for a name a
    locale lacks, babel hands back the unit's id, and "concentr-percent"
    read aloud is worse than "%" — so an id in the output means no name.
    """
    from babel.core import UnknownLocaleError
    from babel.numbers import format_decimal
    from babel.units import format_compound_unit, format_unit

    cldr = HA_UNITS.get(symbol)
    if cldr is None:
        return None
    parts = (cldr,) if isinstance(cldr, str) else cldr
    try:
        if isinstance(cldr, str):
            named = format_unit(value, cldr, "long", locale=code)
        else:
            named = format_compound_unit(
                value, cldr[0], 1, cldr[1], length="long", locale=code
            )
        digits = format_decimal(value, locale=code)
    except UnknownLocaleError:
        return None
    if not named or digits not in named or any(p in named for p in parts):
        return None
    # CLDR joins number and unit with a narrow no-break space in some
    # locales; a model has no reason to know that glyph.
    return named.replace(digits, words, 1).replace(" ", " ").replace("\xa0", " ")
