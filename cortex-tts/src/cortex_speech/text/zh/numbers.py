"""Chinese numeral rendering.

The TTS model has no text normalisation of its own: it pronounces whatever
glyphs it is given. Arabic digits reaching the model come out as noise, so
every number must already be Chinese characters by the time it is synthesised.

Two readings are needed and they are not interchangeable:

- *cardinal* ("二十六") for quantities — temperatures, counts, watts.
- *digit string* ("二零二六") for identifiers — years, versions, addresses.
"""

from __future__ import annotations

DIGITS = "零一二三四五六七八九"
UNITS = ["", "十", "百", "千"]
GROUPS = ["", "萬", "億", "兆"]

MAX_CARDINAL = 10**16


def digit_string(text: str) -> str:
    """Read digits one by one: ``2026`` -> ``二零二六``.

    Used for years, version numbers and anything else that is an identifier
    rather than a quantity.
    """
    return "".join(DIGITS[int(c)] if c.isdigit() else c for c in text)


def _below_10000(value: int) -> str:
    """Render 0-9999 with positional units, collapsing internal zero runs."""
    if value == 0:
        return DIGITS[0]
    out: list[str] = []
    pending_zero = False
    for position in range(3, -1, -1):
        digit = value // (10**position) % 10
        if digit == 0:
            # Only mark a zero once, and never before any significant digit.
            if out:
                pending_zero = True
            continue
        if pending_zero:
            out.append(DIGITS[0])
            pending_zero = False
        out.append(DIGITS[digit] + UNITS[position])
    return "".join(out)


def cardinal(value: int) -> str:
    """Render an integer as a spoken Chinese quantity.

    ``26`` -> ``二十六``, ``148`` -> ``一百四十八``, ``48000`` -> ``四萬八千``.
    Values at or beyond 10^16 fall back to a digit string rather than
    inventing group names past 兆.
    """
    if value < 0:
        return "負" + cardinal(-value)
    if value >= MAX_CARDINAL:
        return digit_string(str(value))
    if value < 10:
        return DIGITS[value]

    groups: list[int] = []
    remaining = value
    while remaining:
        groups.append(remaining % 10000)
        remaining //= 10000

    parts: list[str] = []
    skipped = False
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group == 0:
            skipped = bool(parts)
            continue
        rendered = _below_10000(group)
        # One 零 marks any gap before a lower group, whether the gap is a
        # wholly empty group (100000001 is 一億零一) or a leading zero inside
        # this one (1_0026 is 一萬零二十六) — and never two in a row.
        if parts and (skipped or group < 1000):
            rendered = DIGITS[0] + rendered
        skipped = False
        parts.append(rendered + GROUPS[index])

    text = "".join(parts)
    # Spoken Chinese drops the leading 一 of a bare 十-teens: 15 is 十五.
    if text.startswith("一十"):
        text = text[1:]
    return text


def decimal(text: str) -> str:
    """Render a decimal literal: ``26.5`` -> ``二十六點五``.

    The integer part is a quantity; the fraction is always read digit by
    digit, which is how Chinese says decimals.
    """
    negative = text.startswith("-")
    body = text.lstrip("+-")
    if "." not in body:
        rendered = cardinal(int(body))
    else:
        whole, _, fraction = body.partition(".")
        whole_text = cardinal(int(whole)) if whole else DIGITS[0]
        rendered = whole_text + "點" + digit_string(fraction)
    return ("負" + rendered) if negative else rendered


def hours(value: int) -> str:
    """Render an hour for clock readings, keeping 二 rather than 兩."""
    return cardinal(value)


def minutes(value: int) -> str:
    """Render a minute count, reading a leading zero as 零.

    A whole hour has no minute to say, so zero renders as nothing; the caller
    supplies 整 instead.
    """
    if value == 0:
        return ""
    if value < 10:
        return DIGITS[0] + DIGITS[value]
    return cardinal(value)
