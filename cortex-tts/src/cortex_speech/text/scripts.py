"""What depends on the character, not on the language.

Where a sentence ends and how fast a script is spoken are properties of the
characters on the page: 「।」 closes a sentence whoever wrote it, and one
漢字 is one syllable in Japanese as in Chinese. So they are declared once,
keyed by character, and no request has to name its language for them to
apply — mixed text, an untagged request and a reply arriving in pieces all
read the same table. What depends on the reader — how a number is said,
which stop to add, a word to respell — is the locale's (`locales.py`).

The three derived patterns exist so the batch path, the live buffer and the
terminator pass cannot disagree about what a sentence end is.
"""

from __future__ import annotations

import re

# Sentence-final punctuation. The ASCII set, the CJK fullwidth set, and the
# UAX #29 `Sentence_Break=STerm` members of the scripts a catalog model can
# reach (SentenceBreakProperty.txt, Unicode 18): Devanagari danda and double
# danda, the Arabic question mark and full stop, the Armenian full stop, the
# Myanmar section mark, the Ethiopic full stop, the Khmer khan. Thai has no
# terminator in the property at all; Thai text splits on length alone, as it
# does in ICU and espeak-ng.
TERMINATORS = "。！？；.!?;" + "।॥" + "؟۔" + "։" + "။" + "።" + "។"

# What may end a segment without being a break: an ellipsis trails off.
ENDINGS = TERMINATORS + "…"

# Where an over-long sentence may be cut: the clause commas, and the Arabic
# comma and semicolon (UAX #29 `SContinue`).
CLAUSE_MARKS = "，、," + "،؛"

# The ASCII full stop needs its own alternative: it also ends a decimal, so it
# breaks a sentence only when a non-digit precedes it and whitespace follows.
# Everything else in the set ends a sentence outright, as does a newline.
_OUTRIGHT = "".join(re.escape(c) for c in TERMINATORS if c != ".")
_CLAUSE = "".join(re.escape(c) for c in CLAUSE_MARKS)

SENTENCE_BREAK = re.compile(rf"(?<=[{_OUTRIGHT}\n])|(?<=[^\d\s]\.)(?=\s)")
"""Split a text into sentences, the break falling after the terminator."""

ENDS_SENTENCE = re.compile(rf"[{_OUTRIGHT}\n]$|[^\d\s]\.\s$")
"""Whether a buffer's last sentence is complete: the live form of the split."""

CLAUSE_BREAK = re.compile(rf"(?<=[{_CLAUSE}])")
"""Split a sentence into clauses, the break falling after the mark."""


# Speech-rate priors in characters a second, one per script, on the slow
# side: everything sized from an estimate must over-estimate the audio,
# because past a model's audio ceiling text is not slow but truncated.
# Measured on the Hojo 40M in production: 108 characters of Chinese as
# 26.6 s and 309 of Latin as 21.1 s. Kana and hangul take the Han figure
# because there too one character is one syllable or mora (Pellegrino et
# al. 2011: Mandarin 5.2, Japanese 7.8 syllables a second, Korean ≈5).
HAN_RATE = 4.1
LATIN_RATE = 14.7

RATES: tuple[tuple[re.Pattern[str], float], ...] = (
    # CJK punctuation and kana, Han, hangul, compatibility Han, fullwidth forms
    # (from U+3001: the ideographic space is a space).
    (re.compile(r"[、-鿿가-힯豈-﫿＀-￯]"), HAN_RATE),
    # ASCII, Latin-1 and Latin Extended letters, general punctuation.
    (re.compile(r"[!-~¡-ɏ‐-⁞]"), LATIN_RATE),
)

# A script nobody measured is assumed as slow as the slowest measured one.
# Every language on record speaks 4.7 to 8 syllables a second and no script
# writes a syllable in less than one character, so the floor never
# over-admits; it only splits more (Thai, at about three characters a
# syllable, is split three times as often as it needs).
UNKNOWN_RATE = min(rate for _, rate in RATES)


def _counted(text: str) -> list[tuple[int, float]]:
    """How many characters of each rate the text holds, spaces excluded.

    Returns:
        ``(count, rate)`` pairs for every script with a prior and, last, the
        characters no prior covers at `UNKNOWN_RATE`.
    """
    out: list[tuple[int, float]] = []
    seen = 0
    for pattern, rate in RATES:
        count = len(pattern.findall(text))
        out.append((count, rate))
        seen += count
    total = sum(1 for ch in text if not ch.isspace())
    out.append((max(total - seen, 0), UNKNOWN_RATE))
    return out


def spoken_seconds(text: str) -> float:
    """How long this text takes to say, at the priors."""
    return sum(count / rate for count, rate in _counted(text))


def chars_within(seconds: float, text: str) -> int:
    """How many characters of this text's scripts fit in that much audio.

    Each script speaks at its own rate, so the answer depends on what the
    text is made of, not only on how long it is. The slow-side priors mean
    this under-fills rather than over-fills whatever it is bounding.
    """
    total = sum(count for count, _ in _counted(text))
    if total <= 0 or seconds <= 0:
        return 0
    per_char = spoken_seconds(text) / total
    return max(1, int(seconds / per_char))
