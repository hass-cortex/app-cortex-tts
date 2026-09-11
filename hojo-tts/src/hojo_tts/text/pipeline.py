"""The text path every synthesis request takes.

Order is not negotiable. Normalisation emits Traditional Chinese number words
(二十六點五度), so it has to run before the script conversion that makes those
glyphs pronounceable. Reversing the two would leave freshly-minted Traditional
characters downstream of the only pass that can fix them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from . import english
from .normalize import normalize
from .options import NormalizeOptions

_LOGGER = logging.getLogger(__name__)

# Upper bound on a single synthesis. The LM tops out at 2048 new tokens at a
# 50 Hz token rate — roughly 40 s of audio — and long prompts degrade before
# they truncate, so requests are split into sentences well below that.
MAX_CHARS_PER_SEGMENT = 120

# The ASCII full stop needs its own alternative: it also ends a decimal, so it
# breaks a sentence only when a non-digit precedes it and whitespace follows.
_SENTENCE_BREAK = re.compile(r"(?<=[。！？；!?;\n])|(?<=[^\d\s]\.)(?=\s)")
_CLAUSE_BREAK = re.compile(r"(?<=[，、,])")

_CJK = r"㐀-䶿一-鿿豈-﫿　-〿＀-￯"
# Expanding "26.5°C" into 攝氏二十六點五度 leaves the space that preceded the
# digits stranded between two Chinese characters, and the model reads a space
# as a pause. Spaces bordering Latin text are kept — they are real word gaps.
_CJK_GAP = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])")

# A segment of nothing but punctuation has no sound. Passed to the model it
# generates no audio tokens and surfaces as an opaque engine error, so it is
# dropped here and the caller gets a clean "nothing to say" instead.
_SPEAKABLE = re.compile(r"[0-9A-Za-z㐀-䶿一-鿿぀-ヿ]")

# Without sentence-final punctuation the model misses its cue to stop and
# invents a syllable, so every segment gets one.
_TERMINATORS = "。！？；.!?;…"
_TRAILING_COMMA = "，、,"
_HAS_CJK = re.compile(rf"[{_CJK}]")
# Words, not letters: one 漢字 carries about as much text as one Latin word.
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def _is_chinese(text: str) -> bool:
    """Return whether the text reads as Chinese rather than Latin."""
    return len(_HAS_CJK.findall(text)) > len(_LATIN_WORD.findall(text))


def _terminate(segment: str) -> str:
    """Give a segment the sentence-final punctuation the model needs.

    The stop matches the segment's language, decided the same way the
    normaliser decides it.
    """
    if segment.endswith(tuple(_TERMINATORS)):
        return segment
    stop = "。" if _is_chinese(segment) else "."
    if segment.endswith(tuple(_TRAILING_COMMA)):
        return segment[:-1] + stop
    return segment + stop


@lru_cache(maxsize=1)
def _converter():
    """Return a cached OpenCC Traditional->Simplified converter.

    ``t2s`` is glyph-only. ``tw2sp`` would also rewrite vocabulary (設定 to
    设置), changing the words the model says.
    """
    from opencc import OpenCC

    return OpenCC("t2s")


def to_simplified(text: str) -> str:
    """Convert Traditional glyphs to Simplified, leaving Latin text alone."""
    return _converter().convert(text)


@dataclass(frozen=True)
class TextOptions:
    """Per-request switches for the text path.

    Attributes:
        normalize_text: Expand numbers, units, dates and clock literals.
        convert_script: Apply the Traditional->Simplified glyph conversion.
        normalize_options: Fine-grained normalisation switches.
    """

    normalize_text: bool = True
    convert_script: bool = True
    normalize_options: NormalizeOptions = NormalizeOptions()


DEFAULT_TEXT_OPTIONS = TextOptions()


def _split_long(chunk: str, limit: int) -> list[str]:
    """Break an over-long chunk on clause commas, then hard-wrap the rest.

    Wrapping leaves one character spare: every piece gains sentence-final
    punctuation afterwards, and the limit has to hold for what the model
    actually receives.
    """
    limit -= 1
    if len(chunk) <= limit:
        return [chunk]
    pieces: list[str] = []
    buffer = ""
    for clause in _CLAUSE_BREAK.split(chunk):
        if len(buffer) + len(clause) > limit and buffer:
            pieces.append(buffer)
            buffer = clause
        else:
            buffer += clause
    if buffer:
        pieces.append(buffer)
    # A single clause can still exceed the limit. Cut on a space so the break
    # does not land mid-word; Chinese has none, so the character cut is the floor.
    wrapped: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            cut = piece.rfind(" ", 1, limit + 1)
            if cut <= 0:
                cut = limit
            wrapped.append(piece[:cut].rstrip())
            piece = piece[cut:].lstrip()
        if piece:
            wrapped.append(piece)
    return wrapped


def _joiner(buffer: str, sentence: str) -> str:
    """Return the separator two sentences need when they share a segment.

    Pieces arrive stripped, so Latin text needs its space back. A space beside
    CJK would be read as a pause, so neither side may be CJK.
    """
    if not buffer or not sentence:
        return ""
    if _HAS_CJK.match(buffer[-1]) or _HAS_CJK.match(sentence[0]):
        return ""
    return " "


def segment(text: str, limit: int = MAX_CHARS_PER_SEGMENT) -> list[str]:
    """Split text into synthesis-sized segments on sentence boundaries.

    Args:
        text: Already-normalised text.
        limit: Maximum characters per segment.

    Returns:
        Non-empty segments in reading order. An empty input yields an empty
        list, which callers treat as "nothing to say".
    """
    segments: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(buffer) + len(sentence) > limit and buffer:
            segments.extend(_split_long(buffer, limit))
            buffer = sentence
        else:
            buffer += _joiner(buffer, sentence) + sentence
    if buffer:
        segments.extend(_split_long(buffer, limit))
    return [
        _terminate(s) for s in (seg.strip() for seg in segments) if _SPEAKABLE.search(s)
    ]


def prepare(text: str, options: TextOptions = DEFAULT_TEXT_OPTIONS) -> list[str]:
    """Run the full text path and return synthesis-ready segments.

    Args:
        text: Raw text as supplied by the caller.
        options: Which passes to apply.

    Returns:
        Segments ready to hand to an engine, in reading order.
    """
    prepared = text.strip()
    if not prepared:
        return []

    if options.normalize_text:
        # The caller sends flags, never a language, so the dominant script picks
        # which language the numbers are spelled in.
        if _is_chinese(prepared):
            prepared = normalize(prepared, options.normalize_options)
            # Only Chinese wants the gap closed; Latin spaces are word gaps.
            prepared = _CJK_GAP.sub("", prepared)
        else:
            prepared = english.normalize(prepared, options.normalize_options)
    if options.convert_script:
        prepared = to_simplified(prepared)

    segments = segment(prepared)
    _LOGGER.debug("text path: %r -> %d segment(s)", text, len(segments))
    return segments
