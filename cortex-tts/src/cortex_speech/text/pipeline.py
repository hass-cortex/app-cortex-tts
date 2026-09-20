"""The text path every synthesis request takes.

The request's language tag picks the locale; the text is only sniffed when
there is no tag. Within a locale the order is not negotiable: normalisation
emits words in the language's own script (二十六點五度 is Traditional), so it
runs before any rewrite that makes those glyphs pronounceable, and the
rewrites run in the order the locale lists them — Chinese converts script
before it respells readings, because the readings table is keyed by the
Simplified form.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from . import en, generic, zh
from .locales import LOCALES, Locale, primary, register
from .options import NormalizeOptions
from .scripts import CLAUSE_BREAK, CLAUSE_MARKS, ENDINGS, SENTENCE_BREAK

_LOGGER = logging.getLogger(__name__)

register(zh.LOCALE)
register(en.LOCALE)


_CJK = r"㐀-䶿一-鿿豈-﫿　-〿＀-￯"
# The scripts that write no space between words: 漢字 and kana. Expanding
# "26.5°C" into 攝氏二十六點五度 leaves the space that preceded the digits
# stranded between two of them, and the model reads a space as a pause.
# Spaces bordering Latin text are kept — they are real word gaps.
_UNSPACED = r"㐀-䶿一-鿿豈-﫿぀-ヿ"
_CJK_GAP = re.compile(rf"(?<=[{_UNSPACED}])[ \t]+(?=[{_UNSPACED}])")

# A segment of nothing but punctuation has no sound. Passed to the model it
# generates no audio tokens and surfaces as an opaque engine error, so it is
# dropped here and the caller gets a clean "nothing to say" instead. A letter
# or digit of any script is sound.
_SPEAKABLE = re.compile(r"[^\W_]")

_HAS_CJK = re.compile(rf"[{_CJK}]")
# What votes for Chinese: 漢字 and kana. `_CJK` also spans fullwidth
# punctuation, which must not count — "26.5°C。" is a reading, not a script.
_HAN_OR_KANA = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ]")
_KANA = re.compile(r"[぀-ヿ]")
# Words, not letters: one 漢字 carries about as much text as one Latin word.
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def is_chinese(text: str) -> bool:
    """Return whether the text reads as Chinese rather than Latin.

    With no word of either kind to count, a Chinese stop is the only hint
    left: "80%。" was written in a Chinese sentence.
    """
    han = len(_HAN_OR_KANA.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    if han or latin:
        return han > latin
    return _HAS_CJK.search(text) is not None


def sniff_language(text: str) -> str:
    """The most specific tag the text itself supports, for a request with none.

    Kana is Japanese. 漢字 outnumbering Latin words is Chinese, and Chinese
    written in Traditional glyphs is tagged as such, because that is what
    tells the Taiwan readings to run. Everything else is read as English,
    which is what a Latin-script text with no tag has always meant here.
    """
    if _KANA.search(text):
        return "ja"
    if is_chinese(text):
        return "zh-Hant" if zh.is_traditional(text) else "zh"
    return "en"


def resolve_language(text: str, language: str | None) -> str:
    """The tag the pipeline reads the text as: the request's, else sniffed.

    A bare ``zh`` says only that the text is Chinese; which Chinese, the text
    itself still answers — written in Traditional glyphs it is read as
    ``zh-Hant``, so Taiwan readings follow the script the writer used.
    """
    # BCP 47 hyphens; a POSIX-style zh_TW is the same tag misspelt.
    tag = language.strip().replace("_", "-") if language else ""
    if not tag:
        return sniff_language(text)
    if tag.lower() == "zh" and zh.is_traditional(text):
        return "zh-Hant"
    return tag


def locale_for(tag: str) -> Locale:
    """The locale for a tag: one written for its language, else the generic one."""
    code = primary(tag)
    return LOCALES.get(code) or generic.locale(code)


@dataclass(frozen=True)
class TextOptions:
    """Per-request switches for the text path.

    Attributes:
        normalize_text: Expand numbers, units, dates and clock literals.
        expand_numbers: Read a bare number — no unit, clock or date around
            it — as a quantity too. ``None`` lets the model decide: on for
            one that cannot say a digit, off otherwise.
        convert_script: Chinese only — the Traditional->Simplified glyph
            conversion. ``None`` lets the locale decide (it is always on).
        taiwan_readings: Chinese only — respell words Taiwan reads
            differently. ``None`` lets the locale decide: on for a Taiwanese
            tag, off otherwise.
        normalize_options: Fine-grained normalisation switches.
    """

    normalize_text: bool = True
    expand_numbers: bool | None = None
    convert_script: bool | None = None
    taiwan_readings: bool | None = None
    normalize_options: NormalizeOptions = NormalizeOptions()

    def wants(self, name: str) -> bool | None:
        """The caller's answer for a rewrite, or ``None`` for "you decide"."""
        return getattr(self, name)


DEFAULT_TEXT_OPTIONS = TextOptions()


@dataclass(frozen=True)
class TextPlan:
    """What the pipeline decided for one text, before running it.

    Attributes:
        language: The tag the text is read as, resolved.
        normalize_text: Whether the fixed shapes — units, clock, date — are
            expanded.
        expand_numbers: Whether a bare number is read as a quantity too.
        rewrites: Each rewrite the locale has, and whether it runs — only
            those; a language without script conversion has no entry for it.
        misreads: Words the model declared it misreads; the locale respells
            them if it knows how, after every rewrite.
    """

    language: str
    normalize_text: bool
    expand_numbers: bool
    rewrites: dict[str, bool] = field(default_factory=dict)
    misreads: tuple[str, ...] = ()

    @property
    def locale(self) -> Locale:
        return locale_for(self.language)

    @property
    def passes(self) -> dict[str, bool]:
        """Every switch this language has, and whether it runs."""
        return {
            "normalize_text": self.normalize_text,
            "expand_numbers": self.normalize_text and self.expand_numbers,
            **self.rewrites,
        }


def plan(
    text: str,
    options: TextOptions = DEFAULT_TEXT_OPTIONS,
    language: str | None = None,
    *,
    reads_numerals: bool = False,
    needs_number_words: bool = False,
    misreads: tuple[str, ...] = (),
) -> TextPlan:
    """Decide the locale and the passes for a text without running them.

    Args:
        text: The text, stripped.
        options: The caller's switches.
        language: The request's language tag, or ``None`` to sniff.
        reads_numerals: Whether the model reads digits and unit symbols
            itself. That beats the generic locale's numbers-and-unit-names,
            so normalisation is skipped there; a written locale is kept,
            because it was measured against the model and won.
        needs_number_words: Whether the model cannot say a digit at all, so
            a bare number is expanded unless the caller said otherwise.
        misreads: Words the model reads with the wrong character, which the
            locale respells with a stand-in it reads right.
    """
    tag = resolve_language(text, language)
    locale = locale_for(tag)
    normalize = options.normalize_text and (locale.written or not reads_numerals)
    wanted = options.expand_numbers
    expand = needs_number_words if wanted is None else wanted
    rewrites: dict[str, bool] = {}
    for rewrite in locale.rewrites:
        wanted = options.wants(rewrite.name)
        on = rewrite.default(tag) if wanted is None else wanted
        rewrites[rewrite.name] = on and all(
            rewrites.get(r, False) for r in rewrite.requires
        )
    return TextPlan(tag, normalize, expand, rewrites, misreads)


def _terminate(segment: str, stop: str) -> str:
    """Give a segment the sentence-final punctuation the model needs."""
    if segment.endswith(tuple(ENDINGS)):
        return segment
    if segment.endswith(tuple(CLAUSE_MARKS)):
        return segment[:-1] + stop
    return segment + stop


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
    for clause in CLAUSE_BREAK.split(chunk):
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


def prepared_text(segments: list[str]) -> str:
    """Join prepared segments back into one string, for display and storage.

    Segments arrive stripped, so two Latin ones need their space back; a
    space beside CJK would be read as a pause, so those are joined bare —
    the same rule `segment` used to split them.
    """
    out = ""
    for piece in segments:
        out += _joiner(out, piece) + piece
    return out


def segment(text: str, limit: int | None = None, stop: str | None = None) -> list[str]:
    """Split text into synthesis-sized segments on sentence boundaries.

    Args:
        text: Already-normalised text.
        limit: Maximum characters per segment, or `None` for no bound of
            that kind — the text is then split on sentence boundaries only.
        stop: The sentence-final punctuation to add where a segment has
            none; the locale's when it is known, else the sniffed script's.

    Returns:
        Non-empty segments in reading order. An empty input yields an empty
        list, which callers treat as "nothing to say".
    """
    final = stop if stop is not None else locale_for(sniff_language(text)).stop
    segments: list[str] = []
    buffer = ""
    for sentence in SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if limit is not None and len(buffer) + len(sentence) > limit and buffer:
            segments.extend(_split_long(buffer, limit))
            buffer = sentence
        else:
            buffer += _joiner(buffer, sentence) + sentence
    if buffer:
        segments.extend(_split_long(buffer, limit) if limit is not None else [buffer])
    return [
        _terminate(s, final)
        for s in (seg.strip() for seg in segments)
        if _SPEAKABLE.search(s)
    ]


def run(text: str, decided: TextPlan, options: NormalizeOptions) -> str:
    """Run a plan over the text it was made for; the rewritten text back."""
    locale = decided.locale
    if decided.normalize_text:
        text = locale.normalize(
            text, replace(options, expand_numbers=decided.expand_numbers)
        )
        if locale.close_gaps:
            text = _CJK_GAP.sub("", text)
    for rewrite in locale.rewrites:
        if decided.rewrites[rewrite.name]:
            text = rewrite.apply(text)
    if decided.misreads and locale.misreads is not None:
        text = locale.misreads(text, decided.misreads)
    return text


def prepare_text(
    text: str,
    options: TextOptions = DEFAULT_TEXT_OPTIONS,
    language: str | None = None,
    *,
    reads_numerals: bool = False,
    needs_number_words: bool = False,
    misreads: tuple[str, ...] = (),
) -> str:
    """Run the rewriting passes and return the text, not yet segmented."""
    text = text.strip()
    if not text:
        return ""
    decided = plan(
        text,
        options,
        language,
        reads_numerals=reads_numerals,
        needs_number_words=needs_number_words,
        misreads=misreads,
    )
    return run(text, decided, options.normalize_options)


def prepare(
    text: str,
    options: TextOptions = DEFAULT_TEXT_OPTIONS,
    language: str | None = None,
    *,
    reads_numerals: bool = False,
    needs_number_words: bool = False,
    misreads: tuple[str, ...] = (),
    limit: Callable[[str], int | None] | None = None,
) -> list[str]:
    """Run the full text path and return synthesis-ready segments.

    Args:
        text: Raw text as supplied by the caller.
        options: Which passes to apply.
        language: The request's language tag, or ``None`` to sniff the text.
        reads_numerals: Whether the model reads digits itself; see `plan`.
        needs_number_words: Whether it cannot say a digit at all; see `plan`.
        misreads: Words it reads with the wrong character; see `plan`.
        limit: Asked for the characters one segment may carry, given the
            *prepared* text — `ModelSpec.segment_limit`. A function rather
            than a number because the answer depends on the script, which
            normalisation can change: a number would have to be read off the
            raw text, and a preview computed one way beside a synthesis
            computed the other is two different cuts of the same reply.
            `None` leaves the text split on sentence boundaries alone.

    Returns:
        Segments ready to hand to an engine, in reading order.
    """
    text = text.strip()
    if not text:
        return []
    decided = plan(
        text,
        options,
        language,
        reads_numerals=reads_numerals,
        needs_number_words=needs_number_words,
        misreads=misreads,
    )
    prepared = run(text, decided, options.normalize_options)
    segments = segment(
        prepared,
        limit=limit(prepared) if limit is not None else None,
        stop=decided.locale.stop,
    )
    _LOGGER.debug("text path: %r -> %d segment(s)", text, len(segments))
    return segments
