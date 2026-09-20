"""What the pipeline knows per language.

A language is a `Locale`: how its numbers are read, which stop ends a
sentence, and the rewrites only it needs — Chinese converts script and
respells Taiwan readings; nothing else does. The pipeline picks the locale
from the request's language tag, so adding a language is adding an entry
here, never a branch in the pipeline.

`pipeline` registers the written locales when it is imported, so the table
is complete by the time a request reads it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .options import NormalizeOptions


@dataclass(frozen=True)
class Rewrite:
    """A pass one language has and the others do not.

    Attributes:
        name: The switch it answers to — a `TextOptions` field, and the name
            the API and the UI show. ``None`` there means `default` decides.
        apply: The rewrite itself.
        default: Whether it is on for a given full language tag, when the
            request did not say. Taiwan readings are on for ``zh-TW`` and off
            for ``zh-CN``; both are `zh`.
        requires: Rewrites this one is keyed on; with any of them off it is
            off too, whatever was asked. Taiwan readings match Simplified
            words, so without script conversion they would match nothing
            and still be reported as running.
    """

    name: str
    apply: Callable[[str], str]
    default: Callable[[str], bool]
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class Locale:
    """One language's text path.

    Attributes:
        code: The primary language subtag it serves (``zh``, ``en``).
        normalize: Expands numbers, units, dates and clock literals into the
            language's words.
        stop: The sentence-final punctuation a segment must end in.
        close_gaps: Whether a space between two ideographs is a leftover of
            expansion, not a word gap, and is removed. True for the scripts
            that write no spaces between words.
        rewrites: The language's own passes, run after normalisation in this
            order.
    """

    code: str
    normalize: Callable[[str, NormalizeOptions], str]
    stop: str = "."
    close_gaps: bool = False
    rewrites: tuple[Rewrite, ...] = ()
    written: bool = True
    """Whether someone wrote this locale — its own words for units, dates and
    clock literals — or it is the generic one, which reads the fixed shapes
    Home Assistant emits with num2words and CLDR and nothing else. A model
    that reads numerals itself beats the generic locale, and not a written
    one."""
    misreads: Callable[[str, tuple[str, ...]], str] | None = None
    """Respell the words a model misreads, given which words those are.

    Which words is the model's to declare (`ModelSpec.misreads`); how to
    respell one is the language's. A locale without this ignores the
    declaration."""

    def rewrite(self, name: str) -> Rewrite | None:
        return next((r for r in self.rewrites if r.name == name), None)


LOCALES: dict[str, Locale] = {}


def register(locale: Locale) -> None:
    LOCALES[locale.code] = locale


def primary(tag: str) -> str:
    """The language subtag of a BCP-47 tag: ``zh-Hant-TW`` -> ``zh``."""
    return tag.split("-", 1)[0].lower()
