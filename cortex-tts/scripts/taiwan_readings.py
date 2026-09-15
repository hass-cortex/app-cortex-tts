# /// script
# requires-python = ">=3.12"
# dependencies = ["pypinyin>=0.53"]
# ///
"""Build the Taiwan-readings table the text pipeline substitutes from.

The models are trained on mainland speech, so a word Taiwan reads differently
(垃圾 lè sè, 企業 qì yè, 星期 xīng qí) comes out in the mainland reading. None of
the shipped models reads pinyin reliably, but every one of them reads an
ordinary character reliably — so the fix is a homophone: a character that has
exactly the Taiwan reading and no other, standing in for the one the model
would misread. This script decides which words need it and which stand-in each
gets; the pipeline only looks the words up.

Sources, all fetched from the McBopomofo repository (MIT):

- ``BPMFMappings.txt`` — 140k words with their Taiwan Bopomofo readings.
- ``BPMFBase.txt`` — every character's readings, Bopomofo beside pinyin.
- ``heterophony1.list`` — which reading a character defaults to.
- ``phrase.occ`` — corpus counts, to prefer common words and common stand-ins.

Against those, ``pypinyin`` supplies the mainland reading of the Simplified
form of each word (the pipeline converts script before this pass runs). A word
goes in the table when a syllable differs in more than tone sandhi (一/不) or
the neutral tone, and a stand-in character exists that both sides read one
way only. Run it from the app directory:

    uv run --with pypinyin python scripts/taiwan_readings.py

It writes ``src/cortex_speech/text/zh/taiwan_readings.tsv``.
"""

# pyright: reportMissingImports=false
from __future__ import annotations

import collections
import re
import sys
import urllib.request
from pathlib import Path

from pypinyin import Style, lazy_pinyin

# The pipeline's own conversion, not a bare OpenCC: it also writes every 著
# that is not zhù as 着, and a key made any other way could never match the
# text this table is looked up in.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortex_speech.text.zh.script import to_simplified  # noqa: E402

RAW = "https://raw.githubusercontent.com/openvanilla/McBopomofo/master/Source/Data/"
FILES = ("BPMFMappings.txt", "BPMFBase.txt", "heterophony1.list", "phrase.occ")
OUT = (
    Path(__file__).resolve().parents[1]
    / "src/cortex_speech/text/zh/taiwan_readings.tsv"
)

# A word the corpus never saw is as likely a dictionary artefact as a word.
MIN_WORD_OCCURRENCES = 1
# A stand-in the model has rarely seen is a stand-in it may not read cleanly.
MIN_STANDIN_OCCURRENCES = 100

_HAN = re.compile(r"^[一-鿿]+$")
# A dictionary entry that is a word plus a particle (長的, 倒了, 划著) is a
# fragment whose reading depends on the sentence around it, not a word.
_FRAGMENT = re.compile(r"[的了著地得過]$")
_PINYIN = re.compile(r"^[a-z]+[1-5]?$")

_t2s = to_simplified


def fetch(name: str, cache: Path) -> list[str]:
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RAW + name) as res:  # noqa: S310 - fixed host
            path.write_bytes(res.read())
    return path.read_text(encoding="utf-8").splitlines()


def canonical(syllable: str) -> str:
    """One spelling for ü, and an explicit first tone."""
    s = syllable.replace("ü", "v").replace("lue", "lve").replace("nue", "nve")
    return s if s[-1].isdigit() else s + "1"


def mainland(word: str) -> list[str]:
    return [
        canonical(s)
        for s in lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
    ]


def main() -> int:
    cache = Path(__file__).resolve().parent / ".cache" / "mcbopomofo"
    lines = {name: fetch(name, cache) for name in FILES}

    # Bopomofo syllable -> pinyin, and each character's Taiwan readings.
    bopomofo: dict[str, str] = {}
    taiwan_readings: dict[str, set[str]] = collections.defaultdict(set)
    for line in lines["BPMFBase.txt"]:
        parts = line.split()
        if len(parts) < 3 or not _PINYIN.match(parts[2]):
            continue
        char, zhuyin, py = parts[0], parts[1], canonical(parts[2])
        bopomofo.setdefault(zhuyin, py)
        taiwan_readings[char].add(py)

    def to_pinyin(zhuyin: str) -> str | None:
        if zhuyin in bopomofo:
            return bopomofo[zhuyin]
        if zhuyin.startswith("˙") and zhuyin[1:] in bopomofo:
            return bopomofo[zhuyin[1:]][:-1] + "5"
        return None

    preferred: dict[str, str] = {}
    for line in lines["heterophony1.list"]:
        parts = line.split()
        if len(parts) == 2 and (py := to_pinyin(parts[1])):
            preferred[parts[0]] = py

    occurrences: dict[str, int] = {}
    for line in lines["phrase.occ"]:
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            occurrences[parts[0]] = int(parts[1])

    words: dict[str, set[tuple[str, ...]]] = collections.defaultdict(set)
    for line in lines["BPMFMappings.txt"]:
        parts = line.split()
        if len(parts) < 2 or not _HAN.match(parts[0]) or _FRAGMENT.search(parts[0]):
            continue
        reading = tuple(to_pinyin(z) or "" for z in parts[1:])
        if "" not in reading and len(reading) == len(parts[0]):
            words[parts[0]].add(reading)

    # A stand-in for a syllable: a character both sides read that way by
    # default — the reading Taiwan lists first for it, and the one pypinyin
    # picks with no context — with a one-character Simplified form. Rarer
    # readings are allowed to exist: a stand-in lands in no word, so the model
    # has no context to pick anything but the default.
    standins: dict[str, str] = {}
    candidates: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for char, readings in taiwan_readings.items():
        reading = preferred.get(char) if len(readings) > 1 else next(iter(readings))
        simplified = _t2s(char)
        if reading is None or len(simplified) != 1 or mainland(simplified) != [reading]:
            continue
        count = occurrences.get(char, 0)
        if count >= MIN_STANDIN_OCCURRENCES:
            candidates[reading].append((count, simplified))
    for reading, options in candidates.items():
        standins[reading] = max(options)[1]

    table: dict[str, tuple[str, str, str]] = {}
    skipped = collections.Counter()
    for word, readings in words.items():
        count = occurrences.get(word, 0)
        if count < MIN_WORD_OCCURRENCES:
            skipped["rare"] += 1
            continue
        simplified = _t2s(word)
        if len(simplified) != len(word):
            skipped["script"] += 1
            continue
        cn = mainland(simplified)
        if len(cn) != len(word):
            skipped["script"] += 1
            continue

        # Several readings: keep the one the character defaults to, if that
        # picks exactly one; otherwise the choice is contextual and not ours.
        if len(readings) > 1:
            fits = [
                r
                for r in readings
                if all(preferred.get(c, s) == s for c, s in zip(word, r, strict=True))
            ]
            if len(fits) != 1:
                skipped["ambiguous"] += 1
                continue
            readings = {fits[0]}
        (tw,) = readings

        differing = [
            i
            for i, (a, b) in enumerate(zip(tw, cn, strict=True))
            if a != b and a[-1] != "5" and b[-1] != "5" and word[i] not in "一不"
        ]
        if not differing:
            continue
        # The whole word, or nothing: a half-corrected word helps no one.
        if any(tw[i] not in standins for i in differing):
            skipped["no stand-in"] += 1
            continue
        chars = list(simplified)
        for i in differing:
            chars[i] = standins[tw[i]]
        replacement = "".join(chars)
        if replacement == simplified:
            continue
        table[simplified] = (replacement, " ".join(tw), " ".join(cn))

    OUT.write_text(
        "# Generated by scripts/taiwan_readings.py from McBopomofo (MIT).\n"
        "# word\tstand-in\tTaiwan reading\tmainland reading\n"
        + "".join(
            f"{word}\t{rep}\t{tw}\t{cn}\n"
            for word, (rep, tw, cn) in sorted(table.items())
        ),
        encoding="utf-8",
    )
    print(f"{len(table)} words -> {OUT}", file=sys.stderr)
    print(f"skipped: {dict(skipped)}", file=sys.stderr)
    print(f"stand-ins for {len(standins)} syllables", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
