"""Replay recorded requests and real replies through the pacing rules.

The regression for `STREAM_RTF` and `BANK_S` (`cortex_speech/pacing/release.py`):
a change to either comes with this script's output, and a new host's samples
are added to its inputs. Nothing in the repo carries the data — it is replies
people said to their assistant and measurements of their machines — so every
input is a path.

Inputs, each repeatable with a host label:

    --stats  gpu=/path/stats.json      the app's own store (raw RenderSamples)
    --log    ha=/path/addon.log        an addon log: `rendered` frames, and the
                                       `spoke live …` line each reply ends with
    --bench  ha=/path/bench.json       what scripts/bench_rtf.py wrote
    --replies /path/replies.json       [{"text": …}, …] — real reply texts

What it prints:

1. Per host × model × voice: the median RTF over the newest window, the
   verdict, and — where the log carried `spoke live` lines — what the app
   itself decided and the lowest lead it saw, so the prediction can be held
   against production.
2. Per streaming cell, over the multi-sentence replies: median first word and
   worst gap under the bank against releasing each sentence unheld, so the
   bank's cost and what it buys are both on the table.

    uv run python scripts/replay_pacing.py --stats gpu=stats.json \\
        --log ha=addon.log --replies replies.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

from cortex_speech.pacing.model import RenderSample
from cortex_speech.pacing.release import BANK_S, STREAM_RTF, bank_needed, verdict
from cortex_speech.text.scripts import SENTENCE_BREAK, spoken_seconds
from cortex_tts.stats import MIN_SAMPLES, WINDOW


@dataclass
class Spoken:
    """One reply as the app reported it in its `spoke live` line."""

    verdict: str
    outcome: str
    min_lead_s: float | None


@dataclass
class Cell:
    host: str
    model: str
    voice: str
    samples: list[RenderSample]
    spoken: list[Spoken] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.host} {self.model}/{self.voice}"

    def median(self) -> float | None:
        recent = self.samples[-WINDOW:]
        if len(recent) < MIN_SAMPLES:
            return None
        return st.median(s.rtf for s in recent)

    def line(self) -> tuple[float, float]:
        """(fixed_s, per_audio) by least squares, fixed cost clamped at zero."""
        xs = [s.audio_s for s in self.samples]
        ys = [s.wall_s for s in self.samples]
        n = len(xs)
        if n < 2 or max(xs) - min(xs) < 0.5:
            return 0.0, st.median(s.rtf for s in self.samples)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
        fixed = my - slope * mx
        if fixed < 0:
            return 0.0, sum(ys) / sum(xs)
        return fixed, slope


# --- loaders -----------------------------------------------------------------


def load_stats(host: str, path: Path) -> list[Cell]:
    raw = json.loads(path.read_text())
    models = raw.get("models", raw) if isinstance(raw, dict) else {}
    cells = []
    for model, voices in models.items():
        for voice, entry in voices.items():
            samples = [
                RenderSample(float(r[0]), float(r[1]), str(r[2]) if len(r) > 2 else "")
                for r in entry["renders"]
            ]
            if samples:
                cells.append(Cell(host, model, voice, samples))
    return cells


# The `rendered` frame as uvicorn's frame trace prints it at LOG_LEVEL=debug
# (`uvicorn.error: > TEXT '{...}'`); Starlette writes frames without spaces.
_RENDERED = re.compile(
    r'"type":"rendered","index":\d+,"audio_s":([\d.]+),"render_ms":([\d.]+)'
)
_SPOKE_NEW = re.compile(
    r"spoke live model=(\S+) voice=(\S+) setting=\S+ verdict=(\S+) .*?"
    r"outcome=(\S+) .*?min_lead_s=(\S+)"
)
_SPOKE_OLD = re.compile(r"spoke live as ([\w.\-]+)/([\w\-]+) ->")


def load_log(host: str, path: Path) -> list[Cell]:
    """Requests and outcomes, attributed to the `spoke live` line that ends each reply."""
    pending: list[RenderSample] = []
    cells: dict[tuple[str, str], Cell] = {}
    for line in path.read_text(errors="replace").splitlines():
        if m := _RENDERED.search(line):
            # The log names no provider; a cell is one host's anyway.
            pending.append(
                RenderSample(float(m.group(1)), float(m.group(2)) / 1000, "")
            )
            continue
        if m := _SPOKE_NEW.search(line):
            model, voice = m.group(1), m.group(2)
            lead = None if m.group(5) == "-" else float(m.group(5))
            spoken = Spoken(m.group(3), m.group(4), lead)
        elif m := _SPOKE_OLD.search(line):
            model, voice, spoken = m.group(1), m.group(2), None
        else:
            continue
        cell = cells.setdefault((model, voice), Cell(host, model, voice, []))
        cell.samples.extend(pending)
        if spoken:
            cell.spoken.append(spoken)
        pending = []
    return list(cells.values())


def load_bench(host: str, path: Path) -> list[Cell]:
    raw = json.loads(path.read_text())
    return [
        Cell(
            host,
            model,
            "bench",
            [
                RenderSample(v["audio_s"], v["audio_s"] * v["rtf_median"], "")
                for v in lengths.values()
            ],
        )
        for model, lengths in raw.items()
    ]


# --- the replay ------------------------------------------------------------------


def sentences(text: str) -> list[float]:
    """Audio seconds per sentence, from the app's own splitter and rate priors."""
    out = []
    for s in SENTENCE_BREAK.split(text):
        s = s.strip()
        if s:
            out.append(spoken_seconds(s))
    return out


def simulate(
    audio: list[float], fixed: float, slope: float, bank: float, in_hand: bool = False
) -> tuple[float, float]:
    """One sentence per request, rendered back to back from t=0.

    `in_hand` is the app's rule once the writer has finished: the bank is
    capped at what the rest of the reply needs (`bank_needed`), never more
    than `bank`.

    Returns (first sound, worst gap): a gap is silence between the end of one
    sentence's playback and the moment the next may start.
    """
    ends: list[float] = []
    t = 0.0
    for a in audio:
        t += fixed + slope * a
        ends.append(t)
    gate = ends[-1]
    acc = 0.0
    for i, a in enumerate(audio):
        acc += a
        hold = min(bank, bank_needed(slope, audio[i + 1 :])) if in_hand else bank
        if acc >= hold:
            gate = ends[i]
            break
    ready = [max(gate, e) for e in ends]
    play, worst, prev_end = ready[0], 0.0, 0.0
    for i, a in enumerate(audio):
        play = max(play, ready[i])
        if i:
            worst = max(worst, ready[i] - prev_end)
        prev_end = play + a
        play = prev_end
    return ready[0], worst


def table_verdicts(cells: list[Cell]) -> None:
    print(f"## Verdicts — median of the newest {WINDOW}, threshold {STREAM_RTF}\n")
    print("| cell | n | median | verdict | app said | replies | dry | lowest lead |")
    print("|---|---|---|---|---|---|---|---|")
    for c in cells:
        med = c.median()
        said = "/".join(sorted({s.verdict for s in c.spoken})) or "-"
        leads = [s.min_lead_s for s in c.spoken if s.min_lead_s is not None]
        dry = sum(1 for lead in leads if lead < 0) if leads else "-"
        lowest = f"{min(leads):.2f}" if leads else "-"
        median = f"{med:.2f}" if med is not None else "-"
        print(
            f"| {c.name} | {len(c.samples)} | {median} | {verdict(med)} | {said} |"
            f" {len(c.spoken) or '-'} | {dry} | {lowest} |"
        )


def table_bank(cells: list[Cell], replies: list[list[float]]) -> None:
    multi = [r for r in replies if len(r) > 1]
    print(
        f"\n## Bank {BANK_S} s against unheld — {len(multi)} multi-sentence replies\n"
    )
    print(
        "| cell | line | rule | median first word | worst gap | replies with gap > 0.3 s |"
    )
    print("|---|---|---|---|---|---|")
    for c in cells:
        if verdict(c.median()) != "streaming":
            continue
        fixed, slope = c.line()
        rules = (
            ("unheld", 0.0, False),
            (f"bank {BANK_S:g} s", BANK_S, False),
            (f"bank ≤{BANK_S:g} s, text in hand", BANK_S, True),
        )
        for rule, bank, in_hand in rules:
            res = [simulate(r, fixed, slope, bank, in_hand) for r in multi]
            first = st.median(x[0] for x in res)
            gaps = [x[1] for x in res]
            bad = sum(g > 0.3 for g in gaps)
            print(
                f"| {c.name} | {fixed:.2f}+{slope:.2f}x | {rule} | {first:.1f} s |"
                f" {max(gaps):.1f} s | {bad}/{len(gaps)} |"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stats", action="append", default=[], metavar="HOST=PATH")
    ap.add_argument("--log", action="append", default=[], metavar="HOST=PATH")
    ap.add_argument("--bench", action="append", default=[], metavar="HOST=PATH")
    ap.add_argument("--replies", type=Path, help='JSON list of {"text": …}')
    args = ap.parse_args()

    cells: list[Cell] = []
    for loader, specs in (
        (load_stats, args.stats),
        (load_log, args.log),
        (load_bench, args.bench),
    ):
        for spec in specs:
            host, _, path = spec.partition("=")
            cells.extend(loader(host, Path(path)))
    cells = [c for c in cells if len(c.samples) >= MIN_SAMPLES or c.spoken]
    if not cells:
        ap.error("no cells: pass at least one --stats, --log or --bench")
    table_verdicts(cells)
    if args.replies:
        texts = [r["text"] for r in json.loads(args.replies.read_text())]
        replies = [s for s in (sentences(t) for t in texts) if s]
        table_bank(cells, replies)


if __name__ == "__main__":
    main()
