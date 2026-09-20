"""What this host measured, kept so a model card and a live reply can read it.

Why the catalog carries no figure of its own is in AGENTS.md, under "A
real-time factor belongs to a host, not to a model". What lives here:

One primitive, recorded once per request: a `RenderSample` of raw audio and
wall seconds and the execution provider it ran on. What a caller gets back is
the median real-time factor of a voice's most recent requests on the provider
now in use — one number per model and voice, never pooled across voices and
never borrowed from another, because a clone and a built-in voice on the same
model are different work.

Stored beside the models rather than in the settings file: settings are what a
user chose and stats are what the app observed, and a "reset settings" must
not erase measurements that took real work to gather.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path

from cortex_speech import RenderSample, write_json

_LOGGER = logging.getLogger(__name__)

FILE_NAME = "stats.json"

# The file's layout. Anything older kept samples without a provider, which
# cannot be told from a run on another machine, so it is not carried forward.
FORMAT = 2

# How many recent requests are kept per voice. More than the median reads, so
# a run of one length cannot be all there is once ordinary replies return.
MAX_RENDERS = 24

# How many of the most recent same-provider requests the median is taken over.
# Small enough to follow a host that changed within a few replies.
WINDOW = 8

# Fewer than this and there is no median worth acting on.
MIN_SAMPLES = 3


def split_key(key: str) -> tuple[str, str]:
    """A bucket back into its kind and voice.

    The other half of `StatsStore._key`, kept beside it so the two cannot
    drift: a card reads what the store wrote.
    """
    kind, _, voice = key.partition(":")
    return kind, voice


@dataclass(frozen=True)
class Measured:
    """One voice's real-time factor on this host.

    Attributes:
        kind: The `Voice.source` these were measured with — `builtin`,
            `designed` or `reference`.
        voice: Which voice.
        rtf: The median of `samples` requests' render-over-audio seconds.
        samples: How many requests the median rests on.
        provider: The execution provider they ran on.
        settled: Whether `samples` reaches `MIN_SAMPLES` — before that the
            figure is shown but decides nothing.
    """

    kind: str
    voice: str
    rtf: float
    samples: int
    provider: str
    settled: bool


class StatsStore:
    """Per-model measurements, in memory and on disk.

    Every method is synchronous, and recording one writes the file, so the API
    runs it on a worker thread rather than stalling the event loop mid-reply.
    That makes a record-and-write reachable from more than one thread, which
    the lock is for.
    """

    def __init__(self, path: Path) -> None:
        """Load what was measured before, if anything."""
        self._path = path
        # model id -> "kind:voice" -> requests as they went.
        self._renders: dict[str, dict[str, list[RenderSample]]] = {}
        self._lock = threading.Lock()
        self._read()

    def _read(self) -> None:
        if not self._path.is_file():
            return
        try:
            stored = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            # Measurements are not worth failing a start over.
            _LOGGER.warning("stats unreadable, starting empty: %s", err)
            return
        if not isinstance(stored, dict) or stored.get("format") != FORMAT:
            return
        models = stored.get("models")
        if not isinstance(models, dict):
            return
        for model_id, by_key in models.items():
            if not isinstance(by_key, dict):
                continue
            for key, entry in by_key.items():
                renders = entry.get("renders") if isinstance(entry, dict) else None
                samples = [
                    RenderSample(float(r[0]), float(r[1]), str(r[2]))
                    for r in renders or []
                    if isinstance(r, list) and len(r) == 3
                ]
                if samples:
                    self._renders.setdefault(str(model_id), {})[str(key)] = samples[
                        -MAX_RENDERS:
                    ]

    def _write(self) -> None:
        """Write the file. Callers hold `_lock`."""
        stored = {
            "format": FORMAT,
            "models": {
                model_id: {
                    key: {
                        "renders": [[r.audio_s, r.wall_s, r.provider] for r in renders]
                    }
                    for key, renders in by_key.items()
                }
                for model_id, by_key in self._renders.items()
            },
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self._path, stored, indent=2, sort_keys=True)
        except OSError as err:
            # The measurement still counts for this process; only the memory
            # of it across a restart is lost, which is not worth an error to
            # the caller who only asked for audio.
            _LOGGER.warning("could not write stats: %s", err)

    @staticmethod
    def _key(kind: str, voice_id: str) -> str:
        return f"{kind}:{voice_id}"

    def record(
        self, model_id: str, kind: str, voice_id: str, sample: RenderSample
    ) -> None:
        """Note how one request went.

        Every request counts, short ones included: a short sentence is what a
        streaming reply is made of, and its cost is what the median has to
        describe.
        """
        if sample.audio_s <= 0 or sample.wall_s <= 0:
            return
        with self._lock:
            renders = self._renders.setdefault(model_id, {}).setdefault(
                self._key(kind, voice_id), []
            )
            renders.append(
                RenderSample(
                    round(sample.audio_s, 3), round(sample.wall_s, 3), sample.provider
                )
            )
            del renders[:-MAX_RENDERS]
            self._write()

    @staticmethod
    def _measure(
        key: str, samples: list[RenderSample], provider: str | None
    ) -> Measured | None:
        """The median over the newest `WINDOW` samples on this provider,
        or `None` when the voice has not run on it."""
        if provider is None:
            if not samples:
                return None
            provider = samples[-1].provider
        matching = [s for s in samples if s.provider == provider][-WINDOW:]
        if not matching:
            return None
        kind, voice = split_key(key)
        return Measured(
            kind=kind,
            voice=voice,
            rtf=round(statistics.median(s.rtf for s in matching), 3),
            samples=len(matching),
            provider=provider,
            settled=len(matching) >= MIN_SAMPLES,
        )

    def rtf(
        self, model_id: str, kind: str, voice_id: str, provider: str
    ) -> Measured | None:
        """What this voice measured on `provider`, or `None` until it is
        settled: a verdict rests on `MIN_SAMPLES` requests, never fewer."""
        key = self._key(kind, voice_id)
        with self._lock:
            samples = list((self._renders.get(model_id) or {}).get(key) or [])
        measured = self._measure(key, samples, provider)
        return measured if measured and measured.settled else None

    def get(self, model_id: str, provider: str | None = None) -> list[Measured]:
        """What this host measured for a model, one entry per voice that
        has run, settled or not — a voice spoken once is a voice with a cost.

        `provider` is the one the model is running on now; a model not
        resident is read on whichever provider each voice last ran on. Ordered
        so a card renders the same twice.
        """
        with self._lock:
            by_key = {
                key: list(samples)
                for key, samples in sorted((self._renders.get(model_id) or {}).items())
            }
        return [
            measured
            for key, samples in by_key.items()
            if (measured := self._measure(key, samples, provider)) is not None
        ]

    def clear(self) -> None:
        """Drop every model's measurements."""
        with self._lock:
            if self._renders:
                self._renders = {}
                self._write()

    def forget(self, model_id: str) -> None:
        """Drop a model's measurements, when its weights are deleted.

        Keeping them would carry a figure across a delete and a re-download,
        which is usually right — but the reason to delete a model is often
        that something about it changed.
        """
        with self._lock:
            if self._renders.pop(model_id, None) is not None:
                self._write()
