"""What this host measured, kept so a model card can stop guessing.

Why the catalog carries no figure of its own is in AGENTS.md, under "A
real-time factor belongs to a host, not to a model" — including the
measurements that settled it. Two decisions live here rather than there:

Kept per *voice kind*, because what a model costs depends on how it was asked
to speak, so one figure averaging the kinds would describe neither.

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

from cortex_speech import write_json

_LOGGER = logging.getLogger(__name__)

FILE_NAME = "stats.json"

# How many recent syntheses a model's figure is drawn from. Enough that one
# odd reply cannot define it, few enough that it follows a host that changed —
# a thread count, an execution provider, a busier machine.
MAX_SAMPLES = 12

# Shorter replies than this are measured but not recorded. Every model pays a
# fixed cost per synthesis, so a half-second clip reports an RTF dominated by
# it: measured on OmniVoice, the ten-character sentence came in at 0.99 where
# the other three sat at 0.72-0.80. A card that quoted the short one would
# overstate what a real reply costs.
MIN_AUDIO_SECONDS = 1.0


@dataclass(frozen=True)
class ModelStats:
    """One model-and-voice-kind's recent real-time factors on this host.

    Attributes:
        kind: The `Voice.source` these were measured with — `builtin`,
            `designed` or `reference`.
        samples: Most recent last, at most `MAX_SAMPLES`.
    """

    kind: str
    samples: tuple[float, ...]

    @property
    def rtf(self) -> float:
        """The median, which is what a card shows."""
        return round(statistics.median(self.samples), 2)

    @property
    def count(self) -> int:
        """How many syntheses it rests on, so the UI can say."""
        return len(self.samples)


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
        # model id -> voice kind -> samples.
        self._samples: dict[str, dict[str, list[float]]] = {}
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
        if not isinstance(stored, dict):
            return
        for model_id, by_kind in stored.items():
            if not isinstance(by_kind, dict):
                # An earlier layout kept one flat list per model. It cannot be
                # split after the fact, and a figure that averaged two kinds is
                # exactly what this replaced, so it is dropped rather than
                # carried forward under a label it may not deserve.
                continue
            for kind, samples in by_kind.items():
                if not isinstance(samples, list):
                    continue
                kept = [float(s) for s in samples if isinstance(s, (int, float))]
                if kept:
                    self._samples.setdefault(str(model_id), {})[str(kind)] = kept[
                        -MAX_SAMPLES:
                    ]

    def _write(self) -> None:
        """Write the file. Callers hold `_lock`."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self._path, self._samples, indent=2, sort_keys=True)
        except OSError as err:
            # The measurement still counts for this process; only the memory
            # of it across a restart is lost, which is not worth an error to
            # the caller who only asked for audio.
            _LOGGER.warning("could not write stats: %s", err)

    def record(
        self, model_id: str, kind: str, rtf: float, audio_seconds: float
    ) -> None:
        """Note one synthesis, if it is long enough to be representative.

        Args:
            model_id: Catalog id.
            kind: The `Voice.source` it was rendered with.
            rtf: Render seconds over audio seconds.
            audio_seconds: How much audio came out.
        """
        if rtf <= 0 or audio_seconds < MIN_AUDIO_SECONDS:
            return
        with self._lock:
            samples = self._samples.setdefault(model_id, {}).setdefault(kind, [])
            samples.append(round(float(rtf), 3))
            del samples[:-MAX_SAMPLES]
            self._write()

    def get(self, model_id: str) -> list[ModelStats]:
        """Return what this host measured for a model, one entry per kind.

        Empty when it never has. Ordered so a card renders the same way twice.
        """
        with self._lock:
            by_kind = self._samples.get(model_id) or {}
            return [
                ModelStats(kind=kind, samples=tuple(samples))
                for kind, samples in sorted(by_kind.items())
                if samples
            ]

    def forget(self, model_id: str) -> None:
        """Drop a model's measurements, when its weights are deleted.

        Keeping them would carry a figure across a delete and a re-download,
        which is usually right — but the reason to delete a model is often
        that something about it changed.
        """
        with self._lock:
            if self._samples.pop(model_id, None) is not None:
                self._write()
