"""What this host measured, kept so a model card can stop guessing.

Why the catalog carries no figure of its own is in AGENTS.md, under "A
real-time factor belongs to a host, not to a model" — including the
measurements that settled it. Two decisions live here rather than there:

Kept per *voice kind*, because what a model costs depends on how it was asked
to speak, so one figure averaging the kinds would describe neither.

Stored beside the models rather than in the settings file: settings are what a
user chose and stats are what the app observed, and a "reset settings" must
not erase measurements that took real work to gather.

One primitive, recorded once per request: a `RenderSample` of raw audio and
wall seconds. Everything a caller asks for is fitted from those, so there is
no second series to keep honest and no cadence a transport can get wrong.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from cortex_speech import RenderModel, RenderSample, write_json

_LOGGER = logging.getLogger(__name__)

FILE_NAME = "stats.json"

# How many recent requests are kept, and so how many a fit rests on — this is
# the only bound, `RenderModel.fit` uses whatever it is handed. Enough that one
# odd request cannot define it, few enough that it follows a host that changed
# — a thread count, an execution provider, a busier machine. More than a bare
# figure of merit would need, because a fit wants requests of different lengths
# to find a slope, and a dozen replies to an assistant are mostly one length.
MAX_RENDERS = 24


@dataclass(frozen=True)
class ModelStats:
    """One model-and-voice-kind's cost on this host.

    Attributes:
        kind: The `Voice.source` these were measured with — `builtin`,
            `designed` or `reference`.
        render: The fit those requests produced. Its `per_audio` is the
            real-time factor a card shows, with the per-request fixed cost
            held beside it rather than averaged into it.
    """

    kind: str
    render: RenderModel


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
        # model id -> voice kind -> requests as they went.
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
        if not isinstance(stored, dict):
            return
        for model_id, by_kind in stored.items():
            if not isinstance(by_kind, dict):
                continue
            for kind, entry in by_kind.items():
                # An earlier layout also kept ready-divided ratios under
                # "rtf". The pairs they came from are not in the file, so
                # there is nothing to fit and nothing to carry forward.
                renders = entry.get("renders") if isinstance(entry, dict) else None
                samples = [
                    RenderSample(float(r[0]), float(r[1]), int(r[2]), int(r[3]))
                    for r in renders or []
                    if isinstance(r, list) and len(r) == 4
                ]
                if samples:
                    self._renders.setdefault(str(model_id), {})[str(kind)] = samples[
                        -MAX_RENDERS:
                    ]

    def _write(self) -> None:
        """Write the file. Callers hold `_lock`."""
        stored = {
            model_id: {
                kind: {
                    "renders": [[r.audio_s, r.wall_s, r.cjk, r.latin] for r in renders]
                }
                for kind, renders in by_kind.items()
            }
            for model_id, by_kind in self._renders.items()
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self._path, stored, indent=2, sort_keys=True)
        except OSError as err:
            # The measurement still counts for this process; only the memory
            # of it across a restart is lost, which is not worth an error to
            # the caller who only asked for audio.
            _LOGGER.warning("could not write stats: %s", err)

    def record(self, model_id: str, kind: str, sample: RenderSample) -> None:
        """Note how one request went.

        Every request counts, short ones included: the fixed cost is exactly
        what a short request exposes, and the fit needs both ends of the line.

        Args:
            model_id: Catalog id.
            kind: The `Voice.source` it was rendered with.
            sample: Audio and wall seconds as measured, undivided. What the
                render clock means is the caller's to state and the fit's to
                use; a ratio would hide it.
        """
        if sample.audio_s <= 0 or sample.wall_s <= 0:
            return
        with self._lock:
            renders = self._renders.setdefault(model_id, {}).setdefault(kind, [])
            renders.append(
                RenderSample(
                    round(sample.audio_s, 3),
                    round(sample.wall_s, 3),
                    sample.cjk,
                    sample.latin,
                )
            )
            del renders[:-MAX_RENDERS]
            self._write()

    def render_model(self, model_id: str, kind: str) -> RenderModel | None:
        """What a request to this model and voice kind costs here, if measured."""
        with self._lock:
            renders = list((self._renders.get(model_id) or {}).get(kind) or [])
        return RenderModel.fit(renders)

    def get(self, model_id: str) -> list[ModelStats]:
        """Return what this host measured for a model, one entry per kind.

        Empty until it has served enough requests of one kind to fit a line.
        Ordered so a card renders the same way twice.
        """
        with self._lock:
            by_kind = {
                kind: list(renders)
                for kind, renders in sorted((self._renders.get(model_id) or {}).items())
            }
        fitted = ((kind, RenderModel.fit(s)) for kind, s in by_kind.items())
        return [ModelStats(kind=kind, render=fit) for kind, fit in fitted if fit]

    def clear(self) -> None:
        """Drop every model's measurements.

        For when the host changed under all of them at once — a new execution
        provider, a different thread count — and every stored figure now
        describes a machine that is gone.
        """
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
