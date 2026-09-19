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

from cortex_speech import RenderModel, RenderSample, speech_rates, write_json

_LOGGER = logging.getLogger(__name__)

FILE_NAME = "stats.json"

# How many recent requests are kept, and so how many a fit rests on — this is
# the only bound, `RenderModel.fit` uses whatever it is handed. Enough that one
# odd request cannot define it, few enough that it follows a host that changed
# — a thread count, an execution provider, a busier machine. More than a bare
# figure of merit would need, because a fit wants requests of different lengths
# to find a slope, and a dozen replies to an assistant are mostly one length.
MAX_RENDERS = 24


def split_key(key: str) -> tuple[str, str | None]:
    """A bucket back into its kind and, for a clone, which voice.

    The other half of `StatsStore._key`, kept beside it so the two cannot
    drift: a card reads what the store wrote.
    """
    kind, _, voice = key.partition(":")
    return kind, voice or None


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

    @staticmethod
    def _key(kind: str, voice_id: str) -> str:
        return f"{kind}:{voice_id}"

    def _cost_samples(
        self, model_id: str, kind: str, voice_id: str
    ) -> list[RenderSample]:
        """The renders a cost line is fitted from.

        Pooled across a model's own voices, whose cost differed by 4% where
        that was measured (MOSS), and kept apart for clones, whose reference
        rejoins the prompt every synthesis — 0.354 s of render per second of
        recording on OmniVoice, a tenth of that on MOSS, which is why the
        intercept is not predicted from the recording's length. Pooling is what lets a
        model with eighteen built-in voices have a cost line at all.
        """
        by_key = self._renders.get(model_id) or {}
        if kind == "reference":
            return list(by_key.get(self._key(kind, voice_id)) or [])
        merged: list[RenderSample] = []
        for key, samples in by_key.items():
            if key.split(":", 1)[0] == kind:
                merged.extend(samples)
        return merged

    def record(
        self, model_id: str, kind: str, voice_id: str, sample: RenderSample
    ) -> None:
        """Note how one request went.

        Every request counts, short ones included: the fixed cost is exactly
        what a short request exposes, and the fit needs both ends of the line.

        Args:
            model_id: Catalog id.
            kind: The `Voice.source` it was rendered with.
            voice_id: Which voice. Kept even where the cost is pooled, because
                the speech rate never is.
            sample: Audio and wall seconds as measured, undivided. What the
                render clock means is the caller's to state and the fit's to
                use; a ratio would hide it.
        """
        if sample.audio_s <= 0 or sample.wall_s <= 0:
            return
        with self._lock:
            renders = self._renders.setdefault(model_id, {}).setdefault(
                self._key(kind, voice_id), []
            )
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

    def _stand_in(self, model_id: str, skip: str) -> RenderModel | None:
        """A line for a voice of this model that has none of its own.

        Which lines are eligible is the store's to say; what one line made of
        several should be is `RenderModel.dearest`, with the measurements that
        settled it.
        """
        by_key = self._renders.get(model_id) or {}
        return RenderModel.dearest(
            [
                fit
                for key, samples in by_key.items()
                if key != skip and (fit := RenderModel.fit(samples)) is not None
            ]
        )

    def render_model(
        self, model_id: str, kind: str, voice_id: str
    ) -> RenderModel | None:
        """What a request in this voice costs here, and how fast it speaks.

        Two questions with two right groupings: the cost line comes from every
        voice that shares a cost, the pace only from this one.

        A voice with no line of its own stands in one from the model's other
        voices rather than going without. Without it a voice uploaded a minute
        ago was buffered however long the model had been in use — and buffered
        is what a caller asks for, not what the app should conclude on its own
        while it still has something to go on. `samples` stays 0 on a stand-in,
        so `get` and the card it feeds never show it as measured.
        """
        key = self._key(kind, voice_id)
        with self._lock:
            cost = self._cost_samples(model_id, kind, voice_id)
            own = list((self._renders.get(model_id) or {}).get(key) or [])
            fit = RenderModel.fit(cost) or self._stand_in(model_id, key)
        return fit.with_rates(speech_rates(own)) if fit else None

    def get(self, model_id: str) -> list[ModelStats]:
        """Return what this host measured for a model, one entry per cost line.

        One per clone, because each carries its own recording; one per kind for
        a model's own voices, because they cost the same. Empty until enough
        requests exist to fit a line. Ordered so a card renders the same twice.
        """
        with self._lock:
            keys = sorted((self._renders.get(model_id) or {}).keys())
        seen: set[str] = set()
        out: list[ModelStats] = []
        for key in keys:
            kind, _, voice = key.partition(":")
            bucket = key if kind == "reference" else kind
            if bucket in seen:
                continue
            seen.add(bucket)
            # The pooled line, rates included: a row covering a model's own
            # voices cannot quote one of their paces as if it were the row's.
            # A clone's bucket is one voice, so there is nothing to pool.
            with self._lock:
                cost = self._cost_samples(model_id, kind, voice)
            fit = RenderModel.fit(cost)
            if fit:
                out.append(ModelStats(kind=bucket, render=fit))
        return out

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
