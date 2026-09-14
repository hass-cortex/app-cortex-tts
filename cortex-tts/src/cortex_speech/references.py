"""Storage for the reference recordings that define cloned voices.

A reference is an audio file plus the words spoken in it. The model needs
both: the recording supplies the timbre, the transcript tells it which sounds
in that recording map to which text. A wrong transcript degrades the clone
without failing, so it is validated on the way in rather than debugged later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .store import write_json
from .text.pipeline import TextOptions, prepare, prepared_text

_LOGGER = logging.getLogger(__name__)

INDEX_NAME = "references.json"

# Upstream conditions the speaker encoder on a fixed window and pushes the
# codec tokens of the whole clip into the LM prompt at 50 Hz, so a long
# reference inflates every prompt for no extra fidelity.
MIN_SECONDS = 2.0
MAX_SECONDS = 20.0

# How loud the last of the clip may be, against the clip's own average, before
# it is judged to have been cut rather than to have finished. A recording that
# ends in speech was stopped by a clock, not by the speaker.
#
# Measured on eight real uploads: the seven cut at a recorder's 7.00s limit
# ended between -7.1 and +7.5 dB — one of them louder than its own average,
# which is what stopping mid-vowel looks like — and the single clip that ran
# to its own end was at -25.3 dB. The threshold sits in the gap, nearer the
# bad side, since the cost of a false accept (a clone that drifts, silently)
# is higher than that of asking for a cleaner take.
END_SILENCE_WINDOW_SECONDS = 0.1
END_SILENCE_DB = -15.0

# Peak below this and there is nothing in the file to clone. Checked before
# the ending is judged, because that judgement is a ratio against the clip's
# own level: silence has no level, the ratio is 0 dB, and the reader would be
# told its recording "is still speaking when it ends".
SILENT_PEAK_DBFS = -45.0

_ID_SAFE = re.compile(r"[^a-z0-9_-]+")


class ReferenceError(ValueError):
    """An uploaded reference cannot be used as a voice."""


def slugify(name: str) -> str:
    """Derive a filesystem- and API-safe id from a display name."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = _ID_SAFE.sub("-", folded.strip().lower()).strip("-")
    # A name that survives ASCII-folding only as digits or punctuation carries
    # none of its meaning: "灣灣小何2" becomes "2", which is both unreadable and
    # a collision waiting to happen with every other name ending in 2. Require
    # a letter before trusting the folded form.
    if not any(character.isalpha() for character in slug):
        slug = "voice-" + hashlib.sha256(name.encode()).hexdigest()[:8]
    return slug[:48]


@dataclass(frozen=True)
class Reference:
    """One stored reference recording.

    Attributes:
        id: Stable voice id used in synthesis requests.
        name: Display name.
        transcript: Model-ready transcript (normalised and script-converted).
        raw_transcript: What the user typed, kept for display and editing.
        language: Base language code of the recording.
        gender: ``female``, ``male`` or ``unknown``; label only.
        seconds: Duration of the stored audio.
        created: Unix timestamp of upload.
        audio_path: Where the 24 kHz mono WAV lives.
        fingerprint: Identifies the stored audio. Engines key their cached
            encodings on it, and those depend on the recording only — editing
            the transcript must not force a re-encode.
    """

    id: str
    name: str
    transcript: str
    raw_transcript: str
    language: str
    gender: str
    seconds: float
    created: float
    audio_path: Path
    fingerprint: str

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-serialisable form stored in the index."""
        data = asdict(self)
        data["audio_path"] = self.audio_path.name
        return data


class ReferenceStore:
    """Reference recordings on disk, with a JSON index beside them."""

    def __init__(self, root: Path) -> None:
        """Open (or create) the store rooted at ``root``."""
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._index = self._root / INDEX_NAME
        self._items: dict[str, Reference] = {}
        # A plain lock, not an asyncio one: `add` decodes and hashes a
        # recording, so the API runs it on a worker thread while `update` and
        # `remove` stay on the event loop. Two real threads reach `_items` and
        # the index file, and every mutation here is a read-modify-write.
        self._lock = threading.Lock()
        self._load()

    @property
    def root(self) -> Path:
        """Directory holding the recordings and whatever engines keep beside them."""
        return self._root

    def _load(self) -> None:
        if not self._index.is_file():
            return
        try:
            raw = json.loads(self._index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            # A corrupt index must not take the whole app down: the audio is
            # still on disk and can be re-registered by uploading again.
            _LOGGER.error("reference index unreadable, starting empty: %s", err)
            return
        for entry in raw.get("references", []):
            try:
                path = self._root / entry["audio_path"]
                if not path.is_file():
                    _LOGGER.warning(
                        "reference %s has no audio file, dropping", entry.get("id")
                    )
                    continue
                self._items[entry["id"]] = Reference(
                    id=entry["id"],
                    name=entry["name"],
                    transcript=entry["transcript"],
                    raw_transcript=entry.get("raw_transcript", entry["transcript"]),
                    language=entry.get("language", "zh"),
                    gender=entry.get("gender", "unknown"),
                    seconds=float(entry.get("seconds", 0.0)),
                    created=float(entry.get("created", 0.0)),
                    audio_path=path,
                    fingerprint=entry["fingerprint"],
                )
            except (KeyError, TypeError, ValueError) as err:
                _LOGGER.warning("skipping malformed reference entry: %s", err)

    def _save(self) -> None:
        """Write the index. Callers hold `_lock`."""
        payload = {"references": [ref.to_json() for ref in self._items.values()]}
        write_json(self._index, payload, ensure_ascii=False, indent=2)

    def list(self) -> list[Reference]:
        """Return every stored reference, oldest first."""
        with self._lock:
            return sorted(self._items.values(), key=lambda r: r.created)

    def get(self, reference_id: str) -> Reference | None:
        """Return one reference, or ``None`` when it does not exist."""
        with self._lock:
            return self._items.get(reference_id)

    def add(
        self,
        *,
        name: str,
        transcript: str,
        audio: bytes,
        language: str = "zh",
        gender: str = "unknown",
    ) -> Reference:
        """Validate and store a reference recording.

        Args:
            name: Display name; also seeds the voice id.
            transcript: Exactly what is said in the recording.
            audio: Encoded audio bytes in any format soundfile can read.
            language: Base language code.
            gender: Label shown in the voice picker.

        Returns:
            The stored reference.

        Raises:
            ReferenceError: The audio is unreadable, the wrong length, or the
                transcript is empty.
        """
        if not transcript.strip():
            raise ReferenceError("a reference needs the transcript of what is said")
        # Validated before anything is written: a wrong transcript degrades
        # the clone with no error, and a rejected upload must leave no file.
        prepared = prepared_text(prepare(transcript, TextOptions()))
        if not prepared:
            raise ReferenceError("the transcript has no pronounceable content")

        samples, sample_rate = _decode(audio)
        seconds = len(samples) / sample_rate
        if seconds < MIN_SECONDS:
            raise ReferenceError(
                f"reference is {seconds:.1f}s; at least {MIN_SECONDS:.0f}s is needed"
            )
        if seconds > MAX_SECONDS:
            raise ReferenceError(
                f"reference is {seconds:.1f}s; trim it to {MAX_SECONDS:.0f}s or less "
                "— longer clips inflate every prompt without improving the clone"
            )
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if 20.0 * np.log10(peak + 1e-12) < SILENT_PEAK_DBFS:
            raise ReferenceError("the recording is silent — nothing in it to clone")
        tail = _tail_db(samples, sample_rate)
        if tail > END_SILENCE_DB:
            raise ReferenceError(
                f"the recording is still speaking when it ends ({tail:+.0f} dB in "
                f"the last {END_SILENCE_WINDOW_SECONDS * 1000:.0f} ms against its "
                "own average), so it was cut rather than finished — re-cut it to "
                "end on a completed sentence, with the silence after it"
            )

        # Held from the moment an id is chosen: the id names the file, so
        # picking one and taking it have to be a single step. Two uploads of
        # the same name otherwise agree on a slug, and the second overwrites
        # the first's recording and its entry with no error on either.
        with self._lock:
            reference_id = self._unique_id(slugify(name))
            audio_path = self._root / f"{reference_id}.wav"
            sf.write(audio_path, samples, sample_rate, subtype="PCM_16")
            fingerprint = _fingerprint(audio_path)

            reference = Reference(
                id=reference_id,
                name=name.strip() or reference_id,
                transcript=prepared,
                raw_transcript=transcript.strip(),
                language=language,
                gender=gender,
                seconds=round(seconds, 2),
                created=time.time(),
                audio_path=audio_path,
                fingerprint=fingerprint,
            )
            self._items[reference_id] = reference
            self._save()
        _LOGGER.info(
            "stored reference %s (%.1fs, %s)", reference_id, seconds, reference.name
        )
        return reference

    def update(self, reference_id: str, *, transcript: str) -> Reference:
        """Replace the transcript of a stored reference.

        The recording is untouched, so the cloned voice keeps its timbre; only
        the text the model is told the recording contains changes. That text is
        run through the same pipeline as a fresh upload, because the model
        needs it in the same script and normalisation as the target text.

        Args:
            reference_id: Which reference to edit.
            transcript: Corrected wording of what the recording says.

        Returns:
            The updated reference.

        Raises:
            KeyError: No reference with that id.
            ReferenceError: The transcript is empty.
        """
        if not transcript.strip():
            raise ReferenceError("a reference needs the transcript of what is said")
        prepared = prepared_text(prepare(transcript, TextOptions()))
        if not prepared:
            raise ReferenceError("the transcript has no pronounceable content")

        with self._lock:
            existing = self._items.get(reference_id)
            if existing is None:
                raise KeyError(reference_id)
            updated = replace(
                existing, transcript=prepared, raw_transcript=transcript.strip()
            )
            self._items[reference_id] = updated
            self._save()
        _LOGGER.info("updated transcript for reference %s", reference_id)
        return updated

    def remove(self, reference_id: str) -> bool:
        """Delete a reference and its audio. Returns whether it existed."""
        with self._lock:
            reference = self._items.pop(reference_id, None)
            if reference is None:
                return False
            reference.audio_path.unlink(missing_ok=True)
            # Engines keep their encodings beside the recording; an engine
            # not loaded right now would never be told to drop its own.
            for sidecar in self._root.glob(f"{reference_id}.*"):
                sidecar.unlink(missing_ok=True)
            self._save()
        _LOGGER.info("removed reference %s", reference_id)
        return True

    def _unique_id(self, base: str) -> str:
        if base not in self._items:
            return base
        for suffix in range(2, 100):
            candidate = f"{base}-{suffix}"
            if candidate not in self._items:
                return candidate
        return f"{base}-{int(time.time())}"


def _fingerprint(audio_path: Path) -> str:
    """Identify a stored recording by its bytes."""
    return hashlib.sha256(audio_path.read_bytes()).hexdigest()[:16]


def _tail_db(samples: np.ndarray, sample_rate: int) -> float:
    """How loud the clip's last moments are, against its own average.

    The models that clone best read the whole recording as a worked example
    before they speak, so a clip cut mid-word teaches one: that is how this
    speaker ends a sentence. Nothing downstream can tell — the clone renders,
    and simply drifts and clips its endings.

    Only meaningful on a clip that has speech in it: the measure is a ratio
    against the clip's own level, so a silent file reads 0 dB — "as loud at
    the end as anywhere", which is true and useless. `add` rejects silence
    before asking.

    Returns:
        Decibels relative to the clip's own RMS. Around zero means it ended at
        full speech; well below means it ended in silence, as a finished
        sentence does.
    """
    window = int(sample_rate * END_SILENCE_WINDOW_SECONDS)
    if window <= 0 or samples.size < window * 2:
        return 0.0
    overall = float(np.sqrt(np.mean(np.square(samples))))
    tail = float(np.sqrt(np.mean(np.square(samples[-window:]))))
    if overall <= 0.0:
        return 0.0
    return 20.0 * float(np.log10((tail + 1e-12) / overall))


def _decode(audio: bytes) -> tuple[np.ndarray, int]:
    """Decode uploaded audio to mono float32 at its native rate."""
    import io

    try:
        samples, sample_rate = sf.read(io.BytesIO(audio), dtype="float32")
    except (RuntimeError, sf.LibsndfileError) as err:
        raise ReferenceError(
            "could not read the audio — upload a WAV, FLAC or OGG file"
        ) from err
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.size == 0:
        raise ReferenceError("the audio file is empty")
    return np.asarray(samples, dtype=np.float32), int(sample_rate)
