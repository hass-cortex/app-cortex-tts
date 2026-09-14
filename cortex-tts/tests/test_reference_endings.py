"""A reference must end because the speaker stopped, not because a clock did.

The models that clone best read the whole recording as a worked example
before they say anything: Qwen3-TTS puts its codec frames in the prompt
beside its transcript, and OmniVoice has no speaker encoder at all. A clip cut
mid-word therefore teaches one thing above all — that is how this speaker
finishes a sentence — and nothing downstream can tell. The clone renders
without error, drifts, and clips its own endings.

So it is refused on the way in, like a wrong length and an empty transcript,
and for the same reason: the failure it prevents is silent.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from cortex_speech.references import (
    END_SILENCE_DB,
    ReferenceError,
    ReferenceStore,
    _tail_db,
)

RATE = 24000


def _wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples.astype(np.float32), RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _speech(seconds: float, level: float = 0.2) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return level * np.sin(2 * np.pi * 220 * t)


class TestTheMeasurement:
    def test_a_clip_ending_in_silence_reads_far_below_its_average(self) -> None:
        clip = np.concatenate([_speech(2.5), np.zeros(int(RATE * 0.5))])
        assert _tail_db(clip, RATE) < END_SILENCE_DB

    def test_a_clip_cut_while_speaking_reads_around_zero(self) -> None:
        """No trailing silence at all: the last sample is mid-waveform."""
        assert _tail_db(_speech(3.0), RATE) == pytest.approx(0.0, abs=1.0)

    def test_a_clip_cut_on_a_loud_syllable_reads_above_its_average(self) -> None:
        """What stopping mid-vowel looks like, and the reason the threshold is
        not simply "quieter than average": one real upload measured +7.5 dB."""
        clip = np.concatenate([_speech(2.5, 0.1), _speech(0.5, 0.4)])
        assert _tail_db(clip, RATE) > 0.0

    def test_a_silent_clip_reads_zero_and_must_not_be_judged_on_it(self) -> None:
        """A ratio against a level of zero. It reads "as loud at the end as
        anywhere", which is true and tells you nothing — so `add` rejects
        silence before it gets here. Pinned because the earlier version of
        this test claimed some other check caught silence, and none did: a
        silent upload came back as "still speaking when it ends"."""
        assert _tail_db(np.zeros(RATE), RATE) == 0.0
        assert _tail_db(np.zeros(RATE), RATE) > END_SILENCE_DB

    @pytest.mark.parametrize(
        ("measured_db", "was_cut"),
        [
            # The eight uploads this threshold was calibrated on. Seven were
            # cut at a recorder's 7.00s limit; the eighth ran to its own end.
            (7.5, True),
            (-0.1, True),
            (-0.7, True),
            (-2.5, True),
            (-4.5, True),
            (-6.4, True),
            (-7.1, True),
            (-25.3, False),
        ],
    )
    def test_the_threshold_separates_the_measured_uploads(
        self, measured_db: float, was_cut: bool
    ) -> None:
        assert (measured_db > END_SILENCE_DB) is was_cut


class TestWhatTheStoreAccepts:
    def test_a_finished_take_is_stored(self, tmp_path) -> None:
        store = ReferenceStore(tmp_path)
        clip = np.concatenate([_speech(2.5), np.zeros(int(RATE * 0.5))])
        reference = store.add(
            name="Finished", transcript="客廳的燈打開了。", audio=_wav(clip)
        )
        assert reference.seconds == pytest.approx(3.0, abs=0.05)

    def test_a_cut_take_is_refused(self, tmp_path) -> None:
        store = ReferenceStore(tmp_path)
        with pytest.raises(ReferenceError, match="cut rather than finished"):
            store.add(
                name="Cut", transcript="客廳的燈打開了。", audio=_wav(_speech(3.0))
            )

    def test_a_silent_recording_is_refused_as_silent(self, tmp_path) -> None:
        """Not as a cut one. The two are different problems and the reader can
        only act on the right name for theirs."""
        store = ReferenceStore(tmp_path)
        with pytest.raises(ReferenceError, match="silent"):
            store.add(
                name="Silent",
                transcript="客廳的燈打開了。",
                audio=_wav(np.zeros(int(RATE * 5), dtype=np.float32)),
            )

    def test_the_refusal_leaves_nothing_behind(self, tmp_path) -> None:
        """A rejected upload must not leave a file or an index entry: the
        store's other validators promise the same."""
        store = ReferenceStore(tmp_path)
        with pytest.raises(ReferenceError):
            store.add(
                name="Cut", transcript="客廳的燈打開了。", audio=_wav(_speech(3.0))
            )
        assert store.list() == []
        assert not list(tmp_path.glob("*.wav"))
