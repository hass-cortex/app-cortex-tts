"""Decoding and levelling a reference recording.

A hot reference makes a cloning model generate clipped speech, and the clipping
is baked into the samples the model produces — the output stage cannot undo it.
So the level is fixed on the way in.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cortex_speech.audio import REFERENCE_RMS_DBFS, decode_reference, level


def _tone(seconds: float, rate: int, amplitude: float) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * rate), endpoint=False)
    return (amplitude * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)


def _dbfs(audio: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(audio)))))


def _write(path: Path, audio: np.ndarray, rate: int) -> Path:
    sf.write(path, audio, rate, subtype="PCM_16")
    return path


class TestLevel:
    @pytest.mark.parametrize("amplitude", [0.05, 0.2, 0.9])
    def test_any_starting_level_lands_on_the_target(self, amplitude: float) -> None:
        """Quiet and hot references both end up somewhere the model likes."""
        levelled = level(_tone(1.0, 48000, amplitude))
        assert _dbfs(levelled) == pytest.approx(REFERENCE_RMS_DBFS, abs=0.5)

    def test_a_loud_transient_does_not_get_scaled_into_clipping(self) -> None:
        """RMS alone would push this past full scale; the ceiling stops it."""
        audio = _tone(1.0, 48000, 0.02)
        audio[100] = 0.99  # one spike, very quiet average
        levelled = level(audio)
        assert float(np.max(np.abs(levelled))) <= 0.9

    def test_silence_is_left_alone(self) -> None:
        """Dividing by a zero RMS would produce infinities."""
        silence = np.zeros(1000, dtype=np.float32)
        assert np.array_equal(level(silence), silence)

    def test_an_empty_waveform_is_left_alone(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        assert level(empty).size == 0


class TestDecodeReference:
    def test_resamples_to_the_rate_the_codec_wants(self, tmp_path: Path) -> None:
        """A 44.1 kHz upload must reach a 48 kHz codec as 48 kHz."""
        path = _write(tmp_path / "ref.wav", _tone(0.5, 44100, 0.3), 44100)
        decoded = decode_reference(str(path), sample_rate=48000)
        assert decoded.shape[0] == pytest.approx(24000, rel=0.01)

    def test_downmixes_stereo_to_mono(self, tmp_path: Path) -> None:
        mono = _tone(0.5, 48000, 0.3)
        stereo = np.stack([mono, mono], axis=1)
        path = _write(tmp_path / "stereo.wav", stereo, 48000)
        assert decode_reference(str(path), sample_rate=48000).ndim == 1

    def test_duplicates_mono_into_the_channels_a_codec_expects(
        self, tmp_path: Path
    ) -> None:
        """MOSS's codec is declared stereo even for a mono reference."""
        path = _write(tmp_path / "ref.wav", _tone(0.5, 48000, 0.3), 48000)
        decoded = decode_reference(str(path), sample_rate=48000, channels=2)
        assert decoded.shape[0] == 2
        assert np.array_equal(decoded[0], decoded[1])

    def test_levels_by_default(self, tmp_path: Path) -> None:
        """The default is the whole point: an un-levelled hot reference clips."""
        path = _write(tmp_path / "hot.wav", _tone(0.5, 48000, 0.9), 48000)
        assert _dbfs(decode_reference(str(path), sample_rate=48000)) == pytest.approx(
            REFERENCE_RMS_DBFS, abs=0.5
        )

    def test_levelling_can_be_turned_off(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "quiet.wav", _tone(0.5, 48000, 0.05), 48000)
        decoded = decode_reference(str(path), sample_rate=48000, target_dbfs=None)
        assert _dbfs(decoded) < REFERENCE_RMS_DBFS - 5

    def test_an_undecodable_file_says_so(self, tmp_path: Path) -> None:
        path = tmp_path / "not-audio.wav"
        path.write_bytes(b"this is not a RIFF header")
        with pytest.raises(ValueError, match="could not decode"):
            decode_reference(str(path), sample_rate=48000)
