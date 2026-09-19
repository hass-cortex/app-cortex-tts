"""What a chunked stream is made of, under `/api/speak/live`.

Synthesis itself needs a 729 MB bundle; these cover the parts that break
without one — the container each format opens with, the sample conversion, and
the gain that keeps chunks under the ceiling without ever raising them.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from cortex_speech.audio import (
    MP3_BITRATE,
    STREAM_ENCODERS,
    TARGET_PEAK,
    Mp3Stream,
    StreamGain,
    WavStream,
    pcm_frames,
    wav_header,
)


class TestWavHeader:
    def test_is_a_riff_wave_header_of_the_usual_length(self) -> None:
        header = wav_header(48000)
        assert len(header) == 44
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"

    def test_declares_the_stream_maximal_rather_than_zero(self) -> None:
        """A zero-length WAV is a file players skip; this one they read to EOF."""
        header = wav_header(48000)
        assert struct.unpack("<I", header[4:8])[0] == 0xFFFFFFFF
        assert struct.unpack("<I", header[40:44])[0] == 0xFFFFFFFF

    @pytest.mark.parametrize("rate", [22050, 24000, 48000])
    def test_carries_the_rate_and_derived_fields(self, rate: int) -> None:
        """A wrong byte rate plays at the wrong speed rather than failing."""
        header = wav_header(rate)
        assert struct.unpack("<I", header[24:28])[0] == rate
        assert struct.unpack("<I", header[28:32])[0] == rate * 2  # mono, 16-bit
        assert struct.unpack("<H", header[32:34])[0] == 2


class TestPcmFrames:
    def test_mono_float32_becomes_little_endian_int16(self) -> None:
        frames = pcm_frames(np.zeros(480, dtype=np.float32))
        assert len(frames) == 960

    def test_full_scale_maps_to_the_top_of_the_range(self) -> None:
        assert struct.unpack("<h", pcm_frames(np.array([1.0], np.float32)))[0] == 32767

    def test_out_of_range_input_is_clipped_not_wrapped(self) -> None:
        """Wrapping turns a loud sample into a loud sample of opposite sign."""
        loud = np.array([1.6, -1.6], dtype=np.float32)
        values = struct.unpack("<2h", pcm_frames(loud))
        assert values == (32767, -32767)


class TestStreamGain:
    """What keeps a stream from hard-clipping, without making it pump.

    `encode` peak-normalises a finished waveform; a stream has none, and the
    clamp in `pcm_frames` turned everything the model generated above full
    scale into distortion — measured as 11 clamped runs in 8.6 s of a
    Home-Assistant-shaped MOSS reply, against none from a levelled file.
    """

    @staticmethod
    def _peak(frames: bytes) -> float:
        return float(np.max(np.abs(np.frombuffer(frames, "<i2")))) / 32767.0

    def test_a_chunk_under_the_ceiling_is_untouched(self) -> None:
        """Most speech never approaches it, and must not be quietened."""
        quiet = np.full(64, 0.4, dtype=np.float32)
        assert StreamGain().frames(quiet) == pcm_frames(quiet)

    def test_a_chunk_over_the_ceiling_is_scaled_not_clamped(self) -> None:
        """The failure this exists to remove."""
        loud = np.linspace(-1.3, 1.3, 256, dtype=np.float32)
        frames = StreamGain().frames(loud)
        assert self._peak(frames) <= TARGET_PEAK + 1e-3
        assert self._peak(pcm_frames(loud)) == 1.0, "the clamp still clips"

    def test_the_shape_of_a_loud_chunk_survives(self) -> None:
        """Clamping flattens peaks into each other; scaling keeps them apart."""
        loud = np.array([1.3, 1.1, 0.5], dtype=np.float32)
        values = np.frombuffer(StreamGain().frames(loud), "<i2")
        assert values[0] > values[1] > values[2]

    def test_the_gain_never_rises_again(self) -> None:
        """Recovering after a loud chunk is what makes a stream pump."""
        gain = StreamGain()
        gain.frames(np.array([1.4], dtype=np.float32))
        after = self._peak(gain.frames(np.array([0.8], dtype=np.float32)))
        assert after < 0.8, "a quiet chunk played at full level after a loud one"

    def test_a_louder_chunk_pushes_the_gain_further_down(self) -> None:
        gain = StreamGain()
        gain.frames(np.array([1.1], dtype=np.float32))
        first = self._peak(gain.frames(np.array([1.0], dtype=np.float32)))
        second = self._peak(gain.frames(np.array([1.0], dtype=np.float32)))
        assert first <= TARGET_PEAK + 1e-3
        assert second <= first

    def test_streams_do_not_share_a_gain(self) -> None:
        """One instance per request; a loud reply must not quieten the next."""
        loud = StreamGain()
        loud.frames(np.array([2.0], dtype=np.float32))
        fresh = np.full(8, 0.5, dtype=np.float32)
        assert StreamGain().frames(fresh) == pcm_frames(fresh)

    def test_silence_does_not_divide_by_zero(self) -> None:
        assert StreamGain().frames(np.zeros(16, dtype=np.float32)) == b"\x00" * 32

    def test_an_empty_chunk_is_empty(self) -> None:
        assert StreamGain().frames(np.zeros(0, dtype=np.float32)) == b""


class TestStreamEncoders:
    """What a stream may be encoded as, and why the set is this small.

    A streamed format has to be writable before the length of the audio is
    known. MP3 is a bare sequence of self-describing frames, so it needs
    nothing declared. WAV has to declare a length, and the maximal one it
    declares is read by a general-purpose player as a six-hour file that it
    then waits to buffer — reported from a Windows desktop player as one
    sentence, a long stall, and the rest of the reply at the end.
    """

    def test_only_the_two_that_need_no_length_are_offered(self) -> None:
        assert set(STREAM_ENCODERS) == {"wav", "mp3"}

    def test_flac_and_ogg_are_not_streamable(self) -> None:
        """Both want a size or a seek table in a header written up front."""
        assert "flac" not in STREAM_ENCODERS
        assert "ogg" not in STREAM_ENCODERS


class TestMp3Stream:
    @staticmethod
    def _speech(seconds: float, rate: int = 48000) -> bytes:
        samples = np.sin(np.linspace(0, 400 * seconds, int(rate * seconds)))
        return pcm_frames((samples * 0.3).astype(np.float32))

    def test_it_sends_nothing_before_the_audio(self) -> None:
        """The whole point: no header, so no length to get wrong."""
        assert Mp3Stream().open(48000) == b""

    def test_the_output_starts_with_a_frame_sync(self) -> None:
        """0xFFE is what a decoder scans for; without it nothing plays."""
        encoder = Mp3Stream()
        encoder.open(48000)
        out = encoder.encode(self._speech(1.0)) + encoder.close()
        assert out[0] == 0xFF
        assert out[1] & 0xE0 == 0xE0

    def test_a_byte_count_converts_back_to_a_duration(self) -> None:
        """A consumer sizes its buffer from this without decoding anything.

        Constant bitrate is what makes it true; padding frames put the error
        under a percent, which is inside what any of the arithmetic needs.
        """
        encoder = Mp3Stream()
        encoder.open(48000)
        out = encoder.encode(self._speech(4.0)) + encoder.close()
        assert abs(len(out) * 8 / encoder.bitrate - 4.0) < 0.05

    def test_the_flush_is_not_optional(self) -> None:
        """Dropping it truncates the reply by the last partial frame."""
        encoder = Mp3Stream()
        encoder.open(48000)
        encoder.encode(self._speech(0.5))
        assert encoder.close()

    def test_closing_an_encoder_that_never_opened_is_quiet(self) -> None:
        """A request refused before synthesis still runs the finally block."""
        assert Mp3Stream().close() == b""

    def test_it_declares_a_bitrate_and_the_right_type(self) -> None:
        encoder = Mp3Stream()
        assert encoder.bitrate == MP3_BITRATE
        assert encoder.content_type == "audio/mpeg"


class TestWavStream:
    def test_it_still_opens_with_the_44_byte_header(self) -> None:
        assert WavStream().open(48000) == wav_header(48000)

    def test_samples_pass_through_untouched(self) -> None:
        encoder = WavStream()
        encoder.open(48000)
        pcm = pcm_frames(np.zeros(480, dtype=np.float32))
        assert encoder.encode(pcm) == pcm
        assert encoder.close() == b""

    def test_its_bitrate_is_the_rate_it_was_opened_at(self) -> None:
        """Raw 16-bit mono, so a byte count converts back the same way."""
        encoder = WavStream()
        encoder.open(24000)
        assert encoder.bitrate == 24000 * 16
