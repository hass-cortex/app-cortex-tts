"""Turning a float waveform into bytes Home Assistant will play."""

from __future__ import annotations

import io
import struct
from collections.abc import Callable
from typing import Any, Literal, Protocol

import numpy as np
import soundfile as sf

AudioFormat = Literal["wav", "flac", "ogg", "mp3"]

CONTENT_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
}

# Peak-normalisation target. The model's own output sits around -17 dBFS peak,
# quiet enough that a media player's volume has to be pushed up relative to
# every other source. Leaving headroom below full scale avoids clipping when a
# player applies its own gain.
TARGET_PEAK = 0.89


def normalize_level(audio: np.ndarray) -> np.ndarray:
    """Scale a waveform so its peak sits just below full scale.

    Silence is returned unchanged rather than amplified into noise.
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio
    return (audio * (TARGET_PEAK / peak)).astype(np.float32)


def encode(
    audio: np.ndarray,
    sample_rate: int,
    fmt: AudioFormat = "wav",
    *,
    normalize: bool = True,
) -> bytes:
    """Encode a mono float32 waveform.

    Args:
        audio: Samples in [-1, 1].
        sample_rate: Samples per second.
        fmt: Container to write.
        normalize: Peak-normalise before encoding.

    Returns:
        The encoded file as bytes.
    """
    samples = normalize_level(audio) if normalize else audio
    if fmt == "mp3":
        # soundfile has no MP3 writer; the streaming encoder does, and using
        # it here keeps one implementation of "what our MP3 sounds like".
        encoder = Mp3Stream()
        encoder.open(sample_rate)
        return encoder.encode(pcm_frames(samples)) + encoder.close()
    buffer = io.BytesIO()
    subtype = "PCM_16" if fmt in ("wav", "flac") else None
    sf.write(buffer, samples, sample_rate, format=fmt.upper(), subtype=subtype)
    return buffer.getvalue()


# Level a reference recording to this RMS before conditioning an engine on it.
# Measured on MOSS-TTS-Nano across eight references: output peak tracks the
# reference's loudness monotonically, and references hotter than about
# -16 dBFS RMS made the model generate hard-clipped audio (peaks of 1.05 to
# 1.13 on a float waveform). Levelling the set to this value dropped the worst
# peak to 0.905.
REFERENCE_RMS_DBFS = -21.0

# Never let levelling push a peak above this, however quiet the recording's
# average is. A reference with one loud transient would otherwise be scaled
# until that transient clipped.
REFERENCE_PEAK_CEILING = 0.89


def level(audio: np.ndarray, target_dbfs: float = REFERENCE_RMS_DBFS) -> np.ndarray:
    """Scale a waveform to a target RMS, without letting its peak clip.

    Peak normalisation is the wrong tool here: a recording with one loud
    transient normalises to a quiet average, and it is the average the model
    responds to.

    Args:
        audio: Mono float32 waveform.
        target_dbfs: Desired RMS in dBFS.

    Returns:
        The scaled waveform. Silence is returned unchanged.
    """
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-6 or peak < 1e-6:
        return audio
    gain = min(10.0 ** (target_dbfs / 20.0) / rms, REFERENCE_PEAK_CEILING / peak)
    return (audio * gain).astype(np.float32)


def decode_reference(
    path: str,
    *,
    sample_rate: int,
    channels: int = 1,
    target_dbfs: float | None = REFERENCE_RMS_DBFS,
) -> np.ndarray:
    """Read a reference recording in the shape an engine wants to condition on.

    One place decodes, resamples, downmixes and levels, so engines do not each
    grow their own — and so the levelling that stops a hot reference producing
    clipped speech applies to all of them.

    Args:
        path: The stored recording.
        sample_rate: Rate the engine's codec expects.
        channels: 1 for mono, 2 to duplicate mono into stereo.
        target_dbfs: RMS to level to, or ``None`` to leave the level alone.

    Returns:
        Float32 of shape ``(samples,)`` for mono, else ``(channels, samples)``.

    Raises:
        ValueError: The file could not be decoded.
    """
    try:
        samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as err:
        raise ValueError(f"could not decode reference audio: {err}") from err

    mono = np.asarray(samples, dtype=np.float32).mean(axis=1)
    if source_rate != sample_rate:
        mono = _resample(mono, int(source_rate), sample_rate)
    if target_dbfs is not None:
        mono = level(mono, target_dbfs)
    if channels == 1:
        return mono
    return np.repeat(mono[np.newaxis, :], channels, axis=0)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample with scipy's polyphase filter, which is exact for rate ratios."""
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(source_rate, target_rate)
    resampled = resample_poly(audio, target_rate // divisor, source_rate // divisor)
    return np.asarray(resampled, dtype=np.float32)


# A WAV header must declare its size, and a stream does not know it yet. Every
# player we care about — and Home Assistant's own proxy — reads to end-of-body
# rather than trusting this, so the two size fields are written maximal. The
# alternative, buffering until the length is known, is the latency this whole
# path exists to remove.
_STREAMING_SIZE = 0xFFFFFFFF


def wav_header(sample_rate: int, *, channels: int = 1, bits: int = 16) -> bytes:
    """Return a RIFF/WAVE header for a stream of unknown length.

    Args:
        sample_rate: Samples per second.
        channels: Channel count.
        bits: Bits per sample; only 16 is produced by this app.

    Returns:
        The 44 bytes that precede raw PCM frames.
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", _STREAMING_SIZE),
            b"WAVEfmt ",
            struct.pack(
                "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits
            ),
            b"data",
            struct.pack("<I", _STREAMING_SIZE),
        )
    )


def pcm_frames(audio: np.ndarray) -> bytes:
    """Return one chunk of mono float32 as little-endian 16-bit PCM.

    The clamp is the last line of defence, not the level control: anything it
    actually has to clamp is already audible as distortion. `StreamGain` is
    what keeps samples under the ceiling.
    """
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


# Bitrate of the MP3 a stream is encoded at. Mono speech is transparent well
# below this; the margin is there because the audio is usually transcoded once
# more before it reaches a speaker, and a lossy step feeding a second one is
# where artefacts compound. At 24 kB/s it is still a quarter of raw PCM.
MP3_BITRATE = 192_000


class StreamEncoder(Protocol):
    """How a stream's samples become the bytes a player receives.

    A format belongs here only if it can be written without knowing how long
    the audio will be — which is the whole difficulty of streaming synthesis,
    and why `/api/speak`'s FLAC and OGG are not offered.
    """

    content_type: str
    """What to put in the response's Content-Type."""

    bitrate: int
    """Bits per second of the encoded stream, or 0 when it is not constant.

    A consumer that must know how much audio it holds reads this: a constant
    bitrate turns a byte count into a duration without decoding anything.
    """

    def open(self, sample_rate: int) -> bytes:
        """Return whatever precedes the first sample. May be empty."""
        ...

    def encode(self, pcm: bytes) -> bytes:
        """Return the encoding of one chunk of 16-bit mono PCM."""
        ...

    def close(self) -> bytes:
        """Return whatever the encoder still holds. May be empty."""
        ...


class WavStream:
    """Raw PCM behind a header that declares a length nobody knows yet.

    Kept because it costs nothing and some consumers want exactly this, but
    it is not the default: the maximal length it has to declare is read by a
    general-purpose player as a six-hour file, which it then waits to buffer.
    """

    content_type = "audio/wav"

    def __init__(self) -> None:
        self.bitrate = 0

    def open(self, sample_rate: int) -> bytes:
        """Return the 44-byte header, and fix the rate the bitrate implies."""
        self.bitrate = sample_rate * 16
        return wav_header(sample_rate)

    def encode(self, pcm: bytes) -> bytes:
        """Pass the samples straight through."""
        return pcm

    def close(self) -> bytes:
        """Nothing is held back."""
        return b""


class Mp3Stream:
    """A bare sequence of MPEG frames, which needs no length and no index.

    Every frame carries its own header, so a player decodes what it has and
    asks for more — the behaviour a streamed reply needs and the one a WAV
    header cannot express.
    """

    content_type = "audio/mpeg"

    def __init__(self, bitrate: int = MP3_BITRATE) -> None:
        self.bitrate = bitrate
        self._encoder: Any = None

    def open(self, sample_rate: int) -> bytes:
        """Configure the encoder. MP3 has nothing to send before the audio."""
        import lameenc

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(self.bitrate // 1000)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(1)
        # LAME's own scale, where 2 is best and 7 is fastest. Five is its
        # default and costs 0.23% of a core for ten seconds of speech.
        encoder.set_quality(5)
        self._encoder = encoder
        return b""

    def encode(self, pcm: bytes) -> bytes:
        """Return whatever frames this chunk completed."""
        return bytes(self._encoder.encode(pcm))

    def close(self) -> bytes:
        """Flush the frame the last chunk left half-written."""
        if self._encoder is None:
            return b""
        return bytes(self._encoder.flush())


STREAM_ENCODERS: dict[str, Callable[[], StreamEncoder]] = {
    "wav": WavStream,
    "mp3": Mp3Stream,
}
"""Formats `/api/speak/stream` can produce, by the name a caller asks for."""


class StreamGain:
    """A one-way output gain for a single stream, so no chunk hard-clips.

    `encode` peak-normalises a finished waveform. A stream has no finished
    waveform to measure, and for a while had no level control at all — which
    left `pcm_frames` clamping whatever the model generated above full scale.
    Measured on MOSS-TTS-Nano over a Home-Assistant-shaped reply: 11 clamped
    runs in 8.6 s of audio, against none from `/api/speak` on the same text.

    Scaling each chunk to its own peak is the obvious fix and the wrong one —
    every chunk would arrive at the same loudness and the speech would pump.
    So this gain only ever falls, and only as far as the chunk that asked for
    it needs. A stream that never approaches the ceiling passes through
    untouched; a loud one steps down once or twice, by about a decibel, and
    stays there.

    One instance per stream. Home Assistant sends a request per sentence, so
    two sentences can settle on different gains — a step of that size at a
    sentence boundary is not audible, and clipping is.
    """

    __slots__ = ("_gain",)

    def __init__(self) -> None:
        self._gain = 1.0

    def frames(self, chunk: np.ndarray) -> bytes:
        """Return one chunk as PCM, scaled to keep its peak under the ceiling."""
        samples = np.asarray(chunk, dtype=np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak * self._gain > TARGET_PEAK:
            self._gain = TARGET_PEAK / peak
        return pcm_frames(samples * self._gain if self._gain < 1.0 else samples)
