"""Turning a float waveform into bytes Home Assistant will play."""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import soundfile as sf

AudioFormat = Literal["wav", "flac", "ogg"]

CONTENT_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
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
    buffer = io.BytesIO()
    subtype = "PCM_16" if fmt in ("wav", "flac") else None
    sf.write(buffer, samples, sample_rate, format=fmt.upper(), subtype=subtype)
    return buffer.getvalue()
