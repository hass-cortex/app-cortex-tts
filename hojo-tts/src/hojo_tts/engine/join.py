"""Stitching per-segment waveforms into one utterance."""

from __future__ import annotations

import numpy as np

# Long text is synthesised a sentence at a time, so the joins have to sound
# like sentence breaks rather than splices. A short silence supplies the pause
# the model would otherwise have to be prompted into.
SEGMENT_GAP_MS = 160

# The model occasionally ends a segment on a click. Fading the last and first
# few milliseconds costs nothing audible and removes the discontinuity.
EDGE_FADE_MS = 8


def _fade_edges(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    length = int(sample_rate * EDGE_FADE_MS / 1000)
    if length <= 0 or wave.size < length * 2:
        return wave
    faded = wave.copy()
    ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
    faded[:length] *= ramp
    faded[-length:] *= ramp[::-1]
    return faded


def join_segments(waves: list[np.ndarray], sample_rate: int) -> np.ndarray:
    """Concatenate segment waveforms with a short silence between them.

    Args:
        waves: Mono float32 waveforms in reading order.
        sample_rate: Shared sample rate.

    Returns:
        One float32 waveform. A single segment is returned unchanged apart
        from edge fading.
    """
    if not waves:
        return np.zeros(0, dtype=np.float32)
    if len(waves) == 1:
        return _fade_edges(np.asarray(waves[0], dtype=np.float32), sample_rate)

    gap = np.zeros(int(sample_rate * SEGMENT_GAP_MS / 1000), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, wave in enumerate(waves):
        if index:
            pieces.append(gap)
        pieces.append(_fade_edges(np.asarray(wave, dtype=np.float32), sample_rate))
    return np.concatenate(pieces)
