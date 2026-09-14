"""Judging and trimming what the model produced.

The model stops only when it samples an end-of-speech token, so a rendered
waveform can be cut short or can run past the text. Both are decided from the
audio itself: the engines have no other signal.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from ..text.pipeline import is_chinese

_LOGGER = logging.getLogger(__name__)

# Rough speaking rates, characters per second, by the script the text reads
# as. Only used to notice a generation that stopped far short of the text or
# ran far past it — never to predict a duration. A Latin sentence has about
# three times as many characters per second of speech as a Mandarin one.
CHARS_PER_SECOND = 4.5
LATIN_CHARS_PER_SECOND = 14.0

# Below this fraction of the expected duration, the model almost certainly
# emitted end-of-speech early rather than genuinely finishing.
TRUNCATION_RATIO = 0.6

# Too short to judge: normal variation swamps the signal.
MIN_JUDGEABLE_CHARS = 20

# Seeds tried in order. The first matches the upstream default, so ordinary
# output is unchanged and reproducible; the rest only come into play when a
# generation truncates.
RETRY_SEEDS = (42, 7, 1234)


def expected_seconds(text: str) -> float:
    """Roughly how long the text takes to say, by the script it reads as."""
    rate = CHARS_PER_SECOND if is_chinese(text) else LATIN_CHARS_PER_SECOND
    return len(text) / rate


def estimated_audio_seconds(segments: list[str]) -> float:
    """Roughly how much audio these segments will make, summed by script.

    The same rough rates `expected_seconds` uses to judge truncation, put to a
    second use: estimating a render's length *before* it runs, so a reply too
    long to be worth rendering can be refused rather than discovered at a
    timeout. An estimate, never a promise — a caller acts on the order of
    magnitude, not the digits.
    """
    return sum(expected_seconds(segment) for segment in segments)


def looks_truncated(text: str, seconds: float) -> bool:
    """Whether a generation stopped far short of the text it was given.

    The model can emit its end-of-speech token early, which yields a clean but
    incomplete clip — no error, no warning, just a sentence that stops. There
    is no signal for it in the output, so the length of the text is the only
    thing left to compare against.
    """
    if len(text) < MIN_JUDGEABLE_CHARS:
        return False
    return seconds < expected_seconds(text) * TRUNCATION_RATIO


# A gap this long inside one segment separates utterances rather than words.
BABBLE_GAP_SECONDS = 0.25

# Only look for babble when the clip runs well past what the text needs;
# ordinary sentences vary but never by this much.
OVERRUN_RATIO = 1.5

# How far below the duration estimate the kept audio may fall. Well under the
# estimate on purpose: speech that simply ran fast must not be rejected.
KEEP_FLOOR_RATIO = 0.7


def _speech_runs(audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    """Return (start, end) sample ranges of speech, split on long silences."""
    hop = max(1, sample_rate // 100)
    frames = np.sqrt(
        np.array(
            [
                float((audio[i : i + hop] ** 2).mean())
                for i in range(0, max(0, len(audio) - hop), hop)
            ]
        )
        + 1e-12
    )
    if frames.size == 0:
        return []
    # Relative to this clip's own speech level, so quiet voices are not
    # mistaken for silence.
    threshold = max(float(frames.max()) * 0.06, 1e-4)
    voiced = frames > threshold
    gap_frames = max(1, int(BABBLE_GAP_SECONDS * sample_rate / hop))

    runs: list[tuple[int, int]] = []
    start: int | None = None
    silence = 0
    for index, is_voiced in enumerate(voiced):
        if is_voiced:
            if start is None:
                start = index
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= gap_frames:
                runs.append((start * hop, (index - silence + 1) * hop))
                start = None
    if start is not None:
        runs.append((start * hop, len(audio)))
    return runs


def trim_trailing_babble(audio: np.ndarray, sample_rate: int, text: str) -> np.ndarray:
    """Drop invented speech the model appended after finishing the text.

    On very short input the model often fails to emit end-of-speech promptly
    and carries on with a syllable or two of its own — audible as a stray
    noise after the sentence. Trailing utterances are dropped only while what
    remains still covers the duration the text needs, so a real pause inside a
    sentence is never cut.
    """
    seconds = len(audio) / sample_rate
    expected = expected_seconds(text)
    if not text or seconds <= expected * OVERRUN_RATIO:
        return audio

    runs = _speech_runs(audio, sample_rate)
    if len(runs) < 2:
        return audio

    keep = list(runs)
    floor = expected * KEEP_FLOOR_RATIO
    while len(keep) > 1 and (keep[-2][1] / sample_rate) >= floor:
        keep.pop()
    if len(keep) == len(runs):
        return audio

    end = min(len(audio), keep[-1][1] + int(0.08 * sample_rate))
    return audio[:end]


def render_with_retries(
    text: str, sample_rate: int, generate: Callable[[int], np.ndarray]
) -> np.ndarray:
    """Render one segment, retrying a generation that stopped early.

    Sampling is seeded, so a seed that makes the model emit end-of-speech
    early does so on every retry of the same text; only a different seed
    changes the outcome. When every seed comes up short the best effort is
    returned rather than nothing, the log having said it is incomplete.
    """
    wave = np.zeros(0, dtype=np.float32)
    for attempt, seed in enumerate(RETRY_SEEDS):
        wave = generate(seed)
        seconds = len(wave) / sample_rate
        if not looks_truncated(text, seconds):
            return trim_trailing_babble(wave, sample_rate, text)
        _LOGGER.warning(
            "generation for %d chars stopped at %.1fs (seed %d, attempt %d/%d)",
            len(text),
            seconds,
            seed,
            attempt + 1,
            len(RETRY_SEEDS),
        )
    return wave
