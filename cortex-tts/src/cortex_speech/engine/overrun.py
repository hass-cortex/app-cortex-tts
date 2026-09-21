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

# Seeds tried after the model's own, in order, and only when a generation
# truncates. The first seed is not here because it is the model's: a Hojo LM
# defaults to 42 and other runtimes to 0, so hardcoding one would silently
# change what another produces for an ordinary request.
RETRY_SEEDS = (7, 1234)


def expected_seconds(text: str) -> float:
    """Roughly how long the text takes to say, by the script it reads as."""
    rate = CHARS_PER_SECOND if is_chinese(text) else LATIN_CHARS_PER_SECOND
    return len(text) / rate


def looks_truncated(
    text: str, seconds: float, *, ratio: float = TRUNCATION_RATIO
) -> bool:
    """Whether a generation stopped far short of the text it was given.

    The model can emit its end-of-speech token early, which yields a clean but
    incomplete clip — no error, no warning, just a sentence that stops. There
    is no signal for it in the output, so the length of the text is the only
    thing left to compare against.

    Args:
        text: The segment the generation was asked for.
        seconds: How much audio came back.
        ratio: How far below the estimate counts as short, for engines whose
            pacing sits differently against `expected_seconds` than the
            default was measured on. The number is only as good as that
            estimate, so it is per engine rather than one figure for all.
    """
    if len(text) < MIN_JUDGEABLE_CHARS:
        return False
    return seconds < expected_seconds(text) * ratio


# A gap this long inside one segment separates utterances rather than words.
BABBLE_GAP_SECONDS = 0.25

# Only look for babble when the clip runs well past what the text needs;
# ordinary sentences vary but never by this much.
OVERRUN_RATIO = 1.5

# And only on text short enough for `CHARS_PER_SECOND` to be worth judging by.
# That figure is one number for every voice and every sentence, and a voice's
# own pace moves further than `OVERRUN_RATIO` allows: across the eleven
# requests of one qwen3-tts-0.6b reply the same voice ran 2.81 to 5.19
# characters a second. A 20-character sentence that genuinely took 7.12 s
# therefore read as 1.6x its estimate and lost its last clause — transcribed,
# the kept audio said 「后来他学会了最远的远方。」 where the model had said
# 「后来他学会了：最远的远方不一定在门外。」
#
# The pathology this exists for is at the other end: a model that fails to
# emit end-of-speech promptly on very short input, measured at 2 characters
# ("好了", 0.7 s of text as 2.06 s) and at 5 ("大燈已關閉", 0.90 s as 1.74 s).
# Between the two failures the choice is not close. Babble is a stray syllable
# heard after the sentence; an over-cut is the sentence without its ending,
# and nothing downstream can tell that it happened.
MAX_BABBLE_CHARS = 20

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
    remains still covers the duration the text needs.

    Short input is also the only input this can judge: everything here rests
    on `expected_seconds`, and past `MAX_BABBLE_CHARS` that estimate is looser
    than the ratio it is compared against. The floor is no backstop there —
    it is anchored on the same estimate.
    """
    seconds = len(audio) / sample_rate
    expected = expected_seconds(text)
    if not text or len(text) >= MAX_BABBLE_CHARS:
        return audio
    if seconds <= expected * OVERRUN_RATIO:
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
    text: str,
    sample_rate: int,
    generate: Callable[[int], np.ndarray],
    *,
    seed: int,
    ratio: float = TRUNCATION_RATIO,
    trim: bool = True,
) -> np.ndarray:
    """Render one segment, retrying a generation that stopped early.

    Sampling is seeded, so a seed that makes the model emit end-of-speech
    early does so on every retry of the same text; only a different seed
    changes the outcome. When every seed comes up short the best effort is
    returned rather than nothing, the log having said it is incomplete.

    Args:
        text: The segment, for judging whether what came back is short.
        sample_rate: The engine's, for turning samples into seconds.
        generate: Renders the segment at one seed.
        seed: The model's own default, tried first so an ordinary request
            gets what that model produces unseeded. It is per model — a Hojo
            LM defaults to 42 — and the streaming path, which
            cannot retry, uses the same one.
        ratio: Passed to `looks_truncated`; see there.
        trim: Whether to also drop trailing babble. Only for models the
            pathology was measured on: past `MAX_BABBLE_CHARS` the trimmer is
            inert anyway, and under it an over-cut is the worse of the two
            failures — see the comment on `MAX_BABBLE_CHARS`.
    """
    wave = np.zeros(0, dtype=np.float32)
    # De-duplicated: a model whose own seed is already in `RETRY_SEEDS` would
    # otherwise spend a whole render repeating its first attempt, which
    # sampling being seeded makes identical down to the sample.
    seeds = tuple(dict.fromkeys((seed, *RETRY_SEEDS)))
    for attempt, seed in enumerate(seeds):
        wave = generate(seed)
        seconds = len(wave) / sample_rate
        if not looks_truncated(text, seconds, ratio=ratio):
            return trim_trailing_babble(wave, sample_rate, text) if trim else wave
        _LOGGER.warning(
            "generation for %d chars stopped at %.1fs (seed %d, attempt %d/%d)",
            len(text),
            seconds,
            seed,
            attempt + 1,
            len(seeds),
        )
    return wave
