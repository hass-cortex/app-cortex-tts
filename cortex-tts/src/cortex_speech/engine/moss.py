"""MOSS-TTS-Nano: bundled voices *and* cloning, in one engine.

The first model here that has both. `voice` is resolved against the bundle's
own table first and the reference store second, which is why the protocol's
opaque voice id needed no widening to accommodate it.

Two facts drive the implementation:

- Conditioning is cached. The runtime's own API takes a *path* and re-encodes
  the recording on every call.
- `sample_mode` is left at the runtime's `fixed`. The other two modes are
  slower and stop less reliably.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from ..providers import (
    ExecutionProvider,
    in_use,
    release_sessions,
    requested,
    sessions_of,
    verify,
)
from ..references import Reference, ReferenceStore
from ..vendor.moss_runtime import OnnxTtsRuntime
from .base import (
    Delivery,
    NoAudioError,
    StopCheck,
    Synthesis,
    UnknownVoiceError,
    Voice,
    check_stop,
)
from .conditioning import ConditioningCache
from .join import fade_in, join_segments, segment_gap
from .overrun import looks_truncated, render_with_retries

_LOGGER = logging.getLogger(__name__)

_GENDER_HINTS = {"female": "female", "male": "male"}

# Chunks the render may run ahead by. The codec emits roughly 8 frames at a
# time at 12.5 Hz, so this is a few seconds of audio.
_STREAM_QUEUE_CHUNKS = 8

# The runtime seeds its sampler once, when it is constructed, and every
# generated frame draws from it — so the second synthesis continues where the
# first left off and the same text comes out differently every time. Measured
# on three consecutive requests for one 37-character line: 6.80 s, 7.52 s and
# 9.60 s, a 41% spread in speaking rate and 1.5 dB in level. A reply split
# across requests is several syntheses, so the halves were paced differently
# from each other. Seeding per segment makes a reply reproducible and its
# pieces consistent; the sampling itself is untouched (`sample_mode` stays
# `fixed`, which is the mode that stops reliably).
_SAMPLING_SEED = 1234

# How far below `overrun.expected_seconds` a generation may fall before it is
# taken for one that stopped early. Tighter than the shared default because
# this model's truncations land close to its honest variation: across two
# replies chunked four ways, the generations that kept all their text ran
# 0.83 to 1.23 of the estimate (0.83 and 0.91 the two lowest, both Mandarin)
# and the ones that lost text ran 0.77 and 0.41. The window is (0.77, 0.83)
# and this sits in it. The margin above is 0.03, so the error this makes is a
# needless re-render rather than a sentence delivered without its ending.
# The figures are in `overrun`'s own frame (`CHARS_PER_SECOND` 4.5,
# `LATIN_CHARS_PER_SECOND` 14.0), not `text.scripts`' 4.1 and 14.7.
_TRUNCATION_RATIO = 0.8

# A chunk arrives every few hundred milliseconds, so an abandoned worker
# notices within one chunk; this only bounds a decode that never returns.
_STREAM_JOIN_SECONDS = 5.0


class _StreamAbandonedError(Exception):
    """Raised inside the runtime's decode loop once the consumer is gone."""


def _describe(entry: dict[str, Any]) -> Voice:
    """Turn a bundled voice row into the app's voice metadata.

    Upstream's rows carry a `group` like "Chinese Female" and a `display_name`
    describing the reference clip rather than the voice, so the group is what
    the language and gender come from.
    """
    voice_id = str(entry.get("voice") or "")
    group = str(entry.get("group") or "")
    lower = group.lower()
    language = None
    for code, prefix in (("zh", "chinese"), ("en", "english"), ("ja", "japanese")):
        if lower.startswith(prefix):
            language = code
            break
    gender = next((g for word, g in _GENDER_HINTS.items() if word in lower), "unknown")
    label = group or voice_id
    return Voice(
        id=voice_id,
        name=f"{voice_id} ({label})" if group else voice_id,
        language=language,
        gender=gender,
        source="builtin",
    )


def own_voices(directory: Path) -> list[Voice]:
    """Read the bundle's voice table from its manifest, loading no sessions.

    Named separately from the engine because listing voices must not load a
    model — see `EngineRegistry.voices`. The manifest is the runtime's own
    source for this list, so the two cannot disagree.
    """
    manifest_path = OnnxTtsRuntime._resolve_manifest_path(directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [_describe(entry) for entry in manifest["builtin_voices"]]


def _downmix(waveform: np.ndarray) -> np.ndarray:
    """Reduce the runtime's output to mono, whichever axis holds channels.

    The codec emits `(samples, channels)`; a gap this engine inserts is already
    1-D. Picking the smaller axis is right for both, and for the PyTorch path's
    `(channels, samples)` if it is ever wired up.
    """
    if waveform.ndim != 2:
        return waveform.reshape(-1).astype(np.float32, copy=False)
    axis = 0 if waveform.shape[0] < waveform.shape[1] else 1
    return waveform.mean(axis=axis).astype(np.float32)


class _CodesSidecar:
    """Prompt codes as JSON: exact, and ragged rows would not fit an array."""

    suffix = ".moss.json"

    def dump(self, value: list[list[int]], path: Path) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def load(self, path: Path) -> list[list[int]]:
        return json.loads(path.read_text(encoding="utf-8"))


class MossEngine:
    """Wraps the vendored MOSS runtime, caching per-reference conditioning."""

    def __init__(
        self,
        models_dir: Path,
        references: ReferenceStore,
        *,
        num_threads: int = 0,
        temperature: float = 0.8,
        execution_provider: ExecutionProvider = "auto",
        max_text_tokens: int | None = None,
    ) -> None:
        """Load the MOSS bundle.

        Args:
            models_dir: Directory holding the downloaded ONNX bundle.
            references: Store of uploaded reference recordings.
            num_threads: ONNX Runtime thread count; 0 lets ORT decide.
            temperature: Ignored. This model's sampling is fused into a
                dedicated ONNX graph, which is why its catalog entry declares
                `temperature=False` and the API refuses to accept one.
            execution_provider: Which provider to ask ONNX Runtime for. This is
                the model the choice matters most for — measured at RTF 1.025
                on a laptop i7 against 0.354 on a GTX 1650, which is the
                difference between falling behind playback and outrunning it.
            max_text_tokens: `ModelSpec.max_text_tokens`; a segment carrying
                more is cut into chunks before it reaches the model. `None`
                passes every segment through whole.

        Raises:
            ProviderUnavailableError: `cuda` was required and CPU is what the
                sessions came back on.
        """
        del temperature
        self._references = references
        self._max_text_tokens = max_text_tokens
        started = time.perf_counter()
        self._runtime = OnnxTtsRuntime(
            model_dir=str(models_dir),
            thread_count=num_threads,
            execution_provider=requested(execution_provider),
        )
        self.provider = verify(execution_provider, in_use(sessions_of(self._runtime)))
        self._voices = [_describe(v) for v in self._runtime.list_builtin_voices()]
        self._builtin_ids = {v.id for v in self._voices}
        self._prompts: ConditioningCache[list[list[int]]] = ConditioningCache(
            _CodesSidecar(), references.root
        )
        _LOGGER.info(
            "loaded MOSS bundle from %s in %.2fs on %s (%d built-in voices)",
            models_dir,
            time.perf_counter() - started,
            self.provider,
            len(self._voices),
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return int(self._runtime.codec_meta["codec_config"]["sample_rate"])

    def forget(self, reference_id: str) -> None:
        """Drop cached conditioning for a reference that changed or went away."""
        self._prompts.forget(reference_id)

    def close(self) -> None:
        """Release the sessions; see `Engine.close`."""
        release_sessions(self._runtime)

    def _encode(self, reference: Reference) -> list[list[int]]:
        """Run the codec encoder over a reference recording."""
        started = time.perf_counter()
        codes = self._runtime.encode_reference_audio(reference.audio_path)
        _LOGGER.info(
            "encoded reference %s into %d prompt frames in %.0fms",
            reference.id,
            len(codes),
            (time.perf_counter() - started) * 1000,
        )
        return codes

    def _prompt_codes(self, voice: str) -> list[list[int]]:
        """Resolve a voice id to the codes that condition a render.

        Bundled voices carry their codes in the manifest. A reference is
        encoded once and kept: the runtime's own path would re-read and
        re-encode the recording for every sentence.
        """
        if voice in self._builtin_ids:
            return self._runtime.resolve_prompt_audio_codes(
                voice=voice, prompt_audio_path=None
            )
        reference = self._references.get(voice)
        if reference is None:
            raise UnknownVoiceError(f"unknown voice {voice!r}")
        return self._prompts.get(reference, self._encode)

    def _reseed(self, seed: int = _SAMPLING_SEED) -> None:
        """Put the runtime's sampler back to a known starting state.

        Reaching for the attribute is the vendor boundary's fault: the entry
        point this engine uses, `synthesize_single_chunk`, takes no seed, and
        the one that does is the file-writing path we do not call.
        """
        self._runtime.rng = np.random.default_rng(seed)

    def _chunks(self, segments: list[str]) -> list[str]:
        """Segments cut down to what one call into this model may carry.

        The budget is counted in the model's own text tokens, so the runtime's
        splitter is the one that can apply it; it cuts on clause punctuation,
        which is where a pause belongs anyway. A segment already inside the
        budget is passed through untouched rather than round-tripped, because
        that splitter also normalises what it is given — it appends a stop and
        pads very short Latin text — and nothing but the app's own text path
        may decide what the model is asked to say.
        """
        budget = self._max_text_tokens
        if budget is None:
            return segments
        out: list[str] = []
        for text in segments:
            if self._runtime.count_text_tokens(text) <= budget:
                out.append(text)
                continue
            pieces = self._runtime.split_voice_clone_text(text, max_tokens=budget)
            _LOGGER.debug(
                "segment of %d chars split into %d chunks", len(text), len(pieces)
            )
            out.extend(pieces)
        return out

    def _render(
        self, text: str, codes: list[list[int]], stop: StopCheck | None
    ) -> np.ndarray:
        """Render one chunk, retrying a generation that stopped early.

        Stopping is a token this model samples, so a seed that ends a chunk
        early ends it early on every repeat of the same text and only another
        seed changes the outcome — measured on one 411-character chunk, which
        came back at 12.00 s under the pinned seed and 26.64 s under the next
        one tried.
        """

        def generate(seed: int) -> np.ndarray:
            self._reseed(seed)
            # The callback is the only place inside the runtime's decode loop
            # this code runs, so it is where a lost listener is noticed.
            result = self._runtime.synthesize_single_chunk(
                text=text,
                prompt_audio_codes=codes,
                streaming=True,
                on_audio_chunk=lambda _chunk: check_stop(stop),
            )
            return _downmix(np.asarray(result.get("waveform"), dtype=np.float32))

        return render_with_retries(
            text,
            self.sample_rate,
            generate,
            seed=_SAMPLING_SEED,
            ratio=_TRUNCATION_RATIO,
            # The babble pathology the trimmer exists for has not been measured
            # on this model, and an over-cut is the worse of the two failures.
            trim=False,
        )

    def synthesize(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Synthesis:
        """Render segments with a bundled voice or a cloned one."""
        del delivery  # nothing in it applies; see __init__ and the catalog
        if not segments:
            raise NoAudioError("no segments to synthesize")

        codes = self._prompt_codes(voice)
        chunks = self._chunks(segments)
        started = time.perf_counter()
        waves: list[np.ndarray] = []
        for text in chunks:
            check_stop(stop)
            waves.append(self._render(text, codes, stop))

        if not any(wave.size for wave in waves):
            raise NoAudioError(f"model produced no audio for {segments!r}")

        return Synthesis(
            audio=join_segments(waves, self.sample_rate),
            sample_rate=self.sample_rate,
            segments=len(waves),
            inference_ms=(time.perf_counter() - started) * 1000,
        )

    def synthesize_stream(
        self,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery = Delivery(),
        stop: StopCheck | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """Yield audio as the codec produces it, rather than per segment.

        This is what the `chunk_streaming` capability means.

        The runtime calls back synchronously from inside its decode loop, so
        the synthesis runs on a worker thread and the chunks travel through a
        queue. Collecting them and yielding afterwards would render the whole
        segment first and produce a generator that streams nothing — which is
        the failure this shape exists to avoid.
        """
        del delivery  # nothing in it applies; see __init__ and the catalog
        # `stop` is the registry's to ask between chunks: closing this
        # generator is what stops the worker, and it already does that.
        del stop
        if not segments:
            raise NoAudioError("no segments to synthesize")

        codes = self._prompt_codes(voice)
        chunks_of = self._chunks(segments)
        # From join.py, so a streamed reply pauses between sentences exactly
        # as a rendered one does.
        gap = segment_gap(self.sample_rate)
        # Bounded so the render thread cannot run ahead of a slow consumer and
        # hold a whole utterance in memory — at 48 kHz stereo float32 that is
        # ~11 MB per segment. Blocking the worker costs nothing: it is not the
        # event loop, and the lead it keeps is still seconds of audio.
        chunks: queue.Queue[np.ndarray | None | BaseException] = queue.Queue(
            maxsize=_STREAM_QUEUE_CHUNKS
        )
        # Set when the consumer goes away. The worker is inside the runtime's
        # decode loop and can only notice between chunks, so every put polls
        # it: a put that blocked forever would keep the thread, and the
        # runtime's per-stream state, alive under the next request.
        abandoned = threading.Event()

        def offer(item: np.ndarray | None | BaseException) -> bool:
            while not abandoned.is_set():
                try:
                    chunks.put(item, timeout=0.1)
                except queue.Full:
                    continue
                return True
            return False

        # Samples the chunk being rendered has produced so far. A list because
        # `deliver` runs on the worker thread and only needs to add to it.
        chunk_samples = [0]

        def deliver(chunk: np.ndarray) -> None:
            array = np.asarray(chunk)
            chunk_samples[0] += array.shape[0] if array.ndim == 1 else max(array.shape)
            if not offer(chunk):
                raise _StreamAbandonedError

        def render() -> None:
            try:
                for index, text in enumerate(chunks_of):
                    # Nothing joins these afterwards, so the pause between
                    # sentences is emitted rather than added after.
                    if index and not offer(gap):
                        return
                    self._reseed()
                    chunk_samples[0] = 0
                    self._runtime.synthesize_single_chunk(
                        text=text,
                        prompt_audio_codes=codes,
                        streaming=True,
                        on_audio_chunk=deliver,
                    )
                    # Said rather than fixed: the chunks are already gone, and
                    # only another seed would change what this one produced.
                    seconds = chunk_samples[0] / self.sample_rate
                    if looks_truncated(text, seconds, ratio=_TRUNCATION_RATIO):
                        _LOGGER.warning(
                            "streamed chunk %d/%d of %d chars stopped at %.1fs; "
                            "a streamed generation cannot be retried",
                            index + 1,
                            len(chunks_of),
                            len(text),
                            seconds,
                        )
            except _StreamAbandonedError:
                return
            except Exception as err:  # noqa: BLE001 - re-raised on the consumer
                offer(err)
            finally:
                offer(None)

        worker = threading.Thread(target=render, name="moss-stream", daemon=True)
        worker.start()
        produced = False
        try:
            while True:
                item = chunks.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                mono = _downmix(np.asarray(item, dtype=np.float32))
                if not mono.size:
                    continue
                if not produced:
                    # join.py fades a rendered segment's edges to kill the
                    # click the model sometimes starts on. Only the leading
                    # edge is reachable here; the trailing one would need
                    # lookahead this path cannot afford.
                    mono = fade_in(mono, self.sample_rate)
                produced = True
                yield mono
        finally:
            abandoned.set()
            # Free a put in flight so the worker can see the flag.
            while not chunks.empty():
                try:
                    chunks.get_nowait()
                except queue.Empty:
                    break
            worker.join(timeout=_STREAM_JOIN_SECONDS)
            if worker.is_alive():
                _LOGGER.warning(
                    "moss stream worker did not stop within %ss", _STREAM_JOIN_SECONDS
                )

        if not produced:
            raise NoAudioError(f"model produced no audio for {segments!r}")
