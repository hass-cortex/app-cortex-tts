"""Which models are in memory, and the serialisation around them.

Two constraints shape this. Loading both bundles at once costs roughly 2.8 GB
of resident memory, which is more than a typical Home Assistant host wants to
give a text-to-speech service, so the registry keeps a bounded set and evicts
the least recently used. And ONNX Runtime sessions here drive a stateful
per-token loop, so each engine serves one request at a time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from ..catalog import BY_ID, ModelSpec, inspect, model_dir
from ..device import Memory, exhausted, memory
from ..providers import ExecutionProvider
from ..references import ReferenceStore
from .backends import BuildContext, build, own_voices
from .base import (
    Delivery,
    Engine,
    EngineError,
    StreamingEngine,
    Synthesis,
    Voice,
    reference_voices,
)

_LOGGER = logging.getLogger(__name__)


class ModelNotReadyError(EngineError):
    """The model is not downloaded, so it cannot be loaded."""


class UnknownModelError(EngineError):
    """No catalog entry with that id."""


class OutOfMemoryError(EngineError):
    """The device ran out of memory rendering this, and the engine was dropped."""


def _device(now: Memory | None, before: Memory | None = None) -> str:
    """`, device 2916/4096 MiB (+2907 MiB)`, or nothing without a reading.

    The delta is the point. An arena that grew, an engine that did not give
    its memory back and another process taking the card all leave the same
    absolute number behind; only the movement across a known transition tells
    them apart.
    """
    if now is None:
        return ""
    if before is None:
        return f", device {now}"
    return f", device {now} ({now.used - before.used:+d} MiB)"


def _name(model_id: str) -> str:
    spec = BY_ID.get(model_id)
    return spec.name if spec else model_id


def _exhausted_rendering(model_id: str) -> OutOfMemoryError:
    """What a caller is told when a render ran the device out."""
    return OutOfMemoryError(
        f"{_name(model_id)} ran the device out of memory and has been unloaded; "
        "the next request loads it again. Shorten the text, lower the number of "
        "models kept in memory, or set the execution provider to cpu."
    )


def _exhausted_loading(model_id: str) -> OutOfMemoryError:
    """What a caller is told when there was no room to load at all.

    Nothing to drop — the session never came back, so nothing became resident.
    Not answered by evicting a resident model and trying again: at the default
    of one there is nothing to evict, and the pressure is usually something
    else on the device rather than this app holding two.
    """
    return OutOfMemoryError(
        f"there is not enough memory on the device to load {_name(model_id)}. "
        "Free some, lower the number of models kept in memory, or set the "
        "execution provider to cpu."
    )


class _Slot:
    """A loaded engine plus the lock that serialises access to it."""

    __slots__ = ("engine", "lock", "last_used")

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.lock = asyncio.Lock()
        self.last_used = time.monotonic()


class EngineRegistry:
    """Lazily loads engines, bounds how many stay resident, serialises calls."""

    def __init__(
        self,
        data_dir: Path,
        references: ReferenceStore,
        *,
        num_threads: int = 0,
        max_loaded: int = 1,
        idle_seconds: float = 0,
        temperature: float = 0.8,
        execution_provider: ExecutionProvider = "auto",
    ) -> None:
        """Create the registry.

        Args:
            data_dir: Root holding ``models/``.
            references: Reference store shared with the cloning engine.
            num_threads: ONNX Runtime thread count; 0 lets ORT decide.
            max_loaded: How many engines may stay in memory at once.
            idle_seconds: Drop an engine this long after its last request;
                0 keeps it until something evicts it.
            temperature: Default sampling temperature for both engines.
            execution_provider: Which provider every engine asks ORT for.
        """
        self._data_dir = data_dir
        self._references = references
        self._num_threads = num_threads
        self._max_loaded = max(1, max_loaded)
        self._idle_seconds = max(0.0, idle_seconds)
        self._reaper: asyncio.Task[None] | None = None
        self._temperature = temperature
        self._execution_provider: ExecutionProvider = execution_provider
        self._slots: dict[str, _Slot] = {}
        # Between the decision to load and the engine arriving there is no
        # slot, and a bundle takes seconds to become one. Without this the
        # registry answers "nothing resident" for the whole of that, which
        # reads the same as idle to everything that asks — `/health`, the
        # cards, a log line. The locks already keep that window safe; this is
        # so it can be seen.
        self._loading: set[str] = set()
        self._load_lock = asyncio.Lock()

    @property
    def loaded_ids(self) -> set[str]:
        """Ids of the models currently in memory."""
        return set(self._slots)

    @property
    def loading_ids(self) -> set[str]:
        """Ids of the models being built right now.

        At most one: every build holds the load lock. Disjoint from
        `loaded_ids` — a model is in one set or the other, never both.
        """
        return set(self._loading)

    def is_loaded(self, model_id: str) -> bool:
        """Whether a model is currently in memory."""
        return model_id in self._slots

    def spec(self, model_id: str) -> ModelSpec:
        """Return a catalog entry, raising when the id is unknown."""
        spec = BY_ID.get(model_id)
        if spec is None:
            raise UnknownModelError(f"unknown model {model_id!r}")
        return spec

    def _lru(self) -> str:
        """The resident model that has gone longest without a request."""
        return min(self._slots, key=lambda k: self._slots[k].last_used)

    def _memory(self) -> Memory | None:
        """The card's memory, when this registry is using a card at all.

        Reading it costs a subprocess, and a host set to `cpu` has asked for
        nothing this figure describes. `auto` still looks, because it may well
        have a card — on one that has none the lookup finds no `nvidia-smi`
        and costs a PATH scan, not a process. Once per lifecycle transition
        either way; never per request.
        """
        return memory() if self._execution_provider != "cpu" else None

    def _build(self, spec: ModelSpec) -> Engine:
        """Construct the engine a spec names, via the backend table."""
        return build(
            spec.backend,
            BuildContext(
                directory=model_dir(self._data_dir, spec.id),
                references=self._references,
                num_threads=self._num_threads,
                temperature=self._temperature,
                execution_provider=self._execution_provider,
            ),
        )

    async def acquire(self, model_id: str) -> _Slot:
        """Return the slot for a model, loading it first if needed.

        Loading is guarded by a single lock so two concurrent first-requests
        for the same model do not each pay the load cost — and, more to the
        point, do not both allocate a multi-gigabyte session.
        """
        slot = self._slots.get(model_id)
        if slot is not None:
            slot.last_used = time.monotonic()
            return slot

        spec = self.spec(model_id)
        async with self._load_lock:
            slot = self._slots.get(model_id)
            if slot is not None:
                slot.last_used = time.monotonic()
                return slot

            state = inspect(self._data_dir, spec)
            if not state.downloaded:
                raise ModelNotReadyError(
                    f"{spec.name} is not downloaded ({len(state.missing)} file(s) missing)"
                )

            await self._evict_for(model_id)
            self._loading.add(model_id)
            before = self._memory()
            _LOGGER.info(
                "model %s loading (%d/%d resident)%s",
                model_id,
                len(self._slots),
                self._max_loaded,
                _device(before),
            )
            try:
                engine = await asyncio.to_thread(self._build, spec)
            except Exception as err:
                if not exhausted(err):
                    raise
                raise _exhausted_loading(model_id) from err
            finally:
                self._loading.discard(model_id)
            slot = _Slot(engine)
            self._slots[model_id] = slot
            _LOGGER.info(
                "model %s resident (%d/%d)%s",
                model_id,
                len(self._slots),
                self._max_loaded,
                _device(self._memory(), before),
            )
            self._watch_idle()
            return slot

    def _watch_idle(self) -> None:
        """Start the idle sweep if there is a bound and nothing is sweeping."""
        if not self._idle_seconds or (self._reaper and not self._reaper.done()):
            return
        self._reaper = asyncio.create_task(self._sweep_idle(), name="idle-unload")

    async def _sweep_idle(self) -> None:
        """Drop engines nobody has asked for in `idle_seconds`.

        Runs only while something is resident and a bound is set, and ends
        itself otherwise; the next load starts it again. The bound is read
        every pass, so a settings change takes effect within one pass.
        `last_used` is stamped at both ends of a request, so a long stream is
        not counted as idle for the time it spent rendering.
        """
        while self._idle_seconds and self._slots:
            now = time.monotonic()
            due = [
                model_id
                for model_id, slot in self._slots.items()
                if now - slot.last_used >= self._idle_seconds
            ]
            for model_id in due:
                await self.unload(model_id, f"idle for {self._idle_seconds:.0f}s")
            if not self._slots:
                return
            soonest = min(
                slot.last_used + self._idle_seconds for slot in self._slots.values()
            )
            await asyncio.sleep(min(max(soonest - time.monotonic(), 0.05), 5.0))

    async def close(self) -> None:
        """Unload everything and stop sweeping. For shutdown."""
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
            self._reaper = None
        for model_id in list(self._slots):
            await self.unload(model_id, "shutting down")

    async def _evict_for(self, incoming_id: str) -> None:
        """Unload least-recently-used engines to make room for ``incoming_id``."""
        while len(self._slots) >= self._max_loaded:
            await self._drop(self._lru(), f"evicted for {incoming_id}")

    async def _drop(self, model_id: str, reason: str) -> bool:
        """Remove a model from the resident set and release what it held.

        Callers hold `_load_lock`; `unload` is the entry point that does not.
        `reason` is what the log line says, because an unload is never
        interesting on its own — which of the five things asked for it is.

        The engine is closed here, under its own lock, so the card's memory
        comes back at this line and the figure logged is the real one. Whoever
        else still holds the engine — a traceback, a future — holds a shell.
        """
        slot = self._slots.get(model_id)
        if slot is None:
            return False
        # Never yank an engine out from under an in-flight request; wait for
        # it to finish, which is bounded by one synthesis.
        async with slot.lock:
            if self._slots.get(model_id) is not slot:
                return False  # retired by the request that was holding it
            await self._retire(slot, model_id, reason)
        return True

    async def _retire(self, slot: _Slot, model_id: str, reason: str) -> None:
        """Take a slot out of service and close its engine. Under `slot.lock`."""
        before = self._memory()
        self._slots.pop(model_id, None)
        await asyncio.to_thread(slot.engine.close)
        _LOGGER.info(
            "model %s unloaded (%s)%s",
            model_id,
            reason,
            _device(self._memory(), before),
        )

    async def unload(self, model_id: str, reason: str = "asked") -> bool:
        """Drop a model from memory. Returns whether it was loaded.

        Takes the load lock so a build already in flight cannot insert the
        engine this was asked to remove after it has looked.
        """
        async with self._load_lock:
            return await self._drop(model_id, reason)

    async def voices(self, model_id: str) -> list[Voice]:
        """Return the voices a model offers, loading nothing.

        Both kinds are read off disk: built-in voices from the bundle's own
        manifest or voices file, reference voices from the store's index.
        Constructing the engine instead would pull up to 2 GB into memory to
        read a list of names — and at one resident model, evict the model that
        is speaking, which is what the caller usually wants the names *for*.

        A model can have both kinds, so the two lists are concatenated rather
        than chosen between.

        Raises:
            ModelNotReadyError: The bundle is not on disk to be read.
        """
        spec = self.spec(model_id)
        voices: list[Voice] = []
        if spec.builtin_voices or spec.designed_voices:
            state = inspect(self._data_dir, spec)
            if not state.downloaded:
                raise ModelNotReadyError(
                    f"{spec.name} is not downloaded "
                    f"({len(state.missing)} file(s) missing)"
                )
            voices.extend(
                await asyncio.to_thread(
                    own_voices, spec.backend, model_dir(self._data_dir, spec.id)
                )
            )
        if spec.cloning:
            voices.extend(reference_voices(self._references))
        return voices

    def _with_default_temperature(self, delivery: Delivery | None) -> Delivery:
        """Fill in the registry's own default temperature.

        Passed with every call rather than baked into the engine, so a
        resident engine adopts a changed default without being rebuilt.
        """
        delivery = delivery or Delivery()
        if delivery.temperature is None:
            return replace(delivery, temperature=self._temperature)
        return delivery

    @asynccontextmanager
    async def _lease(self, model_id: str) -> AsyncIterator[_Slot]:
        """Hold a model's engine for one request.

        Waiting for the lock is where a slot can go stale: an eviction or a
        failed request ahead in the queue retires it, and a waiter that then
        ran on the engine it was handed would be rendering on one nobody is
        counting — the card holding two models where it can hold one. So the
        slot is checked again once the lock is held, and a stale one is simply
        acquired afresh, which loads the model again if it has to.
        """
        while True:
            slot = await self.acquire(model_id)
            async with slot.lock:
                if self._slots.get(model_id) is slot:
                    slot.last_used = time.monotonic()
                    try:
                        yield slot
                    finally:
                        slot.last_used = time.monotonic()
                    return

    async def _exhausted(self, slot: _Slot, model_id: str) -> OutOfMemoryError:
        """Drop an engine whose device ran out of memory. Under `slot.lock`.

        An ONNX Runtime arena only grows, and only the session dying returns
        it — so a render that runs the card out leaves it full, and every
        request after it fails the same way. Retired before the lock is
        released, so the next request in the queue meets a fresh engine rather
        than the exhausted one. Measured on a 4 GB GTX 1650 with MOSS-TTS-Nano:
        3716 MiB standing after the failure, 110 MiB once the engine went.

        This request is not retried — a stream has usually sent audio by now,
        and a caller holding bytes cannot be handed a second attempt.
        """
        _LOGGER.warning("%s ran the device out of memory; unloading it", model_id)
        await self._retire(slot, model_id, "out of memory")
        return _exhausted_rendering(model_id)

    def _note_failure(
        self, how: str, model_id: str, segments: list[str], voice: str, err: Exception
    ) -> None:
        """Record what was being rendered when something went wrong.

        The traceback uvicorn prints says where in the code it broke and
        nothing about what was asked for. Every diagnosis of a synthesis
        failure so far has had to infer the endpoint from stack frames and
        guess at the text — so the one line only this layer can write is the
        one naming the model, the voice and how much was being said.

        `EngineError` and its subclasses are left alone: those are answers,
        not incidents, and each already carries its own message to the caller.
        """
        if isinstance(err, EngineError):
            return
        _LOGGER.warning(
            "%s failed on %s/%s (%d segment(s), %d chars): %s",
            how,
            model_id,
            voice,
            len(segments),
            sum(len(segment) for segment in segments),
            err,
        )

    async def synthesize_stream(
        self,
        model_id: str,
        segments: list[str],
        voice: str,
        delivery: Delivery | None = None,
    ) -> AsyncIterator[np.ndarray]:
        """Yield mono float32 chunks, as early as the model allows.

        An engine that satisfies `StreamingEngine` is asked for chunks; one
        that does not renders the whole utterance and yields it as a single
        chunk. A caller therefore never has to know which kind it has — only
        that the first chunk arrives sooner on some models than others.

        The engine lock is held for the whole stream, matching `synthesize`:
        these ONNX sessions drive a stateful per-token loop and interleaving
        two of them corrupts state rather than merely slowing things down.
        """
        wanted = self._with_default_temperature(delivery)
        async with self._lease(model_id) as slot:
            try:
                engine = slot.engine
                if not isinstance(engine, StreamingEngine):
                    synthesis = await asyncio.to_thread(
                        partial(engine.synthesize, segments, voice, delivery=wanted)
                    )
                    yield synthesis.audio
                    return

                # `next()` on a worker thread, not `async for`: the generator
                # is synchronous and blocking it would stall the event loop for
                # the whole render. `None` is the end marker because the
                # contract is arrays, so it cannot collide with a real chunk.
                chunks = engine.synthesize_stream(segments, voice, delivery=wanted)

                def pull() -> np.ndarray | None:
                    return next(chunks, None)

                try:
                    while (chunk := await asyncio.to_thread(pull)) is not None:
                        yield chunk
                finally:
                    # A consumer that stops early (a closed connection) must
                    # stop the engine's render too, and its cleanup joins a
                    # thread, so it does not belong on the event loop.
                    await asyncio.to_thread(chunks.close)
            except Exception as err:
                self._note_failure("stream", model_id, segments, voice, err)
                if not exhausted(err):
                    raise
                raise await self._exhausted(slot, model_id) from err

    async def synthesize(
        self,
        model_id: str,
        segments: list[str],
        voice: str,
        *,
        delivery: Delivery | None = None,
    ) -> Synthesis:
        """Render text with a model, loading and serialising as needed."""
        wanted = self._with_default_temperature(delivery)
        async with self._lease(model_id) as slot:
            try:
                return await asyncio.to_thread(
                    partial(slot.engine.synthesize, segments, voice, delivery=wanted)
                )
            except Exception as err:
                self._note_failure("render", model_id, segments, voice, err)
                if not exhausted(err):
                    raise
                raise await self._exhausted(slot, model_id) from err

    async def forget_reference(self, reference_id: str) -> None:
        """Tell loaded cloning engines a reference changed or was removed.

        Takes each engine's lock: a synthesis on a worker thread may be inside
        the conditioning cache's `get`, and forgetting between its miss and its
        insert puts the stale encoding straight back. And the load lock, so a
        build in flight cannot land an engine after the sweep has passed —
        that engine's cache is empty either way, but "every resident engine was
        told" should not depend on which of those two facts is true today.
        """
        async with self._load_lock:
            for slot in list(self._slots.values()):
                async with slot.lock:
                    slot.engine.forget(reference_id)

    async def reconfigure(
        self,
        *,
        num_threads: int | None = None,
        max_loaded: int | None = None,
        idle_seconds: float | None = None,
        temperature: float | None = None,
        execution_provider: ExecutionProvider | None = None,
    ) -> bool:
        """Change the settings, dropping only what cannot adopt them.

        Thread count and execution provider are bound when ONNX Runtime
        creates a session, so when either changes every resident engine is
        unloaded and the next request pays a rebuild. The default temperature
        travels with each call and needs nothing dropped; a smaller resident
        bound evicts the least recently used down to it.

        Taken under the load lock, because a build in flight is holding the
        settings this is replacing: without it, `_slots` is still empty when
        the sweep looks, the engine lands afterwards carrying the thread count
        and provider that were just discarded, and the caller is told nothing
        needed dropping.

        Returns:
            Whether anything was unloaded, which is what the caller reports as
            "this takes effect on the next reply" rather than "now".
        """
        async with self._load_lock:
            rebuild = False
            if num_threads is not None and num_threads != self._num_threads:
                self._num_threads = num_threads
                rebuild = True
            if (
                execution_provider is not None
                and execution_provider != self._execution_provider
            ):
                self._execution_provider = execution_provider
                rebuild = True
            if max_loaded is not None:
                self._max_loaded = max(1, max_loaded)
            if idle_seconds is not None:
                self._idle_seconds = max(0.0, idle_seconds)
                self._watch_idle()
            if temperature is not None:
                self._temperature = temperature

            dropped = False
            if rebuild:
                for model_id in list(self._slots):
                    dropped |= await self._drop(model_id, "settings changed")
                return dropped
            while len(self._slots) > self._max_loaded:
                dropped |= await self._drop(self._lru(), "resident bound lowered")
            return dropped

    @property
    def providers_in_use(self) -> dict[str, str]:
        """What each resident engine is actually running on.

        Read from the sessions rather than from the setting, because the two
        can disagree: a runtime that lists a CUDA provider it cannot load falls
        back with nothing but a warning. What was asked for is in the config;
        this is what arrived.
        """
        return {
            model_id: getattr(slot.engine, "provider", "cpu")
            for model_id, slot in self._slots.items()
        }
