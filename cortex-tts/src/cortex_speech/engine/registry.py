"""Which models are in memory, and the serialisation around them.

Two constraints shape this. Loading both bundles at once costs roughly 2.8 GB
of resident memory, which is more than a typical Home Assistant host wants to
give a text-to-speech service, so the registry keeps a bounded set and evicts
the least recently used. And ONNX Runtime sessions here drive a stateful
per-token loop, so each engine serves one request at a time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path

import numpy as np

from ..catalog import BY_ID, ModelSpec, inspect, model_dir
from ..providers import ExecutionProvider
from ..references import ReferenceStore
from .backends import BuildContext, build, builtin_voices
from .base import (
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
        temperature: float = 0.8,
        execution_provider: ExecutionProvider = "auto",
    ) -> None:
        """Create the registry.

        Args:
            data_dir: Root holding ``models/``.
            references: Reference store shared with the cloning engine.
            num_threads: ONNX Runtime thread count; 0 lets ORT decide.
            max_loaded: How many engines may stay in memory at once.
            temperature: Default sampling temperature for both engines.
            execution_provider: Which provider every engine asks ORT for.
        """
        self._data_dir = data_dir
        self._references = references
        self._num_threads = num_threads
        self._max_loaded = max(1, max_loaded)
        self._temperature = temperature
        self._execution_provider: ExecutionProvider = execution_provider
        self._slots: dict[str, _Slot] = {}
        self._load_lock = asyncio.Lock()

    @property
    def loaded_ids(self) -> set[str]:
        """Ids of the models currently in memory."""
        return set(self._slots)

    def is_loaded(self, model_id: str) -> bool:
        """Whether a model is currently in memory."""
        return model_id in self._slots

    def spec(self, model_id: str) -> ModelSpec:
        """Return a catalog entry, raising when the id is unknown."""
        spec = BY_ID.get(model_id)
        if spec is None:
            raise UnknownModelError(f"unknown model {model_id!r}")
        return spec

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
            engine = await asyncio.to_thread(self._build, spec)
            slot = _Slot(engine)
            self._slots[model_id] = slot
            return slot

    async def _evict_for(self, incoming_id: str) -> None:
        """Unload least-recently-used engines to make room for ``incoming_id``."""
        while len(self._slots) >= self._max_loaded:
            victim_id = min(self._slots, key=lambda k: self._slots[k].last_used)
            victim = self._slots[victim_id]
            # Never yank an engine out from under an in-flight request; wait
            # for it to finish, which is bounded by one synthesis.
            async with victim.lock:
                self._slots.pop(victim_id, None)
            _LOGGER.info("unloaded %s to make room for %s", victim_id, incoming_id)

    async def unload(self, model_id: str) -> bool:
        """Drop a model from memory. Returns whether it was loaded."""
        slot = self._slots.get(model_id)
        if slot is None:
            return False
        async with slot.lock:
            self._slots.pop(model_id, None)
        _LOGGER.info("unloaded %s", model_id)
        return True

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
        if spec.builtin_voices:
            state = inspect(self._data_dir, spec)
            if not state.downloaded:
                raise ModelNotReadyError(
                    f"{spec.name} is not downloaded "
                    f"({len(state.missing)} file(s) missing)"
                )
            voices.extend(
                await asyncio.to_thread(
                    builtin_voices, spec.backend, model_dir(self._data_dir, spec.id)
                )
            )
        if spec.cloning:
            voices.extend(reference_voices(self._references))
        return voices

    async def synthesize_stream(
        self,
        model_id: str,
        segments: list[str],
        voice: str,
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
        slot = await self.acquire(model_id)
        async with slot.lock:
            slot.last_used = time.monotonic()
            engine = slot.engine
            if not isinstance(engine, StreamingEngine):
                synthesis = await asyncio.to_thread(
                    partial(
                        engine.synthesize,
                        segments,
                        voice,
                        temperature=self._temperature,
                    )
                )
                yield synthesis.audio
                return

            # `next()` on a worker thread, not `async for`: the generator is
            # synchronous and blocking it would stall the event loop for the
            # whole render. `None` is the end marker because the contract is
            # arrays, so it cannot collide with a real chunk.
            chunks = engine.synthesize_stream(segments, voice)

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

    async def synthesize(
        self,
        model_id: str,
        segments: list[str],
        voice: str,
        *,
        temperature: float | None = None,
    ) -> Synthesis:
        """Render text with a model, loading and serialising as needed.

        The registry's default temperature is passed explicitly, so a resident
        engine adopts a changed default without being rebuilt.
        """
        slot = await self.acquire(model_id)
        if temperature is None:
            temperature = self._temperature
        async with slot.lock:
            slot.last_used = time.monotonic()
            return await asyncio.to_thread(
                partial(
                    slot.engine.synthesize, segments, voice, temperature=temperature
                )
            )

    def forget_reference(self, reference_id: str) -> None:
        """Tell loaded cloning engines a reference changed or was removed."""
        for slot in self._slots.values():
            slot.engine.forget(reference_id)

    async def reconfigure(
        self,
        *,
        num_threads: int | None = None,
        max_loaded: int | None = None,
        temperature: float | None = None,
        execution_provider: ExecutionProvider | None = None,
    ) -> bool:
        """Change the settings, dropping only what cannot adopt them.

        Thread count and execution provider are bound when ONNX Runtime
        creates a session, so when either changes every resident engine is
        unloaded and the next request pays a rebuild. The default temperature
        travels with each call and needs nothing dropped; a smaller resident
        bound evicts the least recently used down to it.

        Returns:
            Whether anything was unloaded, which is what the caller reports as
            "this takes effect on the next reply" rather than "now".
        """
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
        if temperature is not None:
            self._temperature = temperature

        dropped = False
        if rebuild:
            for model_id in list(self._slots):
                dropped |= await self.unload(model_id)
            return dropped
        while len(self._slots) > self._max_loaded:
            victim_id = min(self._slots, key=lambda k: self._slots[k].last_used)
            dropped |= await self.unload(victim_id)
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
