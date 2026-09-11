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
from functools import partial
from pathlib import Path

from ..catalog import BY_ID, EngineKind, ModelSpec, inspect, model_dir
from ..refs import ReferenceStore
from .base import Engine, EngineError, Synthesis, Voice, reference_voices
from .preset import PresetEngine

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
    ) -> None:
        """Create the registry.

        Args:
            data_dir: Root holding ``models/``.
            references: Reference store shared with the cloning engine.
            num_threads: ONNX Runtime thread count; 0 lets ORT decide.
            max_loaded: How many engines may stay in memory at once.
            temperature: Default sampling temperature for both engines.
        """
        self._data_dir = data_dir
        self._references = references
        self._num_threads = num_threads
        self._max_loaded = max(1, max_loaded)
        self._temperature = temperature
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
        directory = model_dir(self._data_dir, spec.id)
        if spec.kind is EngineKind.CLONE:
            # Imported here, not at module load: the cloning engine drags in
            # torch and librosa, and a problem in either must not stop the
            # preset model — or the whole app — from starting.
            from .clone import CloneEngine

            return CloneEngine(
                directory,
                self._references,
                num_threads=self._num_threads,
                temperature=self._temperature,
            )
        return PresetEngine(
            directory, num_threads=self._num_threads, temperature=self._temperature
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
        """Return the voices a model offers.

        A cloning model's voices come from the reference store, so listing
        them loads nothing. Asking the engine would pull 2 GB into memory to
        read a JSON index — and at the default of one resident model, evict
        the model that is actually speaking.
        """
        if self.spec(model_id).voices_from_references:
            return reference_voices(self._references)
        slot = await self.acquire(model_id)
        return slot.engine.voices()

    async def synthesize(
        self,
        model_id: str,
        segments: list[str],
        voice: str,
        *,
        temperature: float | None = None,
    ) -> Synthesis:
        """Render text with a model, loading and serialising as needed."""
        slot = await self.acquire(model_id)
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
