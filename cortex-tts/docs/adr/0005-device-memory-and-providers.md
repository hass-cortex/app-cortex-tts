# ADR 0005 — The card is read, never assumed: providers, residency and running out of memory

Status: accepted.

## Context

The app runs on a 4 GB GTX 1650 where MOSS alone takes 2.9 GB and its
streaming path plateaus at 3.7 GB, and on HAOS where there is no NVIDIA
driver at all. `onnxruntime.get_available_providers()` is a claim about the
build, not a promise: on the GTX 1650 host it listed CUDA and then created
every session on the CPU.

## Decision

- **The execution provider is verified from the sessions.** `auto` takes a
  GPU when one answers and the CPU when none does; `cuda` refuses to fall
  back. `providers.in_use` reads what each session actually runs on, and
  `/health` reports what was asked for beside what arrived. A model that
  fits in RAM is not therefore a model that fits on the card.
- **At most `max_loaded_models` engines are resident**, least-recently-used
  evicted, and **one synthesis per engine at a time**: a session carries
  state across a render (a per-token loop on most engines, a fixed
  unmasking schedule on OmniVoice), so concurrency corrupts rather than
  slows.
- **A model's lifecycle is legible from the log alone.** The registry logs
  every transition — loading, resident, unloaded and why — with the card's
  memory and the change across it, because an arena that grew, an engine
  that kept its memory and another process taking the card leave the same
  absolute figure and are told apart only by the movement. Reading the
  figure costs a subprocess, so it is read once per transition and never on
  a CPU host. `loading` is a state `/health` reports (`loading_models`).
- **Running out of memory costs the engine, not the process.** An ONNX
  Runtime arena only grows and only the session losing its last reference
  returns it, so one exhausted render would leave the card full for every
  request after it. `providers.exhausted` recognises the condition and the
  registry drops the engine; the failed request is not retried, because a
  stream has usually sent audio by then. Dropping alone is not enough: an
  exception and its traceback reference each other, and every frame still
  holds the sessions and tensors. `traceback.clear_frames` empties those
  locals and keeps the frames, so the trace still prints and the card comes
  back at once — measured with the collector disabled, 3,716 MiB still held
  without it, 494 with. Exhaustion is recognised by message, in every
  spelling the runtime has: the arena's own, and the cuBLAS and cuDNN
  workspace failures a kernel reports once the arena holds the card
  (`device._EXHAUSTED`). With three models resident on the 4 GB card
  (3,666 MiB) Hojo failed every request with the latter two and, being
  unrecognised, was kept; nothing recovered short of a restart.
- **A failure after `ready` is said on the socket.** A listener whose reply
  simply stops cannot tell a crash from a slow render, so a runtime error
  nothing gave a wire shape to still ends in an `error` frame.
- **A render whose listener left stops within one unit of work.** Every
  engine takes a `stop` check between the units it produces — a decode step
  (Hojo), a diffusion step (OmniVoice), a codec chunk (MOSS) — and the
  registry asks it between streamed chunks. `AbandonedError`
  releases the lock and records nothing against the model. The transports
  supply the check: `/api/speak` polls for a disconnect, `/api/speak/live` a
  closed socket, a `cancel` frame, or a client that has not read for fifteen
  seconds. A player that stopped behind Home Assistant's TTS cache, which
  drains a stream without back-pressure, is past the boundary.
- **OmniVoice's transformer is never materialised.** The int4 ONNX graph
  replaces `forward` outright; the module is built on the meta device and
  the 2.45 GB checkpoint is neither downloaded nor loaded. Peak resident
  1.1 GB against 4.7 GB. Anything reading a weight outside `forward` fails
  with "Cannot copy out of meta tensor", loudly, at first synthesis.
- **A quantised model is not automatically a fast one.** A dynamically
  quantised per-token loop does not scale with threads, and the codec can
  cost as much as the model. Quantisation reliably buys memory, not time.

## Consequences

- What the card buys each model is measured in `docs/models.md` and nowhere
  else.
- A new engine must take the `stop` check, declare a `provider`, and let the
  registry own its memory; an engine with its own cache or its own retry is
  a second rule that will diverge.
