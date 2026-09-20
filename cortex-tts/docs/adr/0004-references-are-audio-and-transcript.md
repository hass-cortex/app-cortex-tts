# ADR 0004 — A reference is audio and transcript, validated on the way in

Status: accepted.

## Context

One recording is a voice on every cloning model at once, under the same id.
The models that clone best read the whole recording as a worked example, so
what is wrong with the recording is reproduced, silently, in every reply.

## Decision

- **A reference is audio _and_ transcript.** Neither alone defines a voice,
  and a wrong transcript degrades the clone with no error. Whether a model
  is told what the recording says is a capability,
  `ModelSpec.reads_reference_transcript`, declared per entry and pinned by a
  test against the engine that would read it; a flag set the permissive way
  promises a caller their transcript matters when nothing will read it.
- **Validation is the strict case's**, whatever the resident model reads:
  2–20 s, non-empty, pronounceable, and **ending in silence**. A clip cut by
  a clock teaches a model that sentences end mid-word, and the clone then
  drifts and clips its own endings. Measured on eight real uploads: the
  seven cut at a recorder's 7.00 s limit ended between −7.1 and +7.5 dB
  against their own average, the one that finished ended at −25.3 dB, so
  the threshold is −15 dB over the last 100 ms.
- **Conditioning is cached once, for every engine.** Encoding a reference is
  the dominant cost of a cloned utterance; `engine/conditioning.py` keeps it
  by reference id and drops it when the recording's fingerprint changes or
  `forget()` says so. An engine keeping its own cache would be a second
  invalidation rule, and the first one to diverge is silent — audio still
  comes out, in the previous voice.
- A transcript is stored as typed; nothing rewrites it. `PATCH
/api/references/{id}` fires the models-changed event only when the gender
  label or the language changed, because a corrected transcript changes
  nothing a picker shows.

## Consequences

- The user-facing rules are in `docs/cloning.md`; this is where they come
  from.
- A new cloning engine declares `reads_reference_transcript` only after
  reading the code path that would consume it.
