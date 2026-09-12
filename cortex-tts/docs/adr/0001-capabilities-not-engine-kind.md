# A model declares capabilities; it does not declare a kind

`EngineKind` had two members, `PRESET` and `CLONE`, and every model was one of
them. That held while both models came from one family: the 40M ships fixed
voices, the 80M clones from a recording, and no model did both. MOSS-TTS-Nano
does both — 18 bundled voices _and_ zero-shot cloning — and the enum has no
member for it. Adding `HYBRID` would not fix the shape, because the enum was
never describing one thing. It was carrying three:

- where a model's voices come from,
- whether a recording can condition it,
- which class to construct.

Those vary independently, so they become independent fields on `ModelSpec`:

    builtin_voices: bool    # ships selectable voices in the bundle
    cloning: bool           # can be conditioned on a reference recording
    backend: str            # which implementation to construct

The three models we ship land at (True, False), (False, True) and (True, True).
A voice list is `builtin_voices` voices from the engine concatenated with the
reference store's voices when `cloning` — which is also what removes the old
`voices_from_references` special case: it was `kind is CLONE` spelled twice.

`backend` is a key into a builder table that each engine module registers
itself in, replacing the `if spec.kind is EngineKind.CLONE:` branch in
`Registry._build`. That branch is the reason this ADR exists: a third engine
could not be added without editing it, and a fourth would make it a ladder.
With a table, an engine is added by writing a module and registering a key.

## What did not change

The `Engine` protocol — `sample_rate`, `voices()`, `synthesize()`, `forget()` —
absorbed MOSS without a single new method. `voice` was already an opaque id
that the engine resolves; MOSS resolves it against its bundled table first and
the reference store second, which is exactly what a both-kinds engine needs.
`forget()` already existed for reference invalidation. The protocol was right;
only the catalog above it was too narrow. We kept the protocol as-is rather
than widening it to fit the new model, and that constraint is what kept the
change small.

## Sampling controls are per-engine and stay out of the protocol

Hojo takes a `temperature`. MOSS takes a `sample_mode` of `greedy`, `fixed` or
`full`, and its `fixed` path — the default, and the fast one — ignores
temperature entirely because the sampling is fused into a dedicated ONNX graph.
Widening `synthesize()` with the union of every engine's knobs would put
`sample_mode` on an engine that has no such concept.

`temperature` stays in the signature because both engines genuinely have it.
Everything else travels in an `options` mapping that the spec declares and the
engine validates, so the admin UI can render a model's real controls without
the protocol knowing what they are.

## Capabilities we declare now and implement later

`chunk_streaming` is declared on the spec and is currently true only for MOSS.
The app streams per sentence today, which makes time-to-first-audio a function
of the first sentence's length; MOSS can emit sub-sentence chunks and measured
286 ms to first audio against Hojo's 1.64 s for the same opening line. The
capability is declared now so the catalog is honest and the API can branch on
it later; the streaming path itself is a separate change.

## What actually happened since

Kept as written above; corrected here rather than edited, because the decision
is the record and only its predictions were wrong.

**The `options` mapping was never built.** No engine needed a knob the protocol
could not carry, so MOSS's `sample_mode` is a constructor argument rather than
a declared control. What the spec grew instead is a fourth capability boolean —
`temperature: bool`, whether this model takes one at all, enforced at the API
edge so a request naming it against MOSS is refused rather than ignored. The
count in this ADR says three; there are four.

**`chunk_streaming` is no longer "declare now, implement later".** The
streaming path shipped: `/api/speak/stream` and `StreamingEngine`. Declaring
the capability separately from implementing the method turned out to be a real
hazard rather than a bookkeeping detail — a spec can claim it without the
engine having the method, and the failure is silent apart from a response
header asserting the opposite. `tests/test_catalog.py` now pins the two to each
other, deriving the engine class from `backends.py`.

**The independence claim held in the library and failed in the UI.** The
capabilities stayed independent everywhere in `cortex_speech`. The admin UI
quietly rebuilt `EngineKind` out of two of them — `cloning && !builtin_voices`
as a kind test — and mislabelled MOSS's 18 bundled voices as cloned for as long
as MOSS shipped. Independence in the data model does not survive on its own in
a consumer that finds it convenient to ask "which kind is this".
