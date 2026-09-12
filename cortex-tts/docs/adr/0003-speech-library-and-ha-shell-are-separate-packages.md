# The speech library and the Home Assistant shell are separate packages

Today the app is one flat package. `engine/`, `text/`, `vendor/`, `catalog.py`
and `refs.py` — which know nothing about Home Assistant, HTTP or the Supervisor
— sit as siblings of `app.py`, `config.py`, `api/`, `discovery.py` and
`supervisor.py`, which know about little else. Reading the tree does not tell
you which side of that line a file is on.

The line itself already holds. Nothing under `engine/`, `text/`, `vendor/`,
`catalog.py` or `refs.py` imports `api`, `app` or `config`, and none of them
mentions FastAPI, bashio, ingress, the Supervisor or Home Assistant. That is
worth stating plainly: this ADR is not repairing a tangle, it is making an
existing boundary visible and enforced before a second engine and a rename
give everyone a reason to cross it.

## What is actually coupled

One thing. `app.py` imports `engine.registry`, `download`, `refs` and
`catalog` directly and wires them together, passing `num_threads` and
`temperature` out of addon `Settings` into an engine registry constructor. The
shell is doing the library's assembly, which makes the library's public surface
"every module it happens to contain". Any internal reshuffle is then a shell
change, and that is the coupling to remove — not an import cycle, but an
undeclared API.

## The split

    src/
      cortex_speech/     # the library: models, engines, text, references
        __init__.py      # the entire public surface
        catalog.py  engine/  text/  vendor/
        references.py  audio.py  download.py  events.py
      cortex_tts/        # the Home Assistant app
        app.py  config.py  api/  discovery.py  supervisor.py  __main__.py

`cortex_speech/__init__.py` exports a narrow surface — a `SpeechService` facade
plus the value types that cross it (`ModelSpec`, `Voice`, `Synthesis`, the
error classes). The shell constructs one `SpeechService` from a library-owned
config object and never reaches past it. Addon options are translated to that
config once, at the boundary, which is the only place that should know both
vocabularies.

The library keeps no dependency on FastAPI, aiohttp or bashio, and the
direction is one-way: `cortex_tts` imports `cortex_speech`, never the reverse.
Splitting it into its own distribution later becomes a packaging decision
rather than a refactor.

## Enforcement is a test, not a convention

A boundary that is only documented erodes. `tests/test_architecture.py` walks
the AST of both packages and fails if `cortex_speech` imports `cortex_tts`, or
if `cortex_tts` imports any `cortex_speech` submodule rather than the facade.
A rule that runs in CI is worth more than a paragraph in CONTRIBUTING, and it
makes the next person's mistake loud and cheap instead of silent and
structural.

## What this buys the MOSS work specifically

MOSS's sampling controls — `sample_mode`, its top-p and top-k — would
otherwise have been described in the HTTP schema layer, putting engine
internals into a pydantic model and into the admin UI's request shape. With a
facade, a model declares its options generically and the API renders whatever a
spec says exists. The API layer stays ignorant of the fact that one engine
fuses sampling into an ONNX graph and another takes a float.

## The other boundary, which is already right

The integration in `hass-cortex/cortex-tts` talks to this app over HTTP from a
separate process and repository. That boundary needs no work; what it needs is
discipline about its contract, which is the API schema and nothing else. The
integration must not grow knowledge of engine ids, voice-id formats or model
internals — everything it shows comes from what the API returns.
