# app-cortex-tts / cortex-tts

On-device text-to-speech for Home Assistant, served by a FastAPI app with an
ingress admin UI. Which models it offers is `catalog.py`, and the README's
**Models** table is the copy written for people; both are edited in one place
when the line-up changes, which is the point.

The models ship no text front-end, so the app carries one: Traditional-to-Simplified
glyph conversion and numeral/unit/date normalisation. That pipeline is the
product, not a detail (32% character error rate against 4%; see
[`docs/text-pipeline.md`](cortex-tts/docs/text-pipeline.md)).

The outer directory is the HA app shell + repo metadata; the inner `cortex-tts/`
subdir is the Python backend, the static admin UI, the Dockerfile and the
rootfs.

## Quick Links

- **Domain vocabulary**: [`cortex-tts/CONTEXT.md`](cortex-tts/CONTEXT.md) — what is
  _Prepared text_? _Segment_ vs _Sentence_? Why is "normalise" two unrelated
  operations in one request? Which of the four meanings of "streaming"?
- **Contributor guide**: [`cortex-tts/CONTRIBUTING.md`](cortex-tts/CONTRIBUTING.md)
  — dev setup, gates, PR flow.
- **User documentation**: [`cortex-tts/DOCS.md`](cortex-tts/DOCS.md) — the HA App
  Store page: install, configure, troubleshoot. It links out (absolute URLs,
  because the App Store renders it alone) to the reference pages under
  [`cortex-tts/docs/`](cortex-tts/docs/): [models](cortex-tts/docs/models.md),
  [the text pipeline](cortex-tts/docs/text-pipeline.md),
  [cloned voices](cortex-tts/docs/cloning.md),
  [keeping up](cortex-tts/docs/streaming.md),
  [running it elsewhere](cortex-tts/docs/standalone.md) and the
  [HTTP API](cortex-tts/docs/api.md). The integration's README points at the
  same pages rather than restating them: model facts live here, once.
- **Release runbook**: `docs/release/` at the root of the hass-cortex workspace —
  not in this repo. The catalog it publishes to is
  [`hass-cortex/repository`](https://github.com/hass-cortex/repository).
- **Primary consumer**: the [`cortex-tts`](https://github.com/hass-cortex/cortex-tts)
  HACS integration (HA TTS platform). It is required — without it Home
  Assistant has no Cortex TTS platform and the discovery record goes nowhere.

## Repository Layout

```
.
├── .github/workflows/
│   ├── app-ci.yaml            hassio-addons app-ci (HA app shell lint)
│   ├── ci.yml                 calls python-checks.yml on PR + push
│   ├── python-checks.yml      reusable: ruff + pyright + pytest
│   ├── deploy.yaml            release-triggered: app-deploy → GHCR + dispatch
│   └── release.yml            tag-triggered: python-checks gate + GitHub Release
├── .yamllint, .mdlrc, .prettierignore    lint configs (consumed by app-ci.yaml)
├── .pre-commit-config.yaml    ruff, yamllint, hadolint, shellcheck, prettier;
│                              pyright + pytest at pre-push
├── README.md                  repo front page
├── images/                    README screenshots
├── LICENSE.md                 MIT
└── cortex-tts/                  ── APP / SOURCE SUBDIR ──
    ├── config.yaml            HA app metadata (slug=cortex_tts, ingress; port 8771 unpublished)
    ├── build.yaml             base image: hassio-addons/debian-base (amd64 only)
    ├── Dockerfile             uv-installed venv baked in; source copied on top
    ├── DOCS.md                HA App Store documentation page
    ├── docs/                  reference pages DOCS.md and the integration link to
    ├── scripts/bench_rtf.py   where docs/models.md's figures come from (one host, one text set)
    ├── README.md              Supervisor reads this as `long_description`
    ├── CONTEXT.md             domain vocabulary
    ├── CONTRIBUTING.md
    ├── icon.png, logo.png     app icons (HA App Store + sidebar)
    ├── translations/en.yaml   app configuration-UI translations
    ├── rootfs/                s6-overlay services (init oneshot + cortex-tts main)
    ├── pyproject.toml/uv.lock ruff + pyright + pytest config live here too
    ├── src/cortex_speech/     the speech library (see Architecture)
    ├── src/cortex_tts/        the Home Assistant app
    ├── tests/                 pytest; no model weights needed
    └── web/                   static admin UI (no build step)
```

## Architecture

Two packages, one dependency direction. `cortex_speech` is the library and
knows nothing about Home Assistant, the Supervisor or HTTP; `cortex_tts` is the
app that serves it and depends on the library through its facade only.
`tests/test_architecture.py` fails the build if either rule is broken.

```
src/cortex_tts/       ── THE HOME ASSISTANT APP ──
├── __main__.py       uvicorn entrypoint
├── config.py         what must be settled before the process starts, from the environment
├── preferences.py    what the user changes while it runs, stored beside the models
├── stats.py          what this host measured, per model — a card shows its own
│                     real-time factor or none, never another machine's
├── app.py            assembly: lifespan, state, routes, ingress UI mount
├── supervisor.py     best-effort POSTs to the Supervisor, and having none
├── discovery.py      announce so HA finds the app by itself
├── events.py         turn a library notification into an event on the HA bus
└── api/
    ├── deps.py       AppState (holds one SpeechService) + the API-key dependency
    ├── schemas.py    request/response bodies
    └── routes.py     every endpoint (see API Endpoints)

src/cortex_speech/    ── THE SPEECH LIBRARY ──
├── __init__.py       SpeechService, SpeechConfig — the whole public surface
├── text/             ── THE TEXT PATH ──
│   ├── pipeline.py   prepare(): normalise → convert → segment
│   ├── passes.py     the ordered pass table both normalisers are built from
│   ├── normalize.py  Chinese readings
│   ├── english.py    Latin readings
│   ├── numbers.py    Chinese numeral rendering
│   └── options.py    NormalizeOptions, shared so neither normaliser imports the other
├── engine/
│   ├── base.py       the Engine and StreamingEngine protocols, Voice, errors
│   ├── backends.py   backend key → builder; how a new engine is added
│   ├── overrun.py    judging and trimming a waveform, and the seed retry
│   ├── registry.py   what is resident, LRU eviction, per-engine serialisation
│   ├── conditioning.py  what an engine derives from a reference, and when it
│   │                 goes stale — one cache, so one invalidation rule
│   ├── preset.py     40M: fixed voices
│   ├── clone.py      80M: every voice is a reference recording
│   ├── moss.py       MOSS-TTS-Nano: both, with cached prompt conditioning
│   ├── qwen3.py      Qwen3-TTS: nine speakers on one checkpoint, cloning on
│   │                 the other; two catalog entries, one engine
│   ├── qwen_tokenizer.py  that checkpoint ships no tokenizer.json — assemble
│   │                 one from vocab.json + merges.txt rather than pull in
│   │                 transformers to do it
│   ├── omni.py       OmniVoice: designed voices from a closed attribute
│   │                 vocabulary, and cloning
│   └── join.py       stitching per-segment waveforms into one utterance
├── catalog.py        the models, their capabilities, and where they come from
├── download.py       Hugging Face fetch in a worker thread, pollable progress
├── references.py     reference recordings: audio + transcript, validated on entry
├── audio.py          waveform ↔ bytes: output encoding, reference decode +
│                     levelling, and the header a chunked stream opens with
├── providers.py      which ONNX Runtime provider was asked for, and got
├── notifications.py  publish "the voice set changed"; the app decides what that means
└── vendor/           upstream inference code, kept diffable — exempt from ruff
                      and pyright file by file, never as `vendor/**`, because
                      two files in there are ours; see vendor/NOTICE.md
```

### Cross-module guarantees

- **Weights are never in the image.** The catalog names what is available;
  bundles arrive on first download into `/data/models`, so HA's "remove with
  data" sweeps them up. A failed download leaves partials in place —
  `catalog.inspect` reports not-downloaded and a retry resumes.
- **Pass order is a contract.** Normalisation emits Traditional number words,
  so it must precede script conversion. Within a normaliser, a construct claims
  its number before the bare-number pass reads that digit as a quantity.
  `text/passes.py` owns the order; each script only supplies readings.
- **Every segment ends in sentence-final punctuation.** Without it the model
  misses its cue to stop and invents a syllable.
- **One synthesis per engine at a time.** The ONNX sessions drive a stateful
  per-token loop, so the registry serialises calls; concurrency would corrupt
  state, not just slow things down.
- **At most `max_loaded_models` engines are resident**, least-recently-used
  evicted. The 40M beside either 2 GB model costs about 2.8 GB; the 80M and
  MOSS together about 4 GB.
- **A bundle may span repositories.** `ModelSpec.sources` is a list;
  MOSS publishes weights and audio codec separately and needs both, and
  `catalog.inspect` reports half a bundle as not downloaded.
- **Conditioning is cached once, for every engine.** Encoding a reference is
  the dominant cost of a cloned utterance, so `engine/conditioning.py` keeps it
  by reference id and drops it when the recording's fingerprint changes or
  `forget()` says so. An engine that kept its own would be a second
  invalidation rule, and the first one to diverge is silent — it still produces
  audio, in the previous voice.
- **A reference is audio _and_ transcript.** Neither alone defines a voice, and
  a wrong transcript degrades the clone with no error — so it is validated on
  the way in (2–20 s, non-empty, pronounceable, and ending in silence). The
  last of those is the same class of silent failure: the models that clone
  best read the whole recording as a worked example, so a clip cut by a clock
  teaches one that sentences end mid-word, and the clone then drifts and clips
  its own endings. Measured on eight real uploads, the seven cut at a
  recorder's 7.00 s limit ended between −7.1 and +7.5 dB against their own
  average and the one that finished ended at −25.3 dB, so the threshold is
  −15 dB over the last 100 ms.
- **An addon option is what a restart is the only way to change.** `config.yaml`
  carries the log level and the discovery key, and nothing else; everything a
  user tunes is a stored setting in `preferences.py`, changed in the admin UI
  and over `PUT /api/settings`. Most take effect on the next request — the
  default temperature travels with every synthesis call, and a smaller
  resident bound evicts down to it. The two ONNX Runtime binds when it
  creates a session (threads, execution provider) are adopted by dropping
  what is resident, and only when they actually changed; `EngineRegistry.reconfigure`
  decides, not the route. `num_threads` is also only half-adopted that way:
  its effect on BLAS and OpenMP is fixed at import and waits for a restart —
  which is why `__main__.py` reads the stored settings before anything
  numeric loads.
- **Ingress bypasses the API key, and only ingress.** `deps.is_ingress`
  requires both the Supervisor's `X-Ingress-Path` header and the Supervisor's
  peer address (`172.30.32.2`); the header alone is forgeable by anything that
  can reach the port. The port is not published by default (`ports: null`),
  and an empty configured key disables the check entirely, which is only sane
  while it stays that way.
- **The UI is served with `cache-control: no-cache`.** A hot-deploy swaps files
  under URLs that never change; without revalidation the browser keeps the old
  panel and the deploy looks like it did nothing.
- **The text path needs no engine.** `/api/preview` answers without loading a
  model, which is what makes the admin UI's right-hand column free to refresh
  on every keystroke.
- **The library never reaches for Home Assistant.** It publishes to listeners
  via `notifications.py`; `cortex_tts/events.py` is what turns that into an
  event on the HA bus. Enforced by `tests/test_architecture.py`.
- **A real-time factor belongs to a host, not to a model.** The catalog used
  to carry an `rtf_hint` measured on the project's reference machine and the
  UI showed it as plain "RTF"; the integration's own comments record it being
  "out by 3x here". It is gone. `cortex_tts/stats.py` keeps what this host
  measured — the median of the last dozen syntheses over a second of audio,
  **kept per voice kind** — and a card shows that or says "not measured". The
  kinds are split because the cost is: on OmniVoice a clone measured 7.17
  against 3.46 for a designed voice on one host, since the reference's codec
  frames rejoin the prompt on every synthesis. `scripts/bench_rtf.py` still
  exists, and what it produces is documentation rather than a figure the app
  repeats back to someone else's machine.
- **A model declares capabilities, not a category.** `builtin_voices`,
  `designed_voices`, `cloning`, `chunk_streaming`, `temperature`,
  `language_choice` and `style_instruction` are independent, so a model can
  have bundled voices _and_ clone. `EngineRegistry.voices` concatenates both
  sources rather than choosing. **One pair is the exception**, and it is an
  exception the code relies on: `builtin_voices` and `designed_voices` are
  mutually exclusive, because `routes._voice_kind` settles which kind a
  rendered voice was from the spec alone rather than looking the voice up.
  `tests/test_catalog.py` pins it, so a future entry that sets both fails the
  build instead of silently mislabelling every measurement it makes.
- **A stream has its own level control.** `encode` peak-normalises a finished
  waveform; a stream has none, so `StreamGain` holds a gain that only ever
  falls, far enough to keep each chunk under the same ceiling. Scaling a chunk
  to its own peak instead would make every chunk equally loud, which pumps.
- **A streamed format must not have to declare a length.** `/api/speak/stream`
  answers MP3 — a bare frame sequence with no container, no length field and no
  index. WAV is still available but is not the default: the maximal length its
  header has to declare is read by a general-purpose player as a six-hour file
  it then waits to buffer. FLAC and OGG are refused. `X-Cortex-Bitrate` is sent
  because it is the one measurement that exists before the first sample, and it
  is what turns a byte count into a duration downstream.
- **Everything that can fail must fail before the first byte.** Once a header
  is out the status is 200 and an error can only truncate the audio, so
  `/api/speak/stream` resolves the model, the text and the voice up front.
- **Streaming is a second protocol, not an optional method.** `StreamingEngine`
  is what MOSS satisfies and the other two do not; the registry asks with
  `isinstance` rather than making every engine decline a method it has no
  answer for. The capability is therefore written down twice — `ModelSpec.chunk_streaming`
  for a caller who has loaded nothing, the method for the registry — so a test
  pins them to each other. Disagreeing is silent in the direction that matters:
  a spec claiming the capability without the method makes `/api/speak/stream`
  answer `X-Cortex-Chunk-Streaming: 1` while whole utterances are rendered.
- **A quantised model is not automatically a fast one.** Check the op types
  before believing a small INT8 export is fast: a dynamically quantised
  per-token loop does not scale with threads, and the codec can cost as much
  as the model. What quantisation reliably buys is memory, not time.
- **The execution provider is verified, never assumed.** `auto` takes a GPU
  when one answers and the CPU when none does; `cuda` refuses to fall back.
  `onnxruntime.get_available_providers()` is a claim about the build, not a
  promise — measured on a GTX 1650 host it listed CUDA and then created every
  session on the CPU — so `providers.in_use` reads the sessions instead, and
  `/health` reports what was asked for beside what arrived. MOSS-TTS-Nano
  measured RTF 1.025 on a laptop i7 against **0.354** on that GTX 1650.
- **A backend is looked up, never branched on.** `ModelSpec.backend` keys into
  `engine/backends.py`; adding an engine is a module plus a registration, and
  builders import lazily so a heavy backend costs nothing until it is used.
- **An engine never lists its own voices.** Both kinds are files and the
  registry reads them: `Engine` has no `voices()` method, because the one it
  had was dead — every path already went through `EngineRegistry.voices`, and a
  protocol method nothing calls is a thing each new engine must write for
  nobody.
- **Listing voices loads nothing.** Both kinds are files: built-in voices live
  in the bundle's manifest or voices npz, reference voices in the store's
  index. A backend whose models bring voices of their own registers an
  `own_voices(directory)` reader alongside its builder — "own" rather than
  "built-in" because the bundled kind and OmniVoice's designed kind are both
  read this way, and a name for one of them lies about the other — so `/api/voices` never constructs an engine — at the
  default of one resident model, doing so evicted whatever was speaking, and
  the integration asks for every model's voices at startup.

## Model behaviour worth knowing

The measurements are in [`docs/models.md`](cortex-tts/docs/models.md); what
matters to the code is:

- **The model stops only when it samples an end-of-speech token.** Stopping is
  probabilistic, so `text/pipeline.py` terminates every segment (without a stop
  the model invented a syllable, measured) and `engine/overrun.py` judges and
  trims what came back. A high temperature makes over-runs likelier; `0` is
  greedy and reproducible, which is why `/api/speak` takes a per-request
  `temperature`.
- **The model pronounces no Arabic numeral and no symbol at all** — an
  unexpanded digit is silent or replaced by an unrelated word. Normalisation
  therefore belongs on for every language, and the integration must not gate
  it on the language tag. Reported upstream as
  [Hojo-TTS-Light#7](https://github.com/HojoAI/Hojo-TTS-Light/issues/7).
- **The 80M's cloning fidelity is capped by its speaker representation.**
  `SPEAKER_EMB_SECONDS = 6.0`: `_encode_speaker` reads the first six seconds
  into one 2048-value vector, while `_encode_ref_codes` encodes the whole
  reference into prompt codes — so audio past six seconds costs time on every
  synthesis and buys nothing. MOSS has no speaker encoder and conditions on the
  whole recording. Not measured against the model's own encoder; the
  conclusion rests on the representation size and on our path matching
  upstream's `generate` step for step.
- **Whether a language can be named is a capability, not a rule.** On most
  models the prompt is the text plus a speaker slot, so picking the voice is
  picking the language; those declare `language_choice = False` and the API
  refuses a `language` rather than accepting one it would ignore. Qwen3-TTS
  and OmniVoice do take one, so a request may name it and the engine narrows
  the whole tag against that model's own list (`_narrowing` in
  `engine/base.py`). Unset, the engine still supplies one: Qwen3-TTS from the
  speaker's own language (upstream's recommendation) or the reference's, and
  OmniVoice from the script the text reads as.
- **A designed voice is a third kind of voice.** OmniVoice has no bundled
  speakers and no free-text style prompt: it takes a short instruction built
  from a closed vocabulary (`vendor/omnivoice/voice_design.py`), and refuses
  anything outside it. The nine on offer are chosen in `engine/omni.py`, so
  its voice reader opens nothing — and `tests/test_omnivoice.py` checks each
  one against the model's own vocabulary, because a typo there is a
  `ValueError` in the middle of a reply rather than a worse voice.
- **Qwen3-TTS must not be run greedy.** It stops by sampling end-of-speech and
  at temperature 0 reliably fails to: a ten-character line ran to the model's
  own 2048-frame limit, 2.7 minutes of invented audio. `engine/qwen3.py` caps
  each segment from the duration the text needs.
- **OmniVoice's transformer is never materialised.** The int4 ONNX graph
  replaces `forward` outright, so the checkpoint's 2.45 GB of weights are
  neither downloaded nor loaded; the module is built on the meta device.
  Measured peak resident 1.1 GB against 4.7 GB. The cost is that anything
  reading a weight outside `forward` fails with "Cannot copy out of meta
  tensor" — loudly, at first synthesis.

## Build

```bash
cd cortex-tts
uv sync --frozen          # dependencies
uv run cortex-tts           # run it (STATIC_DIR + DATA_DIR from the environment)
```

Two extras carry the models that are not pure ONNX Runtime, and the Dockerfile
installs both deliberately. `hojo-80m` pulls torch and librosa for that model's
mel front-end; `omnivoice` adds torchaudio, transformers and pydub for the half
of OmniVoice the ONNX graph does not replace. Together they are most of the
image. Both are named for the model rather than for a capability — MOSS and
Qwen3-TTS clone without touching either, through `audio.decode_reference`.

The admin UI has **no build step** — `web/` is plain ES modules and CSS, served
by `StaticFiles`. Edit and reload.

## Testing

```bash
uv run pytest -q          # 422 tests, no model weights needed
```

`tests/test_text.py` and `tests/test_english.py` pin the text path — the part
that is measurable without a GPU-hour and the part most likely to regress
silently, since a wrong reading produces confident audio rather than an error.
`tests/test_api.py` covers auth, the model list shape and the errors a caller
sees. `tests/test_architecture.py` enforces the library/app boundary by reading
the AST — it loads nothing and runs in milliseconds. `tests/test_catalog.py`
pins the capability model and the backend table. Synthesis itself is exercised
by hand against real bundles.

## API Endpoints

The endpoint table, request fields, headers and every error code are in
[`docs/api.md`](cortex-tts/docs/api.md), which is the reference; `/api/docs`
serves the OpenAPI. Two things the code guarantees and the reference relies on:

- **Every error body is `{"code", "message"}`** — what a route raised, what the
  router could not match (404 on an unknown path) and what pydantic refused
  (422). `app.py` installs both handlers so a client parses one shape.
- **`/health` carries `api_version`**, bumped when a route, field or header the
  integration reads changes shape; the release version says nothing about the
  wire. The integration refuses to set up on a mismatch.

## Home Assistant Discovery

`discovery.announce` publishes a Supervisor discovery record carrying host,
port and API key, so the integration appears on the Devices page with nothing
to type. The key is generated on first start and pushed through the same
channel; clearing `discovery_api_key` and restarting rotates it. Outside the
Supervisor there is no token and nothing to announce to — local development
runs with an empty key and no discovery.

## Live voice sync (HA event)

Downloading or deleting a model and adding or deleting a reference recording
change which voices exist, and each ends in `events.fire_models_changed`
putting one event on the HA bus; the integration adds or removes entities in
place, so a voice uploaded in the admin UI is selectable in a pipeline seconds
later without a config-entry reload or a restart. Two paths reach that event.
Download completion is the library's: `download.py` calls
`notifications.notify_models_changed` and the app subscribes at startup. The
three routes — reference add, reference delete, model delete — call
`fire_models_changed` from the app directly. `PATCH /api/references/{id}` fires
nothing: a corrected transcript changes no voice id. The split still matters
where it exists: the library states a fact, the app decides that Home Assistant
is who hears it.

## How To

### Add a unit to the normaliser

`SUFFIX_UNITS` in `text/normalize.py`. The alternation is built longest-first
so `km/h` is matched before `km`; nothing else needs to change.

### Add an engine

Four declarations, across two files:

1. A module under `engine/` implementing the `Engine` protocol — and, if the
   model declares `builtin_voices` or `designed_voices`, a module-level
   `own_voices(directory)` that reads the list off disk, opening nothing.
2. A builder closure in `engine/backends.py`, importing the engine inside the
   call so a heavy backend costs nothing until it is used.
3. A voice-reader closure there too, when there is one.
4. `backends.register("your-key", builder, voices=…)`, plus a `ModelSpec` in
   `catalog.py` naming that backend with the capabilities it actually has.

Then a `[project.optional-dependencies]` extra and a `--extra` in the
Dockerfile if the backend needs its own dependencies.

**The registry, the API layer and the integration need no change** — the
registry builds through the table, `routes.py` reads `spec.*`, and the
integration reconciles against whatever the server lists. `tests/test_catalog.py`
derives what it checks from `backends.py`, so a new backend is covered by the
provider and chunk-streaming contracts without being added to a list.

**The UI is not automatic.** It renders from the capability booleans and will
be right for any combination of them, but the prose around it — what the
clones panel says, what the model cards claim about memory — is written by
hand and has been wrong for a new engine before.

### Add a pass

Add the pattern to `text/passes.py`, a render to the script that should say it,
and a row to that script's `_PASSES` tuple — in the position the order
requires. A pass that only one script has (the Chinese unit pass) simply does
not appear in the other's table.

## Distribution

CI publishes the image to `ghcr.io/hass-cortex/cortex_tts/{arch}` on release —
amd64 only, matching the published ONNX Runtime builds — and dispatches an
update to the catalog. Stable is
[`hass-cortex/repository`](https://github.com/hass-cortex/repository); beta is
`repository-beta`, and a semver pre-release tag (`0.2.0-beta.1`) routes there
only. The runbook is `docs/release/` at the root of the hass-cortex workspace,
outside this repo.

### Release tags

`X.Y.Z` — `release.yml` runs the Python checks with the uv cache **off** (a tag
build must not read anything a pull request could have written), then creates
the GitHub Release with `DISPATCH_TOKEN` so `deploy.yaml` fires.
