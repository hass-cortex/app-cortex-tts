# app-cortex-tts / cortex-tts

On-device text-to-speech for Home Assistant over three ONNX models — the
[Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) 40M with fifteen
built-in voices, its 80M that clones a voice from a few seconds of reference
audio, and [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano), which
does both and outputs 48 kHz — served by a FastAPI app with an ingress admin
UI. The
models ship no text front-end, so the app carries one: Traditional-to-Simplified
glyph conversion and numeral/unit/date normalisation. That pipeline is the
product, not a detail (32% character error rate against 4%; see `DOCS.md`).

The outer directory is the HA app shell + repo metadata; the inner `cortex-tts/`
subdir is the Python backend, the static admin UI, the Dockerfile and the
rootfs.

## Quick Links

- **Domain vocabulary**: [`cortex-tts/CONTEXT.md`](cortex-tts/CONTEXT.md) — what is
  _Prepared text_? _Segment_ vs _Sentence_? Why is "normalise" two unrelated
  operations in one request? Which of the three meanings of "streaming"?
- **Contributor guide**: [`cortex-tts/CONTRIBUTING.md`](cortex-tts/CONTRIBUTING.md)
  — dev setup, gates, PR flow.
- **User documentation**: [`cortex-tts/DOCS.md`](cortex-tts/DOCS.md) — the HA App
  Store page: install, configure, troubleshoot.
- **Release runbook**: workspace-level [`docs/release/`](../docs/release/README.md).
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
    ├── config.yaml            HA app metadata (slug=cortex_tts, port=8771, ingress)
    ├── build.yaml             base image: hassio-addons/debian-base (amd64 only)
    ├── Dockerfile             uv-installed venv baked in; source copied on top
    ├── DOCS.md                HA App Store documentation page
    ├── README.md              Supervisor reads this as `long_description`
    ├── CONTEXT.md             domain vocabulary
    ├── CONTRIBUTING.md
    ├── icon.png, logo.png     app icons (HA App Store + sidebar)
    ├── translations/en.yaml   app configuration-UI translations
    ├── rootfs/                s6-overlay services (init oneshot + cortex-tts main)
    ├── pyproject.toml/uv.lock ruff + pyright + pytest config live here too
    ├── docs/adr/              architecture decisions (boundary, capabilities)
    ├── src/cortex_speech/     the speech library (see Architecture)
    ├── src/cortex_tts/        the Home Assistant app
    ├── tests/                 pytest; no model weights needed
    └── web/                   static admin UI (no build step)
```

## Architecture

Two packages, one dependency direction. `cortex_speech` is the library and
knows nothing about Home Assistant, the Supervisor or HTTP; `cortex_tts` is the
app that serves it and depends on the library through its facade only.
`tests/test_architecture.py` fails the build if either rule is broken — see
[`docs/adr/0003`](cortex-tts/docs/adr/0003-speech-library-and-ha-shell-are-separate-packages.md).

```
src/cortex_tts/       ── THE HOME ASSISTANT APP ──
├── __main__.py       uvicorn entrypoint
├── config.py         what must be settled before the process starts, from the environment
├── preferences.py    what the user changes while it runs, stored beside the models
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
│   └── join.py       stitching per-segment waveforms into one utterance
├── catalog.py        the models, their capabilities, and where they come from
├── download.py       Hugging Face fetch in a worker thread, pollable progress
├── references.py     reference recordings: audio + transcript, validated on entry
├── audio.py          waveform ↔ bytes: output encoding, reference decode +
│                     levelling, and the header a chunked stream opens with
├── providers.py      which ONNX Runtime provider was asked for, and got
├── notifications.py  publish "the voice set changed"; the app decides what that means
└── vendor/           upstream inference code, kept diffable (ruff: ALL ignored)
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
  evicted. Both bundles at once cost about 2.8 GB.
- **A bundle may span repositories.** `ModelSpec.sources` is a list;
  MOSS publishes weights and audio codec separately and needs both, and
  `catalog.inspect` reports half a bundle as not downloaded.
- **Conditioning is cached once, for every engine.** Encoding a reference is
  the dominant cost of a cloned utterance, so `engine/conditioning.py` keeps it
  by reference id and drops it when the recording's fingerprint changes or
  `forget()` says so. An engine that kept its own would be a second
  invalidation rule, and the first one to diverge is silent — it still produces
  audio, in the previous voice. See [`docs/adr/0002`](cortex-tts/docs/adr/0002-shared-seams-for-reference-conditioning.md).
- **A reference is audio _and_ transcript.** Neither alone defines a voice, and
  a wrong transcript degrades the clone with no error — so it is validated on
  the way in (2–20 s, non-empty, pronounceable).
- **An addon option is what a restart is the only way to change.** `config.yaml`
  carries the log level and the discovery key, and nothing else; everything a
  user tunes is a stored setting in `preferences.py`, changed in the admin UI
  and over `PUT /api/settings`. Most take effect on the next request; the three
  ONNX Runtime binds when it creates a session are adopted by dropping what is
  resident. The one exception is `num_threads`, whose effect on BLAS and
  OpenMP is fixed at import and so waits for a restart — which is why
  `__main__.py` reads the stored settings before anything numeric loads.
- **Ingress bypasses the API key.** `require_api_key` returns early when the
  Supervisor's `X-Ingress-Path` header is present; an empty configured key
  disables the check entirely, which is only sane behind ingress.
- **The UI is served with `cache-control: no-cache`.** A hot-deploy swaps files
  under URLs that never change; without revalidation the browser keeps the old
  panel and the deploy looks like it did nothing.
- **The text path needs no engine.** `/api/preview` answers without loading a
  model, which is what makes the admin UI's right-hand column free to refresh
  on every keystroke.
- **The library never reaches for Home Assistant.** It publishes to listeners
  via `notifications.py`; `cortex_tts/events.py` is what turns that into an
  event on the HA bus. Enforced by `tests/test_architecture.py`.
- **A model declares capabilities, not a category.** `builtin_voices`,
  `cloning` and `chunk_streaming` are independent, so a model can have bundled
  voices _and_ clone. `Registry.voices` concatenates both sources rather than
  choosing. See [`docs/adr/0001`](cortex-tts/docs/adr/0001-capabilities-not-engine-kind.md).
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
  is what turns a byte count into a duration downstream. See
  [`docs/adr/0004`](cortex-tts/docs/adr/0004-a-stream-must-not-declare-a-length.md).
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
  index. A backend whose models declare `builtin_voices` registers a reader
  alongside its builder, so `/api/voices` never constructs an engine — at the
  default of one resident model, doing so evicted whatever was speaking, and
  the integration asks for every model's voices at startup.

## Model behaviour worth knowing

**The model stops only when it samples an end-of-speech token**, so stopping
is a probabilistic event rather than a decision. When the sample goes the other
way it has to emit something, and invents a syllable. Two things make that far
more likely:

- **No sentence-final punctuation.** Measured: `客廳的燈已經打開了` produced a
  stray 「哈」; the same text with a full stop did not. This is why
  `text/pipeline.py` terminates every segment — the guarantee above is not
  cosmetic.
- **A high temperature.** Across five seeds on one phrase, `0.8` produced
  1.96–2.32 s and a stray sound; `0.0` produced exactly 2.02 s every time and
  none. Greedy decoding is reproducible and never over-runs, at the cost of
  flatter delivery. `/api/speak` takes a per-request `temperature` so the two
  can be compared without touching the app's configuration.

**The model pronounces no Arabic numeral and no symbol at all** — an
unexpanded digit is not read wrong, it is silent or replaced by an unrelated
word (`80` came out as "a bay"). This is why normalisation belongs on for every
language and why the integration must not gate it on the language tag. Reported
upstream as [Hojo-TTS-Light#7](https://github.com/HojoAI/Hojo-TTS-Light/issues/7).

**Cloning fidelity is capped by the speaker representation, not by the input.**
`SPEAKER_EMB_SECONDS = 6.0`: `_encode_speaker` reads the first six seconds of a
reference, pads a shorter clip with silence and discards the rest of a longer
one, and compresses it to one 2048-value vector. A clone carries the character
of a voice but not its timbre, and better source audio does not move it.
`_encode_ref_codes`, by contrast, encodes the _whole_ reference into codec
tokens that join the prompt for every sentence — so audio past six seconds
costs time on every synthesis and buys nothing. Not measured against the
model's own speaker encoder; the conclusion rests on the representation size,
on our path matching upstream's `generate` step for step, and on references
that measure clean.

**Neither model takes a language parameter.** The prompt is the text plus a
speaker slot, so pronunciation comes entirely from the voice. Picking the voice
is picking the language, and no switch can make an English voice read Chinese.

## Build

```bash
cd cortex-tts
uv sync --frozen          # dependencies
uv run cortex-tts           # run it (STATIC_DIR + DATA_DIR from the environment)
```

The `hojo-80m` extra pulls torch and librosa for that model's mel front-end and
roughly triples the image; the Dockerfile installs it deliberately. It is named
for the model, not for cloning — MOSS clones without any of it, through
`audio.decode_reference`.

The admin UI has **no build step** — `web/` is plain ES modules and CSS, served
by `StaticFiles`. Edit and reload.

## Testing

```bash
uv run pytest -q          # 247 tests, no model weights needed
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

Everything under `/api` requires the key unless the request came through
ingress. `/health` never does.

| Method | Path                         | Purpose                                                |
| ------ | ---------------------------- | ------------------------------------------------------ |
| GET    | `/health`                    | liveness; version + resident model count               |
| GET    | `/api/defaults`              | the configured default model and voice                 |
| GET    | `/api/settings`              | every stored setting, as it is now in force            |
| PUT    | `/api/settings`              | change some of them; omitted fields keep their value   |
| GET    | `/api/models`                | catalog + per-model state (downloaded/loaded/progress) |
| POST   | `/api/models/{id}/download`  | start a download; poll `/api/models`                   |
| DELETE | `/api/models/{id}`           | remove the bundle from disk                            |
| POST   | `/api/models/{id}/load`      | make it resident                                       |
| POST   | `/api/models/{id}/unload`    | evict it                                               |
| GET    | `/api/voices`                | voices across downloaded models, or one model's        |
| POST   | `/api/preview`               | run the text path only; no model is loaded             |
| POST   | `/api/speak`                 | synthesise; returns WAV with `X-Cortex-*` headers      |
| POST   | `/api/speak/stream`          | the same, sent as it is produced (chunked MP3)         |
| POST   | `/v1/audio/speech`           | the same, OpenAI-shaped                                |
| GET    | `/api/references`            | cloned-voice reference recordings                      |
| POST   | `/api/references`            | add one (multipart: audio + transcript + metadata)     |
| PATCH  | `/api/references/{id}`       | correct a transcript                                   |
| DELETE | `/api/references/{id}`       | remove it, and the voice it defined                    |
| GET    | `/api/references/{id}/audio` | play the recording back                                |

`/api/speak` answers with the synthesis measurements in headers —
`X-Cortex-Audio-Seconds`, `X-Cortex-Inference-Ms`, `X-Cortex-Rtf`, `X-Cortex-Segments`
— which is what the integration's diagnostic sensors report. Full OpenAPI at
`/api/docs`.

## Home Assistant Discovery

`discovery.announce` publishes a Supervisor discovery record carrying host,
port and API key, so the integration appears on the Devices page with nothing
to type. The key is generated on first start and pushed through the same
channel; clearing `discovery_api_key` and restarting rotates it. Outside the
Supervisor there is no token and nothing to announce to — local development
runs with an empty key and no discovery.

## Live voice sync (HA event)

Downloading a model or adding, editing or deleting a reference recording
changes which voices exist. The library calls `notifications.notify_models_changed`;
the app subscribes at startup and `events.fire_models_changed` puts one event on
the HA bus, and the integration adds or removes entities in place — so a voice
uploaded in the admin UI is selectable in a pipeline seconds later without a
config-entry reload or a restart. The split matters: the library states the
fact, the app decides that Home Assistant is who hears it.

## How To

### Add a unit to the normaliser

`SUFFIX_UNITS` in `text/normalize.py`. The alternation is built longest-first
so `km/h` is matched before `km`; nothing else needs to change.

### Add an engine

Four declarations, across two files:

1. A module under `engine/` implementing the `Engine` protocol — and, if the
   model declares `builtin_voices`, a module-level `builtin_voices(directory)`
   that reads the list off disk, opening nothing.
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
only. See [`docs/release/`](../docs/release/README.md).

### Release tags

`X.Y.Z` — `release.yml` runs the Python checks with the uv cache **off** (a tag
build must not read anything a pull request could have written), then creates
the GitHub Release with `DISPATCH_TOKEN` so `deploy.yaml` fires.
