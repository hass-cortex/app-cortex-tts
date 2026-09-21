# app-cortex-tts / cortex-tts

On-device text-to-speech for Home Assistant: a FastAPI app with an ingress
admin UI, three catalog models on ONNX Runtime, and a text pipeline that is the
product rather than a detail (32% character error rate without it, 4% with;
[`docs/text-pipeline.md`](cortex-tts/docs/text-pipeline.md)).

The outer directory is the HA app shell and repo metadata; `cortex-tts/` is
the Python backend, the static admin UI, the Dockerfile and the rootfs.

This file is a map. Each rule below is one line with a pointer; the reasoning
and the measurement behind it live in
[`cortex-tts/docs/adr/`](cortex-tts/docs/adr/README.md), and every measured
figure lives in [`docs/models.md`](cortex-tts/docs/models.md) and nowhere else.

## Where things are

| Need                                                                           | File                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Domain vocabulary — _segment_ vs _sentence_, the three meanings of "streaming" | [`cortex-tts/CONTEXT.md`](cortex-tts/CONTEXT.md)                                                                                                                                                                                         |
| Why a rule is what it is                                                       | [`cortex-tts/docs/adr/`](cortex-tts/docs/adr/README.md)                                                                                                                                                                                  |
| Dev setup, gates, PR flow                                                      | [`cortex-tts/CONTRIBUTING.md`](cortex-tts/CONTRIBUTING.md)                                                                                                                                                                               |
| The HA App Store page: install, settings, troubleshooting                      | [`cortex-tts/DOCS.md`](cortex-tts/DOCS.md)                                                                                                                                                                                               |
| Reference pages the App Store page and the integration link to                 | [`cortex-tts/docs/`](cortex-tts/docs/): models, text pipeline, cloning, delivery, standalone, HTTP API                                                                                                                                   |
| How a release happens and where it lands                                       | `.github/workflows/release.yml` (tag → checks → GitHub Release) and `deploy.yaml` (→ GHCR, then a dispatch to the [stable](https://github.com/hass-cortex/repository) or [beta](https://github.com/hass-cortex/repository-beta) catalog) |
| The consumer this app exists for                                               | the [`cortex-tts`](https://github.com/hass-cortex/cortex-tts) HACS integration — required; without it HA has no Cortex TTS platform                                                                                                      |

## Working here

- **Never `git push`, tag, or create a repository without being asked.** A push is a release step here: a `X.Y.Z` tag publishes an image.
- **`uv`, never `pip`**: `pyproject.toml` and `uv.lock` are the truth; `uv run` for everything, `uv sync --frozen` to install. One ONNX Runtime build at a time, chosen by dependency group (see Build below).
- **"app", not "add-on", in anything a user reads** (UI, docs, translations); `addon` stays in code identifiers, paths and the Supervisor's API.
- **Docs carry rules, not history**: no "used to" or "was removed"; keep the measurement that justifies the current rule (`docs/models.md`, `docs/adr/`).
- **The integration is a separate repository and release**: anything that changes what it reads bumps `api_version`, and the app releases first.

## Repository layout

```
.
├── .github/workflows/        app-ci (shell lint), ci → python-checks (ruff+pyright+pytest),
│                             release (tag → checks → GitHub Release), deploy (→ GHCR + catalog)
├── .pre-commit-config.yaml   ruff, yamllint, hadolint, shellcheck, prettier; pyright + pytest at pre-push
├── README.md, LICENSE.md, images/
└── cortex-tts/
    ├── config.yaml, build.yaml, Dockerfile, rootfs/   HA app shell; venv baked in, source on top
    ├── DOCS.md, README.md, docs/, samples/            user documentation; README is the Supervisor's long_description
    ├── CONTEXT.md, CONTRIBUTING.md
    ├── pyproject.toml, uv.lock                        ruff, pyright, pytest config
    ├── scripts/                bench_rtf.py (docs/models.md's figures), replay_pacing.py (ADR 0001's regression)
    ├── src/cortex_speech/      the speech library
    ├── src/cortex_tts/         the Home Assistant app
    ├── tests/                  pytest, no model weights needed
    └── web/                    admin UI: plain ES modules + CSS, no build step
```

## Architecture

Two packages, one dependency direction. `cortex_speech` knows nothing about
Home Assistant, the Supervisor or HTTP; `cortex_tts` serves it and depends on
it through its facade only. `tests/test_architecture.py` fails the build if
either rule is broken, and pins the facade's surface by name.

```
src/cortex_tts/                 ── THE APP ──
├── __main__.py                 uvicorn entrypoint; reads stored settings before anything numeric loads
├── config.py / preferences.py  settled before start (environment) / changed while running (stored beside the models)
├── stats.py                    per model and voice: the newest requests and their median real-time factor
├── app.py                      assembly: lifespan, state, routes, ingress UI mount
├── supervisor.py / discovery.py / events.py   Supervisor POSTs; the discovery record; library notification → HA event
├── updates.py                  the hub that tells open UI sockets which read went stale
└── api/  deps.py (AppState, API key)  schemas.py  routes.py (HTTP)  live.py (the reply socket and its bank)  updates.py (/api/events)

src/cortex_speech/              ── THE LIBRARY ──
├── __init__.py                 SpeechService, SpeechConfig — the whole public surface
├── text/                       pipeline.py (plan/prepare), locales.py (per language), scripts.py (per character:
│                               terminators, speech rates, chars_within), passes.py (the pass order), units.py,
│                               generic.py (any unwritten language), zh/ (normalise, numbers, t2s, Taiwan readings), en/
├── pacing/                     model.py (RenderSample), sentences.py, release.py (verdict + pacer)
├── engine/                     base.py (protocols, stop check), backends.py (key → builder), registry.py (residency,
│                               LRU, one synthesis per engine), overrun.py, conditioning.py, join.py,
│                               preset.py (40M) moss.py omni.py
├── catalog.py                  the models, their capabilities, their sources
├── download.py / references.py / audio.py / providers.py / device.py / store.py / notifications.py
└── vendor/                     upstream inference code, kept diffable; exempt from ruff and pyright file by file
```

### Rules the code keeps

Each links to the ADR that argues it.

**Text** — [ADR 0002](cortex-tts/docs/adr/0002-text-pipeline-keyed-by-language.md)

- The locale is picked by the request's `language` tag; sniffing is the fallback.
- A rewrite one language needs lives with that locale and is absent elsewhere.
- Pass order is a contract owned by `text/passes.py`.
- A pass that can misjudge is opt-in; bare-number expansion is off in every locale.
- A word a model misreads is the model's to declare (`ModelSpec.misreads`) and the locale's to respell (`text/zh/standins.tsv`); the other models are handed the word untouched.
- Every segment ends in sentence-final punctuation.
- What depends on the character — where a sentence ends, how fast a script is spoken — is keyed by script in `text/scripts.py`, once, for the batch splitter, the live buffer and the terminator pass alike; a script nobody measured is assumed slow.
- The text path needs no engine: `/api/preview` never loads a model.

**Rendering** — [ADR 0003](cortex-tts/docs/adr/0003-overrun-trimming-and-stopping.md), [ADR 0005](cortex-tts/docs/adr/0005-device-memory-and-providers.md)

- An autoregressive model is trimmed only on short text (`MAX_BABBLE_CHARS`), and only where the babble pathology was measured on it (`render_with_retries(trim=...)`).
- A generation that stopped early is retried at another seed, at a ratio the engine chooses; a streamed one cannot be and is logged instead, while a buffered live reply is rendered through `EngineRegistry.synthesize` so that it can be.
- The execution provider is read from the sessions, never assumed from the build; one ONNX Runtime build is installed, chosen by dependency group, and the app refuses to start with two.
- At most `max_loaded_models` engines are resident, LRU-evicted; one synthesis per engine at a time.
- A model's lifecycle is legible from the log alone.
- Running out of device memory costs the engine, not the process.
- A render whose listener left stops within one unit of work; every engine takes the `stop` check.
- A backend is looked up in `engine/backends.py`, never branched on; builders import lazily.
- An engine never lists its own voices; listing voices loads nothing.
- Weights are never in the image; a bundle may span repositories (`ModelSpec.sources`).

**Voices** — [ADR 0004](cortex-tts/docs/adr/0004-references-are-audio-and-transcript.md)

- A reference is audio _and_ transcript, validated on the way in (2–20 s, ending in silence).
- Conditioning is cached once, for every engine, in `engine/conditioning.py`.
- A model declares capabilities, not a category; `builtin_voices` and `designed_voices` are the one exclusive pair.
- Whether a language can be named is a capability (`language_choice`), not a rule.

**Measuring and pacing** — [ADR 0007](cortex-tts/docs/adr/0007-measurements-belong-to-a-host.md), [ADR 0001](cortex-tts/docs/adr/0001-rtf-threshold-pacing.md)

- A real-time factor belongs to a host: the catalog carries none; the card shows this host's or "not measured".
- One measurement per request, a median per model and voice, one provider's; nothing pooled or borrowed.
- A live reply is paced by that median against the host's threshold (`stream_rtf` in settings, default `STREAM_RTF` 0.8): under it one sentence per request from a `BANK_S` (6 s) bank — capped by `bank_needed` at what the rest of the reply needs once the writer has finished — otherwise buffered — rendered as written, released when done. The card, the `ready` frame and the `spoke live` line all name the threshold the verdict was made against.
- Three settings (`auto`, `streaming`, `buffered`), two outcomes (`streaming`, `buffered`); `auto` takes the verdict.
- `scripts/replay_pacing.py` is the constants' regression; a change to either comes with its output.

**Transport** — [ADR 0006](cortex-tts/docs/adr/0006-live-socket-transport.md)

- Chunked audio is the WebSocket's alone; `/v1/audio/speech` answers a finished file.
- The stream is MP3 with no length; one frame is bounded at `MAX_FRAME_BYTES`, not one reply.
- Everything that can fail fails before the first byte.
- The delivery is legible from outside: `batch`, `rendered` and `done` say what the audio cannot.
- The mode is settled at `ready` and never revised; its two words are why `api_version` moves.
- Ingress bypasses the API key, and only ingress; the UI is served with `cache-control: no-cache`.
- The admin UI never polls: `/api/events` says which read went stale (`cortex_tts/updates.py`), and the data stays on the route that owns it.

**Settings**

- An addon option is what a restart is the only way to change (`config.yaml`: log level, discovery key); everything else is a stored setting, and `EngineRegistry.reconfigure` decides what a change evicts.
- The library never reaches for Home Assistant: it publishes to `notifications.py`; `cortex_tts/events.py` turns that into an HA event.

## Model behaviour the code relies on

The figures are in [`docs/models.md`](cortex-tts/docs/models.md).

- `ModelSpec.max_audio_s` is the generator's ceiling in audio and it truncates rather than slows; `segment_limit` converts it to characters with the slow-side priors so a segment never depends on which voice says it. OmniVoice declares no ceiling and is given none.
- `ModelSpec.max_text_tokens` is the other bound and not a ceiling: a model that samples its own stop can end anywhere, so the engine — the only thing holding a tokenizer — cuts an over-budget segment into chunks. MOSS declares 50.
- Hojo 40M declares `needs_number_words`; no model declares `reads_numerals`.
- MOSS and OmniVoice condition on the whole recording; neither has a speaker encoder.
- OmniVoice takes a designed voice from a closed attribute vocabulary; the nine offered are checked by test against that vocabulary.
- OmniVoice's transformer is built on the meta device; reading a weight outside `forward` fails at first synthesis.

## Build, test, release

```bash
cd cortex-tts
uv sync --frozen --extra omnivoice   # CPU build; the one extra adds torch, torchaudio, transformers, librosa, pydub
                                     # a card: --no-default-groups --group dev --group cuda (or cuda12) — one ONNX Runtime build, never both
uv run cortex-tts                    # STATIC_DIR + DATA_DIR from the environment
uv run pytest -q                     # ~710 tests, no model weights needed
```

Tests: `test_text.py`/`test_english.py` pin the readings; `test_architecture.py` the package boundary and the facade; `test_catalog.py` capabilities and the backend table; `test_stats.py` the median; `test_pacing.py` the buffer, the verdict and the pacer; `test_live.py` the WebSocket over a fake engine, bank included; `test_abandon.py` that a lost listener stops a render; `test_api.py` auth, shapes and errors. Synthesis itself is exercised by hand against real bundles.

Wire: every error body is `{"code", "message"}`; `/health` carries `api_version` (5), bumped when anything the integration reads changes shape, and the integration refuses a mismatch — so the app releases first and the integration second. The reference is [`docs/api.md`](cortex-tts/docs/api.md); `/api/docs` serves the OpenAPI.

Discovery: `discovery.announce` publishes a Supervisor discovery record with host, port and API key; clearing `discovery_api_key` and restarting rotates the key. Outside the Supervisor there is no token and nothing to announce to.

Live voice sync: a download finishing, a model deleted, a reference added or deleted, or a reference relabelled ends in `events.fire_models_changed`, and the integration adds or removes entities in place — no reload, no restart.

Distribution: CI publishes `ghcr.io/hass-cortex/cortex_tts/{arch}` (amd64 only) on a `X.Y.Z` tag and dispatches the catalog; a pre-release tag routes to the beta catalog only. `release.yml` runs the checks with the uv cache off.

## How to

**Add a unit**: `text/units.py` maps Home Assistant's symbol to its CLDR unit id (or a numerator/denominator pair); `UNNAMED` lists symbols CLDR has no unit for. Chinese readings go in `SUFFIX_UNITS` / `WORD_UNITS` of `text/zh/normalize.py`. Alternations are built longest-first, so nothing else changes.

**Add an engine**: a module under `engine/` implementing `Engine` (and `own_voices(directory)` if the model brings voices); a builder closure and, if any, a voice-reader closure registered in `engine/backends.py`; a `ModelSpec` in `catalog.py` with the capabilities it actually has, `reads_numerals`/`needs_number_words`/`misreads` declared only after measuring; an extra in `pyproject.toml` and the Dockerfile if it needs dependencies, and its row in `engine/backends.py`'s `_EXTRAS` so a host without it answers `BACKEND_MISSING` instead of an import error. The registry, the API and the integration need no change; the UI's prose does.

**Add a language**: a locale package beside `text/zh/` and `text/en/` whose `LOCALE` names its normaliser, its stop and its rewrites, registered in `text/pipeline.py`. Until it exists the language gets `text/generic.py`. If its script's sentence terminator or speech rate is missing from `text/scripts.py`, that is a row there — keyed by character, with the measurement that justifies the rate — not a field on the locale.

**Add a pass**: a pattern in `text/passes.py`, a render per script, a row in that script's `_PASSES` tuple in the position the order requires.

**Change a pacing constant**: edit `pacing/release.py`, run `scripts/replay_pacing.py` against recorded stats, logs and replies, and put its output in the change.
