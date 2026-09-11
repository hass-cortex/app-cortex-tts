# app-hojo-tts / hojo-tts

On-device text-to-speech for Home Assistant over the
[Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) ONNX models — a 40M
with fifteen built-in voices and an 80M that clones a voice from a few seconds
of reference audio — served by a FastAPI app with an ingress admin UI. The
models ship no text front-end, so the app carries one: Traditional-to-Simplified
glyph conversion and numeral/unit/date normalisation. That pipeline is the
product, not a detail (32% character error rate against 4%; see `DOCS.md`).

The outer directory is the HA app shell + repo metadata; the inner `hojo-tts/`
subdir is the Python backend, the static admin UI, the Dockerfile and the
rootfs.

## Quick Links

- **Domain vocabulary**: [`hojo-tts/CONTEXT.md`](hojo-tts/CONTEXT.md) — what is
  _Prepared text_? _Segment_ vs _Sentence_? Why is "normalise" two unrelated
  operations in one request? Which of the three meanings of "streaming"?
- **Contributor guide**: [`hojo-tts/CONTRIBUTING.md`](hojo-tts/CONTRIBUTING.md)
  — dev setup, gates, PR flow.
- **User documentation**: [`hojo-tts/DOCS.md`](hojo-tts/DOCS.md) — the HA App
  Store page: install, configure, troubleshoot.
- **Release runbook**: workspace-level [`docs/release/`](../docs/release/README.md).
- **Primary consumer**: the [`hojo-tts`](https://github.com/hass-cortex/hojo-tts)
  HACS integration (HA TTS platform). It is required — without it Home
  Assistant has no Hojo TTS platform and the discovery record goes nowhere.

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
└── hojo-tts/                  ── APP / SOURCE SUBDIR ──
    ├── config.yaml            HA app metadata (slug=hojo_tts, port=8771, ingress)
    ├── build.yaml             base image: hassio-addons/debian-base (amd64 only)
    ├── Dockerfile             uv-installed venv baked in; source copied on top
    ├── DOCS.md                HA App Store documentation page
    ├── README.md              Supervisor reads this as `long_description`
    ├── CONTEXT.md             domain vocabulary
    ├── CONTRIBUTING.md
    ├── icon.png, logo.png     app icons (HA App Store + sidebar)
    ├── translations/en.yaml   app configuration-UI translations
    ├── rootfs/                s6-overlay services (init oneshot + hojo-tts main)
    ├── pyproject.toml/uv.lock ruff + pyright + pytest config live here too
    ├── src/hojo_tts/          the app (see Architecture)
    ├── tests/                 pytest; no model weights needed
    └── web/                   static admin UI (no build step)
```

## Architecture

```
src/hojo_tts/
├── __main__.py       uvicorn entrypoint
├── config.py         Settings, read from the environment the s6 run script exports
├── app.py            assembly: lifespan, state, routes, ingress UI mount
├── api/
│   ├── deps.py       AppState + the API-key dependency
│   ├── schemas.py    request/response bodies
│   └── routes.py     every endpoint (see API Endpoints)
├── text/             ── THE TEXT PATH ──
│   ├── pipeline.py   prepare(): normalise → convert → segment
│   ├── passes.py     the ordered pass table both normalisers are built from
│   ├── normalize.py  Chinese readings
│   ├── english.py    Latin readings
│   ├── numbers.py    Chinese numeral rendering
│   └── options.py    NormalizeOptions, shared so neither normaliser imports the other
├── engine/
│   ├── base.py       the Engine protocol, Voice, Synthesis, error types
│   ├── overrun.py    judging and trimming a waveform, and the seed retry
│   ├── registry.py   what is resident, LRU eviction, per-engine serialisation
│   ├── preset.py     40M: fixed voices
│   ├── clone.py      80M: every voice is a reference recording
│   └── join.py       stitching per-segment waveforms into one utterance
├── catalog.py        the two models, their files, and where they come from
├── download.py       Hugging Face fetch in a worker thread, pollable progress
├── refs.py           reference recordings: audio + transcript, validated on entry
├── audio.py          waveform → bytes (level normalisation, resampling, encoding)
├── supervisor.py     best-effort POSTs to the Supervisor, and having none
├── discovery.py      announce so HA finds the app by itself
├── events.py         fire a models-changed event so the voice list syncs live
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
- **A reference is audio _and_ transcript.** Neither alone defines a voice, and
  a wrong transcript degrades the clone with no error — so it is validated on
  the way in (2–20 s, non-empty, pronounceable).
- **Ingress bypasses the API key.** `require_api_key` returns early when the
  Supervisor's `X-Ingress-Path` header is present; an empty configured key
  disables the check entirely, which is only sane behind ingress.
- **The UI is served with `cache-control: no-cache`.** A hot-deploy swaps files
  under URLs that never change; without revalidation the browser keeps the old
  panel and the deploy looks like it did nothing.
- **The text path needs no engine.** `/api/preview` answers without loading a
  model, which is what makes the admin UI's right-hand column free to refresh
  on every keystroke.

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
cd hojo-tts
uv sync --frozen          # dependencies
uv run hojo-tts           # run it (STATIC_DIR + DATA_DIR from the environment)
```

The `clone` extra pulls torch and librosa for the 80M's mel front-end and
roughly triples the image; the Dockerfile installs it deliberately.

The admin UI has **no build step** — `web/` is plain ES modules and CSS, served
by `StaticFiles`. Edit and reload.

## Testing

```bash
uv run pytest -q          # 134 tests, no model weights needed
```

`tests/test_text.py` and `tests/test_english.py` pin the text path — the part
that is measurable without a GPU-hour and the part most likely to regress
silently, since a wrong reading produces confident audio rather than an error.
`tests/test_api.py` covers auth, the model list shape and the errors a caller
sees. Synthesis itself is exercised by hand against real bundles.

## API Endpoints

Everything under `/api` requires the key unless the request came through
ingress. `/health` never does.

| Method | Path                         | Purpose                                                |
| ------ | ---------------------------- | ------------------------------------------------------ |
| GET    | `/health`                    | liveness; version + resident model count               |
| GET    | `/api/defaults`              | the configured default model and voice                 |
| GET    | `/api/models`                | catalog + per-model state (downloaded/loaded/progress) |
| POST   | `/api/models/{id}/download`  | start a download; poll `/api/models`                   |
| DELETE | `/api/models/{id}`           | remove the bundle from disk                            |
| POST   | `/api/models/{id}/load`      | make it resident                                       |
| POST   | `/api/models/{id}/unload`    | evict it                                               |
| GET    | `/api/voices`                | voices across downloaded models, or one model's        |
| POST   | `/api/preview`               | run the text path only; no model is loaded             |
| POST   | `/api/speak`                 | synthesise; returns WAV with `X-Hojo-*` headers        |
| POST   | `/v1/audio/speech`           | the same, OpenAI-shaped                                |
| GET    | `/api/references`            | cloned-voice reference recordings                      |
| POST   | `/api/references`            | add one (multipart: audio + transcript + metadata)     |
| PATCH  | `/api/references/{id}`       | correct a transcript                                   |
| DELETE | `/api/references/{id}`       | remove it, and the voice it defined                    |
| GET    | `/api/references/{id}/audio` | play the recording back                                |

`/api/speak` answers with the synthesis measurements in headers —
`X-Hojo-Audio-Seconds`, `X-Hojo-Inference-Ms`, `X-Hojo-Rtf`, `X-Hojo-Segments`
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
changes which voices exist. `events.fire_models_changed` puts one event on the
HA bus and the integration adds or removes entities in place, so a voice
uploaded in the admin UI is selectable in a pipeline seconds later without a
config-entry reload or a restart.

## How To

### Add a unit to the normaliser

`SUFFIX_UNITS` in `text/normalize.py`. The alternation is built longest-first
so `km/h` is matched before `km`; nothing else needs to change.

### Add a pass

Add the pattern to `text/passes.py`, a render to the script that should say it,
and a row to that script's `_PASSES` tuple — in the position the order
requires. A pass that only one script has (the Chinese unit pass) simply does
not appear in the other's table.

## Distribution

CI publishes the image to `ghcr.io/hass-cortex/hojo_tts/{arch}` on release —
amd64 only, matching the published ONNX Runtime builds — and dispatches an
update to the catalog. Stable is
[`hass-cortex/repository`](https://github.com/hass-cortex/repository); beta is
`repository-beta`, and a semver pre-release tag (`0.2.0-beta.1`) routes there
only. See [`docs/release/`](../docs/release/README.md).

### Release tags

`X.Y.Z` — `release.yml` runs the Python checks with the uv cache **off** (a tag
build must not read anything a pull request could have written), then creates
the GitHub Release with `DISPATCH_TOKEN` so `deploy.yaml` fires.
