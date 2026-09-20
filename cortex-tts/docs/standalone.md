# Running it elsewhere

The app is a Home Assistant app first, but nothing in it needs the Supervisor:
the server is a Python process that reads its settings from the environment,
and the integration can be pointed at any address by hand. That is how you
give it a faster machine than the one Home Assistant runs on — a bigger CPU,
or a GPU, which the Home Assistant OS host cannot offer.

## Why you would

- **Home Assistant OS cannot use a GPU.** It ships no NVIDIA driver, and an
  app cannot bring one, so on HAOS every model runs on the CPU whatever the
  execution provider is set to.
- **The HA host is usually the slowest machine in the house**, and a GPU
  changes the answer for some models by enough to move one from stuttering
  to outrunning the speaker
  ([Models](models.md#hardware-and-running-it-elsewhere)).

## What you get, and what you do not

Running elsewhere keeps everything the API offers — every model, cloning, the
text pipeline, the admin UI at the root URL — and the integration's whole
feature set once it is pointed at the address. What it loses is what the
Supervisor provided:

- **No discovery.** The integration is added by hand with the address and the
  key (the Container/Core path in its README).
- **No live entity sync.** `cortex_tts_models_changed` reaches the Home
  Assistant event bus through the Supervisor; without one, a new model or
  reference reaches the integration on its next reload.
- **No ingress.** The admin UI is served on the port itself: its first API
  call answers 401, and the page asks for the key once and keeps it in that
  browser. The static page is open, so keep the port on a network you trust
  or behind your own proxy.

## Running from source

```bash
git clone https://github.com/hass-cortex/app-cortex-tts
cd app-cortex-tts/cortex-tts
uv sync --frozen --extra omnivoice   # drop the extra to skip OmniVoice

API_KEY=choose-a-long-random-string \
DATA_DIR=/srv/cortex-tts STATIC_DIR="$PWD/web" HOST=0.0.0.0 PORT=8771 \
  uv run python -m cortex_tts
```

`DATA_DIR` receives the model bundles, `settings.json` and — unless
`REFERENCES_DIR` points elsewhere — the reference recordings; the first start
downloads the default model unless `{"preload": false}` is written there
first. Open `http://<host>:8771/` for the admin UI, then add the integration
in Home Assistant with that address and `API_KEY`.

## With a GPU

ONNX Runtime is one build or the other, never both — the CPU and the GPU
wheel unpack into the same package directory and whichever installs later
shadows the first — so the build is a dependency group in `pyproject.toml`,
not a package you swap by hand. `cpu` is the default group; `cuda` and
`cuda12` replace it, and the lockfile pins all three:

```bash
# CUDA 13: onnxruntime-gpu 1.29, for a driver of 580 or newer
uv sync --frozen --no-default-groups --group dev --group cuda \
  --extra omnivoice
# CUDA 12: onnxruntime-gpu 1.26, for a driver that cannot (the GTX 1650 host runs 575)
uv sync --frozen --no-default-groups --group dev --group cuda12 \
  --extra omnivoice

uv run --no-sync cortex-tts     # or .venv/bin/cortex-tts
```

Then set the execution provider to `cuda` in the admin UI, which refuses to
fall back rather than run on the CPU behind the label. The groups carry the
CUDA and cuDNN libraries the wheel wants and the app loads them before
creating a session, so only the driver has to be on the host.

An environment that already holds both builds needs one extra flag on that
first sync, `--reinstall-package onnxruntime-gpu`: removing the CPU wheel
takes the files the two share with it, and the GPU wheel left behind has a
record and no package.

A plain `uv sync` or `uv run` puts the default `cpu` group back, and the app
then refuses to start: both builds installed is the one state it will not run
in, and the message says which command to use. `uv run --no-sync` runs what
is there. A driver that cannot serve the wheel's CUDA is reported as
`PROVIDER_UNAVAILABLE`, with the runtime's own line in the log naming the
library it could not load; `/health` shows what each loaded model actually
got beside what was asked for. A model whose extra was not synced answers
`BACKEND_MISSING` and names it.

**Which build.** 1.26.0 (`cuda12`) is what the GPU figures in
[Models](models.md) were measured with. 1.29 (`cuda`) runs the same graphs on
a driver that serves CUDA 13 — Hojo 40M and OmniVoice were exercised on an RTX
5070 Ti at driver 616.

**A small card needs the arena kept honest.** ONNX Runtime's CUDA allocator
defaults to `kNextPowerOfTwo`, which rounds every allocation up and holds it
for the life of the process, so a model unloaded is not memory returned;
`providers.CUDA_OPTIONS` asks for `kSameAsRequested`. Measured on a 4 GB GTX
1650: a resident cloning model fell from 3222 MiB to 1626, a model's residue
after unloading from ~750 MiB to ~100, and a model loaded after another loads
instead of ending in `CUBLAS failure 3: the resource allocation failed`; RTF
unchanged (2.82 against 2.78). The runtimes carried from upstream keep their
own provider lists; only `vendor/omnivoice_ort.py`, which this project wrote,
passes the option.

Measure before you trust it: a GTX 1650 beside a 4-vCPU i7-9750H took every
model well under what the CPU managed, while an RTX 5070 Ti on Windows beside
a 16-core Ryzen was slower than the CPU — the per-token loop is bound by
kernel launch latency, which a fast CPU beats and a Windows GPU scheduler
makes jittery ([Models](models.md#hardware-and-running-it-elsewhere) has the
pairs). Prefer native Linux for a GPU deployment. OmniVoice's torch half stays
on the CPU either way; only the ONNX sessions move.

## Not yet

There is no published standalone or CUDA image. The addon image's init
scripts talk to the Supervisor and will not start without it, so a container
deployment elsewhere is a Dockerfile of your own around the steps above.
