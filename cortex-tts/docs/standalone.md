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
- **The HA host is usually the slowest machine in the house.** Measured on
  the 4-core HAOS VM this project runs on, every model is two to three times
  slower than on a laptop CPU, and MOSS-TTS-Nano does not keep up with
  playback there at all — see [Models](models.md).
- **A GPU changes the answer.** MOSS-TTS-Nano measured RTF 1.025 on a laptop
  i7 against **0.354** on a GTX 1650: the difference between a long reply
  stuttering and outrunning the speaker.

## What you get, and what you do not

Running elsewhere keeps everything the API offers — every model, cloning, the
text pipeline, the admin UI at the root URL — and the integration's whole
feature set once it is pointed at the address. What it loses is what the
Supervisor provided:

- **No discovery.** The integration is added by hand with the address and the
  key (the Container/Core path in its README).
- **No live entity sync.** The app fires `cortex_tts_models_changed` on the
  Home Assistant event bus through the Supervisor; without one, a model
  downloaded or a reference uploaded reaches the integration on its next
  reload rather than in seconds.
- **No ingress.** The admin UI is served on the port itself. The first API
  call it makes answers 401, and the page then asks for the key once and keeps
  it in that browser; the static page itself is open, so keep the port on a
  network you trust or behind your own proxy.

## Running from source

```bash
git clone https://github.com/hass-cortex/app-cortex-tts
cd app-cortex-tts/cortex-tts
uv sync --frozen --extra hojo-80m        # drop the extra to skip the 80M and torch

API_KEY=choose-a-long-random-string \
DATA_DIR=/srv/cortex-tts STATIC_DIR="$PWD/web" HOST=0.0.0.0 PORT=8771 \
  uv run python -m cortex_tts
```

`DATA_DIR` receives the model bundles, the reference recordings and
`settings.json`; the first start downloads the default model unless
`{"preload": false}` is written there first. Open `http://<host>:8771/` for the
admin UI, then add the integration in Home Assistant with that address and
`API_KEY`.

## With a GPU

The published image and the lockfile carry the CPU build of ONNX Runtime.
On a machine with an NVIDIA card, swap it for the GPU build in the same
environment and set the execution provider to `cuda` in the admin UI, which
refuses to fall back rather than run on the CPU behind the label:

```bash
uv pip install --python .venv/bin/python "onnxruntime-gpu==1.22.0" \
  nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12
```

The wheels ship the CUDA and cuDNN libraries and the app loads them before
creating a session, so only the driver has to be on the host. The version is
pinned on purpose: the CUDA 13 builds (1.29, 1.30) crashed on session
creation on the development laptop, and 1.22 is the last CUDA 12 build. A driver too old for
the wheel's CUDA is the usual failure; the app reports it as
`PROVIDER_UNAVAILABLE` rather than silently landing on the CPU, and `/health`
shows what each loaded model actually got beside what was asked for.

Measure before you trust it. Beside a 4-vCPU i7-9750H a GTX 1650 took the
40M from 0.54 to 0.31 and MOSS from 1.04 to 0.37; beside a 16-core Ryzen an
RTX 5070 Ti on Windows was slower than the CPU for both — the per-token
loop is bound by kernel launch latency, which a fast CPU beats and a Windows
GPU scheduler makes jittery. Prefer native
Linux for a GPU deployment. The 80M does not load on CUDA
at all (a bfloat16 fusion without a kernel). The numbers are in
[Models](models.md). The 80M's torch dependency stays on the CPU either way;
only the ONNX sessions move.

## Not yet

There is no published standalone or CUDA image. The addon image's init
scripts talk to the Supervisor and will not start without it, so a container
deployment elsewhere is a Dockerfile of your own around the steps above.
