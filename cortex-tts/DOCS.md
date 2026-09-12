# Cortex TTS

On-device text-to-speech for Home Assistant. Runs
[Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) and
[MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) ONNX models on CPU —
no GPU, no cloud, no API bill.

Three models are available:

| Model         | Voices                                    | Speed    | Memory  | Disk   |
| ------------- | ----------------------------------------- | -------- | ------- | ------ |
| **40M**       | 15 built-in (2 Chinese, 13 English)       | RTF 0.21 | ~780 MB | 241 MB |
| **80M**       | none built in; clones from your recording | RTF 0.79 | ~2 GB   | 437 MB |
| **MOSS Nano** | 18 built-in (6 Chinese) **and** clones    | RTF 0.35 | ~2 GB   | 729 MB |

RTF is the real-time factor: wall-clock time divided by the length of the audio
produced. Below 1 means it speaks faster than the audio plays. The figures were
measured on a modern desktop core; a slower host scales roughly linearly.

Start with the 40M. It is the cheapest by a wide margin and, in a measured ASR
round trip, no less accurate than the 80M.

Reach for **MOSS Nano** when you want a choice of voice: it is the only model
here with both a voice library and cloning, it speaks Japanese as well as
Chinese and English, and it outputs 48 kHz. It costs about 2 GB of memory, and
it reads Latin words poorly — measured at 32% character error rate on a reply
containing a product name, against 0-9% for the other models — so it suits
replies that are Chinese throughout.

Reach for the **80M** only when you want one specific person's voice and the
40M's built-ins will not do.

## Why the text pipeline exists

This model has no text front-end. It pronounces the glyphs it is given, and
two things it is routinely given are glyphs it cannot pronounce.

**Traditional Chinese.** The tokenizer covers Traditional characters, so nothing
fails — the model simply produces the wrong sounds. Measured over 14
Home-Assistant-shaped sentences, scored by transcribing the output and comparing
it to the input:

| Input                                  | Character error rate |
| -------------------------------------- | -------------------- |
| Human recording (measurement floor)    | ~0%                  |
| 40M, Traditional text as-is            | 32.4%                |
| 80M clone, Traditional text as-is      | 39.1%                |
| **40M, converted to Simplified first** | **4.4%**             |

At 32% the content words are gone: 客廳的燈 comes out as something like
「确定的的」. The app converts Traditional glyphs to Simplified before synthesis,
using OpenCC's `t2s` — a **glyph-only** conversion. The phrase-aware variants
would rewrite 設定 to 设置 and 訊號 to 信号, changing the words spoken aloud;
they score no better, so the one that preserves Taiwanese wording is used.

**Numbers, units and times.** `26.5°C`, `68%`, `14:35` and `2026-09-06` are read
as noise — 36–50% character error rate on their own. Home Assistant emits these
constantly, so the app expands them into Chinese words before synthesis:

| Input         | Spoken as          |
| ------------- | ------------------ |
| `26.5°C`      | 攝氏二十六點五度   |
| `68%`         | 百分之六十八       |
| `14:35`       | 十四點三十五分     |
| `2026-09-06`  | 二零二六年九月六日 |
| `3.2 kWh`     | 三點二度電         |
| `2026.9 版本` | 二零二六點九版本   |

Latin words are left alone — the model reads English natively, so `Home
Assistant` and `Roborock` pass through untouched.

Both passes are on by default and can be turned off per request. Turning them
off for ordinary Traditional Chinese will make the voice unintelligible; the
switches exist for callers whose text is already prepared.

The app's UI shows the prepared text — **What the model is asked to say** —
beside the composer, so you can read it before spending a synthesis on it.

![The composer and the prepared text](https://raw.githubusercontent.com/hass-cortex/app-cortex-tts/main/images/composer.png)

Left is what you typed; right is what the model receives. Each row under
**Rewrites** is one change: `number` for an expansion, `script` for the glyph
conversion with the count of glyphs it touched, `stop` for punctuation added at
the end. A pass that did not fire leaves no row — which is how you tell "nothing
needed rewriting" from "the switch is off".

## Installation

[![Open this app in your Home Assistant instance.][app-badge]][app]

1. Click **Install**, then **Start**.
2. Check the log to see that it came up.
3. Click **OPEN WEB UI**.

## After installing

### 1. Download a model

Nothing is baked into the image, so the first download takes a minute or two.
Open the panel → **Models** → **Download** on the **40M**. Start there; the
table above says why.

### 2. Install the companion integration

[![Open your Home Assistant instance and add the Cortex TTS integration via HACS.][hacs-badge]][hacs]

Press **Add**, then restart Home Assistant. (Requires
[HACS](https://hacs.xyz).) Home Assistant has no Cortex TTS platform without it,
and nothing will be discovered.

### 3. Pair with the running app

After the restart a **Cortex TTS discovered** card appears under **Settings →
Devices & services**. Click **Configure** and confirm — the address and the API
key come from the app itself, so there is nothing to type. Each downloaded
model becomes its own TTS entity.

### 4. Assign it to a voice pipeline

[![Open your Home Assistant instance and manage your voice assistants.][va-badge]][va]

Pick (or create) a pipeline → **Text-to-speech** → choose the Cortex TTS entity,
then the voice. The voice is what picks the language: the model takes no
language parameter, so a Chinese voice is the only thing that makes it read
Chinese.

## Cloned voices (80M)

1. Download the **Hojo TTS Light 80M** model.
2. In **Cloned voices**, upload about **6 seconds** of clean speech that
   starts talking immediately, and type _exactly_ what is said in it. The transcript is not a label — it tells the model which
   sounds in the recording map to which text, and a wrong one degrades the clone
   without any error.
3. The voice appears in Home Assistant within seconds; no reload needed.

Recordings shorter than 2 s or longer than 20 s are rejected, but six seconds
is the number that matters: the speaker encoder reads exactly that much, padding
a shorter clip with silence and discarding the rest of a longer one. Meanwhile
the _whole_ recording is encoded into codec tokens that join the prompt for
every sentence — so audio past the six-second mark slows down every synthesis
while contributing nothing to the voice.

## Configuration

Two options are set in the app's configuration tab, because both have to be
settled before the process starts. Everything else is in the app's own UI,
under **Settings** — changing one there takes effect on the next request, and
adding a model to the list no longer needs the app rebuilt.

### Option: `log_level`

How much the app writes to its log. `debug` additionally prints the prepared
text for every request — what the model was actually asked to say — which is
the fastest way to see whether the text passes did what you expected.

### Option: `discovery_api_key`

The key the Cortex TTS integration uses. Generated on first start and pushed to
Home Assistant through the discovery service, so it normally needs no
attention. Clear the field and restart to rotate it; the integration picks up
the new value by itself.

## Settings

Open the app and scroll to **Settings**. Three of these are bound when ONNX
Runtime creates a session, so adopting one drops whatever is resident and the
next reply loads it again; the rest are read afresh on every request.

### Inference threads

ONNX Runtime threads per synthesis; `0` lets the runtime decide. **More is not
faster.** The decode loop is Python-bound, so past a couple of threads the
runtime spends its time synchronising them rather than working. Measured on a
four-core host with MOSS-TTS-Nano, rendering 40 seconds of speech:

| Threads | Time to render, against real time |
| ------- | --------------------------------- |
| 1       | 1.17x                             |
| 2       | **1.05x**                         |
| 4       | 1.80x                             |

Leave it at `2`. Spare cores are not a reason to raise it — four threads on a
four-core host was the slowest setting tested, by a wide margin.

### Execution provider

`auto` takes a GPU when one answers and the CPU when none does. `cuda` refuses
to fall back, which is what you want on a host that has a card: a GPU build
quietly running on the CPU is the failure nobody notices. What each loaded
model actually got is printed on its card, beside **loaded**.

MOSS-TTS-Nano measured 1.025x real time on a laptop i7 against **0.354x** on a
GTX 1650 — the difference between a long reply outrunning the speaker and not.
A CUDA-capable image is needed for `cuda` to answer at all.

### Models kept in memory

How many models may stay in memory at once. The 40M needs about 780 MB and the
80M about 2 GB, so the default of `1` swaps between them on demand. Raise it to
`2` only if the host can hold both — about 2.8 GB.

### Default model and voice

Used when a request does not name one. The 40M has built-in voices and is
roughly four times cheaper; the 80M clones a voice from a recording you upload.
A voice the chosen model does not offer is ignored and the first available one
is used instead, so leaving the voice empty always takes the first.

### Sampling temperature

How randomly the model picks each step. It stops speaking only when it
_samples_ its end-of-speech token, so a higher value occasionally over-runs the
text with an invented syllable. `0` is greedy: reproducible, never over-runs,
at the cost of flatter delivery.

### Load the default model at startup

Load the default model when the app starts rather than on the first request.
Costs about a second of startup and roughly 780 MB of memory, and removes that
delay from the first thing you ask it to say.

## API

The app is usable on its own, not only through the integration. Everything
under `/api` takes `Authorization: Bearer <discovery_api_key>`; requests
arriving through ingress are already authenticated and skip the check.
`POST /v1/audio/speech` is served too, so OpenAI-shaped clients work unchanged.

Full OpenAPI, with every endpoint and an in-browser console, at `/api/docs`.

## Limitations

- **The app's API does not stream.** One `/api/speak` call synthesises all of
  the text it was given before returning a byte. Through Home Assistant you do
  not wait for all of it — the integration splits the reply into sentences and
  plays the first while the rest is still being made — but one long sentence
  is still one long wait.
- **A cloned voice is a likeness, not a match.** The 80M compresses a speaker
  into a single 2048-value vector taken from the first **6 seconds** of the
  reference. The result carries the general character of a voice — rough pitch
  range and delivery — but the timbre is audibly not the same person. A longer
  or cleaner recording does not change this: the encoder ignores everything
  past six seconds, and the limit is the model's capacity, not the input.
- **No speed, pitch or emotion control.** The model exposes none.
- **Deterministic.** Sampling uses a fixed seed, so the same text in the same
  voice always produces the same audio — including when you dislike it.
- **Chinese and English only.**
- **amd64 only**, matching the published ONNX Runtime builds.

## Troubleshooting

**No voices in the pipeline picker.** The model is probably not downloaded —
check the app's **Models** cards. A downloaded 80M with no reference recording
also shows no voices, by design.

**Chinese sounds like the wrong words.** Check that `convert_script` was not
turned off for that call. Compare **What the model is asked to say** against
what you hear.

**Numbers are read as gibberish.** Same check for `normalize_text`. If the text
reaches the app already containing an unusual unit, it will pass through
unexpanded; the unit table is fixed, so an unusual unit needs a code change.

**The voice adds a syllable that is not in the text.** The model stops only
when it samples an end-of-speech token, so stopping is probabilistic. Set
**Sampling temperature** to 0 for output that is identical every time and
never over-runs.

**First request is slow, later ones are fast.** That is the model load. Turn on
**Load the default model at startup**, or raise **Models kept in memory** if
you switch between models often.

**Memory pressure.** Keep **Models kept in memory** at 1, and prefer the 40M.

## Support & Source

- Source: <https://github.com/hass-cortex/app-cortex-tts>
- Issues: <https://github.com/hass-cortex/app-cortex-tts/issues>
- Integration (HACS): <https://github.com/hass-cortex/cortex-tts>

[app]: https://my.home-assistant.io/redirect/supervisor_addon/?addon=24127962_cortex_tts&repository_url=https%3A%2F%2Fgithub.com%2Fhass-cortex%2Frepository
[app-badge]: https://my.home-assistant.io/badges/supervisor_addon.svg
[hacs]: https://my.home-assistant.io/redirect/hacs_repository/?owner=hass-cortex&repository=cortex-tts&category=integration
[hacs-badge]: https://my.home-assistant.io/badges/hacs_repository.svg
[va]: https://my.home-assistant.io/redirect/voice_assistants/
[va-badge]: https://my.home-assistant.io/badges/voice_assistants.svg
