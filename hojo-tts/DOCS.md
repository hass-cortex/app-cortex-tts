# Hojo TTS

On-device text-to-speech for Home Assistant. Runs the
[Hojo TTS Light](https://github.com/HojoAI/Hojo-TTS-Light) ONNX models on CPU —
no GPU, no cloud, no API bill.

Two models are available:

| Model   | Voices                                    | Speed    | Memory  | Disk   |
| ------- | ----------------------------------------- | -------- | ------- | ------ |
| **40M** | 15 built-in (2 Chinese, 13 English)       | RTF 0.21 | ~780 MB | 241 MB |
| **80M** | none built in; clones from your recording | RTF 0.79 | ~2 GB   | 437 MB |

RTF is the real-time factor: wall-clock time divided by the length of the audio
produced. Below 1 means it speaks faster than the audio plays. Both figures were
measured on a modern desktop core; a slower host scales roughly linearly.

Start with the 40M. It is about four times cheaper and, in a measured ASR round
trip, no less accurate than the 80M. Reach for the 80M only when you actually
want a specific person's voice.

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

[![Open your Home Assistant instance and add the Hojo TTS integration via HACS.][hacs-badge]][hacs]

Press **Add**, then restart Home Assistant. (Requires
[HACS](https://hacs.xyz).) Home Assistant has no Hojo TTS platform without it,
and nothing will be discovered.

### 3. Pair with the running app

After the restart a **Hojo TTS discovered** card appears under **Settings →
Devices & services**. Click **Configure** and confirm — the address and the API
key come from the app itself, so there is nothing to type. Each downloaded
model becomes its own TTS entity.

### 4. Assign it to a voice pipeline

[![Open your Home Assistant instance and manage your voice assistants.][va-badge]][va]

Pick (or create) a pipeline → **Text-to-speech** → choose the Hojo TTS entity,
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

| Option              | Default        | Notes                                                                                                        |
| ------------------- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| `log_level`         | `info`         | `debug` shows the prepared text for every request                                                            |
| `num_threads`       | `2`            | Scaling is nearly flat — the decode loop is Python-bound — so raising this mostly costs the rest of the host |
| `max_loaded_models` | `1`            | Both models resident costs ~2.8 GB. At 1 they swap on demand.                                                |
| `default_model`     | `hojo-40m`     | Used when a request names no model                                                                           |
| `default_voice`     | `hojo_zh_f_01` | Ignored when the model does not offer it                                                                     |
| `temperature`       | `0.8`          | How randomly each step is chosen. `0` is reproducible and never over-runs, at the cost of flatter delivery   |
| `preload`           | `true`         | Loads the default model at startup so the first request is not slow                                          |
| `discovery_api_key` | generated      | Clear it and restart to rotate                                                                               |

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
`temperature: 0` for output that is identical every time and never over-runs.

**First request is slow, later ones are fast.** That is the model load. Turn on
`preload`, or raise `max_loaded_models` if you switch between models often.

**Memory pressure.** Keep `max_loaded_models` at 1, and prefer the 40M.

## Support & Source

- Source: <https://github.com/hass-cortex/app-hojo-tts>
- Issues: <https://github.com/hass-cortex/app-hojo-tts/issues>
- Integration (HACS): <https://github.com/hass-cortex/hojo-tts>

[app]: https://my.home-assistant.io/redirect/supervisor_addon/?addon=24127962_hojo_tts&repository_url=https%3A%2F%2Fgithub.com%2Fhass-cortex%2Frepository
[app-badge]: https://my.home-assistant.io/badges/supervisor_addon.svg
[hacs]: https://my.home-assistant.io/redirect/hacs_repository/?owner=hass-cortex&repository=hojo-tts&category=integration
[hacs-badge]: https://my.home-assistant.io/badges/hacs_repository.svg
[va]: https://my.home-assistant.io/redirect/voice_assistants/
[va-badge]: https://my.home-assistant.io/badges/voice_assistants.svg
