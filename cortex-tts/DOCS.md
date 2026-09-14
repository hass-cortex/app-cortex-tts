# Cortex TTS

On-device text-to-speech for Home Assistant. A catalog of models running
locally — on the CPU, or on a GPU where one answers — no cloud, no API bill,
with the Chinese text front-end none of them ship.

| Model               | Voices                            | Languages  | Relative cost | Memory  | Disk   |
| ------------------- | --------------------------------- | ---------- | ------------- | ------- | ------ |
| **Hojo 40M**        | 15 built in (2 zh, 13 en)         | zh, en     | **0.55**      | ~780 MB | 241 MB |
| **MOSS Nano**       | 18 built in (6 zh) **and** clones | zh, en, ja | 1.07          | ~2 GB   | 729 MB |
| **Hojo 80M**        | clones only                       | zh, en     | 1.51          | ~2 GB   | 437 MB |
| **OmniVoice**       | 9 designed **and** clones         | 800+       | 3.83          | ~1.1 GB | 1.4 GB |
| **Qwen3-TTS**       | 9 built in (5 zh)                 | 10         | 6.72          | ~1.6 GB | 1.0 GB |
| **Qwen3-TTS clone** | clones only                       | 10         | 6.87          | ~2.1 GB | 1.3 GB |

**That column is not a prediction about your machine.** It is render time over
audio time with every model measured on one host — a VM with 4 vCPU of an
Intel Core i7-9750H, two inference threads, CPU — so it ranks the models
against each other and nothing else; below 1 means the model outran playback
_there_. Once the app is running, each model's card shows what **your** host
measured, or says it has none yet. Start at the top of the table; the lower
entries want a faster machine or a GPU. Which model suits what, how each one
clones, and what a faster CPU or a GPU changes is in [Models][models].

None of these models can pronounce Traditional Chinese glyphs or an Arabic
numeral, so the app rewrites both before synthesis — 32% character error rate
against 4% once converted. [The text pipeline][text] shows exactly what the
model is asked to say, and the app's own UI shows it beside the composer.

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
[HACS](https://hacs.xyz) and Home Assistant 2026.3.0 or later.) Home Assistant
has no Cortex TTS platform without it, and nothing will be discovered.

### 3. Pair with the running app

After the restart a **Cortex TTS discovered** card appears under **Settings →
Devices & services**. Click **Configure** and confirm — the address and the API
key come from the app itself, so there is nothing to type. Each downloaded
model becomes its own TTS entity (`tts.hojo_tts_light_40m`,
`tts.hojo_tts_light_80m_voice_cloning`, `tts.moss_tts_nano`).

### 4. Assign it to a voice pipeline

[![Open your Home Assistant instance and manage your voice assistants.][va-badge]][va]

Pick (or create) a pipeline → **Text-to-speech** → choose the Cortex TTS entity,
then the voice. On most models the voice is what picks the language — a
Chinese voice is the only thing that makes them read Chinese. Qwen3-TTS and
OmniVoice take a language of their own, and there the pipeline's language is
sent with every reply, so the speaker is a timbre rather than a language.

How to call it from `tts.speak`, find a voice id, and choose a speaking mode
per model is the [integration's documentation][integration]. Uploading a
recording to clone a voice is [Cloned voices][cloning]; the recordings live in
`/share/cortex-tts/references`, so they are part of your backups and can be
copied off over the `share` folder.

## Configuration

Two options are set in the app's configuration tab, because both have to be
settled before the process starts. Everything else is in the app's own UI,
under **Settings** — changing one there takes effect on the next request.

### Option: `log_level`

How much the app writes to its log. `debug` additionally logs the input text
and the segment count for every request. To see what the model was actually
asked to say, use the admin UI's **What the model is asked to say** panel or
`POST /api/preview`.

### Option: `discovery_api_key`

The key the Cortex TTS integration uses. Generated on first start and pushed to
Home Assistant through the discovery service, so it normally needs no
attention. Clear the field and restart to rotate it; the integration picks up
the new value by itself.

### Network

The port is not published by default: the integration reaches the app over
the Supervisor's internal network, and the admin UI comes through ingress.
Publish 8771 under **Network** only to call [the API][api] from elsewhere on
your network; every request then needs the key above.

## Settings

Open the app and scroll to **Settings**, in two groups: the defaults a request
falls back to when it names none, and what this host spends on answering it.
Two of these — threads and execution provider — are bound when ONNX Runtime
creates a session, so changing one drops whatever is resident and the next
reply loads it again; a smaller **Models kept in memory** evicts down to the
new bound; the rest are read afresh on every request.

### Default model and voice

Used when a request does not name one. A voice the chosen model does not offer
is ignored and the first available one is used instead, so leaving the voice
empty always takes the first.

### Preload

Load the default model when the app starts rather than on the first request.
It takes that model's memory from the moment the app comes up whether or not
anything asks it to speak, and in exchange the first reply does not pay the
load — several seconds on the larger models.

### Sampling temperature

How randomly the model picks each step. Default `0.8`. A model stops speaking
only when it _samples_ its end-of-speech token, so a higher value occasionally
over-runs the text with an invented syllable.

`0` is greedy: reproducible, flatter, and on the Hojo models it never
over-runs. **Not on Qwen3-TTS** — there, greedy decoding often fails to sample
end-of-speech at all, and the reply is cut off at the model's own ceiling
instead.

The Hojo models and Qwen3-TTS read this setting. MOSS and OmniVoice fuse their
sampling into a dedicated graph and have no temperature at all, so a request
naming one for those is refused rather than silently ignored.

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

A change here is only half adopted: the ONNX sessions pick it up on the next
load, but the BLAS and OpenMP thread caps are fixed when the process starts, so
the full effect waits for a restart of the app.

### Execution provider

`auto` takes a GPU when one answers and the CPU when none does. `cuda` refuses
to fall back, which is what you want on a host that has a card: a GPU build
quietly running on the CPU is the failure nobody notices. What each loaded
model actually got is printed on its card, beside **loaded**.

MOSS-TTS-Nano measured 1.025x real time on a laptop i7 against **0.354x** on a
GTX 1650 — the difference between a long reply outrunning the speaker and not.
A CUDA-capable image is needed for `cuda` to answer at all, and Home Assistant
OS ships no NVIDIA driver.

### Models kept in memory

How many models may stay in memory at once. The 40M needs about 780 MB, the
80M and MOSS about 2 GB each, so the default of `1` swaps between them on
demand. Raise it to `2` only if the host can hold two — about 2.8 GB for the
40M beside either of the others, about 4 GB for the 80M beside MOSS.

### Unload when idle

Seconds after its last request a model is dropped from memory; `0`, the
default, keeps it until something evicts it. The next reply then pays the load
again — about a second for the 40M, about 4 s for MOSS on a GPU. Worth setting
on a card another workload shares: MOSS holds about 2.5 GB of a GPU for as
long as it is resident, whether or not anyone is speaking.

### Refuse over-long replies

Seconds; a reply whose estimated render would take longer than this on the
chosen model's measured speed is refused with a clear error rather than
rendered. `0`, the default, accepts any length. What it stops is a reply long
enough to render past the caller's own timeout: the audio then finishes into a
connection nobody is reading, having held the model for the whole of it — one
514-character story measured at over seven minutes on a CPU that renders
OmniVoice at 4.6x. The estimate needs the model to have been measured on this
host at least once, so the very first long reply on a fresh model still runs.

## Troubleshooting

**No voices in the pipeline picker.** The model is probably not downloaded —
check the app's **Models** cards. A cloning model with no reference recording
uploaded yet also shows no voices, by design; the 80M has no built-in ones at
all.

**I do not know what to write for `voice:`.** The app's UI lists every voice
with its id; in Home Assistant, `cortex_tts.list_voices` does. Ids differ per
model — see [Models][models].

**It stutters near the end of long replies.** The model is not keeping up with
playback on this host. Read `sensor.<model>_playback_margin`; negative means
the renderer lost the race, and [Keeping up][streaming] lists the five things
that fix it, starting with setting that model back to buffered.

**Chinese sounds like the wrong words.** Check that `convert_script` was not
turned off for that call. The integration turns it on whenever the pipeline
language is Chinese; a `tts.speak` call can override it under `options:`, and
that is the only place it can be off. Compare **What the model is asked to
say** against what you hear.

**Numbers are read as gibberish.** Same check for `normalize_text`, which the
integration turns on for every language. If the text reaches the app already
containing an unusual unit, it will pass through unexpanded; the unit table is
fixed, so an unusual unit needs a code change.

**The voice adds a syllable that is not in the text.** The model stops only
when it samples an end-of-speech token, so stopping is probabilistic. Set
**Sampling temperature** to 0 for output that is identical every time and
never over-runs.

**First request is slow, later ones are fast.** That is the model load. Turn on
**Preload**, or raise **Models kept in memory** if
you switch between models often.

**Memory pressure.** Keep **Models kept in memory** at 1, and prefer the 40M.

## Documentation

- [Models][models] — the line-up, what each costs, how each clones, which to
  pick.
- [The text pipeline][text] — why Traditional Chinese and numbers are
  rewritten, and into what.
- [Cloned voices][cloning] — the recording, the transcript, the name.
- [Keeping up][streaming] — buffered, streamed, the sensors that decide it.
- [Running it elsewhere][standalone] — a faster CPU or a GPU outside HAOS.
- [HTTP API][api] — using the app without the integration.
- [Integration][integration] — `tts.speak`, voice ids, speaking mode, the
  sensors.

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
[models]: https://github.com/hass-cortex/app-cortex-tts/blob/main/cortex-tts/docs/models.md
[text]: https://github.com/hass-cortex/app-cortex-tts/blob/main/cortex-tts/docs/text-pipeline.md
[cloning]: https://github.com/hass-cortex/app-cortex-tts/blob/main/cortex-tts/docs/cloning.md
[streaming]: https://github.com/hass-cortex/app-cortex-tts/blob/main/cortex-tts/docs/streaming.md
[api]: https://github.com/hass-cortex/app-cortex-tts/blob/main/cortex-tts/docs/api.md
[standalone]: https://github.com/hass-cortex/app-cortex-tts/blob/main/cortex-tts/docs/standalone.md
[integration]: https://github.com/hass-cortex/cortex-tts#readme
