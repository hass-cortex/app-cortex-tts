# Cortex TTS

On-device text-to-speech for Home Assistant. A catalog of models running
locally — on the CPU, or on a GPU where one answers — no cloud, no API bill,
with the Chinese text front-end none of them ship.

Cheapest first:

| Model               | Voices                            | Languages  |
| ------------------- | --------------------------------- | ---------- |
| **Hojo 40M**        | 15 built in (2 zh, 13 en)         | zh, en     |
| **MOSS Nano**       | 18 built in (6 zh) **and** clones | zh, en, ja |
| **Hojo 80M**        | clones only                       | zh, en     |
| **OmniVoice**       | 9 designed **and** clones         | 800+       |
| **Qwen3-TTS**       | 9 built in (5 zh)                 | 10         |
| **Qwen3-TTS clone** | clones only                       | 10         |

**What each one costs is in [Models][models]** — real-time factor, memory and
disk, every model measured on one reference host so the figures rank against
each other. They are measured and edited there and nowhere else, so this page
does not repeat them. **They are not a prediction about your machine** either;
they say only which model outran playback _there_.

Once the app is running, each model's card shows what **your** host measured,
or says it has none yet: one figure covering the model's own voices, which
cost the same, and one for each cloned voice, whose recording rejoins the
prompt on every synthesis and so costs by its own length. The card fits the
per-second part apart from the fixed cost every request pays, so it reads a
little under the reference figure even on the same machine — level with it
where a model's requests are all about one length, because there is no slope
to fit and the plain ratio is taken instead. Each figure needs three requests
before it says anything, and a reply spoken as it is written is rendered in
several of them.

Start at the top of the table; the lower entries want a faster machine or a
GPU. Which model suits what, how each one clones, and what a faster CPU or a
GPU changes is in [Models][models].

None of these models reads a unit symbol, a time or a date, and Traditional
Chinese glyphs come out as the wrong words, so the app rewrites all of it
before synthesis — 32% character error rate against 4% once converted,
measured on the 40M. A bare digit is the one part that depends on the model:
only the two Hojo models cannot say one. [The text pipeline][text] shows exactly what the
model is asked to say, and the app's own UI shows it beside the composer.

## Installation

[![Open this app in your Home Assistant instance.][app-badge]][app]

1. Click **Install**, then **Start**.
2. Check the log to see that it came up.
3. Click **OPEN WEB UI**.

## After installing

### 1. Download a model

Nothing is baked into the image, so the first download takes a minute or two.
With **Preload** on — it is on by default — the app fetches the default model,
the **40M**, by itself on first start and there is nothing to do but wait. To
take a different one, or if you turned Preload off, open the panel →
**Models** → **Download**. Start with the 40M; the table above says why.

### 2. Install the companion integration

[![Open your Home Assistant instance and add the Cortex TTS integration via HACS.][hacs-badge]][hacs]

Press **Add**, then restart Home Assistant. (Requires
[HACS](https://hacs.xyz) and Home Assistant 2026.3.0 or later.) Home Assistant
has no Cortex TTS platform without it, and nothing will be discovered.

### 3. Pair with the running app

After the restart a **Cortex TTS discovered** card appears under **Settings →
Devices & services**. Click **Configure** and confirm — the address and the API
key come from the app itself, so there is nothing to type. Each downloaded
model becomes its own TTS entity, named after the model —
`tts.hojo_tts_light_40m`, `tts.moss_tts_nano`, and one for each of the others
in the table above.

### 4. Assign it to a voice pipeline

[![Open your Home Assistant instance and manage your voice assistants.][va-badge]][va]

Pick (or create) a pipeline → **Text-to-speech** → choose the Cortex TTS entity,
then the voice. The pipeline's language is sent with every reply and decides
how the text is prepared. On most models the voice is what picks the language
the model speaks — a Chinese voice is the only thing that makes them read
Chinese. Qwen3-TTS and OmniVoice take the language themselves, so there the
speaker is a timbre rather than a language.

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

Open the app and scroll to **Settings**, in three groups: the defaults a
request falls back to when it names none, what the text switches default to
per model and language, and what this host spends on answering it.
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
On by default. It takes that model's memory from the moment the app comes up whether or not
anything asks it to speak, and in exchange the first reply does not pay the
load — several seconds on the larger models.

### Sampling temperature

How randomly the model picks each step. Default `0.8`. A model stops speaking
only when it _samples_ its end-of-speech token, so a higher value occasionally
over-runs the text with an invented syllable.

`0` is greedy: reproducible, flatter, and on the Hojo models it never
over-runs. **Not on Qwen3-TTS** — there greedy decoding reliably fails to
sample end-of-speech, so every sentence runs on to the app's own ceiling: two
and a half times the time the text needs, plus four seconds. A finished file
has that invented tail trimmed off; a reply spoken while it is written does
not, because the audio has already gone.

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

`auto` takes a GPU when one answers and the CPU when none does. `cpu` never
looks for one, which is what you want where a card is present but spoken for.
`cuda` refuses to fall back, which is what you want on a host that has a card: a GPU build
quietly running on the CPU is the failure nobody notices. What each loaded
model actually got is printed on its card, beside **loaded**.

A card is the difference between a long reply outrunning the speaker and not;
what it buys each model is in [Models][models], measured rather than
estimated. A CUDA-capable image is needed for `cuda` to answer at all, and
Home Assistant OS ships no NVIDIA driver.

### Models kept in memory

How many models may stay in memory at once. What each one holds, and what two
of them hold together, is in [Models][models]; the default of `1` swaps
between them on demand. Raise it to `2` only if the host can hold both at
once.

### Unload when idle

Seconds after its last request a model is dropped from memory; `0`, the
default, keeps it until something evicts it. The next reply then pays the load
again — about a second for the 40M, about 4 s for MOSS on a GPU. Worth setting
on a card another workload shares: MOSS holds about 2.5 GB of a GPU for as
long as it is resident, whether or not anyone is speaking.

### Silence a sentence end may carry

Seconds, `1.5` by default, 0–10. Playback that catches the renderer at a
sentence end is heard as a longer pause between sentences, which is where a
pause belongs — so the opening need not be held against it, and the first word
comes sooner. A cut inside a sentence never earns this, and neither does a
model that streams audio while it renders. `0` holds until nothing can ever run
dry.

### Text switches, per model and language

What a request gets for `normalize_text`, `expand_numbers`, `convert_script`
and `taiwan_readings` when it does not say. Left alone, each is the
pipeline's call — bare numbers are read only for a model that cannot say a
digit (Hojo), the two Chinese rewrites follow the language — and a rule here
answers instead, for one model, one language, or both. The most specific rule
that says something wins; a language covers every tag it prefixes (`zh`
covers `zh-TW`). The Home Assistant integration's `options:` still override
a rule for that one call.

## Troubleshooting

**No voices in the pipeline picker.** The model is probably not downloaded —
check the app's **Models** cards. A cloning model with no reference recording
uploaded yet also shows no voices, by design; the 80M and the Qwen3-TTS
cloning entry have no built-in ones at
all.

**I do not know what to write for `voice:`.** The app's UI lists every voice
with its id; in Home Assistant, `cortex_tts.list_voices` does. Ids differ per
model — see [Models][models].

**It stutters near the end of long replies.** The model is not keeping up with
playback on this host, and the app's estimate of it was too optimistic. Read
`sensor.<model>_playback_margin`; negative means the renderer lost the race.
The app learns from every request and paces the next reply from what it
measured, and it watches the reply in hand too: once one has produced a second
of audio its own pace is believed over the fit's, so a reply that meets a busy
moment widens its own hold. One stutter usually corrects itself; a model that keeps losing is one to name
a **Speaking mode** for in the integration rather than leave on automatic —
_speak as it renders_ for the first word as soon as one exists, _wait for the whole reply_
for no stutter at all ([Keeping up][streaming]).

**Chinese sounds like the wrong words.** Check that `convert_script` was not
turned off for that call. The integration turns it on whenever the pipeline
language is Chinese; a `tts.speak` call can override it under `options:`, and
that is the only place it can be off. Compare **What the model is asked to
say** against what you hear.

**A word is read the mainland way (垃圾 as lā jī).** The app respells such
words with homophones so any model reads them the Taiwan way, and lists each
one as a `reading` row under **What the model is asked to say**. If the word
has no row, it is not in the table — the table is generated from a dictionary,
and a word missing from it needs a code change, not a setting. If you wanted
the mainland reading, turn `taiwan_readings` off under `options:`.

**Numbers are read as gibberish, or not at all.** A number with a unit, a
percent sign, a clock colon or a date around it is expanded whenever
`normalize_text` is on, which the integration keeps on for every language. A
number with nothing around it is left as digits on purpose — it is as often a
room, a phone number or a model as a count, and read as a count it would
mislead — so it goes unread on a model that cannot read digits; a call that
knows its numbers are counts sets `expand_numbers: true` under `options:`,
and a template that formats a sensor should write the unit. An unusual unit
passes through unexpanded; the unit table is fixed, so it needs a code change.

**The voice adds a syllable that is not in the text.** A model that samples
its own end-of-speech token stops probabilistically. On the **Hojo** models,
setting **Sampling temperature** to 0 gives output that is identical every
time and never over-runs, at the cost of flatter delivery. Do **not** do it on
**Qwen3-TTS** — run greedy it reliably fails to stop at all, and a short line
can run to minutes of invented audio. **MOSS** and **OmniVoice** have no
temperature setting to change.

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
- [Keeping up][streaming] — how the app paces a reply, where the RTF
  threshold is, and the sensors that show it.
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
[streaming]: https://github.com/hass-cortex/app-cortex-tts/blob/main/cortex-tts/docs/delivery.md
[api]: https://github.com/hass-cortex/app-cortex-tts/blob/main/cortex-tts/docs/api.md
[standalone]: https://github.com/hass-cortex/app-cortex-tts/blob/main/cortex-tts/docs/standalone.md
[integration]: https://github.com/hass-cortex/cortex-tts#readme
