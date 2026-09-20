# Cortex TTS

On-device text-to-speech for Home Assistant. A catalog of models running
locally — on the CPU, or on a GPU where one answers — no cloud, no API bill,
with the Chinese text front-end none of them ship.

Cheapest first:

| Model         | Voices                            | Languages  |
| ------------- | --------------------------------- | ---------- |
| **Hojo 40M**  | 15 built in (2 zh, 13 en)         | zh, en     |
| **MOSS Nano** | 18 built in (6 zh) **and** clones | zh, en, ja |
| **OmniVoice** | 9 designed **and** clones         | 800+       |

Start at the top; the lower entries want a faster machine or a GPU. What each
one costs, how each clones and which to pick is [Models][models]. Once the app
is running, each model's card shows the real-time factor **your** host has
measured per voice, or says it has none yet, and the verdict that figure
gives.

None of these models reads a unit symbol, a time or a date, and Traditional
Chinese glyphs come out as the wrong words, so the app rewrites the text
before synthesis. [The text pipeline][text] says into what, and the app's own
UI shows the result beside the composer.

## Installation

[![Open this app in your Home Assistant instance.][app-badge]][app]

1. Click **Install**, then **Start**.
2. Check the log to see that it came up.
3. Click **OPEN WEB UI**.

## After installing

### 1. Download a model

Nothing is baked into the image, so the first download takes a minute or two.
With **Preload** on — the default — the app fetches the 40M by itself on first
start. For a different model, or with Preload off, open the panel →
**Models** → **Download**.

### 2. Install the companion integration

[![Open your Home Assistant instance and add the Cortex TTS integration via HACS.][hacs-badge]][hacs]

Press **Add**, then restart Home Assistant. Requires [HACS](https://hacs.xyz)
and Home Assistant 2026.3.0 or later. Without it Home Assistant has no Cortex
TTS platform, and nothing is discovered.

### 3. Pair with the running app

After the restart a **Cortex TTS discovered** card appears under **Settings →
Devices & services**. Click **Configure** and confirm — the address and the
API key come from the app itself. Each downloaded model becomes its own TTS
entity, named after the model: `tts.hojo_tts_light_40m`, `tts.moss_tts_nano`,
and one for each of the others in the table above.

### 4. Assign it to a voice pipeline

[![Open your Home Assistant instance and manage your voice assistants.][va-badge]][va]

Pick (or create) a pipeline → **Text-to-speech** → the Cortex TTS entity, then
the voice. The pipeline's language travels with every reply and decides how
the text is prepared. On most models the voice is what picks the language the
model speaks; OmniVoice takes the language itself, so there the voice is a
timbre.

Calling it from `tts.speak`, finding a voice id and choosing a speaking mode
per model are the [integration's documentation][integration]. Cloning a voice
from a recording is [Cloned voices][cloning]; the recordings live in
`/share/cortex-tts/references`, so they are part of your backups.

## Configuration

Two options are set in the app's configuration tab, because both have to be
settled before the process starts. Everything else is under **Settings** in
the app's own UI, where a change takes effect on the next request.

### Option: `log_level`

How much the app writes to its log. `debug` additionally logs the input text
and the segment count for every request. What the model was actually asked
to say is the admin UI's **What the model is asked to say** panel, or
`POST /api/preview`.

### Option: `discovery_api_key`

The key the Cortex TTS integration uses. Generated on first start and pushed
to Home Assistant through the discovery service. Clear the field and restart
to rotate it; the integration picks up the new value by itself.

### Network

The port is not published by default: the integration reaches the app over
the Supervisor's internal network, and the admin UI comes through ingress.
Publish 8771 under **Network** only to call [the API][api] from elsewhere on
your network; every request then needs the key above.

## Settings

Open the app and scroll to **Settings**. Threads and execution provider are
bound when ONNX Runtime creates a session, so changing either drops whatever
is resident and the next reply loads it again. A smaller **Models kept in
memory** evicts down to the new bound. The rest are read afresh on every
request.

### Default model and voice

Used when a request names none. A voice the chosen model does not offer is
ignored and the first available one is used instead, so leaving the voice
empty always takes the first.

### Preload

Load the default model when the app starts rather than on the first request.
On by default. It holds that model's memory from the moment the app comes up,
and in exchange the first reply does not pay the load — several seconds on
the larger models.

### Sampling temperature

How randomly the model picks each step. Default `0.8`. A model stops speaking
only when it _samples_ its end-of-speech token, so a higher value
occasionally over-runs the text with an invented syllable. `0` is greedy:
reproducible, flatter, and on Hojo 40M it never over-runs. MOSS and
OmniVoice have no temperature at all; a request naming one for those is
refused.

### Inference threads

ONNX Runtime threads per synthesis; `0` lets the runtime decide. **More is
not faster.** The decode loop is Python-bound, so past a couple of threads
the runtime spends its time synchronising them. Leave it at `2`; what each
model gains or loses from four is measured in [Models][models]. The ONNX
sessions adopt a change on their next load, but the BLAS and OpenMP thread
caps are fixed when the process starts, so the full effect waits for a
restart of the app.

### Execution provider

`auto` takes a GPU when one answers and the CPU when none does. `cpu` never
looks for one, which is what you want where a card is present but spoken
for. `cuda` refuses to fall back, so a GPU build quietly running on the CPU is
an error rather than a surprise. What each loaded model actually got is on
its card, beside **loaded**. A CUDA-capable image is needed for `cuda` to
answer at all, and Home Assistant OS ships no NVIDIA driver — see
[Running it elsewhere][standalone].

### Models kept in memory

How many models may stay in memory at once. The default of `1` swaps between
them on demand. Raise it to `2` only if the host can hold both at once; what
two of them hold together is in [Models][models].

### Unload when idle

Seconds after its last request a model is dropped from memory; `0`, the
default, keeps it until something evicts it. The next reply then pays the
load again. Worth setting on a GPU another workload shares, since a resident
model holds its memory whether or not anyone is speaking.

### Text switches, per model and language

What a request gets for `normalize_text`, `expand_numbers`, `convert_script`
and `taiwan_readings` when it does not say. Left alone, each is the
pipeline's call; a rule here answers instead, for one model, one language,
or both. The most specific rule that says something wins, and a language
covers every tag it prefixes (`zh` covers `zh-TW`). The integration's
`options:` still override a rule for that one call. What each switch does is
[The text pipeline][text].

## Troubleshooting

**No voices in the pipeline picker.** The model is not downloaded — check the
app's **Models** cards. A cloned voice appears only once its recording has
been uploaded.

**I do not know what to write for `voice:`.** The app's UI lists every voice
with its id; in Home Assistant, `cortex_tts.list_voices` does. Ids differ per
model — see [Models][models].

**It stutters near the end of long replies.** The model is not keeping ahead
of playback on this host, in that voice. Read
`sensor.<model>_playback_margin`; negative means the renderer lost the race,
and by how much. Set that model's **Speaking mode** to _Wait for the whole
reply_, which never stalls, or pick a faster model; [Keeping up][streaming]
has the rest.

**Chinese sounds like the wrong words.** `convert_script` was turned off for
that call. The integration turns it on whenever the pipeline language is
Chinese; a `tts.speak` call can override it under `options:`, and that is the
only place it can be off. Compare **What the model is asked to say** against
what you hear.

**A word is read the mainland way (垃圾 as lā jī).** The app respells such
words with homophones and lists each one as a `reading` row under **What the
model is asked to say**. A word with no row is missing from the generated
dictionary and needs a code change, not a setting. For the mainland reading,
turn `taiwan_readings` off under `options:`.

**Numbers are read as gibberish, or not at all.** A number with a unit, a
percent sign, a clock colon or a date around it is expanded whenever
`normalize_text` is on. A bare number is left as digits on purpose — it is as
often a room, a phone number or a model as a count — so it goes unread on a
model that cannot read digits. A call whose numbers are counts sets
`expand_numbers: true` under `options:`; a template that formats a sensor
should write the unit. The unit table is fixed, so an unusual unit needs a
code change.

**The voice adds a syllable that is not in the text.** See **Sampling
temperature** above: `0` cures it on Hojo 40M; MOSS and OmniVoice have no
temperature to change.

**First request is slow, later ones are fast.** That is the model load. Turn
on **Preload**, or raise **Models kept in memory** if you switch between
models often.

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
