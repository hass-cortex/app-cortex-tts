# Cortex TTS — Domain Language

On-device text-to-speech over ONNX models that have no text front-end of their
own. The vocabulary below covers what callers reach for across the codebase;
general programming patterns (locks, caches, dataclasses) are not listed.

Most of the confusion this file exists to prevent comes from one place: almost
every noun here has a second, unrelated meaning somewhere else in the same
request path.

## Language

### The text path

**Text path**:
Everything `prepare()` does between the caller's string and the model's input:
normalise, convert, split. It is the product, not a preprocessing detail — the
models read glyphs, so whatever this produces is what gets spoken.
_Avoid_: "preprocessing", "cleanup" (both suggest something optional)

**Normalisation**:
Expanding the fixed shapes a sensor produces — units, percentages, clock
literals, ISO dates — into words: `26.5°C` into 攝氏二十六點五度. A bare `80`
is read as "eighty" only when `expand_numbers` asks, or the model cannot say
a digit at all (`needs_number_words`, Hojo). Which language the words
come out in is the **Locale**'s, chosen by the request's language tag.
_Avoid_: "conversion" (that is the other pass), "normalize the audio" (see
Flagged ambiguities)

**Script conversion**:
The Traditional-to-Simplified glyph swap (OpenCC `t2s`). Glyph shape only —
`tw2sp` would also rewrite vocabulary (設定 to 设置) and change the words the
model says.
_Avoid_: "translation" (a different language), "normalisation", "correction"
(the Traditional text was not wrong; this model just cannot pronounce it)

**Pass**:
One substitution in the ordered table in `text/passes.py`: a pattern, a
rendering, and the option that switches it off. Order is the contract — a
construct must claim its number before the bare-number pass reads that digit
as a quantity.

**Prepared text**:
What the model is actually asked to say, after every pass. The admin UI shows
it under that name because it is the only place the rewrite is visible before
it is audible.
_Avoid_: "output" (the output is audio), "result"

**Segment**:
One synthesis-sized piece of Prepared text — at most `MAX_CHARS_PER_SEGMENT`
characters, one model call, and always ending in sentence-final punctuation
because without it the model invents a syllable. A segment may hold several
sentences, and one long sentence may be split across several segments.
_Avoid_: "sentence" (a **Sentence** is what the splitter sees on the way in; a
segment is what the engine gets on the way out)

**Locale**:
What the pipeline knows about one language: how its numbers are read, which
stop ends a sentence, and the rewrites only it has (`text/locales.py`).
Chinese and English are written (`text/zh/`, `text/en/`); every other language
gets the generic one — numbers from `num2words`, unit names and date layout
from CLDR via `babel`, clock literals as hour words then minute words, and
nothing it is not sure of. Adding a language is
adding a locale, never a branch in the pipeline. A model that reads digits
itself (`reads_numerals` in the catalog) makes the generic locale stand
aside; a written one is kept.

**Sniffed language**:
The tag the pipeline settles on when a request carries none and the voice
declares none — kana is `ja`, 漢字 outnumbering Latin _words_ is `zh` (or
`zh-Hant` in Traditional glyphs), anything else `en`. A fallback, not the
rule: the request's tag wins when there is one.
_Avoid_: "dominant script" (the old name, from when the request had no
language), "detected language"

**Identifier reading** / **Quantity reading**:
Two ways to say the same digits. A number welded after a letter is a label and
is read digit by digit (`P0` → "P zero"); one before a letter carries a unit
and stays a quantity (`24V` → "twenty-four V"). Leaving either as digits is not
neutral — the model speaks no numeral at all, so an unexpanded digit is silent.

**Request flag**:
`normalize_text` / `expand_numbers` / `convert_script` / `taiwan_readings` on
the wire. The last two are Chinese's, and absent means the language decides.
`normalize_text` and `convert_script` are a capability for a caller whose
text is already prepared, not a preference: turning conversion off for
ordinary Traditional Chinese makes the voice unintelligible.
`expand_numbers` is off unless asked — a bare number is as often a room or a
phone number as a count — except for a model that cannot say a digit at all,
where digits are noise and it is on; `taiwan_readings` is a preference.

**Taiwan reading** / **Stand-in**:
A word Taiwan reads differently from the mainland (垃圾 lè sè), and the
homophone it is respelled with so the model says it that way (乐色). The table
is generated from the McBopomofo dictionary by `scripts/taiwan_readings.py`,
never edited by hand.
_Avoid_: "pinyin hint" (measured fragile; not what the app does)

### Models and voices

**Catalog model**:
An entry in `catalog.py` — id, ONNX file list, Hugging Face repo, and the cost
figures the UI shows. The set of _downloadable_ models. Weights are never in
the image; they arrive on first download and live under `/data`.
_Avoid_: "built-in model" (nothing is built in)

**Engine**:
A loaded model behind the `Engine` protocol (`PresetEngine` for the 40M,
`CloneEngine` for the 80M, `MossEngine` for MOSS-TTS-Nano and `Qwen3TtsEngine`
for both Qwen3-TTS entries, which also satisfy `StreamingEngine`, and
`OmniVoiceEngine` for OmniVoice). One engine serves one synthesis at a time:
the ONNX sessions drive a stateful per-token loop.
_Avoid_: "model" for the loaded thing — a _model_ is files on disk, an _engine_
is the runtime holding ~780 MB or ~2 GB of it

**Resident**:
Loaded into memory. The registry keeps at most `max_loaded_models` engines and
evicts the least recently used: the 40M beside either 2 GB model costs about
2.8 GB, the 80M and MOSS together about 4 GB.
_Avoid_: "cached" (eviction is about memory, not staleness)

**Voice**:
A selectable speaker, always belonging to exactly one model. On the 40M a
built-in slot (`hojo_zh_f_01`); on the 80M a **Reference recording**; on MOSS
and Qwen3-TTS either (`Yuewen`, `vivian`, or a reference); on OmniVoice a
**Designed voice** or a reference. The voice is also the only place a language
is declared — the model takes no language parameter, so picking the voice is
picking the language.
_Avoid_: "speaker" (that is the embedding slot inside the model)

**Designed voice**:
OmniVoice's third kind. Not a slot in the bundle and not a recording: a short
instruction built from the model's **closed** attribute vocabulary — a sex, an
age band, a pitch band, whisper, an accent or a dialect. The nine on offer are
chosen in `engine/omni.py` rather than read off disk, and the model refuses an
attribute outside its vocabulary instead of approximating it. They are
reported as `source: designed`, a third value beside `builtin` and
`reference`, because the three cost different amounts to render — a clone
measured 7.17 against 3.46 for a designed voice on the same model and host —
and a caller comparing them has to know which it is looking at.
_Avoid_: "prompt", "style" (the model validates them as a fixed list, not free
text), "preset" (that is the 40M's word)

**Reference recording**:
An audio file _and_ the words spoken in it — both, always. The recording
supplies the timbre; the transcript tells the model which sounds map to which
text. A wrong transcript degrades the clone with no error, which is why it is
validated on the way in.
_Avoid_: "sample" (suggests something disposable), "voice file"

**Conditioning** / **Sidecar**:
What an engine derives from a Reference recording before it can clone it —
codec frames, a speaker vector, a prompt. Encoding it is the dominant cost of a
cloned utterance, so it is cached by reference id and, per engine, written
beside the recording as `<id>.<fingerprint>.<engine>.*`: the _sidecar_. The
fingerprint in the name is the invalidation; a replaced recording misses by
construction. Sidecars are disposable.
_Avoid_: "cache file" (says nothing about what is in it or when it is stale)

**Raw transcript** / **Transcript**:
Two fields, deliberately. The _raw transcript_ is what a person typed; the
_transcript_ is what the model is told, after the Text path has run over it.
The UI shows the raw one for editing and reports the other on save.

**Cloned voice**:
The voice a Reference recording defines. It exists the moment the reference is
stored — no training, no build step — and Home Assistant sees it within
seconds via the models-changed event.

### Latency

**Streamed synthesis**:
Speaking a reply sentence by sentence, so playback starts on the first one
while the rest is still being made. The _integration_ does this by calling
`/api/speak` per sentence. Needs no engine support: it is the _integration_
splitting the reply, not the model streaming it.
_Avoid_: "streaming" unqualified (see Flagged ambiguities)

**Chunk streaming**:
An engine emitting audio before a whole segment is finished. A `ModelSpec`
capability and a separate `StreamingEngine` protocol, both true only for
MOSS-TTS-Nano. `/api/speak/stream` carries it over HTTP as chunked MP3, and
the integration consumes it inside each sentence. The measurement is in the
`/api/speak/stream` docstring.
Distinct from **Streamed synthesis**, which is per _sentence_ and needs no
engine support at all.

**Time to first audio**:
The wait a listener actually experiences: request in, first frame out. The
number that matters, and the only one that changes when streaming is turned on.
_Avoid_: "latency" (unqualified), "inference time"

**Inference time**:
What the model cost, summed across segments. It is not a wait: segment one is
already playing while segment three is being generated.

**RTF**:
Real-time factor — inference time divided by the length of audio produced.
Below 1 means it speaks faster than the audio plays, which is what a streamed
reply needs in order not to run dry mid-sentence.

### Configuration

**Option**:
A field in the app's configuration tab, stored by the Supervisor and declared
in `config.yaml`. Only two are left — the log level and the discovery key —
because both have to be settled before the process starts. Changing one costs
a restart; changing the _schema_ of one costs a rebuild.
_Avoid_: calling a Setting an option, which promises a restart that no longer
happens and a place that no longer has it.

**Setting**:
A field the app stores itself, in `settings.json` beside the models, changed in
the admin UI and over `PUT /api/settings`. Read afresh on the next request —
except the two ONNX Runtime binds when it creates a session (threads,
execution provider), which are adopted by dropping whatever is resident. The
thread count is only half of that: its BLAS/OpenMP side is fixed at import and
waits for a restart.

## Relationships

- A **Catalog model** becomes an **Engine** when the registry loads it; the
  engine is **Resident** until evicted. Downloading, loading and evicting are
  three separate transitions and the UI shows all three.
- A **Voice** belongs to exactly one model. Where it comes from is the model's
  `builtin_voices` and `cloning` capabilities, which are independent: the 40M
  has only bundled voices, the 80M only reference recordings, and MOSS has
  both, so its voice list is the two concatenated. `chunk_streaming` and
  `temperature` are the other two capabilities, equally independent.
- The **Text path** runs before any engine is touched, so `/api/preview`
  answers without loading a model at all — which is what makes the admin UI's
  right-hand column free.
- **Normalisation** runs before **Script conversion**: normalisation emits
  Traditional number words, so reversing the two would leave freshly-minted
  Traditional glyphs downstream of the only pass that can fix them.
- **Segment** count drives **Streamed synthesis**: one segment means one model
  call and no streaming benefit, however the flags are set.
- A **Reference recording** and the models-changed event are the two things
  that alter the voice list without a config-entry reload.

## Example dialogue

> **Dev:** "The request said `normalize_text: true` but the numbers came out in
> English. Is that a bug?"
> **Maintainer:** "Not by itself — the flag says _whether_ to expand, the
> The **Locale** says _into what_. If the sentence had more Latin words
> than 漢字, English is the correct reading."

> **Dev:** "Can I turn off **Script conversion** for a Traditional voice?"
> **Maintainer:** "You can, and it will sound like noise — 32% character error
> rate against 4%. The flag is for callers whose text is already Simplified,
> not a style preference."

## Flagged ambiguities

- **"normalise" is two unrelated operations in one request.**
  `text/zh/normalize.py` rewrites numerals into words; `audio.normalize_level`
  scales a waveform's amplitude. They run in the same call and share no
  vocabulary. Say "text normalisation" or "level normalisation" — never the
  bare verb.
- **"streaming" is four things.** Home Assistant's _streaming input_ (the
  conversation agent feeding text in as it is written), our **Streamed
  synthesis** (one `/api/speak` per sentence), **Chunk streaming** (an engine
  emitting audio mid-segment, which only MOSS can do),
  and HTTP chunked responses, which `/api/speak/stream` does and `/api/speak`
  does not. A sensor that
  measured the first of these was removed precisely because the name promised
  one and the clock measured another.
- **"voice" without a model is meaningless.** `hojo_zh_f_01` exists on the 40M
  and nowhere else, `Yuewen` only on MOSS, and the 80M's voices are whatever
  references have been uploaded. A reference is a voice on _every_ model that
  can clone, under the same id — so even a reference id needs a model beside
  it. Always say which model.
- **"transcript" is two wire fields** — `raw_transcript` and `transcript`. See
  the entry above; they differ exactly when the text path changed something.
- **"temperature"** is the sampling temperature of the language model. Nothing
  in this codebase measures heat, but the unit table does expand `°C`, and both
  appear in the same file.
- **"default_voice"** is a stored _name_, not a resolved voice. It is
  ignored when the chosen model does not offer it, and both the synthesis path
  and the admin UI fall back to the first available voice.
- **"model" in the admin UI** means a Catalog model; **"model" in a log line
  from the engine** usually means the loaded ONNX session. The distinction is
  Catalog model versus **Engine**.
