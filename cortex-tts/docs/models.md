# Models

Which models Cortex TTS can run, what each costs, and how to choose one.
`src/cortex_speech/catalog.py` is the source of truth for the line-up; this
page is the copy written for people, and the two are edited together.

Nothing is baked into the image: each model is downloaded from the app's UI on
first use into `/data/models`, so removing the app with its data removes the
weights. Every model runs locally, on the CPU or on a GPU where one answers
and the execution provider setting allows it. The figures below are CPU
figures.

## The line-up

| Model         | Voices                            | Languages  | RTF      | Memory  | Disk   |
| ------------- | --------------------------------- | ---------- | -------- | ------- | ------ |
| **Hojo 40M**  | 15 built in (2 zh, 13 en)         | zh, en     | **0.55** | ~780 MB | 241 MB |
| **MOSS Nano** | 18 built in (6 zh) **and** clones | zh, en, ja | 1.06     | ~2 GB   | 729 MB |
| **OmniVoice** | 9 designed **and** clones         | 800+       | 3.71     | ~1.1 GB | 1.4 GB |

RTF — real-time factor, render seconds divided by audio seconds; below 1 means
the model speaks faster than the audio plays. One measurement, every model on
the same host with the same text and settings, so the figures are comparable
with each other and **not a prediction about your machine**. A model card
shows what your host measured instead, per voice, fixed cost included
([Delivering a reply](delivery.md)); a card built from short sentences reads
above this column. Upstream figures are not used; each model's authors
measured on a different machine.

**Memory is per resident engine**; **Models kept in memory** decides how many
may be resident at once. The 40M beside MOSS costs about 2.8 GB; a card holds
more
([Hardware](#hardware-and-running-it-elsewhere)). **Only the 40M keeps ahead
of playback** — read the column as what each costs, not as a ranking, and
read [Which model](#which-model) before reaching past the top two.

## Measured across the models

### Hardware, and running it elsewhere

The same script, the same text, four environments: two machines, each one's
CPU then its own GPU. Every figure is the median over the four sentences above.

| Model         | 4 vCPU of an i7-9750H, CPU | GTX 1650 Laptop, CUDA | Ryzen 9 9955HX, CPU | RTX 5070 Ti Laptop, CUDA |
| ------------- | -------------------------- | --------------------- | ------------------- | ------------------------ |
| **Hojo 40M**  | 0.57                       | 0.31                  | **0.26**            | 0.48                     |
| **MOSS Nano** | 1.09                       | 0.38                  | **0.35**            | 0.75                     |
| **OmniVoice** | 3.77                       | 0.74                  | 1.67                | **0.18**                 |

Column one is the Home Assistant OS VM, which ran all three
([The other VM on the same CPU](#the-other-vm-on-the-same-cpu)). Column two
is the reference VM on that same i7-9750H with its GTX 1650 passed through
(4 vCPU, 11 GB, Ubuntu, driver 575), the provider set to `cuda`; its CPU
figures are the reference host's, 0.55, 1.06 and 3.71. Columns three and four
are the development laptop (16-core AMD Ryzen 9 9955HX, 39 GB, RTX 5070 Ti
Laptop GPU, Windows), CPU at two threads then GPU. All four were measured
2026-09-21 with two inference threads and one model resident; the two i7
columns ran Cortex TTS 0.7.0 on ONNX Runtime 1.26.0, the laptop 0.8.0 on
1.29.0 with CUDA 13.4 and cuDNN 9.26.

- **Home Assistant OS cannot use a GPU.** It ships no NVIDIA driver, so on
  HAOS every model runs on the CPU whatever the provider is set to.
- **A faster CPU is the reliable win**: two to three times faster on the Ryzen
  for every model, enough to put the 40M and MOSS ahead of playback.
- **A GPU is what makes OmniVoice usable**: 3.77 to **0.18** on the 5070 Ti,
  by far the largest gain from a card here and the fastest figure in the
  table. A cloned voice re-encodes its reference on every request (about 1.5 s
  on the GTX 1650), which keeps it past the streaming threshold, so a live
  reply in it is buffered.
- **VRAM on load**, one model at a time on the 4 GB GTX 1650: 556 MiB for the
  40M, 634 MiB for OmniVoice, **2414 MiB for MOSS**. MOSS is the one with no
  room beside anything else, and it fails to allocate if another process holds
  about 1.5 GB.
- **A GPU pays beside a weak CPU, not a strong one.** The GTX 1650 takes the
  i7-9750H VM's 40M from 0.55 to 0.31 and MOSS from 1.06 to 0.38; the RTX 5070
  Ti was slower than the Ryzen's own CPU for both — 0.48 against 0.26, 0.75
  against 0.35. The decode loop is thousands of tiny kernels with a host sync
  per token, bound by launch latency: one 64² ONNX MatMul takes 80 µs on the
  5070 Ti and 53 µs on the 1650, against 6.7 µs on the Ryzen's CPU and 11 µs
  on the i7's. The card only wins once the matrices are far larger than these
  models use — at 4096² the 5070 Ti takes 18.8 ms against the Ryzen's 267 ms.
  Windows adds a tail (p99 262 µs, spikes past 2 ms, against 76 µs and a 113 µs
  worst case on the native-Linux GTX 1650), so the laptop's GPU figures wander
  (40M 0.37–0.58, MOSS 0.56–0.98) while the 1650's repeat to the hundredth.
  Kept busy the 5070 Ti is 5.5x the 1650; fed one token at a time it is not.
- **Threads are a trade between models.** Four against two on the same four
  cores: OmniVoice 3.71 to **2.65** (1.4x), MOSS 60% _slower_, the 40M
  unmoved (0.55 to 0.54).

### What a benchmark cannot measure

Every figure above came from requests issued back to back on an idle host —
right for comparing models, wrong for the per-voice median a live reply is
spoken from ([Delivering a reply](delivery.md)). A benchmark is a run of one
length, so afterwards the median describes that length and nothing else — a
clone's short sentences cost more than its average — and its 24 samples per
voice evict the real traffic. `DELETE /api/models/{id}/stats` afterwards, on
any host whose figures are in use.

### How much text one synthesis takes

A segment is one call into the model. What one call may **produce** is the
model's own and is declared; what it should **cost** on this machine is not.
Two declared bounds keep a call inside what the model can do, and they fail
differently.

**The ceiling** is `ModelSpec.max_audio_s`, counted in audio: past it the call
returns what it had and the rest of the text is never spoken — it does not
slow down first — so the text path splits to stay under it.

| Model         | Ceiling    | Where it comes from                   |
| ------------- | ---------- | ------------------------------------- |
| Hojo 40M      | **41 s**   | 2048 new tokens at a 50 Hz codec      |
| MOSS-TTS-Nano | **30 s**   | `max_new_frames` 375 at 12.5 Hz       |
| OmniVoice     | none known | nothing in the model or its code says |

MOSS's was found by being hit: seven inputs from 155 to 284 Chinese characters
each came back as exactly 30.0 s, the figure its manifest gives. Hojo's is
read off the generation limit the model declares. OmniVoice is left
without one, and without a character bound — a figure neither the model's nor
this host's will be wrong on some machine; at 108 characters it returned
everything it was given.

**The budget** is `ModelSpec.max_text_tokens`, counted in the model's own text
tokens, and it is not a ceiling at all. A model that stops when it samples an
end-of-speech token can stop anywhere, and a longer call is more chances to
stop early; the budget only makes each call short enough for that to be rare.
It is applied by the engine, because counting it needs the model's tokenizer
and characters per token is a property of the script (measured on MOSS: 3.8 to
4.0 for Latin, about 1 for Han).

| Model         | Budget        | Where it comes from                                                   |
| ------------- | ------------- | --------------------------------------------------------------------- |
| MOSS-TTS-Nano | **50 tokens** | largest chunk that kept all its text: 49; smallest that lost some: 61 |
| Hojo 40M      | none          | its early stops have not needed one                                   |
| OmniVoice     | none          | decodes a fixed number of steps                                       |

The same 411-character English text at MOSS's pinned seed: one call, 12.00 s
of the 28 s it needed; cut to the budget, all of it. Upstream's own splitter
defaults to 75, which is a different code path's figure and too loose here —
at 75 the first chunk came back at 0.77 of the duration its text needed.
What survives the budget is caught afterwards: a generation judged to have
stopped early is retried at another seed, and where it cannot be (a streamed
reply, whose chunks have already gone) it is logged
([ADR 0003](adr/0003-overrun-trimming-and-stopping.md)).

A ceiling in seconds becomes text through the slow-side speech-rate priors in
`text.scripts` — **4.1 characters a second** for Han, kana and hangul, **14.7**
for Latin, the slower of the two for any script not yet measured —
priors because a segment must not depend on which voice says it; MOSS's 30 s
is about 122 Chinese characters or 440 Latin ones. What bounds a request is
not a figure: one sentence when streaming, every sentence that had arrived
when buffered, split at the ceiling and nowhere else
([Delivering a reply](delivery.md)). Long prompts degrade before they
truncate, so a ceiling is not a number to approach. On a small card memory
can end a segment before the ceiling does — a visible error; the registry
drops the engine and gives the card back. A buffered reply, which renders
every waiting sentence as one request, is the case to watch there.

## The models

### Hojo TTS Light 40M

[HojoAI/Hojo-TTS-Light](https://github.com/HojoAI/Hojo-TTS-Light), the 40M
export. The cheapest model by a wide margin and the one to start with.

- **Voices**: 15 built in, named by language, sex and number (`hojo_zh_f_01`,
  `hojo_en_f_04`). No cloning. **Languages**: Chinese and English; the voice
  picks which.
- **Sampling temperature** applies (default 0.8; 0 is greedy and never
  over-runs). No chunk streaming. Ceiling 41 s.
- **Cannot say a digit** (`needs_number_words`): bare numbers are expanded for
  it by default.

### MOSS-TTS-Nano

[OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano), the 100M
ONNX export, two bundles (weights and audio codec) downloaded together.

- **Voices**: 18 built in, given names (`Yuewen`, `Junhao`; six Chinese),
  **and** clones. **Languages**: Chinese, English and Japanese — the only
  model here that reads Japanese.
- **Cloning**: no speaker encoder; the whole recording (2–20 s) as codec tokens
  is the voice, paid once per reference and cached. **Never reads the
  transcript** (`reads_reference_transcript` is false for it alone) — type it
  properly anyway; the same recording is a voice on OmniVoice, which does.
- **Chunk streaming**, the only model here with it: first audio 143–178 ms
  after the request against 1.8 s for the whole utterance.
- **No sampling temperature** (fused into the ONNX graph); a request naming one
  is refused (`NO_TEMPERATURE`). Ceiling 30 s, budget 50 tokens. Slower at four
  threads (above).
- **Stops when it samples the stop**, which is why it has a budget at all: the
  same text at a different seed runs to a different length, measured on one
  411-character segment at 12.00 s, 26.64 s, 11.60 s, 26.32 s and 27.52 s for
  five seeds. A rendered reply that came back short is retried at another
  seed; a streamed one says so in the log instead
  ([ADR 0003](adr/0003-overrun-trimming-and-stopping.md)).
- **Reads Latin words poorly**: 32% character error on a reply with a product
  name, against 0–9% for Hojo. Suits replies that are Chinese throughout.
- **A GPU takes it under real time**: 1.06 here, **0.38** on a GTX 1650 —
  which means [running outside HAOS](standalone.md).

### OmniVoice 0.8B

[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice): a 0.6B Qwen3
language model over a 0.2B Higgs Audio V2 tokenizer. The language model runs from an
int4 ONNX export; the rest is PyTorch, which is why the image carries torch,
torchaudio and transformers.

- **Voices**: nine designed **and** clones, no bundled speakers. Measured apart
  on the HA VM — a designed voice at RTF 3.46, a ten-second clone at 7.17 — so
  the nine share one line and every clone gets its own.
- **Languages**: 800+ claimed upstream, 646 in the engine's own table; the
  catalog and admin UI list the ten this project has exercised, the rest
  reachable over the API (`yue`, `th`, `vi`, …), unmeasured rather than
  disabled.
- **Designed voices**: an instruction from a **closed vocabulary** — sex, age
  band, pitch band, whisper, an English accent or a Chinese dialect — refused
  outside it. The model translates it for Chinese text, so a designed voice
  claims no language.
- **Cloning**: the whole recording as codec tokens in the prompt, as MOSS, no
  speaker encoder to cap fidelity. Asks for 3–10 s, warns past 20; a 70 s
  reference measured 11.4 GB resident.
- **No sampling temperature**: the decoder's two (5.0 picks which positions to
  unmask next, the token pick was tuned at 0) are not the setting's quantity;
  a request naming one is refused (`NO_TEMPERATURE`).
- **The 2.45 GB of transformer weights are never downloaded**: the ONNX graph
  replaces `forward` on a meta-device module — 1.1 GB peak resident against
  4.7 GB loaded whole.
- **`cuda` covers the ONNX graph only**; tokenizer, prompt handling and decoder
  stay in torch on the CPU, and `/health` reports the session. Still 3.77 to
  0.18, because the graph is where the time goes.
- **No chunk streaming** (a fixed number of unmasking steps over the whole
  utterance). No ceiling declared. Gains the most from threads (above).

## Voice ids

Every model names its voices differently, so an id from one is meaningless on
another:

| Model     | Looks like                     | Where the name comes from                   |
| --------- | ------------------------------ | ------------------------------------------- |
| Hojo 40M  | `hojo_zh_f_01`, `hojo_en_f_04` | Fixed in the bundle: language, sex, number  |
| MOSS Nano | `Yuewen`, `Junhao`             | Given names, as the bundle ships them       |
| OmniVoice | `female-young`                 | The nine designed voices, fixed in the app  |
| A clone   | `xiaohe`                       | Whatever you named the recording, slugified |

A cloned voice is the same id on **every** model that clones, because the
recording defines it. The app's UI lists every voice; in Home Assistant
`cortex_tts.list_voices` returns them with their ids.

## Which model

Two questions decide it, and neither is "which is best".

**Is anyone waiting?** The silence before the first word is roughly RTF times
the seconds of speech that have to exist first: at RTF 1.0 a three-second
announcement means three seconds of nothing, a nine-second Assist answer nine.
For an announcement nobody is standing there; for a conversation the wait is
the product. **Is the reply long?** Past about half a minute keeping up
matters more than the wait: a voice under the host's threshold (0.8 unless
changed in Settings) is spoken as the reply is written, one at or above it
is held until rendered, and a model at RTF 1.1 falls a second behind for
every ten seconds it speaks ([Delivering a reply](delivery.md)).

- **Announcements, notifications, timers**: Hojo 40M — cheapest by a wide
  margin, 15 voices without uploading anything.
- **A voice assistant answering questions**: Hojo 40M, or MOSS Nano if you
  want the voice; MOSS costs 2 GB.
- **Japanese**: MOSS Nano, the cheapest that reads it; OmniVoice does too, at
  several times the cost.
- **Korean, German, French, Russian, …**: OmniVoice, on a fast machine or a
  GPU; it designs or clones its voice rather than shipping speakers.
- **One specific person's voice**: MOSS Nano — clones, with built-in voices to
  fall back on.
- **A voice nobody has recorded**: OmniVoice — no upload, no training.
- **Long replies with a short wait**: whichever your host measures **under
  its threshold**. Read `sensor.<model>_real_time_factor`, not the table.

OmniVoice measured 3.71 on the reference host — 7x the 40M, a nine-second
answer taking 33 seconds to render. It is reasonable on a fast desktop CPU or
a GPU and does not belong in a conversation on a Home Assistant box:
[Running it elsewhere](standalone.md).

## What no model here does

- **A cloned voice is a likeness, not a match.** MOSS conditions on the whole
  recording, but at 100M parameters it is still a likeness; OmniVoice does
  too, and is where a careless transcript shows.
- **A wrong transcript fails silently**: confident audio, simply less like the
  person ([Cloned voices](cloning.md)).
- **No speed, pitch or emotion dial.** The nearest is OmniVoice's pitch and
  whisper attributes, which describe a voice being designed rather than
  control one you have.
- **Stopping is probabilistic on the model that samples it.** Hojo 40M stops
  when it _samples_ end-of-speech, so at a high temperature it occasionally
  over-runs with an invented syllable; temperature `0` cures it
  (above). MOSS and OmniVoice have no temperature to set.
- **Not every model reads a numeral.** Hojo cannot say a digit (`80` came out
  as "a bay"); no model declares `reads_numerals`, a stricter claim about unit
  symbols in a language nobody has written a locale for; Traditional Chinese
  glyphs come out as the wrong words on every model. The
  [text pipeline](text-pipeline.md) exists because of this.
- **A language tag means two things.** On every model it picks the text
  pipeline's locale; on OmniVoice it is also what the model is told to read
  the text as. On Hojo and MOSS the prompt is the text plus a
  speaker slot, so the tag never reaches the model.
- **amd64 only**, matching the published ONNX Runtime builds.
