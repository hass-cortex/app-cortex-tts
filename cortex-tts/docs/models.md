# Models

Which models Cortex TTS can run, what each costs, and how to choose one.
`src/cortex_speech/catalog.py` is the source of truth for the line-up; this
page is the copy written for people, and the two are edited together.

Nothing is baked into the image. Each model is downloaded from the app's own
UI on first use and lives under `/data/models`, so removing the app with its
data removes the weights too. All of them run locally: on the CPU, or on a
GPU where one answers and the execution provider setting allows it — the
figures below are CPU figures, and MOSS in particular gains the most from a
card.

## The line-up

| Model               | Voices                            | Languages  | RTF      | Memory  | Disk   |
| ------------------- | --------------------------------- | ---------- | -------- | ------- | ------ |
| **Hojo 40M**        | 15 built in (2 zh, 13 en)         | zh, en     | **0.55** | ~780 MB | 241 MB |
| **MOSS Nano**       | 18 built in (6 zh) **and** clones | zh, en, ja | 1.07     | ~2 GB   | 729 MB |
| **Hojo 80M**        | clones only                       | zh, en     | 1.51     | ~2 GB   | 437 MB |
| **OmniVoice**       | 9 designed **and** clones         | 800+       | 3.83     | ~1.1 GB | 1.4 GB |
| **Qwen3-TTS**       | 9 built in (5 zh)                 | 10         | 6.72     | ~1.6 GB | 1.0 GB |
| **Qwen3-TTS clone** | clones only                       | 10         | 6.87     | ~2.1 GB | 1.3 GB |

RTF — real-time factor, render seconds divided by audio seconds; below 1 means
the model speaks faster than the audio plays. The column is one measurement,
not one per model: every model on the same host, the same text, the same settings, so
the figures are comparable with each other. **They are not a prediction about
your machine**, and the app does not pretend otherwise: a model card shows the
real-time factor that host has measured, or says it has none yet. This table
is for choosing between models before you have run any of them. That host is a 4-vCPU virtual
machine on an Intel Core i7-9750H running the app at two threads on the CPU —
deliberately the kind of machine Home Assistant usually lives on, not a
desktop. Upstream figures are not used: each model's authors measured on a
different machine, and those numbers cannot be lined up.

**Only the 40M keeps ahead of playback**, and the gap to the rest is not
close. MOSS is just over real time, the 80M half again over, and the two
newest models several times over — read the column as what each costs, not as
a ranking, and read [Which model](#which-model) before reaching past the top
two.

### The reference host

|              |                                                                                                                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Machine      | Ubuntu virtual machine, KVM, 4 vCPU on an Intel Core i7-9750H, 11 GB RAM, amd64 — the second VM on the host the Home Assistant VM itself runs on                                                              |
| App          | Cortex TTS 0.2.2, ONNX Runtime 1.29.0, CPU execution provider                                                                                                                                                 |
| Settings     | Inference threads 2, one model resident, temperature 0.8 where the model takes one                                                                                                                            |
| Load         | Nothing else running; 1-minute load 0.0 before the run                                                                                                                                                        |
| Method       | `scripts/bench_rtf.py`, 2026-09-13: one model at a time, a warm-up synthesis discarded so model load and reference conditioning are not counted, then three runs per sentence, the server's own `X-Cortex-Rtf` |
| Text         | Four Chinese sentences of 10, 31, 44 and 92 characters (2–21 s of audio)                                                                                                                                      |
| Voices       | `hojo_zh_f_01` (40M), `Yuewen` (MOSS), `vivian` (Qwen3-TTS), `female-young` (OmniVoice), a 10-second `zh` reference for all three cloning entries                                                              |

The median of three runs per sentence:

| Sentence   | Chars | Hojo 40M | MOSS Nano | Hojo 80M | OmniVoice | Qwen3-TTS | Qwen3-TTS clone |
| ---------- | ----- | -------- | --------- | -------- | --------- | --------- | --------------- |
| One clause | 10    | 0.50     | 1.12      | 1.58     | 3.73      | 6.89      | 7.40            |
| Two facts  | 31    | 0.54     | 1.06      | 1.47     | 3.63      | 6.73      | 6.92            |
| A forecast | 44    | 0.56     | 1.06      | 1.47     | 3.93      | 6.64      | 6.76            |
| A story    | 92    | 0.63     | 1.08      | 1.55     | 4.00      | 6.71      | 6.82            |

Only the 40M outruns playback on this host, and its lead narrows on long
text. MOSS is just over real time and the 80M well over: neither can stream
here, and both are fine for announcements, where nobody is waiting on the
first word. OmniVoice and Qwen3-TTS are a different order of cost — a
three-second announcement takes twelve and twenty seconds to render — and
belong on a faster machine or on a GPU, which is [Running it
elsewhere](standalone.md). Your host will differ; the integration's
`sensor.<model>_real_time_factor` is your number, and [Keeping
up](streaming.md) says what to do with it.

### Why the numbers moved

The first three figures were previously 0.67, 1.06 and 1.42, measured on the
Home Assistant OS VM beside this one on the same physical CPU. The run above
reproduces MOSS to 0.01 and puts the 40M 18% faster and the 80M 6% slower.
Both VMs are quoted because neither is wrong: the one above can hold all six
models, and the Home Assistant VM — 8 GB, shared with Home Assistant itself —
is where the first three were measured and is the constraint that matters for
what you can actually run there.

## Hardware, and running it elsewhere

The same script, the same text, four environments. Every figure is the median
over the four sentences above; the reference host is the first column.

| Model                | 4 vCPU of an i7-9750H, CPU | Ryzen 9 9955HX, CPU | RTX 5070 Ti Laptop, CUDA | GTX 1650, CUDA |
| -------------------- | -------------------------- | ------------------- | ------------------------ | -------------- |
| **Hojo 40M**         | 0.67                       | **0.28**            | 0.50                     | 0.31           |
| **MOSS Nano**        | 1.06                       | **0.38**            | 0.58                     | 0.37           |
| **Hojo 80M**         | 1.42                       | **0.70**            | not loadable             | not loadable   |
| **OmniVoice**        | 3.83                       | 1.72                | —                        | **0.80**       |
| **Qwen3-TTS**        | 6.72                       | 2.89                | —                        | **2.84**       |
| **Qwen3-TTS clone**  | 6.87                       | 2.99                | —                        | **3.07**       |

The first column is two VMs on that one CPU — the Home Assistant OS VM for the
first three rows, the VM beside it for the last three, which is the difference
[Why the numbers moved](#why-the-numbers-moved) measures. The GTX 1650 column
is one run of `bench_rtf.py` against the app with the execution provider set
to `cuda`, which reproduced the two figures already in it (0.31 and 0.37) to
the hundredth. Neither new model has been run on the 5070 Ti, and their Ryzen
figures are single sentences rather than the four.

**A GPU is what makes OmniVoice usable**: 3.83 to **0.80**, from four times
real time to comfortably under it, and the largest gain any model here gets
from a card. Qwen3-TTS gains less than half as much in relative terms — 6.72
to 2.84 — and is still nearly three times real time on the card, for the
reason in its section below.

Both need more VRAM than the older models: **1818 MiB for Qwen3-TTS, 1102 MiB
for OmniVoice**, measured on load. On a 4 GB card that is one model at a time
with nothing else on it; MOSS failed to allocate outright while another
process held 1.1 GB.

The second and third columns are one machine — the development laptop, a
16-core AMD Ryzen 9 9955HX with 39 GB and an RTX 5070 Ti Laptop GPU, on
Windows — measured 2026-09-13 on the CPU at two threads and then on the GPU.
The
last column is a second VM on the reference host's own i7-9750H with its
GTX 1650 passed through (4 vCPU, 11 GB, Ubuntu, driver 575), measured the
same day; on that VM's CPU alone the figures were 0.54, 1.04 and 1.38, in
line with the HA VM beside it.

What the table says:

- **Home Assistant OS cannot use a GPU.** It ships no NVIDIA driver, so on
  HAOS the models run on the CPU whatever the execution provider is set to,
  and the 40M is the model to reach for.
- **A faster CPU is the reliable win.** Every model runs two to three times
  faster on the Ryzen than on the VM, and all three keep ahead of playback
  there — including MOSS, which the VM cannot stream.
- **A GPU pays beside a weak CPU and not beside a strong one.** On the
  i7-9750H VM the GTX 1650 takes the 40M from 0.54 to 0.31 and MOSS from 1.04
  to 0.37 — from cannot-stream to comfortably streaming. On the Ryzen laptop
  the RTX 5070 Ti was slower than the same machine's CPU for both, and the
  reason is measurable: the decode loop is thousands of tiny kernels with a
  host sync per token, so it is bound by launch latency, not by compute. One
  small ONNX call costs about 30 µs on either card and 5–20 µs on the Ryzen's
  CPU; the card only pulls ahead at matrices far larger than these models use
  (4096², 199 µs against 463). Windows adds a tail on top: p99 123 µs and
  spikes past 400, against 66 on the native-Linux GTX 1650 host — which is
  why the laptop's GPU figures wander from run to run (40M 0.36–0.60, MOSS
  0.50–0.68) while the GTX 1650 host's repeat to the hundredth. Kept busy,
  the 5070 Ti is 1.8x the 1650; fed one token at a time through a Windows
  GPU scheduler, it is not.
- **The 80M does not load on CUDA at all**, on either card: its language
  model carries a bfloat16 `QuickGelu` fusion that ONNX Runtime 1.22 has no
  CUDA kernel for.

Both of the last two placements are [Running it elsewhere](standalone.md).

## Hojo TTS Light 40M

[HojoAI/Hojo-TTS-Light](https://github.com/HojoAI/Hojo-TTS-Light), the 40M
export. The cheapest model by a wide margin and the one to start with.

- **Voices**: 15 built in, named by language, sex and number — `hojo_zh_f_01`,
  `hojo_en_f_04`. No cloning.
- **Languages**: Chinese and English. The voice picks the language: a `zh`
  voice reads Chinese, an `en` voice reads English, and no setting changes that.
- **Sampling temperature** applies (default 0.8; 0 is greedy and never
  over-runs).
- Renders whole segments; no mid-sentence chunk streaming.

## Hojo TTS Light 80M (voice cloning)

The same family's 80M export, which speaks only in a voice cloned from a
[reference recording](cloning.md). It pulls in torch and librosa for its mel
front-end, which is why the image is as large as it is.

- **Voices**: every uploaded reference, under the id its name was slugified to
  (`anna-su`). No built-in voices at all.
- **Languages**: Chinese and English, decided by the recording.
- **How it clones**: a speaker encoder reads the first **six seconds** of the
  recording — padding a shorter clip with silence, discarding the rest of a
  longer one — and compresses it to a single 2048-value vector. That vector is
  the whole of what carries the voice, so a clone has the character of a
  speaker but not the timbre, and a longer or cleaner recording does not move
  it. The whole recording is also encoded into codec tokens that join the
  prompt for every sentence, so audio past six seconds costs time on every
  synthesis and buys nothing. Trim to six or seven seconds.
- **Sampling temperature** applies.
- Renders whole segments; no chunk streaming.

## MOSS-TTS-Nano

[OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano), the 100M
ONNX export, shipped as two bundles (weights and audio codec) that are
downloaded together.

- **Voices**: 18 built in, named as given names (`Yuewen`, `Junhao`; six of
  them Chinese), **and** clones from a reference recording.
- **Languages**: Chinese, English and Japanese — the only model here that
  reads Japanese.
- **How it clones**: there is no speaker encoder. The whole recording (2–20 s),
  as codec tokens, is the voice, so its length and content do shape the clone.
  Every second is paid for once per reference, then cached.
- **Chunk streaming**: the only model that emits audio before a sentence is
  finished. `/api/speak/stream` starts 143–178 ms after the request against
  1.8 s for the whole utterance.
- **No sampling temperature**: sampling is fused into a dedicated ONNX graph.
  The setting is ignored for it, and a request that names one is refused
  (`NO_TEMPERATURE`).
- **Reads Latin words poorly** — measured at 32% character error on a reply
  containing a product name, against 0–9% for the Hojo models — so it suits
  replies that are Chinese throughout.
- **Gains the most from a GPU**: RTF 1.025 on a laptop i7 against **0.354** on
  a GTX 1650. Home Assistant OS ships no NVIDIA driver, so that needs the app
  [run outside HAOS](standalone.md).

## OmniVoice 0.8B

[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice): a 0.6B Qwen3 language
model over a 0.2B Higgs Audio V2 tokenizer. The language model runs from an
int4 ONNX export; the rest of the pipeline is PyTorch, which is why this model
is the reason the image carries torch, torchaudio and transformers.

- **Voices**: nine designed ones, **and** clones from a reference recording.
  There are no bundled speakers. The two are not the same cost: measured on
  the Home Assistant VM, a designed voice rendered at RTF 3.46 and a clone of
  a ten-second recording at 7.17, because the reference's codec frames rejoin
  the prompt on every synthesis. The app measures them apart for that reason.
- **Languages**: upstream claims more than 800, and the engine will ask for
  any of the 646 its own table names. The catalog lists the ten this project
  has actually exercised, and those are what the admin UI offers — the rest
  are reachable over `/api/speak` with a `language` of `yue`, `th`, `vi` and
  so on, unmeasured rather than disabled.
- **How a designed voice works**: the model takes a short instruction built
  from a **closed vocabulary** — a sex, an age band, a pitch band, whisper,
  and either an English accent or a Chinese dialect. Anything outside it is
  refused rather than approximated, so the nine voices here are attribute sets
  chosen from that list, not free text. The model translates the attributes
  itself when the text reads as Chinese, so one definition covers both scripts
  and a designed voice claims no language.
- **How it clones**: the whole recording becomes codec tokens that join the
  prompt, as MOSS does, with no speaker encoder to cap the fidelity. Trim to
  the 3–10 s the model asks for: past 20 s it warns, and a 70 s reference was
  measured taking 11.4 GB of resident memory before it was cut back.
- **No sampling temperature.** This decoder has two, and neither is the
  quantity the app's setting names: one picks which positions to unmask next
  (5.0, and where the variation between renders comes from), the other picks
  the token at a chosen position and was tuned at 0, greedy. Mapping the app's
  0.8 onto the second would be a guess, so the model declares no temperature
  and a request that names one is refused (`NO_TEMPERATURE`), as MOSS's is.
- **The 2.45 GB of transformer weights are never downloaded.** The ONNX graph
  replaces `forward` outright, so the checkpoint's own copy would be fetched,
  held and never read. The module is built on the meta device instead —
  measured peak resident 1.1 GB against 4.7 GB for the pipeline loaded whole,
  which is the difference between fitting on a Home Assistant VM and not.
- **`cuda` covers the graph and not the rest of it.** This is the one model
  where the execution provider is partial: the audio tokenizer, the prompt
  handling and the decoder stay in torch on the CPU whatever it is set to, so
  `/health` reporting `cuda` for OmniVoice means its ONNX session got CUDA and
  says nothing about the other half. That is still worth having — 3.83 to
  0.80, the biggest gain from a card in this table — because the graph is
  where the time goes.
- **No chunk streaming**: the decoder unmasks the whole utterance over a fixed
  number of steps, so no audio exists until all of it does.
- **It is the one model that wants more threads.** Measured on the same four
  cores: **3.83 at two threads, 2.66 at four** — 1.4x for the setting alone.
  MOSS moves the other way on the same hardware, so "Inference threads" is a
  trade between the two rather than a number to raise.

## Qwen3-TTS 0.6B

[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), the 12 Hz 0.6B
checkpoints, from the `onnx-community` int4 export. Two catalog entries, two
checkpoints, one engine: **custom voice** carries nine speakers, **voice
cloning** carries the two encoders a reference needs and no speakers at all.

- **Voices**: nine built in — Vivian, Serena and Uncle_Fu (Chinese), Dylan
  (Beijing) and Eric (Sichuan), Ryan and Aiden (English), Ono_Anna (Japanese),
  Sohee (Korean) — or every uploaded reference on the cloning entry.
- **It is the one model that takes a style instruction.** A plain-language
  note beside the speaker — *speak slowly, in a warm tone* — carried as
  `instruct` on
  `/api/speak`, offered in the admin UI on this model alone, and reaching Home
  Assistant as a `tts.speak` option on its entity. The cloning checkpoint
  refuses one: upstream gates it as a CustomVoice feature.
- **Voice design is a different checkpoint, and Qwen only built it at 1.7B.**
  There is no 0.6B VoiceDesign. At 1.7B on this host it would be several times
  slower again than the 6.72 the 0.6B measures, so it belongs on a GPU or not
  at all.
- **Languages**: ten, and no more — unlike OmniVoice, the catalog's list is
  the whole of it. A speaker is told to read in its own by default, which is
  upstream's recommendation, and a cloned voice in the one its reference
  declares; a request naming a `language` overrides either.
- **How it clones**: an ECAPA speaker encoder turns the recording into one
  x-vector, *and* the recording's codec frames join the prompt beside its
  transcript as a worked example. Both, unlike the 80M, which has only the
  first, and unlike MOSS, which has only the second.
- **Sampling temperature** applies. Do not set it to 0: greedy decoding here
  does not reliably emit end-of-speech, and a ten-character line was measured
  running to the model's own 2048-frame limit — 2.7 minutes of invented audio.
  The engine caps each segment at roughly two and a half times the duration
  the text needs, so the failure is a clipped reply rather than a hung request.
- **Chunk streaming**: the codec decoder is exported at a fixed 25 frames,
  which at 12.5 Hz is exactly two seconds, and a block decoded alone is the
  same audio as the same block inside a longer decode. So audio can leave
  every two seconds — but at an RTF well above 1 the listener still waits for
  the first block and then falls behind. It is a real capability on a fast
  host and cosmetic on a slow one.
- **The cost is the code predictor, and it does not go away.** One frame is
  one language-model step plus **fifteen** predictor calls, because each of the
  sixteen residual codes is conditioned on the ones before it. At 12.5 frames
  a second that is about 190 model calls per second of speech, they cannot be
  batched, and they were 81% of the time on a CPU and 89% on a GTX 1650. An
  export with a cached predictor would change this; the one published does not
  have one.
- **Threads buy less than they cost.** Measured on the four-core reference
  host: two threads gave 6.72 while the process used 2.1 of the four cores,
  four gave **5.22** with all four flat out. Twice the machine for 22% — because what the model waits on is
  the next of those 190 calls, not the arithmetic inside one. It is the same
  reason a GPU only took it from 6.72 to 2.84.

## Voice ids

Every model names its voices differently, so an id from one is meaningless on
another, and there is nothing to guess from:

| Model     | Looks like                     | Where the name comes from                   |
| --------- | ------------------------------ | ------------------------------------------- |
| Hojo 40M  | `hojo_zh_f_01`, `hojo_en_f_04` | Fixed in the bundle: language, sex, number  |
| Hojo 80M  | `anna-su`, `ke-min-hsun`       | Whatever you named the recording, slugified |
| MOSS Nano | `Yuewen`, `Junhao`             | Given names, as the bundle ships them       |

A cloned voice is the same id on **every** model that clones, because it is the
recording that defines it rather than the model. The app's UI lists every
voice; in Home Assistant the integration's `cortex_tts.list_voices` action
returns them with their ids.

## Which model

Two questions decide it, and neither is "which is best".

**Is anyone waiting?** Nothing plays until enough has been rendered, so the
silence before the first word is roughly RTF times the seconds of speech that
have to exist first. At a real 1.0 RTF, a ten-second answer means ten seconds
of nothing.

| Situation                      | Typical reply | At RTF 0.5 | At RTF 1.0 | At RTF 1.5 |
| ------------------------------ | ------------- | ---------- | ---------- | ---------- |
| Doorbell or alarm announcement | ~3 s          | 1.5 s      | 3 s        | 5 s        |
| Assist answering a question    | ~9 s          | 4.5 s      | 9 s        | 14 s       |
| Assist reading a summary       | ~27 s         | 13 s       | 27 s       | 41 s       |
| Reading a story aloud          | ~80 s         | 40 s       | 80 s       | 120 s      |

For an announcement nobody is standing there waiting, so even a slow model is
fine. For a conversation the wait is the product, and it is the one number to
optimise.

**Is the reply long?** Past about half a minute the wait stops being the
problem and keeping up starts. The integration can stream — speak the opening
while the rest is still being made — which removes almost all of the wait, but
only if the model renders faster than the audio plays. A model at RTF 1.1
falls a second behind for every ten seconds it speaks, and a long reply
stutters to a halt near the end. [Keeping up](streaming.md) is about that.

Putting both together:

| You want                                 | Use                                                  | Because                                                                       |
| ---------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| Announcements, notifications, timers     | **Hojo 40M**                                         | Cheapest by a wide margin, and 15 voices without uploading anything           |
| A voice assistant that answers questions | **Hojo 40M**, or **MOSS Nano** if you want the voice | The wait is what you feel; both are usable, MOSS costs 2 GB                   |
| Japanese                                 | **MOSS Nano**                                        | The 40M and 80M speak Chinese and English only, MOSS adds Japanese            |
| Korean, German, French, Russian, …       | **Qwen3-TTS**, on a fast machine                     | Ten languages against three, and it is the only one that reads most of them   |
| One specific person's voice              | **MOSS Nano**                                        | Clones, and still has built-in voices to fall back on                         |
| A voice nobody has recorded              | **OmniVoice**                                        | Sex, age, pitch, whisper, an accent or a dialect — no upload, no training     |
| Long replies read aloud without stalling | whichever your host measures **under 0.5**           | Check `sensor.<model>_real_time_factor`, not the table                        |

The last two rows come with a bill. OmniVoice and Qwen3-TTS measured 3.83 and
6.72 on the reference host, which is 7x and 12x the 40M — a nine-second answer
takes 35 and 60 seconds to render there. Both are reasonable on a fast desktop
CPU or a GPU and neither belongs in a conversation on a Home Assistant box.
[Running it elsewhere](standalone.md) is how you get them.

## What no model here does

- **A cloned voice is a likeness, not a match.** On the 80M the reason is the
  speaker vector above; MOSS conditions on the whole recording, but at 100M
  parameters it is still a likeness.
- **A wrong transcript fails silently.** Give a reference the wrong text and
  you get confident audio that is simply less like the person. Nothing warns
  you; see [Cloned voices](cloning.md).
- **No speed, pitch or emotion control.** None of the models expose any.
- **Stopping is probabilistic.** A model stops when it _samples_ an
  end-of-speech token, so at a high sampling temperature it occasionally
  over-runs the text with an invented syllable. Temperature 0 is reproducible
  and never over-runs, at the cost of flatter delivery; MOSS has no
  temperature to set.
- **No numerals, no symbols.** An unexpanded digit is not read wrong, it is
  silent or replaced by an unrelated word (`80` came out as "a bay"), and
  Traditional Chinese glyphs come out as the wrong words. The
  [text pipeline](text-pipeline.md) exists because of this.
- **No language parameter.** The prompt is the text plus a speaker slot, so
  pronunciation comes entirely from the voice.
- **amd64 only**, matching the published ONNX Runtime builds.
