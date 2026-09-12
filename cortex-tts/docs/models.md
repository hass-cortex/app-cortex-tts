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

| Model         | Voices                            | Languages  | RTF on the HA VM | Memory  | Disk   |
| ------------- | --------------------------------- | ---------- | ---------------- | ------- | ------ |
| **Hojo 40M**  | 15 built in (2 zh, 13 en)         | zh, en     | **0.67**         | ~780 MB | 241 MB |
| **Hojo 80M**  | clones only                       | zh, en     | 1.42             | ~2 GB   | 437 MB |
| **MOSS Nano** | 18 built in (6 zh) **and** clones | zh, en, ja | 1.06             | ~2 GB   | 729 MB |

RTF — real-time factor, render seconds divided by audio seconds; below 1 means
the model speaks faster than the audio plays. The column is one measurement,
not three: every model on the same host, the same text, the same settings, so
the figures are comparable with each other. That host is the project's
reference, a 4-core Home Assistant OS VM (KVM) running the app at two threads
on the CPU — deliberately the kind of machine Home Assistant usually lives on,
not a desktop. Upstream figures are not used: each model's authors measured on
a different machine, and those numbers cannot be lined up.

### The reference host

|                |                                                                                                                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Machine        | Home Assistant OS 18.2 virtual machine, KVM, 4 vCPU on an Intel Core i7-9750H, 8 GB RAM, amd64                                                                                                                 |
| Home Assistant | Core 2026.9.2, Supervisor 2026.09.0                                                                                                                                                                            |
| App            | Cortex TTS 0.1.0, ONNX Runtime 1.29.0, CPU execution provider                                                                                                                                                  |
| Settings       | Inference threads 2, one model resident, temperature 0.8 (Hojo)                                                                                                                                                |
| Load           | The household's ordinary apps running beside it; 1-minute load 0.0 before the run                                                                                                                              |
| Method         | `scripts/bench_rtf.py`, 2026-09-13: one model at a time, a warm-up synthesis discarded so model load and reference conditioning are not counted, then three runs per sentence, the server's own `X-Cortex-Rtf` |
| Text           | Four Chinese sentences of 10, 31, 44 and 92 characters (2–21 s of audio)                                                                                                                                       |
| Voices         | `hojo_zh_f_01` (40M), `Yuewen` (MOSS), a 7-second `zh` reference (80M)                                                                                                                                         |

The median of three runs per sentence:

| Sentence   | Chars | Hojo 40M | MOSS Nano | Hojo 80M |
| ---------- | ----- | -------- | --------- | -------- |
| One clause | 10    | 0.62     | 1.10      | 1.43     |
| Two facts  | 31    | 0.66     | 1.04      | 1.40     |
| A forecast | 44    | 0.69     | 1.04      | 1.41     |
| A story    | 92    | 0.78     | 1.07      | 1.49     |

Only the 40M outruns playback on this host, and its lead narrows on long
text. MOSS is just over real time and the 80M well over: neither can stream
here, and both are fine for announcements, where nobody is waiting on the
first word. Your host will differ — the integration's
`sensor.<model>_real_time_factor` is your number, and [Keeping up](streaming.md)
says what to do with it.

## Hardware, and running it elsewhere

The same script, the same text, four environments. Every figure is the median
over the four sentences above; the reference host is the first column.

| Model         | HA VM, 4 vCPU of an i7-9750H, CPU | Ryzen 9 9955HX, CPU | RTX 5070 Ti Laptop, CUDA | GTX 1650, CUDA |
| ------------- | --------------------------------- | ------------------- | ------------------------ | -------------- |
| **Hojo 40M**  | 0.67                              | **0.28**            | 0.50                     | 0.31           |
| **MOSS Nano** | 1.06                              | **0.38**            | 0.58                     | 0.37           |
| **Hojo 80M**  | 1.42                              | **0.70**            | not loadable             | not loadable   |

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

| You want                                 | Use                                                  | Because                                                             |
| ---------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| Announcements, notifications, timers     | **Hojo 40M**                                         | Cheapest by a wide margin, and 15 voices without uploading anything |
| A voice assistant that answers questions | **Hojo 40M**, or **MOSS Nano** if you want the voice | The wait is what you feel; both are usable, MOSS costs 2 GB         |
| Japanese                                 | **MOSS Nano**                                        | The 40M and 80M speak Chinese and English only, MOSS adds Japanese  |
| One specific person's voice              | **MOSS Nano**                                        | Clones, and still has built-in voices to fall back on               |
| Long replies read aloud without stalling | whichever your host measures **under 0.5**           | Check `sensor.<model>_real_time_factor`, not the table              |

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
