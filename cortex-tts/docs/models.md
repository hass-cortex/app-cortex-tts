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
the model speaks faster than the audio plays. It is a figure at a length, not a
constant: a model card fits `fixed_s + per_audio × audio` to what this host
served and quotes the factor at the mean request it saw, because a line with a
per-request cost is dearer per second on a short request than on a long one. The column is one measurement,
not one per model: every model on the same host, the same text, the same settings, so
the figures are comparable with each other. **They are not a prediction about
your machine**, and the app does not pretend otherwise: a model card shows
what this host measured, or says it has none yet — one line for the model's
own voices, whose cost differs by 4% where it was measured (MOSS, one host),
and one for each cloned voice, whose
recording rejoins the prompt on every synthesis. The number on it is the
fitted per-second part with each request's fixed cost held separately, so on a
comparable host it reads at or under this column, which divides the whole cost
of each sentence. This table
is for choosing between models before you have run any of them. That host is a 4-vCPU virtual
machine on an Intel Core i7-9750H running the app at two threads on the CPU —
deliberately the kind of machine Home Assistant usually lives on, not a
desktop. Upstream figures are not used: each model's authors measured on a
different machine, and those numbers cannot be lined up.

**Memory is per resident engine**, and **Models kept in memory** decides how
many may be resident at once. The 40M beside either 2 GB model costs about
2.8 GB; the 80M and MOSS together about 4 GB. Those are host figures — a card
holds more, which is under [Hardware](#hardware-and-running-it-elsewhere).

**Only the 40M keeps ahead of playback**, and the gap to the rest is not
close. MOSS is just over real time, the 80M half again over, and the two
newest models several times over — read the column as what each costs, not as
a ranking, and read [Which model](#which-model) before reaching past the top
two.

### The reference host

|          |                                                                                                                                                                                                                |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Machine  | Ubuntu virtual machine, KVM, 4 vCPU on an Intel Core i7-9750H, 11 GB RAM, amd64 — the second VM on the host the Home Assistant VM itself runs on                                                               |
| App      | Cortex TTS 0.2.2, ONNX Runtime 1.29.0, CPU execution provider                                                                                                                                                  |
| Settings | Inference threads 2, one model resident, temperature 0.8 where the model takes one                                                                                                                             |
| Load     | Nothing else running; 1-minute load 0.0 before the run                                                                                                                                                         |
| Method   | `scripts/bench_rtf.py`, 2026-09-13: one model at a time, a warm-up synthesis discarded so model load and reference conditioning are not counted, then three runs per sentence, the server's own `X-Cortex-Rtf` |
| Text     | Four Chinese sentences of 10, 31, 44 and 92 characters (2–21 s of audio)                                                                                                                                       |
| Voices   | `hojo_zh_f_01` (40M), `Yuewen` (MOSS), `vivian` (Qwen3-TTS), `female-young` (OmniVoice), a 10-second `zh` reference for all three cloning entries                                                              |

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
`sensor.<model>_real_time_factor` is your number, and [Delivering a reply](delivery.md) says what to do with it.

### The other VM on the same CPU

The Home Assistant OS VM beside this one, on the same physical CPU, measures
the first three at 0.67, 1.06 and 1.42. MOSS agrees to 0.01; the reference
host is 18% faster on the 40M and 6% slower on the 80M. Both are quoted
because neither is wrong: the reference VM can hold all six models, and the
Home Assistant VM — 8 GB, shared with Home Assistant itself — is the
constraint that matters for what you can actually run there.

## Measured across the models

Everything below is a figure, measured on a host named here and re-measured
when it changes. What each model _is_ follows after it, one section each.

### Hardware, and running it elsewhere

The same script, the same text, four environments. Every figure is the median
over the four sentences above; the reference host is the first column.

| Model               | 4 vCPU of an i7-9750H, CPU | Ryzen 9 9955HX, CPU | RTX 5070 Ti Laptop, CUDA | GTX 1650, CUDA |
| ------------------- | -------------------------- | ------------------- | ------------------------ | -------------- |
| **Hojo 40M**        | 0.67                       | **0.28**            | 0.50                     | 0.31           |
| **MOSS Nano**       | 1.06                       | **0.38**            | 0.58                     | 0.37           |
| **Hojo 80M**        | 1.42                       | **0.70**            | not loadable             | not loadable   |
| **OmniVoice**       | 3.83                       | 1.72                | —                        | **0.80**       |
| **Qwen3-TTS**       | 6.72                       | 2.89                | —                        | **2.84**       |
| **Qwen3-TTS clone** | 6.87                       | 2.99                | —                        | **3.07**       |

The first column is two VMs on that one CPU — the Home Assistant OS VM for the
first three rows, the VM beside it for the last three, which is the difference
[The other VM on the same CPU](#the-other-vm-on-the-same-cpu) measures. The GTX 1650 column
is one run of `bench_rtf.py` against the app with the execution provider set
to `cuda`, which reproduced the two figures already in it (0.31 and 0.37) to
the hundredth. Neither new model has been run on the 5070 Ti, and their Ryzen
figures are single sentences rather than the four.

**A GPU is what makes OmniVoice usable**: 3.83 to **0.80**, from four times
real time to comfortably under it, and the largest gain any model here gets
from a card. With a cloned voice the fixed cost of re-encoding the reference
on every request puts it past the streaming threshold even so — measured at
about 1.5 s per request on the GTX 1650 — so a live reply on it is planned
rather than streamed ([Delivering a reply](delivery.md)). Qwen3-TTS gains less than half as much in relative terms — 6.72
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
  faster on the Ryzen than on the VM. It is enough to put the 40M, the 80M and
  MOSS ahead of playback there — including MOSS, which the VM cannot stream —
  but not OmniVoice or Qwen3-TTS, which stay well over real time on it.
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

### What batching costs

How a reply is cut into requests is the app's decision, and [Delivering a reply](delivery.md) explains it. These are the measurements it rests on.

The same six-sentence paragraph, sent back to back, with playback starting on
the first byte; the column is the least audio the listener still held when the
next request's audio landed.

| Model, host                       | Streams inside a request | Cost measured             | One request per sentence | Grouped to 9 s | Grouped to 14 s |
| --------------------------------- | ------------------------ | ------------------------- | ------------------------ | -------------- | --------------- |
| OmniVoice, GTX 1650, cloned voice | no                       | ~1.5 s fixed + 1.0× audio | −0.58 s                  | −0.25 s        | **−4.35 s**     |
| Hojo 40M, GTX 1650                | no                       | 0.35×                     | +2.8 s                   | +2.8 s         | +2.9 s          |
| Hojo 40M, HA VM CPU               | no                       | 0.7×                      | +0.9 s                   | +1.0 s         | +0.65 s         |
| MOSS-TTS-Nano, HA VM CPU, cloned  | yes                      | 1.14–1.21×                | −4.9 s                   | −4.9 s         | −4.8 s          |

Splitting a reply does not cost consistency at this granularity: rendered as
one, three and six requests it changed pitch wander by nothing measurable (f0
variation over 4-second windows: Hojo 9–11%, OmniVoice 7–10% in every
condition; MOSS drifted _more_ in one long generation, 17.6%, than in six
sentences, 10.5%).

Coalescing sentences removes the fixed cost of sending one at a time — 0.37 s
of dead air per boundary on MOSS — but a request that is too long costs more
than it saves. Measured on MOSS-TTS-Nano, the same 55-second story sent in
pieces of different sizes, as the fraction by which rendering fell behind
playback:

| Per request         | Behind playback |
| ------------------- | --------------- |
| ~1 s (one sentence) | +41%            |
| ~9 s                | +4.1%           |
| ~19 s               | +5.6%           |
| ~28 s               | +6.5%           |
| ~55 s (one request) | +17.8%          |

On a GTX 1650 the same 48-character reply has OmniVoice speaking at 5.32 s in
three requests against 8.92 s in one, for the same total render, while MOSS
gains nothing by splitting and stays in one at 0.37 s.

The valley is the same shape on a whole-render model, and steeper. One
124-character sentence — a list of what is in a room, 20.9 s of speech, no stop
until the end — cut at clause marks into equal pieces on OmniVoice, GTX 1650,
cloned voice. Total render for the whole reply, two runs each, medians within
0.3 s of one another:

| Requests | Speech each | Total render | Real-time factor |
| -------- | ----------- | ------------ | ---------------- |
| 1        | 20.9 s      | 27.9 s       | 1.33×            |
| 2        | ~10.4 s     | 17.0 s       | 0.82×            |
| 3        | ~7.0 s      | **16.5 s**   | **0.79×**        |
| 4        | ~5.2 s      | 18.4 s       | 0.87×            |
| 5        | ~4.2 s      | 20.7 s       | 0.98×            |
| 6        | ~3.5 s      | 21.8 s       | 1.02×            |

The floor is 7–10 s a request, and the sides are two different costs: below it
each extra boundary repays the ~1.5 s fixed cost of a cloned request, above it
the model's own attention does. A fit taken in the valley says 0.61× and is
right there and nowhere else — at 20.9 s it under-predicts by half, which is
why the cap is a constant and not something a measurement is allowed to move.

### What a first request can promise

A model this host has served nothing of has no line, and its reply's first
request is the only measurement that will exist before something has to be
released. Simulated against the OmniVoice clone at its measured figures
(1.5 s fixed, 0.61x, the 20.9 s reply in three requests), with the host turned
dearer only after that first request lands:

| Host after request 1 | First word | Lead at the tightest moment |
| -------------------- | ---------- | --------------------------- |
| As measured          | 6.8 s      | +1.72 s                     |
| 30% dearer           | 6.8 s      | +0.07 s                     |
| 60% dearer           | 6.8 s      | **−1.58 s**                 |
| Held whole instead   | 16.5 s     | never at risk               |

So the tolerance is about 30%, against a measured 0.9% across back-to-back
requests on an idle GPU and 1.8–7.8% for a fitted line's own spread. What sits
outside the table is another process taking the card part-way through a reply,
which has not been measured here. It applies to a model's first reply only —
by the fourth there is a fit, and a `spread_s` with it.

### What a benchmark cannot measure

Every figure above came from requests issued back to back on an idle host,
which is the right shape for a cost curve and the wrong one for a spread. The
app keeps both: `per_audio` is the line, `spread_s` is one standard deviation
of what the line failed to explain, and every hold is widened by it.

| Where the 24 samples came from     | `spread_s` |
| ---------------------------------- | ---------- |
| Back-to-back, one length, idle GPU | 0.017 s    |
| OmniVoice designed, real replies   | 0.156 s    |
| OmniVoice cloned, real replies     | 0.305 s    |
| MOSS cloned, real replies          | 0.200 s    |

A benchmark under-reports it by an order of magnitude, because the thing it
has to cover is the host getting busy part-way through a reply and a clean run
never does. Nothing here should be used to tune a figure that guards
playback — measure that against traffic, not against a loop.

Two further cautions about running one at all. A run of same-length requests
destroys the line it measures: with no length to fit a slope against, the fit
folds the per-request cost into the factor, and afterwards the model looks
right at that one length and wrong at every other. And the samples are kept
per model and voice, only 24 of them, so a benchmark evicts the real traffic
that was there — `DELETE /api/models/{id}/stats` afterwards and let it learn
again.

### Where the line breaks

The cost line is fitted, and a fit is only true over the lengths it was fitted
from. One request at a time on OmniVoice, GTX 1650, cloned voice, each rendered
twice and scored against a line fitted from requests under 8 s
(`2.29 + 0.450 x audio`):

| Speech in one request | Render     | The line says | Over by |
| --------------------- | ---------- | ------------- | ------- |
| 3.9 s                 | 4.04 s     | 4.04 s        | 0%      |
| 6.3 s                 | 5.14 s     | 5.14 s        | 0%      |
| **9.2 s**             | **6.48 s** | **6.42 s**    | **1%**  |
| **10.7 s**            | **8.44 s** | **7.11 s**    | **19%** |
| 12.7 s                | 9.43 s     | 8.02 s        | 18%     |
| 15.2 s                | 12.10 s    | 9.12 s        | 33%     |
| 19.1 s                | 17.41 s    | 10.90 s       | 60%     |
| 23.6 s                | 25.81 s    | 12.93 s       | 100%    |

The line holds to 9.2 s and has broken by 10.7 s. That is what sets
`BATCH_CAP_S`, and the reason is not the render time — it is that every hold
is computed from this line. At 10.7 s the line under-predicts by 1.33 s while
the planner's whole fixed allowance is 0.5 s, so a request that long is one
whose hold cannot be right.

It is a property of the model rather than of the host, but not the same figure
for every model: MOSS-TTS-Nano is 4% over its line at 15.0 s and 12% over by
21.8 s, then flat — +10%, +8% and +12% at 26.9, 32.3 and 43.7 s. A step rather
than OmniVoice's curve, and not the 28 s an earlier note claimed. That is why
it is the one model to declare its own `batch_cap_s`, 15 s against the
default's 9.
Nine is the earliest break measured, and a model that breaks later pays almost
nothing for it — MOSS costs 0.05 s a boundary against OmniVoice's 1.2 s.

### How much text one synthesis takes

A segment is one call into the model. What one call may **produce** is the
model's own and is declared; what one call should **cost** on this machine is
not the model's, and nothing here pretends otherwise.

**The ceiling that truncates** is `ModelSpec.max_audio_s`, counted in audio.
Past it the call returns what it had and the rest of the text is never spoken —
it does not slow down first, so the text path splits to stay under it.

| Model              | Ceiling    | Where it comes from                   |
| ------------------ | ---------- | ------------------------------------- |
| Hojo 40M, Hojo 80M | **41 s**   | 2048 new tokens at a 50 Hz codec      |
| MOSS-TTS-Nano      | **30 s**   | `max_new_frames` 375 at 12.5 Hz       |
| Qwen3-TTS (both)   | **164 s**  | the talker's 2048 frames at 12.5 Hz   |
| OmniVoice          | none known | nothing in the model or its code says |

MOSS's is the one that was found by being hit: seven inputs from 155 to 284
Chinese characters each came back as exactly 30.0 s with the rest missing,
which is the figure its shipped manifest gives. The others are read off the
generation limit each model declares. **OmniVoice is left without one**, and
without a character bound either — a figure that is neither the model's nor
this host's is one that will be wrong on some machine. Measured at 108
characters it returned everything it was given, which bounds the question
without answering it.

**What bounds a request here** is `ModelSpec.batch_cap_s`, and that one is
measured rather than declared: `RenderModel.fit` replaces the catalog's figure
with this host's own as soon as its samples can say anything. That is the
number that knows about this machine.

A ceiling in seconds becomes a length of text through the speech rate of the
script — the slow-side priors in `pacing.model`, **4.1 characters a second**
for CJK and **14.7** for Latin, and priors rather than a fitted line because a
segment must not depend on which voice happens to say it. So the same ceiling
is a different amount of text in each language: MOSS's 30 s is about 122
Chinese characters or 440 Latin ones.

What ends a Qwen3-TTS segment on a small card is memory rather than frames: at
378 Chinese characters on a 4 GB GTX 1650 it exhausted the device and the
request failed. That is a visible error and not a silent loss — the registry
recognises the condition, drops the engine and gives the card back — but on
that machine the ceiling above is far out of reach, and it is `batch_cap_s`
rather than the ceiling that keeps ordinary replies inside it.

Long prompts degrade before they truncate, so a ceiling is not a number to
approach. What raising one buys is a reply held to the end — **buffered** —
that is not cut into pieces it never needed; see
[Delivering a reply](delivery.md).

### Which half of a cost line belongs to the voice

Every line one host had fitted, in one table. It is what decides whether a
voice with no line of its own can borrow another's:

| Model          | Kind     | Voice      | Slope (per audio second) | Intercept (per request) |
| -------------- | -------- | ---------- | ------------------------ | ----------------------- |
| MOSS-TTS-Nano  | built-in | —          | 0.401                    | 0.00 s                  |
| MOSS-TTS-Nano  | clone    | hsiao-chen | 0.360                    | 0.05 s                  |
| OmniVoice      | designed | —          | 0.717                    | 0.31 s                  |
| OmniVoice      | clone    | ya-ping    | 0.718                    | 1.16 s                  |
| OmniVoice      | clone    | hsiao-chen | 0.608                    | 1.50 s                  |
| Qwen3-TTS 0.6B | clone    | hsiao-chen | 2.767                    | 0.87 s                  |

The slope holds across kinds on one model; the intercept is where cloning is
paid for — 4–5× on OmniVoice. So "a clone costs about twice a designed voice"
is a statement about the intercept, not about the real-time factor, and a
voice this host has not heard yet can borrow the slope safely. It cannot
predict its own intercept from the recording's length: 1.50 s came from a
3.5 s reference and 1.16 s from a 4.52 s one, the wrong way round.

## The models

What each one does, what it cannot, and the quirks that decide whether it
suits a use. The figures are the ones measured above.

### Hojo TTS Light 40M

[HojoAI/Hojo-TTS-Light](https://github.com/HojoAI/Hojo-TTS-Light), the 40M
export. The cheapest model by a wide margin and the one to start with.

- **Voices**: 15 built in, named by language, sex and number — `hojo_zh_f_01`,
  `hojo_en_f_04`. No cloning.
- **Languages**: Chinese and English. The voice picks the language: a `zh`
  voice reads Chinese, an `en` voice reads English, and no setting changes that.
- **Sampling temperature** applies (default 0.8; 0 is greedy and never
  over-runs).
- Renders whole segments; no mid-sentence chunk streaming.

### Hojo TTS Light 80M (voice cloning)

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

### MOSS-TTS-Nano

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
- **Chunk streaming**: audio leaves before the sentence is finished —
  measured over a chunked stream, the first audio arrives 143–178 ms after the
  request against 1.8 s for the whole utterance. Qwen3-TTS streams too, but
  only MOSS is cheap enough here for it to shorten the wait rather than
  merely pace it.
- **No sampling temperature**: sampling is fused into a dedicated ONNX graph.
  The setting is ignored for it, and a request that names one is refused
  (`NO_TEMPERATURE`).
- **Reads Latin words poorly** — measured at 32% character error on a reply
  containing a product name, against 0–9% for the Hojo models — so it suits
  replies that are Chinese throughout.
- **A GPU takes it under real time**: 1.06 on the reference host against
  **0.37** on a GTX 1650 — the difference between a model that cannot stream
  here and one that streams comfortably. The largest gain any model here gets
  from a card is OmniVoice's, 3.83 to 0.80. Home Assistant OS ships no NVIDIA driver, so that needs the app
  [run outside HAOS](standalone.md).

### OmniVoice 0.8B

[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice): a 0.6B Qwen3 language
model over a 0.2B Higgs Audio V2 tokenizer. The language model runs from an
int4 ONNX export; the rest of the pipeline is PyTorch, which is why this model
is the reason the image carries torch, torchaudio and transformers.

- **Voices**: nine designed ones, **and** clones from a reference recording.
  There are no bundled speakers. The two are not the same cost: measured on
  the Home Assistant VM, a designed voice rendered at RTF 3.46 and a clone of
  a ten-second recording at 7.17, because the reference's codec frames rejoin
  the prompt on every synthesis. The app measures them apart for that reason —
  the nine designed voices share one line, and every clone gets its own,
  because what a clone costs follows the length of its own recording.
- **Languages**: upstream claims more than 800, and the engine will ask for
  any of the 646 its own table names. The catalog lists the ten this project
  has actually exercised, and those are what the admin UI offers — the rest
  are reachable over the API with a `language` of `yue`, `th`, `vi` and so
  on, unmeasured rather than disabled.
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
- **It gains the most from more threads.** Measured on the same four cores:
  **3.83 at two threads, 2.66 at four** — 1.4x for the setting alone. Qwen3-TTS
  also gains, though far less (22%, below); MOSS moves the other way on the
  same hardware. "Inference threads" is a trade between the models in use
  rather than a number to raise.

### Qwen3-TTS 0.6B

[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), the 12 Hz 0.6B
checkpoints, from the `onnx-community` int4 export. Two catalog entries, two
checkpoints, one engine: **custom voice** carries nine speakers, **voice
cloning** carries the two encoders a reference needs and no speakers at all.

- **Voices**: nine built in — Vivian, Serena and Uncle_Fu (Chinese), Dylan
  (Beijing) and Eric (Sichuan), Ryan and Aiden (English), Ono_Anna (Japanese),
  Sohee (Korean) — or every uploaded reference on the cloning entry.
- **It is the one model that takes a style instruction.** A plain-language
  note beside the speaker — _speak slowly, in a warm tone_ — carried as
  `instruct` on a
  request, offered in the admin UI on this model alone, and reaching Home
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
  x-vector, _and_ the recording's codec frames join the prompt beside its
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
problem and keeping up starts. The app speaks the opening while the rest is
still being written and sizes every request after it to the lead the listener
actually holds — but only a model that renders faster than the audio plays
can gain lead that way. One that cannot is not streamed at all: the whole
reply is planned and enough audio is banked before playback starts that the
speaker never catches the renderer, so it never stutters, it waits. A model
at RTF 1.1 falls a second behind for every ten seconds it speaks, and that
second is paid up front. [Delivering a reply](delivery.md) is about that.

Putting both together:

| You want                                  | Use                                                  | Because                                                                                                                                                                                |
| ----------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Announcements, notifications, timers      | **Hojo 40M**                                         | Cheapest by a wide margin, and 15 voices without uploading anything                                                                                                                    |
| A voice assistant that answers questions  | **Hojo 40M**, or **MOSS Nano** if you want the voice | The wait is what you feel; both are usable, MOSS costs 2 GB                                                                                                                            |
| Japanese                                  | **MOSS Nano**                                        | The 40M and 80M speak Chinese and English only. MOSS is the cheapest that adds Japanese; Qwen3-TTS and OmniVoice read it too, at several times the cost                                |
| Korean, German, French, Russian, …        | **Qwen3-TTS**, on a fast machine                     | Ten languages against three. OmniVoice lists the same ten and reaches further still, but designs or clones its voice rather than shipping speakers for them                            |
| One specific person's voice               | **MOSS Nano**                                        | Clones, and still has built-in voices to fall back on                                                                                                                                  |
| A voice nobody has recorded               | **OmniVoice**                                        | Sex, age, pitch, whisper, an accent or a dialect — no upload, no training                                                                                                              |
| Long replies read aloud with a short wait | whichever your host measures **under about 0.8**     | Below that the opening is spoken while the rest still renders; above it the reply is planned and the wait grows with its length. Read `sensor.<model>_real_time_factor`, not the table |

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
- **No speed, pitch or emotion dial.** Nothing here takes a rate, a pitch or
  an emotion as a number. The nearest are Qwen3-TTS's style instruction — a
  plain-language note the model interprets as it sees fit — and OmniVoice's
  pitch and whisper attributes, which describe the voice being designed
  rather than control one you already have.
- **Stopping is probabilistic, on the models that sample it.** The two Hojo
  models and Qwen3-TTS stop when they _sample_ an end-of-speech token, so at a
  high sampling temperature they occasionally over-run the text with an
  invented syllable. On the Hojo pair temperature 0 is reproducible and never
  over-runs, at the cost of flatter delivery. **On Qwen3-TTS it is the
  opposite**: greedy decoding there reliably fails to emit end-of-speech, and
  a ten-character line ran to the model's own frame limit. MOSS and OmniVoice
  have no temperature to set at all.
- **Not every model reads a numeral.** The two Hojo models declare
  `needs_number_words`: an unexpanded digit is not read wrong by them, it is
  silent or replaced by an unrelated word (`80` came out as "a bay"), so the
  pipeline expands for them by default. MOSS, OmniVoice and Qwen3-TTS are left
  to read a bare digit as they find it — none of them declares
  `reads_numerals`, which is a separate and stricter claim about unit symbols
  in a language nobody has written a locale for. Traditional Chinese glyphs come out as the wrong words on
  every model. The [text pipeline](text-pipeline.md) exists because of this.
- **A language tag means two things.** On every model it picks the text
  pipeline's locale — which words numbers become, which rewrites run. On
  Qwen3-TTS and OmniVoice it is also what the model is told to read the text
  as; on the Hojo models and MOSS the prompt is the text plus a speaker slot,
  so pronunciation comes from the voice and the tag never reaches the model.
- **amd64 only**, matching the published ONNX Runtime builds.
