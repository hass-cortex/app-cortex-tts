# app-cortex-tts / cortex-tts

On-device text-to-speech for Home Assistant, served by a FastAPI app with an
ingress admin UI. Which models it offers is `catalog.py`, and the README's
**Models** table is the copy written for people; both are edited in one place
when the line-up changes, which is the point.

**What a model costs is `docs/models.md`, and only there.** Every measured
figure — real-time factors, what a card buys, what cloning adds — lives in its
tables, and every other page points at them rather than quoting one. A number
a reader can find in two places is a number that will disagree with itself:
the same MOSS-on-a-GPU measurement was carried by four pages in three
versions, and the page that had drifted furthest used its stale pair to draw a
conclusion its own table contradicted. The test is whether a re-measurement
can be applied by editing one file.

The models ship no text front-end, so the app carries one: numeral/unit/date
normalisation, and for Chinese also Traditional-to-Simplified glyph conversion
and Taiwan readings by homophone (垃圾 → 乐色, from a generated table — no
model reads a pinyin hint reliably). That pipeline is the product, not a
detail (32% character error rate against 4%; see
[`docs/text-pipeline.md`](cortex-tts/docs/text-pipeline.md)). It is built to
take languages the way the catalog takes engines: a language is a locale
added beside the others, and a rewrite only one language needs lives with that
locale — never a branch in the pipeline or a global switch in the API, the UI
or the integration.

The outer directory is the HA app shell + repo metadata; the inner `cortex-tts/`
subdir is the Python backend, the static admin UI, the Dockerfile and the
rootfs.

## Quick Links

- **Domain vocabulary**: [`cortex-tts/CONTEXT.md`](cortex-tts/CONTEXT.md) — what is
  _Prepared text_? _Segment_ vs _Sentence_? Why is "normalise" two unrelated
  operations in one request? Which of the four meanings of "streaming"?
- **Contributor guide**: [`cortex-tts/CONTRIBUTING.md`](cortex-tts/CONTRIBUTING.md)
  — dev setup, gates, PR flow.
- **User documentation**: [`cortex-tts/DOCS.md`](cortex-tts/DOCS.md) — the HA App
  Store page: install, configure, troubleshoot. It links out (absolute URLs,
  because the App Store renders it alone) to the reference pages under
  [`cortex-tts/docs/`](cortex-tts/docs/): [models](cortex-tts/docs/models.md),
  [the text pipeline](cortex-tts/docs/text-pipeline.md),
  [cloned voices](cortex-tts/docs/cloning.md),
  [delivering a reply](cortex-tts/docs/delivery.md),
  [running it elsewhere](cortex-tts/docs/standalone.md) and the
  [HTTP API](cortex-tts/docs/api.md). The integration's README points at the
  same pages rather than restating them: model facts live here, once.
- **Release runbook**: `docs/release/` at the root of the hass-cortex workspace —
  not in this repo. The catalog it publishes to is
  [`hass-cortex/repository`](https://github.com/hass-cortex/repository).
- **Primary consumer**: the [`cortex-tts`](https://github.com/hass-cortex/cortex-tts)
  HACS integration (HA TTS platform). It is required — without it Home
  Assistant has no Cortex TTS platform and the discovery record goes nowhere.

## Repository Layout

```
.
├── .github/workflows/
│   ├── app-ci.yaml            hassio-addons app-ci (HA app shell lint)
│   ├── ci.yml                 calls python-checks.yml on PR + push
│   ├── python-checks.yml      reusable: ruff + pyright + pytest
│   ├── deploy.yaml            release-triggered: app-deploy → GHCR + dispatch
│   └── release.yml            tag-triggered: python-checks gate + GitHub Release
├── .yamllint, .mdlrc, .prettierignore    lint configs (consumed by app-ci.yaml)
├── .pre-commit-config.yaml    ruff, yamllint, hadolint, shellcheck, prettier;
│                              pyright + pytest at pre-push
├── README.md                  repo front page
├── images/                    README screenshots
├── LICENSE.md                 MIT
└── cortex-tts/                  ── APP / SOURCE SUBDIR ──
    ├── config.yaml            HA app metadata (slug=cortex_tts, ingress; port 8771 unpublished)
    ├── build.yaml             base image: hassio-addons/debian-base (amd64 only)
    ├── Dockerfile             uv-installed venv baked in; source copied on top
    ├── DOCS.md                HA App Store documentation page
    ├── docs/                  reference pages DOCS.md and the integration link to
    ├── scripts/bench_rtf.py   where docs/models.md's figures come from (one host, one text set)
    ├── README.md              Supervisor reads this as `long_description`
    ├── CONTEXT.md             domain vocabulary
    ├── CONTRIBUTING.md
    ├── icon.png, logo.png     app icons (HA App Store + sidebar)
    ├── translations/en.yaml   app configuration-UI translations
    ├── rootfs/                s6-overlay services (init oneshot + cortex-tts main)
    ├── pyproject.toml/uv.lock ruff + pyright + pytest config live here too
    ├── src/cortex_speech/     the speech library (see Architecture)
    ├── src/cortex_tts/        the Home Assistant app
    ├── tests/                 pytest; no model weights needed
    └── web/                   static admin UI (no build step)
```

## Architecture

Two packages, one dependency direction. `cortex_speech` is the library and
knows nothing about Home Assistant, the Supervisor or HTTP; `cortex_tts` is the
app that serves it and depends on the library through its facade only.
`tests/test_architecture.py` fails the build if either rule is broken.

```
src/cortex_tts/       ── THE HOME ASSISTANT APP ──
├── __main__.py       uvicorn entrypoint
├── config.py         what must be settled before the process starts, from the environment
├── preferences.py    what the user changes while it runs, stored beside the models
├── stats.py          what this host measured, per model — one RenderSample per
│                     request, fitted for both the card and the planner; a card shows its own
│                     real-time factor or none, never another machine's
├── app.py            assembly: lifespan, state, routes, ingress UI mount
├── supervisor.py     best-effort POSTs to the Supervisor, and having none
├── discovery.py      announce so HA finds the app by itself
├── events.py         turn a library notification into an event on the HA bus
└── api/
    ├── deps.py       AppState (holds one SpeechService) + the API-key dependency
    ├── schemas.py    request/response bodies
    ├── routes.py     every HTTP endpoint (see API Endpoints)
    └── live.py       the WebSocket: a reply spoken while it is written, planned by
                      the library's planner; owns releasing audio and noticing
                      that the listener has gone

src/cortex_speech/    ── THE SPEECH LIBRARY ──
├── __init__.py       SpeechService, SpeechConfig — the whole public surface
├── text/             ── THE TEXT PATH ──
│   ├── pipeline.py   plan()/prepare(): language → locale → normalise → rewrites → segment
│   ├── locales.py    Locale and Rewrite: what a language is to the pipeline; the registry
│   ├── units.py      Home Assistant's unit symbols as CLDR unit ids, named through babel
│   ├── generic.py    the locale any unwritten language gets: num2words + CLDR units and dates
│   ├── passes.py     the ordered pass table the written normalisers are built from
│   ├── options.py    NormalizeOptions, shared so no normaliser imports another
│   ├── zh/           Chinese: normalize.py, numbers.py, script.py (t2s), readings.py + taiwan_readings.tsv
│   └── en/           English: normalize.py
├── pacing/           ── WHEN TO RENDER WHAT ──
│   ├── model.py      RenderModel: what a request costs here and how fast a voice
│   │                 speaks, both fitted from the requests served
│   ├── sentences.py  sentences out of text arriving in pieces
│   └── planner.py    streaming / planned / buffered, the first request's floor
│                     and ceiling, batches sized to the lead, the opening hold
├── engine/
│   ├── base.py       the Engine and StreamingEngine protocols, Voice, errors,
│                     and the `stop` check every render takes
│   ├── backends.py   backend key → builder; how a new engine is added
│   ├── overrun.py    judging and trimming a waveform, and the seed retry
│   ├── registry.py   what is resident, LRU eviction, per-engine serialisation
│   ├── conditioning.py  what an engine derives from a reference, and when it
│   │                 goes stale — one cache, so one invalidation rule
│   ├── preset.py     40M: fixed voices
│   ├── clone.py      80M: every voice is a reference recording
│   ├── moss.py       MOSS-TTS-Nano: both, with cached prompt conditioning
│   ├── qwen3.py      Qwen3-TTS: nine speakers on one checkpoint, cloning on
│   │                 the other; two catalog entries, one engine
│   ├── qwen_tokenizer.py  that checkpoint ships no tokenizer.json — assemble
│   │                 one from vocab.json + merges.txt rather than pull in
│   │                 transformers to do it
│   ├── omni.py       OmniVoice: designed voices from a closed attribute
│   │                 vocabulary, and cloning
│   └── join.py       stitching per-segment waveforms into one utterance
├── catalog.py        the models, their capabilities, and where they come from
├── download.py       Hugging Face fetch in a worker thread, pollable progress
├── references.py     reference recordings: audio + transcript, validated on entry
├── audio.py          waveform ↔ bytes: output encoding, reference decode +
│                     levelling, and the header a chunked stream opens with
├── providers.py      which ONNX Runtime provider was asked for, and got, and
│                     what to hand CUDA beside the name
├── device.py         the card: how much memory it has left, and how a failure
│                     that means it has none is recognised
├── store.py          replacing a small JSON file without a reader seeing half
├── notifications.py  publish "the voice set changed"; the app decides what that means
└── vendor/           upstream inference code, kept diffable — exempt from ruff
                      and pyright file by file, never as `vendor/**`, because
                      two files in there are ours; see vendor/NOTICE.md
```

### Cross-module guarantees

- **Weights are never in the image.** The catalog names what is available;
  bundles arrive on first download into `/data/models`, so HA's "remove with
  data" sweeps them up. A failed download leaves partials in place —
  `catalog.inspect` reports not-downloaded and a retry resumes.
- **The text pipeline is keyed by language, not sniffed from script.** A
  request's `language` tag picks the locale — its normaliser, and any
  rewrite that language needs (script conversion and Taiwan readings exist
  only under `zh`) — and sniffing the text is the fallback for a request that
  carries no tag. A language the pipeline has no locale for still gets numbers
  read (`num2words`) and a sentence-final stop, never another language's
  words.
- **A pass that can misjudge is opt-in.** Script conversion and Taiwan
  readings cannot read a word wrong, and a number with a unit, a clock colon
  or a date around it says what it is; those run by default. A bare number
  does not say what it is — `撥打 110`, `302號房`, `RTX 4090`, `John 3:16`
  were all read as quantities — so `expand_numbers` is off unless the caller
  says otherwise, in every locale. A rule that guesses is not a fix for what
  a model cannot read: wrong misleads, unread merely goes unheard, and the
  place to say what a number is remains the caller that knows.
- **Pass order is a contract.** Normalisation emits Traditional number words,
  so it must precede script conversion, and Taiwan readings are keyed by the
  Simplified form so they run last. Within a normaliser, a construct claims
  its number before the bare-number pass reads that digit as a quantity.
  `text/passes.py` owns the order; each locale only supplies readings.
- **Every segment ends in sentence-final punctuation.** Without it the model
  misses its cue to stop and invents a syllable.
- **A model's lifecycle is legible from the log alone.** The registry logs
  every transition it owns — loading, resident, unloaded and why — with the
  card's memory and the change across it, because an arena that grew, an engine
  that did not give its memory back and another process taking the card leave
  the same absolute figure and are told apart only by the movement. `loading`
  is a state `/health` reports (`loading_models`), not a four-second gap in
  which the registry says nothing is resident. Reading the figure costs a
  subprocess, so it is read once per transition and never on a host configured
  for the CPU.
- **A device that runs out of memory costs the engine, not the process.** An
  ONNX Runtime arena only grows, and only the session losing its last reference
  returns it, so one exhausted render would otherwise leave the card full and
  every request after it failing the same way. `providers.exhausted` recognises
  the condition and the registry drops the engine holding the arena; the failed
  request is not retried, because a stream has usually sent audio by then.
  Dropping it is not enough on its own: an exception and its traceback
  reference each other, so one that reaches a logger and stops there is freed
  by a collection and never by refcounting — and every frame in it is still
  holding the engine and its sessions. `traceback.clear_frames` empties those
  locals and keeps the frames, so the trace still prints and the card comes
  back at once. Measured with the collector disabled: 3716 MiB still held
  without it, 494 with. It cannot reach the frame that raised — that one is
  still executing — but that frame dies with its coroutine, and the frames it
  clears are the deep ones holding the sessions and the tensors.
- **One synthesis per engine at a time.** A session carries state across a
  render — a stateful per-token loop on most engines, a fixed unmasking
  schedule on OmniVoice — so the registry serialises calls; concurrency would
  corrupt state, not just slow things down.
- **A render whose listener left stops within one unit of work.** Every
  engine takes a `stop` check and asks it between the units it produces — a
  decode step (Hojo, through a marked deviation in the vendored loop), a
  diffusion step (OmniVoice, through its generation config), a frame
  (Qwen3-TTS), a codec chunk (MOSS); the registry asks it between streamed
  chunks itself. `AbandonedError` comes back, the lock is released, and
  nothing is recorded against the model. The transports supply the check:
  `/api/speak` polls the request for a disconnect, and `/api/speak/live` a
  closed socket, a `cancel` frame or a client that has not read for fifteen
  seconds. What none
  of them can see is a player that stopped behind Home Assistant's TTS cache,
  which drains a stream without back-pressure; that is the boundary.
- **The app paces a live reply; the integration forwards words.** Over
  `/api/speak/live` the planner in `cortex_speech/pacing` decides, per reply,
  between streaming (batches sized to the listener's lead), planned (the whole
  reply known, so the first batch carries no figure: the transport banks the
  opening audio and releases it once it covers what the rest is predicted to
  lose, asked again as each chunk arrives) and buffered (an unmeasured model, or a
  caller asking), from a `RenderModel` fitted to the requests this host has
  served — fixed cost plus a per-second factor, and a speech rate per script.
  Measured on the two production hosts before this existed: a model that hands
  requests over whole at real time (OmniVoice with a clone) stalled −4.35 s
  when batches were allowed to grow while the lead did not; a chunk-streaming
  one at 1.15× (MOSS on a CPU) lost a second per sentence at any batching;
  splitting a reply into six requests changed pitch wander by nothing
  measurable. The numbers that came out of that — a three-second floor on the
  first request and a six-second ceiling — are in `planner.py` with the
  measurements beside them. How a planned reply is
  cut is not among them: `Schedule.best_batches` groups the sentences at every
  limit that moves a boundary and takes the plan whose first word comes
  soonest, a tie going to the fewest boundaries. One length is not a
  measurement and is not searched: `ModelSpec.batch_cap_s` is where that
  model's fitted line stops describing it. Measured on OmniVoice against a
  line fitted from requests under 8 s, a single request is on that line at
  9.2 s and 19% over it by 10.7 s, 60% by 19 s and 100% by 23.6 s — so its cap
  is nine, the longest length seen still on it, and `BATCH_CAP_S` is that
  figure as the default for a model nobody has measured. What it buys is not
  mainly a cheaper render (the same 20.9 s reply cost 27.9 s whole, 17.0 s in
  halves, 16.5 s in thirds) but a hold computed from arithmetic that still
  holds: at 10.7 s the line under-predicts the render by 1.33 s and `MARGIN_S`
  is 0.5. **The cap belongs to the model and to the host, so it is
  measured.** An autoregressive decode's quadratic term is attention over a
  growing cache, and where it surfaces depends on whether the machine is short
  of compute or of bandwidth — a figure carried from another host is a guess.
  `RenderModel.fit` finds it: `holds_to_s` is the longest sampled request
  whose real-time factor is still within `LINE_TOLERANCE` of the shortest
  requests', and `_cap_from` prefers it to anything declared. Ten per cent
  reproduces every break measured by hand on the GTX 1650 box — Hojo 40M 8%
  over at 9.4 s, OmniVoice 1% over at 9.2 s and 19% by 10.7 s, MOSS 4% over at
  15.0 s and 12% by 21.8 s, so nine, nine and fifteen — and the store's own
  lines then landed on 9.18, 9.36 and 15.28. **A cap bounds the evidence that
  would move it**, which is the difficulty: every sample was itself cut to the
  cap, so a store left alone can confirm a break or find an earlier one and
  never a later one. `PROBE` is the way out — where the line held to the top of
  what was tried, the next reply may reach 1.25x past it, and the first request
  that misses shortens `holds_to_s` again. What it does not survive is the
  rolling window: the samples that proved a break age out after 24 requests and
  the cap climbs until it finds it again, so it oscillates around the break
  rather than settling on it. `ModelSpec.batch_cap_s` is what stands until this
  host can say, and it is declared per model. MOSS
  declares fifteen, measured the same way the default was: one request at a
  time against the cost of the short ones, it is 4% over its line at 15.0 s
  and 12% over by 21.8 s, then flat (+10%, +8%, +12% at 26.9, 32.3 and
  43.7 s) — a step rather than OmniVoice's curve. The cost is not the render
  time the default's own reasoning counted — 0.05 s a boundary there against
  OmniVoice's 1.2 — but the cuts: a sentence over the cap is split at its
  clause marks and every segment is terminated, so those land in the audio as
  full stops. Measured on one 63 s reply, held to nine MOSS took ten requests
  instead of three, 11% more render and two cuts inside sentences, for 0.09 s
  of opening it does not gain — a chunk-streaming engine is audible from its
  fixed cost whatever the cut, and that 0.09 s was a safety allowance
  shrinking, not a listener hearing anything. `Schedule.speaks_at` therefore
  compares plans on the arithmetic alone and leaves the margin and the spread
  to `Schedule.hold_for`, which is what actually pays them; including a
  render-scaled spread in the comparison made the search prefer thirteen
  requests over one. A sentence longer than the cap is cut at
  its own clause marks, since grouping can join sentences but never cut below
  one — which is how a reply written as a single long enumeration reached the
  renderer whole whatever limit was tried.
- **Three deliveries, and a caller may name one.** `auto` is what anything
  serving a listener sends, and the planner chooses; `buffered`, `planned` and
  `streaming` insist. Insisting exists because the three cannot be compared
  any other way — the admin UI's live panel puts all four on one reply, which
  is how the panel answers "would another delivery have been better here". It
  is honoured as far as the reply allows: forced streaming needs a cost line
  before the first byte, so a host that has measured nothing still paces, and
  a reply shorter than an opening is planned because there is no first batch
  to send. The `done` frame rather than the request is what says how it went.

  **An insisted-on delivery does not depend on when `end` arrived.** Whether
  the words are all in at the first decision is a scheduling outcome, not a
  property of the reply: the transport creates its reader task and then plans
  before that task has run. So `Planner._first` hands a finished reply to
  `_plan_whole_reply` only when streaming was not insisted on, and a caller
  that sends the whole reply in one go gets the delivery it named. `auto` is
  outside this by design — it decides on what is known when it is asked, and a
  reply already written out gives it a better option than one still arriving,
  so that caller may see either.

- **Buffered is asked for, never concluded.** It is one render of the whole
  reply, released when it is done, and it is what the integration or a caller
  gets for setting `mode: buffered` — nothing else reaches it. The cap does
  not apply and neither does the clause cut: both exist to buy an earlier
  first word, and a held reply has none to buy, so the only thing a cut could
  buy is a shorter total render (20.9 s of speech cost 27.9 s whole against
  16.5 s in thirds) at the price of an unmeasured change in how the reply
  sounds — every segment is terminated, so a clause mark cut at is spoken as a
  full stop. Buffered is the one mode that makes no such trade on anyone's
  behalf.
- **A model this host has served nothing of is planned, not held.** Nothing
  measured is not nothing to go on: the reply's own first request is a
  measurement of this host, in this voice, a moment ago. So the reply is cut
  to the cap from `spoken_seconds`, nothing is released until that first
  request lands (`bank_needed` is infinite until then), and
  `Planner._learn_from` reads it as the whole cost of a request that size —
  the only reading of one point that invents no second number, and the cut
  makes the requests that size. Simulated against the OmniVoice clone as
  measured, the first word lands at 6.8 s against 16.5 s held, with 1.72 s of
  lead at the tightest moment; the tolerance that represents is a host turning
  30% dearer after the first request, where everything measurable sits at
  0.9-7.8%. It cannot stream — sizing to the lead needs a line before the
  first byte — and it is the first reply only: by the fourth there is a fit.
- **An unmeasured voice borrows, it is not buffered.** Buffered is what a
  caller asks for, not what the app concludes on its own while it still has
  something to go on. A voice with no cost line of its own stands in
  `RenderModel.dearest` of the model's other lines, so a reference uploaded a
  minute ago is planned rather than held whole however long the model has been
  in use. What makes that sound is that the slope belongs to the model and
  only the intercept to the voice: measured on one host, MOSS ran 0.401
  built-in against 0.360 cloned and OmniVoice 0.717 designed against 0.718 and
  0.608 for two clones, while their intercepts ran 0.308 against 1.157 and
  1.496 — so refusing a new clone every part of a sibling's line to protect
  the small half left it with nothing at all. Every part is taken at its
  dearest, and `samples` stays 0 so `StatsStore.get` and the card it feeds
  never show a stand-in as measured. Only a model with no line of any voice on
  this host is still buffered.
- **At most `max_loaded_models` engines are resident**, least-recently-used
  evicted. What two of them cost together is in `cortex-tts/docs/models.md`.
  Those are host figures and a card's are larger:
  measured on a 4 GB GTX 1650, MOSS alone takes 2.9 GB and its streaming path
  plateaus at 3.7 GB, which is the whole card. A model that fits in RAM is not
  therefore a model that fits on the GPU.
- **A bundle may span repositories.** `ModelSpec.sources` is a list;
  MOSS publishes weights and audio codec separately and needs both, and
  `catalog.inspect` reports half a bundle as not downloaded.
- **Conditioning is cached once, for every engine.** Encoding a reference is
  the dominant cost of a cloned utterance, so `engine/conditioning.py` keeps it
  by reference id and drops it when the recording's fingerprint changes or
  `forget()` says so. An engine that kept its own would be a second
  invalidation rule, and the first one to diverge is silent — it still produces
  audio, in the previous voice.
- **A reference is audio _and_ transcript.** Neither alone defines a voice, and
  a wrong transcript degrades the clone with no error — so it is validated on
  the way in (2–20 s, non-empty, pronounceable, and ending in silence). The
  last of those is the same class of silent failure: the models that clone
  best read the whole recording as a worked example, so a clip cut by a clock
  teaches one that sentences end mid-word, and the clone then drifts and clips
  its own endings. Measured on eight real uploads, the seven cut at a
  recorder's 7.00 s limit ended between −7.1 and +7.5 dB against their own
  average and the one that finished ended at −25.3 dB, so the threshold is
  −15 dB over the last 100 ms.
- **Streaming admission is whether the model gains lead, and nothing else.**
  A model that does not gain lead per request would have to bank against a
  reply whose length nobody yet knows, so it is planned once written. Nothing
  tests the opening's own length: a gaining model opens on the margin and the
  spread alone, which is 0.81-0.89 s across the five lines fitted on the two
  production hosts, so a ceiling over it decides nothing and a ceiling under it
  refuses every model. `_open` asks `gains_lead` and stops.
- **A planned reply's hold is whatever never running dry takes.** The plan was
  grouped for exactly that, so cutting the hold short leaves a plan chosen
  never to stall, played in a way that does. `_plan_whole_reply` gives every
  planned reply its deadline, chunk-streaming engines included, and
  `Schedule.of` measures it from each engine's own first byte. The bank stays
  as the early release for a host running ahead of its fit, and below about
  1.2x it is what releases: the two agree to within the sampling across factors
  0.4 to 1.2. Past that the bank stops being enough on its own — at 1.6x the
  first request banks 13.0 s against the 17.0 s the rest is predicted to lose —
  and the deadline is the only thing that speaks on the plan's own terms.

  **unheld** is the answer for a reply too slow to wait out, and it is asked
  for rather than concluded. Switching per reply — from a length nobody can
  see — would give one model and one setting two behaviours, and leave anyone
  who heard the difference no way to tell which they got or to reproduce it.
  `Planner._plan_whole_reply` groups for whichever was named and nothing
  else.

- **A sentence end carries its own pause, and only a sentence end.** Playback
  that catches the renderer at a sentence end is heard as a longer gap between
  sentences, so `Schedule.of` credits `SENTENCE_PAUSE_S` against the bank
  for every boundary already crossed and the hold covers only the remainder.
  Measured on one 313-character reply, OmniVoice with a clone: holding until
  nothing could ever run dry put the first word at 6.29 s, and the allowance
  put it at 4.87 s with one 0.55 s gap, after the first sentence — so
  `sensor.<model>_playback_margin` goes negative by whatever the allowance was
  spent, and a reply that spends it is doing what it was asked to. The credit
  is refused to a batch `Schedule.cut_to_cap` produced at a clause mark, because that
  silence falls inside a sentence, and to a chunk-streaming engine, whose audio
  arrives continuously so a dry moment lands wherever it ran out.
  `Settings.max_sentence_pause` is the stored default, read per reply.
- **A planned reply releases on a deadline; the bank is only the early way out.**
  `_schedule` gives the moment playback may start, and `_apply_hold`
  sends it as `hold_wall_s` for a timer armed on the first audio byte.
  `bank_needed` is still asked as each request lands, so a host running ahead
  of its fit speaks sooner — but on its own it can only ever release where a
  request happens to land, and a whole-render engine lands one batch at a time.
  Measured on OmniVoice with a clone, that put the first word at 8.73 s where
  the same plan's own arithmetic allowed 5.62 s, and it hid the sentence
  allowance completely: 8.10 s against 8.28 s with and without it, because both
  figures were the second batch landing rather than either plan's deadline.
- **An addon option is what a restart is the only way to change.** `config.yaml`
  carries the log level and the discovery key, and nothing else; everything a
  user tunes is a stored setting in `preferences.py`, changed in the admin UI
  and over `PUT /api/settings`. Most take effect on the next request — the
  default temperature travels with every synthesis call, and a smaller
  resident bound evicts down to it. The two ONNX Runtime binds when it
  creates a session (threads, execution provider) are adopted by dropping
  what is resident, and only when they actually changed; `EngineRegistry.reconfigure`
  decides, not the route. `num_threads` is also only half-adopted that way:
  its effect on BLAS and OpenMP is fixed at import and waits for a restart —
  which is why `__main__.py` reads the stored settings before anything
  numeric loads.
- **Ingress bypasses the API key, and only ingress.** `deps.is_ingress`
  requires both the Supervisor's `X-Ingress-Path` header and the Supervisor's
  peer address (`172.30.32.2`); the header alone is forgeable by anything that
  can reach the port. The port is not published by default (`ports: null`),
  and an empty configured key disables the check entirely, which is only sane
  while it stays that way.
- **The UI is served with `cache-control: no-cache`.** A hot-deploy swaps files
  under URLs that never change; without revalidation the browser keeps the old
  panel and the deploy looks like it did nothing.
- **The text path needs no engine.** `/api/preview` answers without loading a
  model, which is what makes the admin UI's right-hand column free to refresh
  on every keystroke.
- **The library never reaches for Home Assistant.** It publishes to listeners
  via `notifications.py`; `cortex_tts/events.py` is what turns that into an
  event on the HA bus. Enforced by `tests/test_architecture.py`.
- **A real-time factor belongs to a host, not to a model.** The catalog
  carries no figure of its own: one measured on the project's reference
  machine read 3x out elsewhere. `cortex_tts/stats.py` keeps what this host
  measured, **per voice kind**, and a card shows that or says "not measured".
  The kinds are split because the cost is: a clone runs about twice what a
  designed voice does on the same model, since the reference's codec frames
  rejoin the prompt on every synthesis — `docs/models.md` has the figures. `scripts/bench_rtf.py` still
  exists, and what it produces is documentation rather than a figure the app
  repeats back to someone else's machine.
- **One measurement, fitted — not two series averaged.** The store keeps one
  primitive: a `RenderSample` of raw audio and wall seconds, recorded once per
  request by every transport, with the model made resident before the clock
  starts — a load is not what a request costs, and folding one in taught the
  fit a figure no later request would reproduce. The card's figure and the planner's model are
  both `RenderModel.fit` of those, so the per-request fixed cost is held beside
  the slope rather than averaged into it. That
  matters because a planned reply is many short requests of one length: an
  average of their ratios charges each the whole fixed cost and reads high,
  while the same points fitted are just the short end of a line the long
  requests already define.

  **`per_audio` is the slope, not the real-time factor.** The factor is what
  the docs say it is — render seconds over audio seconds — and with a fixed
  cost a request also pays `fixed_s / audio`, so it falls as the request
  grows and the two agree only where the line has no fixed cost.
  `RenderModel.real_time_factor` is the one to show a person, quoted at
  `audio_ref_s`, the mean request this host served; quoting a factor without
  its length says nothing. Printing the slope under that heading read 0.29 for
  a line whose own requests cost 1.54 times their audio. It also leaves no cadence for a transport to get
  wrong — there is no per-reply call to remember, because a request is the
  only thing anyone records. Two things a benchmark does to that store and a
  reply does not: a run of one length leaves the fit no slope to find, so the
  per-request cost folds into the factor and the line is right at that length
  and wrong at every other; and back-to-back requests on an idle host
  under-report `spread_s` by an order of magnitude — 0.017 s against the 0.156
  and 0.305 the same model's lines carry from real traffic — which matters
  because every hold is widened by it. Measured again on the standalone host
  after a session of benchmark-shaped traffic: 1.652 s for one voice, 0.198 s
  once the window had refilled with ordinary replies, and that 1.45 s was most
  of a 5.07 s wait before a 7.6 s reply's first word.

- **Scatter is not drift, and a hold has to survive the second.** `spread_s`
  is how far the requests sit from their line; `RenderModel.drift` is how far
  the line itself is from the requests that come after it, fitted on the older
  half of the window and tested against the newer. They are different sizes
  and different shapes: across eight lines on one host the scatter ran
  2.0-9.1% of a render while the older half missed the newer half's total by
  up to 7.9% on lines whose scatter was 2.6%.

  A hold is where the difference bites, because it is the one figure that sums
  the line's predictions over a whole reply. Scatter cancels over a sum, a
  systematic offset does not, and the opening ends up short by that fraction
  of the entire render — measured on a 199 s render fitted 4.2% cheap, seven
  seconds short and three gaps in a delivery whose promise is that it never
  stalls. So `Schedule.hold_for` costs the plan on `model.scaled(1 + drift)`.
  Only under-prediction counts: a line that over-predicts has already bought
  the silence. `MAX_DRIFT` bounds it, because a hold widened in proportion to
  one bad window is minutes of it.

  `bank_needed` keeps the undrifted line. It is re-asked as the reply runs and
  has the reply's own pace to go on, which is better evidence than a window
  measured before it started.

- **The spread is charged by the render it insures, not flat.** `spread_s` is
  one figure over a sample of requests, and charged flat a two-second render
  pays what a ten-second one is worth. Measured on that host's own stored
  samples the residual is closer to proportional: it grew with the render on
  three of four cost lines (MOSS r=+0.67, the OmniVoice clone r=+0.38), and on
  the fourth the _relative_ error was flat instead — 21.9% of the render over
  the short half of its samples against 20.7% over the long. So `fit` records
  `spread_ref_s`, the mean render the spread was measured over, and
  `RenderModel.spread_for` scales between them, capped at twice the measured
  figure because past the lengths sampled proportion is an extrapolation and a
  hold has to end. `Schedule.of` returns the render of the batch that
  set the deepest point, which is the one whose over-running the hold exists
  to survive. What this does not buy is a shorter wait on a short reply: the
  binding render there is usually the _next_ batch, not the first, so the
  measured case moved 7.27 s to 7.24 s. It is a guard against a noisy host
  overcharging a small request, not an optimisation. `scripts/bench_rtf.py` goes through
  `/v1/audio/speech`, so it records like any other request and does both of those
  things to the model it benchmarks — `DELETE /api/models/{id}/stats`
  afterwards, on any host whose figures are in use.
- **A model declares capabilities, not a category.** `builtin_voices`,
  `designed_voices`, `cloning`, `chunk_streaming`, `temperature`,
  `language_choice` and `style_instruction` are independent, so a model can
  have bundled voices _and_ clone. `EngineRegistry.voices` concatenates both
  sources rather than choosing. **One pair is the exception**, and it is an
  exception the code relies on: `builtin_voices` and `designed_voices` are
  mutually exclusive, because `routes.voice_kind` settles which of those two a
  rendered voice was from the spec alone rather than looking the voice up. A
  stored recording is a clone whichever model spoke it, so that case is
  decided before the spec is consulted.
  `tests/test_catalog.py` pins it, so a future entry that sets both fails the
  build instead of silently mislabelling every measurement it makes.
- **A stream has its own level control.** `encode` peak-normalises a finished
  waveform; a stream has none, so `StreamGain` holds a gain that only ever
  falls, far enough to keep each chunk under the same ceiling. Scaling a chunk
  to its own peak instead would make every chunk equally loud, which pumps.
- **A streamed format must not have to declare a length.** `/api/speak/live`
  answers MP3 — a bare frame sequence with no container, no length field and no
  index. WAV is still available but is not the default: the maximal length its
  header has to declare is read by a general-purpose player as a six-hour file
  it then waits to buffer. FLAC and OGG are not offered at all. The `ready`
  frame carries `bitrate` because it is the one measurement that exists before
  the first sample, and it is what turns a byte count into a duration
  downstream.
- **One audio frame is bounded, not one reply.** A receiver buffers a
  WebSocket frame whole before it sees any of it, so every client caps one —
  aiohttp's default is 4 MB — and the opening hold of a buffered reply is the
  entire reply in a single frame. `MAX_FRAME_BYTES` slices it at 512 KB on the
  way out; the bytes are a stream and the receiver concatenates them, so a
  slice may fall anywhere. Fixed here so no client has to be configured for
  this server.
- **The delivery is legible from outside, or it is not reviewable.** Over
  `/api/speak/live` a `batch` frame says what a request carries and whether a
  gap after it would be heard as a pause, and a `rendered` frame says what it
  cost. Neither is recoverable from the audio, and while the opening is held
  back there is no audio to recover anything from — which is exactly the
  window every pacing question has been about. The admin UI's live panel draws
  its timeline from those two and from what the audio clock did, and it plays
  the reply while it arrives — it asks for WAV and schedules the raw PCM on a
  `Web Audio` clock rather than feeding an `<audio>` element, because an
  element decides the timing for itself (it buffers, it stalls, it catches up)
  and the timing is the whole subject. A buffer that arrives after its slot is
  a gap the listener hears and the panel measures, which is what makes the
  chart evidence rather than a second drawing of the planner's own arithmetic.
  A host with no audio device reports a running context whose clock never
  advances — measured in headless Chrome — so the panel withdraws every
  playback figure rather than count a playback that is not happening. A
  browser reaches the socket through the handshake's subprotocol list
  (`["cortex-tts", key]`), because a `WebSocket` constructor sets no headers;
  behind ingress it sends none.
- **Everything that can fail must fail before the first byte.** Once audio has
  started an error can only truncate it, so `/api/speak/live` resolves the
  model, the voice and the encoder before it sends `ready`.
- **Chunked audio is the WebSocket's alone.** `/api/speak/live` is the only
  route that sends audio as it is produced, and every reply the integration
  speaks goes through it — including a buffered one, which the app holds to
  the end and levels like a file before releasing. `/v1/audio/speech` answers
  a finished file for callers that are not Home Assistant. Two routes because
  those are two different
  questions, not because one is a remnant.
- **Streaming is a second protocol, not an optional method.** `StreamingEngine`
  is what MOSS and Qwen3-TTS satisfy and the rest do not; the registry asks
  with `isinstance` rather than making every engine decline a method it has no
  answer for. The capability is therefore written down twice — `ModelSpec.chunk_streaming`
  for a caller who has loaded nothing, the method for the registry — so a test
  pins them to each other. Disagreeing is silent in the direction that matters:
  a spec claiming the capability without the method makes `/api/speak/live`
  report `chunk_streaming: true` while whole utterances are rendered.
- **A quantised model is not automatically a fast one.** Check the op types
  before believing a small INT8 export is fast: a dynamically quantised
  per-token loop does not scale with threads, and the codec can cost as much
  as the model. What quantisation reliably buys is memory, not time.
- **The execution provider is verified, never assumed.** `auto` takes a GPU
  when one answers and the CPU when none does; `cuda` refuses to fall back.
  `onnxruntime.get_available_providers()` is a claim about the build, not a
  promise — measured on a GTX 1650 host it listed CUDA and then created every
  session on the CPU — so `providers.in_use` reads the sessions instead, and
  `/health` reports what was asked for beside what arrived. What the card
  buys each model is measured in `docs/models.md` and nowhere else.
- **A backend is looked up, never branched on.** `ModelSpec.backend` keys into
  `engine/backends.py`; adding an engine is a module plus a registration, and
  builders import lazily so a heavy backend costs nothing until it is used.
- **An engine never lists its own voices.** Both kinds are files and the
  registry reads them: `Engine` has no `voices()` method, because the one it
  had was dead — every path already went through `EngineRegistry.voices`, and a
  protocol method nothing calls is a thing each new engine must write for
  nobody.
- **Listing voices loads nothing.** Both kinds are files: built-in voices live
  in the bundle's manifest or voices npz, reference voices in the store's
  index. A backend whose models bring voices of their own registers an
  `own_voices(directory)` reader alongside its builder — "own" rather than
  "built-in" because the bundled kind and OmniVoice's designed kind are both
  read this way, and a name for one of them lies about the other — so `/api/voices` never constructs an engine — at the
  default of one resident model, doing so evicted whatever was speaking, and
  the integration asks for every model's voices at startup.

## Model behaviour worth knowing

The measurements are in [`docs/models.md`](cortex-tts/docs/models.md); what
matters to the code is:

- **What one call may produce is the model's; what it should cost is the
  host's.** `ModelSpec.max_audio_s` is the generator's ceiling, counted in
  audio, and it truncates rather than slowing: past it the call returns what it
  had and the rest of the text is never spoken. Hojo declares 41 s (2048 new
  tokens at a 50 Hz codec), MOSS 30 s (its manifest's `max_new_frames` of 375
  at 12.5 Hz, and measured here — seven inputs from 155 to 284 Chinese
  characters each came back as exactly 30.0 s), Qwen3-TTS 164 s (the talker's
  2048 frames at 12.5 Hz). A model's runtime splitting the _text_ by a token
  budget does not save it, because the budget is spent on audio.

  **A model that establishes no ceiling is given no number on its behalf.**
  OmniVoice declares neither bound, so nothing splits it by length — a figure
  that is neither the model's nor this host's is one that will be wrong on some
  machine, and there is already a figure that knows about the machine:
  `batch_cap_s`, which `RenderModel.fit` replaces with this host's own as soon
  as its samples can say anything. `max_chars_per_segment` therefore has no
  default and is declared only where a model's tokenizer, rather than a guess,
  says one.

  The ceiling is in seconds and the splitter counts characters, so
  `ModelSpec.segment_limit` converts with the slow-side priors in
  `pacing.model` — priors rather than a fitted line, because a segment must not
  depend on which voice says it. Every caller asks the spec rather than reading
  a field: `prepare` takes the method, so the limit is read off the _prepared_
  text, which is what the engine gets and what normalisation may have changed
  the script of. Splitting a reply the model could have taken whole is not free
  — every segment is terminated, so a cut lands as a full stop — and it is what
  **buffered** exists not to do. The figures are in
  [`docs/models.md`](cortex-tts/docs/models.md#how-much-text-one-synthesis-takes).

- **An autoregressive model stops only when it samples an end-of-speech
  token.** Stopping is therefore probabilistic, so `text/pipeline.py`
  terminates every segment (without a stop the 40M invented a syllable,
  measured) and `engine/overrun.py` judges and trims what came back. This is
  the Hojo pair and Qwen3-TTS: `render_with_retries` is imported by
  `preset.py`, `clone.py` and `qwen3.py` and nowhere else. OmniVoice decodes a
  fixed number of steps and MOSS folds sampling into its runtime, which is why
  neither declares `temperature`. A high temperature makes over-runs likelier
  on the three that have one — but `0` is not the cure everywhere, because
  Qwen3-TTS run greedy reliably fails to emit end-of-speech at all (below).
  That is what a per-request `temperature` on a reply is for.
- **Whether a bare digit survives is the model's; a symbol is nobody's.**
  The two Hojo models declare `needs_number_words` — an unexpanded digit is
  silent or replaced by an unrelated word, reported upstream as
  [Hojo-TTS-Light#7](https://github.com/HojoAI/Hojo-TTS-Light/issues/7) — so
  bare numbers are expanded for them by default and left alone elsewhere. Unit
  symbols, times and dates are a separate question and a settled one: no model
  here declares `reads_numerals`, so normalisation belongs on for every
  language and the integration must not gate it on the language tag.
- **The 80M's cloning fidelity is capped by its speaker representation.**
  `SPEAKER_EMB_SECONDS = 6.0`: `_encode_speaker` reads the first six seconds
  into one 2048-value vector, while `_encode_ref_codes` encodes the whole
  reference into prompt codes — so audio past six seconds costs time on every
  synthesis and buys nothing. MOSS has no speaker encoder and conditions on the
  whole recording. Not measured against the model's own encoder; the
  conclusion rests on the representation size and on our path matching
  upstream's `generate` step for step.
- **Whether a language can be named is a capability, not a rule.** On most
  models the prompt is the text plus a speaker slot, so picking the voice is
  picking the language; those declare `language_choice = False` and the API
  refuses a `language` rather than accepting one it would ignore. Qwen3-TTS
  and OmniVoice do take one, so a request may name it and the engine narrows
  the whole tag against that model's own list (`narrowing` in
  `engine/base.py`). Unset, the engine still supplies one: Qwen3-TTS from the
  speaker's own language (upstream's recommendation) or the reference's, and
  OmniVoice from the script the text reads as.
- **A designed voice is a third kind of voice.** OmniVoice has no bundled
  speakers and no free-text style prompt: it takes a short instruction built
  from a closed vocabulary (`vendor/omnivoice/voice_design.py`), and refuses
  anything outside it. The nine on offer are chosen in `engine/omni.py`, so
  its voice reader opens nothing — and `tests/test_omnivoice.py` checks each
  one against the model's own vocabulary, because a typo there is a
  `ValueError` in the middle of a reply rather than a worse voice.
- **Qwen3-TTS must not be run greedy.** It stops by sampling end-of-speech and
  at temperature 0 reliably fails to: a ten-character line ran to the model's
  own 2048-frame limit, 2.7 minutes of invented audio. `engine/qwen3.py` caps
  each segment from the duration the text needs.
- **OmniVoice's transformer is never materialised.** The int4 ONNX graph
  replaces `forward` outright, so the checkpoint's 2.45 GB of weights are
  neither downloaded nor loaded; the module is built on the meta device.
  Measured peak resident 1.1 GB against 4.7 GB. The cost is that anything
  reading a weight outside `forward` fails with "Cannot copy out of meta
  tensor" — loudly, at first synthesis.

## Build

```bash
cd cortex-tts
uv sync --frozen          # dependencies
uv run cortex-tts           # run it (STATIC_DIR + DATA_DIR from the environment)
```

Two extras carry the models that are not pure ONNX Runtime, and the Dockerfile
installs both deliberately. `hojo-80m` pulls torch and librosa for that model's
mel front-end; `omnivoice` adds torchaudio, transformers and pydub for the half
of OmniVoice the ONNX graph does not replace. Together they are most of the
image. Both are named for the model rather than for a capability — MOSS and
Qwen3-TTS clone without touching either, through `audio.decode_reference`.

The admin UI has **no build step** — `web/` is plain ES modules and CSS, served
by `StaticFiles`. Edit and reload.

## Testing

```bash
uv run pytest -q          # ~700 tests, no model weights needed
```

`tests/test_text.py` and `tests/test_english.py` pin the text path — the part
that is measurable without a GPU-hour and the part most likely to regress
silently, since a wrong reading produces confident audio rather than an error.
`tests/test_api.py` covers auth, the model list shape and the errors a caller
sees. `tests/test_architecture.py` enforces the library/app boundary by reading
the AST — it loads nothing and runs in milliseconds. `tests/test_catalog.py`
pins the capability model and the backend table. `tests/test_stats.py` covers
the one series both the card and the planner read — that it is fitted rather
than averaged, and that a reply split into many short requests cannot skew it.
`tests/test_pacing.py` drives the planner with synthetic render models, text a
character at a time included; `tests/test_live.py` runs the WebSocket over a
fake engine; `tests/test_abandon.py` pins that a lost listener stops a render
and is never relabelled as a failure.
Synthesis itself is exercised by hand against real bundles. The planner's
behaviour across render speeds was covered by a one-off simulation — its
figures are in `cortex-tts/docs/delivery.md`, but the script itself was not
kept.

## API Endpoints

The endpoint table, request fields, headers and every error code are in
[`docs/api.md`](cortex-tts/docs/api.md), which is the reference; `/api/docs`
serves the OpenAPI. Two things the code guarantees and the reference relies on:

- **Every error body is `{"code", "message"}`** — what a route raised, what the
  router could not match (404 on an unknown path) and what pydantic refused
  (422). `app.py` installs both handlers so a client parses one shape.
- **The mode is said twice on the socket and only the last is final.**
  `ready` carries what the reply starts as, before any decision has been
  taken; `batch` carries the plan in force and goes out before each request;
  `done` replaces it with what happened, forced to `whole` whenever the reply
  fitted one request — `whole` is about where the reply was cut, and one
  request has nowhere to cut. **Buffered is the exception**, because it is not
  a statement about cutting but about releasing, and it is true of a
  one-request reply as much as of any other. Measured on MOSS, which emits
  audio while it renders: the same reply in one request was heard at 4.53 s
  when that audio went out as it came and at 26.96 s when it was held to the
  end, and reporting `whole` for both had the sensor say one word about two
  experiences twenty-two seconds apart. The integration writes its mode sensor
  from the last two,
  so every value either can carry has to be in that sensor's options — an enum
  handed a state outside them raises, and the reply never plays at all.
- **`/health` carries `api_version`**, bumped when a route, field or header the
  integration reads changes shape; the release version says nothing about the
  wire. The integration refuses to set up on a mismatch. It is 4 — three
  when the third delivery was called `paced`. Renaming it was a wire change
  and not only a word: the integration reads that string into an enum sensor,
  and a value outside its options raises rather than degrades, so the reply
  would never have played at all. `planned` says what actually separates it
  from `streaming` — when the plan is made, not how the audio goes out.

## Home Assistant Discovery

`discovery.announce` publishes a Supervisor discovery record carrying host,
port and API key, so the integration appears on the Devices page with nothing
to type. The key is generated on first start and pushed through the same
channel; clearing `discovery_api_key` and restarting rotates it. Outside the
Supervisor there is no token and nothing to announce to — local development
runs with an empty key and no discovery.

## Live voice sync (HA event)

Downloading or deleting a model and adding or deleting a reference recording
change which voices exist, and each ends in `events.fire_models_changed`
putting one event on the HA bus; the integration adds or removes entities in
place, so a voice uploaded in the admin UI is selectable in a pipeline seconds
later without a config-entry reload or a restart. Two paths reach that event.
Download completion is the library's: `download.py` calls
`notifications.notify_models_changed` and the app subscribes at startup. The
three routes — reference add, reference delete, model delete — call
`fire_models_changed` from the app directly. `PATCH /api/references/{id}` fires
it only when the gender label or the language changed: a corrected transcript
changes nothing the picker shows, a relabelled voice does. The split still matters
where it exists: the library states a fact, the app decides that Home Assistant
is who hears it.

## How To

### Add a unit to the normaliser

The symbols are Home Assistant's (`homeassistant/const.py`, `UnitOf*`).
`text/units.py` maps each to its CLDR unit id (or a numerator/denominator
pair for a rate), which names it in English and in every generic-locale
language through `babel`; `UNNAMED` there lists the symbols CLDR has no
unit for, so the number before them is still read. `SUFFIX_UNITS` in
`text/zh/normalize.py` carries the Chinese reading of the same symbols
(`kWh` → 度電); `WORD_UNITS` there the Chinese unit words read as written
(`分鐘`), in both scripts. Every alternation is built longest-first so
`km/h` is matched before `km`; nothing else needs to change.

### Add an engine

Four declarations, across two files:

1. A module under `engine/` implementing the `Engine` protocol — and, if the
   model declares `builtin_voices` or `designed_voices`, a module-level
   `own_voices(directory)` that reads the list off disk, opening nothing.
2. A builder closure in `engine/backends.py`, importing the engine inside the
   call so a heavy backend costs nothing until it is used.
3. A voice-reader closure there too, when there is one.
4. `backends.register("your-key", builder, voices=…)`, plus a `ModelSpec` in
   `catalog.py` naming that backend with the capabilities it actually has —
   including what it does to the text path: `reads_numerals` lets the
   generic locale stand aside for a model that reads digits itself, and
   `needs_number_words` turns bare-number expansion on for one that cannot
   say a digit at all. Both are declared only after measuring (every model
   so far, Qwen3-TTS included, read German and Japanese digits as noise on
   the sentences tried; only Hojo could not read Chinese or English ones).

Then a `[project.optional-dependencies]` extra and a `--extra` in the
Dockerfile if the backend needs its own dependencies.

**The registry, the API layer and the integration need no change** — the
registry builds through the table, `routes.py` reads `spec.*`, and the
integration reconciles against whatever the server lists. `tests/test_catalog.py`
derives what it checks from `backends.py`, so a new backend is covered by the
provider and chunk-streaming contracts without being added to a list.

**The UI is not automatic.** It renders from the capability booleans and will
be right for any combination of them, but the prose around it — what the
clones panel says, what the model cards claim about memory — is written by
hand and has been wrong for a new engine before.

### Add a language

A locale, never a branch: a package beside `text/zh/` and `text/en/` whose
`LOCALE` names its normaliser, its sentence-final stop and the rewrites only
it needs, registered in `text/pipeline.py`. Until it exists the language gets
`text/generic.py` — numbers from `num2words`, nothing else touched — so a
locale is only worth writing when a language needs more than that. A rewrite
only that language needs (a script conversion, a readings table) lives with
the locale and is absent — not switched off — everywhere else: `/api/preview`
lists only the switches the language has, and the UI's chips follow.

### Add a pass

Add the pattern to `text/passes.py`, a render to the script that should say it,
and a row to that script's `_PASSES` tuple — in the position the order
requires. A pass that only one script has (the Chinese unit pass) simply does
not appear in the other's table.

## Distribution

CI publishes the image to `ghcr.io/hass-cortex/cortex_tts/{arch}` on release —
amd64 only, matching the published ONNX Runtime builds — and dispatches an
update to the catalog. Stable is
[`hass-cortex/repository`](https://github.com/hass-cortex/repository); beta is
`repository-beta`, and a semver pre-release tag (`0.2.0-beta.1`) routes there
only. The runbook is `docs/release/` at the root of the hass-cortex workspace,
outside this repo.

### Release tags

`X.Y.Z` — `release.yml` runs the Python checks with the uv cache **off** (a tag
build must not read anything a pull request could have written), then creates
the GitHub Release with `DISPATCH_TOKEN` so `deploy.yaml` fires.
