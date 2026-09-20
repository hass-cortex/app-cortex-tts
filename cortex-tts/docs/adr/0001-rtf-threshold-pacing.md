# ADR 0001 — A live reply is paced by one measured real-time factor and one threshold

Status: accepted (2026-09-20). Supersedes the cost-line planner in `cortex_speech/pacing`.

## Context

A live reply (`/api/speak/live`) is spoken while the conversation agent is
still writing it, and the app has to decide when to render what and when to
release the audio. The previous design fitted a two-part cost line per model
and voice (fixed cost plus a per-audio-second slope, with a scatter term, a
drift term, a per-reply slip correction, a measured batch cap and a probe
past it), and from that line derived, per reply, one of four deliveries
(streaming, planned, unheld, buffered), an opening hold sized as a bank of
audio seconds or a wall-clock deadline, a sentence-pause credit and a grouping
search over batch boundaries. It was about 2,260 lines and a fifth of the
first-party source, carried 127 tests of its own, and its documentation was
longer than its code.

It still ran dry. Measured on the two production hosts on 2026-09-20:

- On the GPU host (GTX 1650) every model was set to `auto`; OmniVoice was
  delivered `planned` and stalled mid-reply.
- On the Home Assistant host (4 vCPU i7-9750H, CPU) MOSS-TTS-Nano `planned`
  replies reported `min_lead_s` of −0.208 and −1.131 the same morning.

The mechanisms that were meant to prevent that — the bank, the deadline, the
drift allowance — each exist because an earlier one had failed, and each
added a way to be wrong. The objective the project actually holds is that a
reply never runs dry; time to first audio is allowed to regress to buy that.

## Decision

Replace the planner with three rules and two constants.

1. **One number per model and voice: the median real-time factor of the last
   `WINDOW = 8` requests served under the execution provider now in use.**
   RTF is render seconds over audio seconds of one request, fixed cost
   included. `null` until `MIN_SAMPLES = 3` exist. Built-in voices, designed
   voices and each cloned reference are separate series; nothing is pooled
   or borrowed. Loading a model is outside every request. Samples record the
   provider (`cpu` / `cuda`); samples from a different provider are ignored,
   and samples written before this change carry no provider and are ignored
   too.

2. **Verdict: `streaming` if the median is under this host's threshold, else
   `buffered`.** Made once per reply at `ready`, never revised. The setting
   `auto` takes the verdict; `streaming` and `buffered` overrule it. The
   threshold is `stream_rtf` in the app's settings, default `STREAM_RTF =
0.8`, range 0.1 to 3.0: it belongs to the host like the measurement it is
   compared with, and moving it is the one way to say "I would rather wait
   less and risk a gap" without setting every model's mode by hand.

3. **Two ways to release.**
   - _Streaming_: one sentence per request, rendered as sentences arrive.
     Nothing plays until `BANK_S = 6.0` seconds of audio have been rendered
     or the whole reply has; from then on each sentence goes out as it
     finishes. Chunk-streaming engines count their chunks toward the bank
     like any other audio. **Once the writer has finished, the bank is
     capped at what the rest needs**: with the remaining sentences in hand
     and the voice's median RTF, `bank_needed` gives the least lead under
     which no remaining sentence is reached before it is rendered
     (`rtf × aᵢ + (rtf − 1) × Σ before`), the sentences' audio estimated
     with the slow-side priors. The 6 s is for what is still unknown; a
     two-sentence reply at 1.05x was held 6.4 s for the sake of 0.4 s of
     deficit, and now waits for the first sentence alone.
   - _Buffered_: rendered as the words arrive — whenever the engine is free
     it takes every complete sentence waiting, as one request, split into
     segments only at `ModelSpec.segment_limit` — and released once
     everything has been rendered.

Everything else goes: the fitted line, spread, drift, slip, `gains_lead`,
the bank arithmetic and deadline, the first-request floor and ceiling, clause
cuts, the sentence-pause credit and its setting, `batch_cap_s` and `PROBE`,
the `dearest` stand-in, the `planned` and `unheld` deliveries and the `whole`
outcome. `api_version` becomes 5 and the integration ships in step.

## Why these numbers

`scripts/replay_pacing.py` replays recorded requests and real replies through
candidate rules. Its inputs are a host's `stats.json`, an app log written at
`LOG_LEVEL=debug` (uvicorn's frame trace carries every `rendered` frame) and
the reply texts; none of them is in the repository, being measurements of
someone's machine and things people said to their assistant. The figures
below are the run of 2026-09-20 over 420 production replies.

- **0.7** is where the median stops misclassifying any of 21 measured
  host × model × voice cells (HA host bench and log, Ryzen 9955HX bench, GTX
  1650 stats, RTX 5070 Ti log) against a hand verdict from the same data. At
  0.6 it would refuse Hojo 40M on the HA host (0.67 in the bench, 0.58 in
  production) and Hojo 80M on the Ryzen (0.65). The p90 needs 0.8 for the
  same result; the median at 8 and at 24 samples agree everywhere.
- **The default is 0.8 all the same**, because the two cells 0.7 keeps out
  and 0.8 lets in — OmniVoice designed on the GTX 1650 and MOSS on the 5070
  Ti, both 0.78 — were judged "stalls" from the unheld planner's logs, not
  under the bank, and what buffered costs them is the whole render before
  the first word: at 0.78 a median multi-sentence reply waits 23 s buffered
  against 7 s streaming. A gap is the cheaper failure there.
- **Whether a gap comes depends on how the engine releases audio.** One that
  hands a segment over whole (Hojo, OmniVoice) needs the lead to outlast
  the next sentence's whole render, so a long sentence can drain the bank
  even under 1.0. One that releases within a sentence (MOSS) loses lead at
  `rtf − 1` per second of audio and runs dry only once the reply is longer
  than about `6 s · rtf² / (rtf − 1)`: at 1.06, 112 s of audio, which 9 of
  420 production replies exceed; at 1.12, 63 s and 46 of them. That is why
  MOSS on the HA host (1.06–1.12) is a reasonable thing to stream from a
  host that sets its threshold at 1.2, and why the threshold is a host's
  setting rather than one number for every engine. If a segment-whole
  engine ever measures between 0.8 and 1.2 on a host that raised the line,
  the threshold should become the engine's to declare; it is not yet.
- **6 s** comes from 182 multi-sentence production replies replayed under
  each cell's measured cost. Releasing every sentence unheld stalls 27–33% of
  them on a 0.6–0.7 host (worst gap 7 s); a one-sentence lookahead is worse
  than any bank; a 3 s bank still stalls 15–20%; 6 s leaves 1–3 of 182 with
  a gap under 1.8 s and none on a GPU host; 8 s leaves none but costs
  another 1.2 s of first word. Median first word on multi-sentence replies at
  6 s: 5.8 s on the HA host's Hojo 40M, 2.9 s on the GPU host's. Single-
  sentence replies — 57% of production — are unaffected by the rule.
- **Per-voice, unpooled**: a clone's fixed cost makes its short sentences
  slower than its average (OmniVoice clone on the GTX 1650: 0.8 at medium
  length, 1.1–1.2 on short sentences). Below 0.7 that voice is buffered
  anyway, so no intercept model is needed — but a pooled figure could let a
  cheap built-in voice carry an expensive clone across the line.

## Consequences

- A voice needs three requests on this host before it can stream; until
  then, and for every voice at or above the host's threshold, replies are
  buffered — rendered as written, released when done. At the default, on the
  HA host only Hojo 40M streams; on the GTX 1650 Hojo 40M and MOSS do;
  OmniVoice is buffered everywhere measured.
- The card shows the comparison (`1.06 ≥ 0.80 → buffered`), the `ready`
  frame and the `spoke live` log line carry `threshold`, so a verdict reads
  as a setting applied and a stall report says which line it was under.
- Time to first audio on a streaming multi-sentence reply rises to roughly
  the bank; a buffered reply's wait is about max(writer, render) plus the
  last request, not their sum.
- A slow writer (an LLM producing text more slowly than it is spoken) is not
  handled: the bank does not cover it, and a pause then lands at a sentence
  boundary. Out of scope by decision.
- The wire changes: `start.mode` and the subentry setting are `auto` /
  `streaming` / `buffered` (default `auto`); `batch.mode` and `done.mode` are
  `streaming` / `buffered`; `ready` loses `pause_s` and gains `rtf` and
  `samples`; `batch` loses `ends_sentence`; the measured-RTF card is
  `{kind, voice, rtf, samples, provider, verdict}`. Stored subentry values
  `sentence`, `coalesced`, `planned` and `unheld` read as `auto`.
- The measurement store keeps raw `[audio_s, wall_s, provider]` samples, 24
  per voice; the fit and the character counts go.
- The replay script is kept as the regression for the two constants: a new
  host's samples are added to its inputs, and a change to either constant
  is argued from its output.
