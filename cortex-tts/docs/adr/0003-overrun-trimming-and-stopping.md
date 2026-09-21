# ADR 0003 — An autoregressive model is stopped by its punctuation, retried when it stops early, and trimmed only on short text

Status: accepted.

## Context

Hojo 40M and MOSS-TTS-Nano stop only when they sample an end-of-speech token,
so stopping is probabilistic in both directions: a segment without a stop
invents a syllable, and a generation can also end in the middle of the text it
was given — a clean, complete-sounding clip that is simply missing its ending,
with no error and nothing in the audio to say so. OmniVoice decodes a fixed
number of steps and has neither failure.

MOSS is the case that shows what "probabilistic" costs, because its sampling
is fused into an exported ONNX graph and its uniforms are drawn from one
seeded generator: 17 per frame, the first of which decides the token that
carries the stop. Seeding that generator per segment is what makes a reply
reproducible (see `_SAMPLING_SEED`) — and it makes a bad draw reproducible
too. One 411-character English segment, one voice, one runtime, varying only
the seed:

| seed | audio                        |
| ---- | ---------------------------- |
| 1234 | 12.00 s, ending mid-sentence |
| 7    | 26.64 s                      |
| 42   | 11.60 s, ending mid-sentence |
| 0    | 26.32 s                      |
| 99   | 27.52 s                      |

Three of five seeds say all of it. The engine happens to pin one of the other
two, so that text lost its last 180 characters on every request, every time.
Upstream reports the same failure from CPU and CUDA alike
([MOSS-TTS-Nano#58](https://github.com/OpenMOSS/MOSS-TTS-Nano/issues/58)).

Length does not cause it; length compounds it. Every frame is another chance
to draw the stop, so the same 411 characters fed as three chunks came back
whole while the single chunk did not.

## Decision

- `text/pipeline.py` terminates every segment with sentence-final
  punctuation, in the locale's script.
- **What one call may carry is a declared model fact.** `ModelSpec.max_text_tokens`
  is counted in the model's own text tokens, which is why `segment_limit`
  cannot settle it and the engine applies it: only the engine holds a
  tokenizer, and characters per token is a property of the script. Measured at
  the chunk rather than the request — the largest MOSS chunk that kept all its
  text was 49 tokens, the smallest that lost some was 61 — so MOSS declares
  50, well under the 75 upstream's own splitter defaults to.
- **A generation that stopped early is retried at another seed.**
  `engine/overrun.py` judges a rendered waveform against the duration its text
  needs; `render_with_retries` re-seeds and retries. `preset.py` and `moss.py`
  both use it. The seeds are de-duplicated: a model whose own seed is already
  in `RETRY_SEEDS` would otherwise spend a whole render repeating an attempt
  that seeded sampling makes identical to the sample.
- **The ratio is per engine, because the estimate it is compared against is.**
  `TRUNCATION_RATIO` (0.6) stands for Hojo. MOSS passes 0.8: across two
  replies chunked four ways its intact generations ran 0.83 to 1.23 of the
  estimate and its truncated ones 0.77 and 0.41, a window of (0.77, 0.83).
  The margin is 0.03, so the error it makes is a needless re-render rather
  than a sentence delivered without its ending.
- **A streamed generation is not retried, only reported.** Its chunks are
  already on the wire, and buffering a chunk to make the retry possible would
  cost the time-to-first-audio that ADR 0001 prices. It logs instead. A
  _buffered_ live reply releases nothing until the end, so it is rendered
  through `EngineRegistry.synthesize` and does get the retry at no cost to the
  listener.
- **Trimming judges short text only**: `MAX_BABBLE_CHARS` (20) bounds it to
  the inputs the pathology is actually on, measured at 2 characters ("好了",
  0.7 s of text as 2.06 s) and 5 ("大燈已關閉", 0.90 s as 1.74 s). Past that
  the estimate is looser than the ratio it is compared against: across the
  eleven requests of one reply (measured on qwen3-tts-0.6b) the same voice
  ran 2.81 to 5.19 characters a second, and a 20-character sentence it
  genuinely took 7.12 s over read as 1.6x its estimate and came back trimmed
  to 4.89 s —
  「后来他学会了最远的远方。」 where the model had said
  「后来他学会了：最远的远方不一定在门外。」. `KEEP_FLOOR_RATIO` is no
  backstop, being anchored on the same estimate.
- **Trimming is opt-in per engine** (`render_with_retries(trim=...)`), and
  only for a model the babble pathology was measured on. MOSS opts out.
- A per-request `temperature` is the other lever where a model takes one: at
  `0` Hojo 40M decodes greedily and never over-runs. MOSS takes none — its
  sampling constants are baked into the exported graph and can only be
  changed by re-exporting it.

## Consequences

- Between the two failures the choice is not close: babble is a stray
  syllable after the sentence; an over-cut is the sentence without its
  ending, and nothing downstream can tell that it happened.
- A model whose stopping is not probabilistic is routed through neither the
  retry nor the trimmer, and a model the babble pathology was not measured on
  is routed through the retry alone. Both are decided per engine at the call,
  not by which module imports what.
- Truncation is never silent again: retried where a retry is possible,
  logged at WARNING where it is not.
- A smaller token budget means more chunk boundaries. The runtime's splitter
  cuts on clause punctuation, so they land where a short pause belongs, and
  the gap between them is `join.segment_gap` — the same one a rendered reply
  puts between sentences.
