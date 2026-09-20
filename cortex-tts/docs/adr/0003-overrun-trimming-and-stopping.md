# ADR 0003 — An autoregressive model is stopped by its punctuation and trimmed only on short text

Status: accepted.

## Context

Hojo 40M stops only when it samples an end-of-speech token, so stopping
is probabilistic: a segment without a stop invents a syllable, and a short
input sometimes runs on into babble. OmniVoice decodes a fixed number of
steps and MOSS folds sampling into its runtime, so neither has the problem
and neither declares `temperature`.

## Decision

- `text/pipeline.py` terminates every segment with sentence-final
  punctuation, in the locale's script.
- `engine/overrun.py` judges a rendered waveform against the duration its
  text needs (`CHARS_PER_SECOND`, `OVERRUN_RATIO`) and trims what overran;
  `render_with_retries` re-seeds and retries a render that came back
  unusable. Only `preset.py` imports it.
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
- A per-request `temperature` is the other lever: at `0` Hojo 40M decodes
  greedily and never over-runs.

## Consequences

- Between the two failures the choice is not close: babble is a stray
  syllable after the sentence; an over-cut is the sentence without its
  ending, and nothing downstream can tell it happened.
- A model whose stopping is not probabilistic must not be routed through
  the trimmer; the import list is the contract.
