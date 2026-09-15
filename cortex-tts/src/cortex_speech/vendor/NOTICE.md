# Provenance

What each file here came from, and which of them are not upstream's work.

The point of the directory is diffability: upstream code is carried
byte-for-byte (apart from an import path or a header) so a fix released there
can be diffed in rather than re-derived. `pyproject.toml` exempts those files
from ruff and pyright for the same reason — restyling them would destroy the
property the directory exists for.

## Carried from upstream

| Path              | Source                                                      | Licence    |
| ----------------- | ----------------------------------------------------------- | ---------- |
| `hojo40.py`       | `HojoAI/Hojo-TTS-Light` — `Hojo-TTS-Light-40M/onnx_model.py` | Apache-2.0 |
| `hojo80.py`       | `HojoAI/Hojo-TTS-Light` — `Hojo-TTS-Light-80M/onnx_model.py` | Apache-2.0 |
| `moss_runtime.py` | `OpenMOSS/MOSS-TTS-Nano` @ `8b7bcc9`                          | see repo   |
| `moss_ort.py`     | `OpenMOSS/MOSS-TTS-Nano` @ `8b7bcc9`                          | see repo   |
| `omnivoice/`      | `omnivoice` 0.2.1 (`k2-fsa/OmniVoice`), six modules flattened | Apache-2.0 — `omnivoice/LICENSE` |

## Local deviations in carried files

Each is marked `DEVIATION` at the line, so a diff against upstream reads as
three known edits rather than noise:

| Path                    | Deviation                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- |
| `moss_runtime.py`       | `on_audio_chunk` callback on `synthesize_single_chunk`: publishes each decoded chunk as it appears          |
| `hojo40.py`, `hojo80.py` | `on_step` on `generate` / `_generate_coarse_tokens`: called before every decode step; raising from it aborts |
| `omnivoice/modeling.py` | `on_step` on `OmniVoiceGenerationConfig`: called before every iterative decoding step; raising aborts       |

The two `on_step` hooks exist for one reason: a render whose listener has gone
must stop within one unit of work, and the unit is inside these loops.

## Ours, and therefore linted

These two sit here for provenance, not because upstream would recognise them.
They are **not** exempt from ruff or pyright; the exemption in
`pyproject.toml` names files individually so that a blanket `vendor/**` cannot
quietly take our own code out of both checkers again.

`omnivoice_ort.py` carries one file-level suppression, `reportMissingImports`:
torch and transformers arrive with the `omnivoice` extra and CI does not
install it, so every import of them is unresolved there. Anything narrower
than a file is suppressed at the line that needs it. A file-level pragma has
to sit **above** the module docstring — pyright reads one only before any
other code — and one written below it is ignored without a word, which is how
this file spent a while type-checking less than it claimed.

| Path                | What it is                                                                                                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qwen3_tts_ort.py`  | Derived from `inference.py` in `onnx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice` (**Apache-2.0**), which mirrors `Qwen3TTSForConditionalGeneration.generate`. Reshaped from a demo script into a runtime: no manifest, provider chosen by the caller, tokenizer injected, generation yields frames. The file's own docstring lists every departure. |
| `omnivoice_ort.py`  | Original. Builds OmniVoice with the int4 ONNX graph in place of its transformer, on the meta device, so the checkpoint's 2.45 GB of weights are never downloaded or loaded.                                                                                                                                                                      |
