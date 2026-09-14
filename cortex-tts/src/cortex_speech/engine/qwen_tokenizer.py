"""Assemble Qwen3-TTS's text tokenizer from the pieces its checkpoint ships.

Every other model here loads a `tokenizer.json` and is done. Qwen publishes
none: the checkpoints carry `vocab.json`, `merges.txt` and
`tokenizer_config.json`, which `transformers` knows how to assemble and the
`tokenizers` library does not. Pulling `transformers` in for a byte-level BPE
would cost more than the model's weights, so the assembly happens here.

The recipe is Qwen2's, which Qwen3-TTS inherits: byte-level BPE, the GPT-2
split pattern below, no prefix space, and the special tokens from the config
added at the ids it gives them. `tests/test_qwen_tokenizer.py` pins the result
against ids taken from a published `tokenizer.json` for the same checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    pre_tokenizers,
    processors,
)
from tokenizers.models import BPE

# Qwen2's pre-tokenizer split, verbatim from the published tokenizer.json.
# Contractions first, then letters, digits, punctuation runs and whitespace —
# the order is the pattern's meaning, so it is copied rather than rewritten.
_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _merges(path: Path) -> list[tuple[str, str]]:
    """Read merges.txt, dropping the version banner git-style tooling writes."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        (left, right)
        for line in lines
        if line and not line.startswith("#version")
        for left, _, right in [line.partition(" ")]
        if right
    ]


def build(directory: Path) -> Tokenizer:
    """Return the checkpoint's text tokenizer.

    Args:
        directory: A model bundle holding `vocab.json`, `merges.txt` and
            `tokenizer_config.json`.

    Returns:
        A tokenizer whose ids match the checkpoint's own.
    """
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer(
        BPE(
            vocab,
            _merges(directory / "merges.txt"),
            fuse_unk=False,
            byte_fallback=False,
        )
    )
    # `use_regex=False` on the ByteLevel stage: the split above already did
    # that job, and leaving both on would split twice.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(_SPLIT), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    config = json.loads(
        (directory / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    # Sorted by id: `add_special_tokens` assigns ids in order, and the control
    # tokens the prompt is built from (`<|im_start|>`, `<tts_pad>`) have to
    # land on the ids the model was trained with.
    added = [
        AddedToken(
            row["content"],
            special=row.get("special", True),
            normalized=row.get("normalized", False),
            lstrip=row.get("lstrip", False),
            rstrip=row.get("rstrip", False),
            single_word=row.get("single_word", False),
        )
        for _, row in sorted(
            config["added_tokens_decoder"].items(), key=lambda item: int(item[0])
        )
    ]
    tokenizer.add_special_tokens(added)
    return tokenizer
