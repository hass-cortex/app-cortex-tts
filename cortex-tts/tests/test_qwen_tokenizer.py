"""The tokenizer Qwen3-TTS ships in pieces, assembled.

Parity with the real checkpoint was established once, out of band: the
assembled tokenizer produced identical ids to a published `tokenizer.json` for
the same checkpoint family on Chinese, English, mixed and whitespace-heavy
probes, at vocab size 151676. That comparison needs 15 MB of vocabulary and is
not a unit test. What is pinned here is the assembly itself — the three places
it can go quietly wrong and still return ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex_speech.engine.qwen_tokenizer import build

# A byte-level vocabulary padded to exactly 100 entries, so the special tokens
# below land on the ids the config gives them rather than wherever the vocab
# happens to end — which is what goes wrong silently on the real checkpoint.
_BASE_TOKENS = ["h", "e", "l", "o", "Ġ", "w", "r", "d", "he", "hel", "Ġw"]
_VOCAB = {token: index for index, token in enumerate(_BASE_TOKENS)}
_VOCAB.update({f"<pad{index}>": index for index in range(len(_BASE_TOKENS), 100)})
_MERGES = "#version: 0.2\nh e\nhe l\nĠ w\n"
_ADDED = {
    "100": {"content": "<|im_start|>", "special": True, "normalized": False},
    "101": {"content": "<|im_end|>", "special": True, "normalized": False},
}


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    (tmp_path / "vocab.json").write_text(json.dumps(_VOCAB), encoding="utf-8")
    (tmp_path / "merges.txt").write_text(_MERGES, encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": _ADDED}), encoding="utf-8"
    )
    return tmp_path


def test_merges_apply_in_order(checkpoint: Path) -> None:
    """`h`+`e`, then `he`+`l`: the longest merge wins, leaving `l` and `o`."""
    tokenizer = build(checkpoint)
    assert tokenizer.encode("hello", add_special_tokens=False).ids == [9, 2, 3]


def test_the_version_banner_is_not_a_merge(checkpoint: Path) -> None:
    """Taking `#version: 0.2` for a merge pair shifts every id after it."""
    tokenizer = build(checkpoint)
    assert tokenizer.get_vocab_size() == 102


def test_a_leading_space_becomes_the_byte_level_marker(checkpoint: Path) -> None:
    """Byte-level encoding is what makes ` w` a token at all."""
    tokenizer = build(checkpoint)
    assert tokenizer.encode(" w", add_special_tokens=False).ids == [10]


def test_special_tokens_land_on_the_ids_the_config_gives(checkpoint: Path) -> None:
    """The prompt is built from these ids; a shifted one is a silent wrong prompt."""
    tokenizer = build(checkpoint)
    ids = tokenizer.encode("<|im_start|>hel<|im_end|>", add_special_tokens=False).ids
    assert ids == [100, 9, 101]


def test_a_round_trip_returns_the_text(checkpoint: Path) -> None:
    """The byte-level decoder has to be wired up too, or ids will not read back."""
    tokenizer = build(checkpoint)
    encoded = tokenizer.encode("hello world", add_special_tokens=False)
    assert tokenizer.decode(encoded.ids) == "hello world"
