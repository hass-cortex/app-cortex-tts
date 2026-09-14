"""A corrected transcript reaches the model even when its encoding is cached.

The conditioning cache is keyed by the recording's fingerprint and survives on
disk, so a transcript fixed through PATCH would otherwise be read from the
cached prompt forever on the two engines whose prompt object carries it.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cortex_speech.engine.omni import retranscribed
from cortex_speech.vendor.qwen3_tts_ort import ReferenceConditioning

SRC = Path(__file__).resolve().parent.parent / "src/cortex_speech/engine"


def test_omnivoice_prompt_takes_the_current_transcript() -> None:
    """The vendored prompt is a plain mutable dataclass; a stand-in suffices.

    CI has no torch, so the real `VoiceClonePrompt` is not constructed here.
    """
    prompt = SimpleNamespace(ref_audio_tokens=None, ref_text="舊的", ref_rms=0.1)
    assert retranscribed(prompt, "新的。") is prompt
    assert prompt.ref_text == "新的。"


def test_qwen3_conditioning_carries_the_transcript_and_is_frozen() -> None:
    """Why the engine has to rebuild it rather than assign into it."""
    conditioning = ReferenceConditioning(
        codes=np.zeros((1, 16)), x_vector=np.zeros((1, 1, 4)), transcript="舊的"
    )
    with pytest.raises(FrozenInstanceError):
        conditioning.transcript = "新的。"  # type: ignore[misc]


def test_qwen3_engine_rebuilds_it_on_every_use() -> None:
    source = (SRC / "qwen3.py").read_text(encoding="utf-8")
    assert "replace(conditioning, transcript=reference.transcript)" in source
