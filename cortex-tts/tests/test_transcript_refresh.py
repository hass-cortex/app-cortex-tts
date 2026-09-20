"""A corrected transcript reaches the model even when its encoding is cached.

The conditioning cache is keyed by the recording's fingerprint and survives on
disk, so a transcript fixed through PATCH would otherwise be read from the
cached prompt forever on the two engines whose prompt object carries it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cortex_speech.engine.omni import retranscribed

SRC = Path(__file__).resolve().parent.parent / "src/cortex_speech/engine"


def test_omnivoice_prompt_takes_the_current_transcript() -> None:
    """The vendored prompt is a plain mutable dataclass; a stand-in suffices.

    CI has no torch, so the real `VoiceClonePrompt` is not constructed here.
    """
    prompt = SimpleNamespace(ref_audio_tokens=None, ref_text="舊的", ref_rms=0.1)
    assert retranscribed(prompt, "新的。") is prompt
    assert prompt.ref_text == "新的。"
