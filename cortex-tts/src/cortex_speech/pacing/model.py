"""What one request cost.

A request's cost is recorded raw — audio seconds, wall seconds, the execution
provider it ran on — and the app keeps those (`cortex_tts.stats`) to compute a
real-time factor per model and voice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderSample:
    """One request as it actually went.

    Attributes:
        audio_s: Seconds of audio it produced.
        wall_s: Seconds from asking to the last byte, prefill included, with
            the model already resident — a load is not what a request costs.
        provider: The execution provider the engine ran on (`cpu`, `cuda`).
            A sample from another provider describes another machine.
    """

    audio_s: float
    wall_s: float
    provider: str

    @property
    def rtf(self) -> float:
        """Render seconds over audio seconds, fixed cost included."""
        return self.wall_s / self.audio_s
