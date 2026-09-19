"""Deciding when to render what, so a spoken reply neither stalls nor waits.

Three pieces, all pure: `RenderModel` is what this host has measured about a
model (how long a request takes to render, how fast the voice speaks),
`SentenceBuffer` turns text arriving in pieces into sentences, and `Planner`
turns the two into decisions — send this much now, wait, hold the opening
this long. Nothing here knows about sockets or clocks; the caller supplies
the lead and reads the decision.
"""

from .model import RenderModel, RenderSample
from .planner import (
    BUFFERED,
    PLANNED,
    STREAMING,
    UNHELD,
    Decision,
    Finished,
    Planner,
    Send,
    Wait,
)
from .sentences import SentenceBuffer, clause_pieces, ends_sentence

__all__ = [
    "BUFFERED",
    "PLANNED",
    "STREAMING",
    "UNHELD",
    "Decision",
    "Finished",
    "Planner",
    "RenderModel",
    "RenderSample",
    "Send",
    "SentenceBuffer",
    "Wait",
    "clause_pieces",
    "ends_sentence",
]
