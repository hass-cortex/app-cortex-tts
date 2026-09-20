"""Deciding what each request of a live reply carries.

All pure: `SentenceBuffer` turns text arriving in pieces into sentences,
`verdict` turns a measured real-time factor into a way of speaking, and
`Pacer` hands out the next request's text under that verdict. Nothing here
knows about sockets or clocks; releasing audio is the transport's.
"""

from .model import RenderSample
from .release import (
    BANK_S,
    BUFFERED,
    STREAM_RTF,
    STREAMING,
    Mode,
    Pacer,
    bank_needed,
    verdict,
)
from .sentences import SentenceBuffer

__all__ = [
    "BANK_S",
    "BUFFERED",
    "Mode",
    "Pacer",
    "RenderSample",
    "STREAMING",
    "STREAM_RTF",
    "SentenceBuffer",
    "bank_needed",
    "verdict",
]
