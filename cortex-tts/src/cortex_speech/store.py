"""Writing a small JSON file so a reader never sees half of one.

Three files here are rewritten in place while the app is serving — the
reference index, the settings and the measurements — and each is read back at
startup by a process that must not find a truncated one. Write to a temporary
and rename: on the same filesystem that is atomic, so a reader sees the old
file or the new one.

The temporary's name is the part worth stating. A fixed `.tmp` is itself
shared mutable state: `references.add` runs on a worker thread while `update`
and `remove` stay on the event loop, so two writers would open the same
temporary, truncate each other's bytes in it, and then rename whatever was
left. One name per write costs nothing and removes the question.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def write_json(path: Path, payload: Any, **dumps: Any) -> None:
    """Replace `path` with `payload` as JSON, atomically.

    Args:
        path: The file to replace. Its parent must exist.
        payload: Anything `json.dumps` accepts.
        dumps: Passed through to `json.dumps` — each caller has its own
            preference about indentation and key order, and those are about
            how the file reads, not about how it is written.

    Raises:
        OSError: The write or the rename failed. Callers differ on whether
            that is worth an error to their own caller, so it is not decided
            here.
    """
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, **dumps), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
