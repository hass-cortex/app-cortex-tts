"""Runtime settings, read from the environment the s6 run script exports."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from cortex_speech import EXECUTION_PROVIDERS, ExecutionProvider


def _env_provider(name: str) -> ExecutionProvider:
    """Read an execution provider, falling back to `auto` on anything else.

    A typo must not become a silent `cpu`: `auto` is the honest default, and
    what the engines actually got is reported at `/health` either way.
    """
    raw = os.environ.get(name, "").strip().lower()
    return cast(ExecutionProvider, raw) if raw in EXECUTION_PROVIDERS else "auto"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, "").strip() or default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """The few things that must be settled before the process starts.

    Everything a user changes while it runs lives in `preferences.py` — those
    used to be addon options, and each change cost a restart while a change to
    their *schema* cost a rebuild.

    Attributes:
        host: Bind address.
        port: Bind port.
        data_dir: Root for models and state.
        references_dir: Where reference recordings for cloned voices live.
            Defaults to ``references/`` under ``data_dir``; the Home Assistant
            app points it at ``/share`` so user-made recordings are backed up
            and reachable over Samba.
        static_dir: Directory served as the ingress UI.
        api_key: Bearer token required on every ``/api`` route. Empty disables
            authentication, which is only sane behind ingress.
    """

    host: str = "0.0.0.0"
    port: int = 8771
    data_dir: Path = field(default_factory=lambda: Path("/data"))
    static_dir: Path = field(default_factory=lambda: Path("/app/web"))
    api_key: str = ""
    references_dir: Path = field(default_factory=lambda: Path("/data/references"))

    @property
    def models_dir(self) -> Path:
        """Directory holding downloaded model bundles."""
        return self.data_dir / "models"


def load() -> Settings:
    """Build settings from the process environment."""
    data_dir = _env_path("DATA_DIR", "/data")
    return Settings(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8771),
        data_dir=data_dir,
        static_dir=_env_path("STATIC_DIR", "/app/web"),
        api_key=os.environ.get("API_KEY", "").strip(),
        references_dir=_env_path("REFERENCES_DIR", str(data_dir / "references")),
    )
