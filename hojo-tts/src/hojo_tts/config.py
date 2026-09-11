"""Runtime settings, read from the environment the s6 run script exports."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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
    """Everything the app needs to start.

    Attributes:
        host: Bind address.
        port: Bind port.
        data_dir: Root for models, references and state.
        static_dir: Directory served as the ingress UI.
        api_key: Bearer token required on every ``/api`` route. Empty disables
            authentication, which is only sane behind ingress.
        num_threads: ONNX Runtime thread count; 0 lets ORT decide. Scaling is
            flat because the decode loop is Python-bound, so raising it mostly
            costs the rest of the host.
        max_loaded_models: How many engines may stay resident at once.
        default_model: Model used when a request names none.
        default_voice: Voice used when a request names none.
        preload: Load the default model at startup instead of on first use.
        temperature: Sampling temperature. Stopping is a sampled event, so a
            higher value occasionally over-runs the text with an invented
            syllable; 0 is greedy and never does, at the cost of flatter
            prosody.
    """

    host: str = "0.0.0.0"
    port: int = 8771
    data_dir: Path = field(default_factory=lambda: Path("/data"))
    static_dir: Path = field(default_factory=lambda: Path("/app/web"))
    api_key: str = ""
    num_threads: int = 2
    max_loaded_models: int = 1
    default_model: str = "hojo-40m"
    default_voice: str = "hojo_zh_f_01"
    preload: bool = True
    temperature: float = 0.8

    @property
    def models_dir(self) -> Path:
        """Directory holding downloaded model bundles."""
        return self.data_dir / "models"

    @property
    def references_dir(self) -> Path:
        """Directory holding reference recordings for cloned voices."""
        return self.data_dir / "references"


def load() -> Settings:
    """Build settings from the process environment."""
    return Settings(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8771),
        data_dir=_env_path("DATA_DIR", "/data"),
        static_dir=_env_path("STATIC_DIR", "/app/web"),
        api_key=os.environ.get("API_KEY", "").strip(),
        num_threads=_env_int("NUM_THREADS", 2),
        max_loaded_models=_env_int("MAX_LOADED_MODELS", 1),
        default_model=os.environ.get("DEFAULT_MODEL", "").strip() or "hojo-40m",
        default_voice=os.environ.get("DEFAULT_VOICE", "").strip() or "hojo_zh_f_01",
        preload=_env_flag("PRELOAD", True),
        temperature=_env_float("TEMPERATURE", 0.8),
    )
