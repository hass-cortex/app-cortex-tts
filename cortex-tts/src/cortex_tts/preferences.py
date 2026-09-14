"""The settings a user changes while the app is running.

Everything here used to be an addon option, which meant every change went
through the Supervisor and cost a restart — and a change to the *schema* cost
a rebuild. None of these need that: what they configure is either read afresh
on every request or bound when an engine's sessions are created, and the
registry can drop those on demand.

What stays an addon option is what has to be settled before the process
starts: the log level, and the key the Supervisor pushes through discovery.

Stored beside the models rather than in `addon_config/`, because it is state
the app owns and Home Assistant's "remove with data" should sweep up with the
weights.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

from cortex_speech import (
    BY_ID,
    CATALOG,
    EXECUTION_PROVIDERS,
    ExecutionProvider,
    write_json,
)

_LOGGER = logging.getLogger(__name__)

FILE_NAME = "settings.json"


SWITCHES = ("normalize_text", "expand_numbers", "convert_script", "taiwan_readings")


@dataclass(frozen=True)
class TextRule:
    """What the text switches default to for a model, a language, or both.

    A request that leaves a switch out gets the rule's answer; ``None`` in
    the rule leaves the switch to the pipeline — the model's need for number
    words, the language's own rewrites. Rules cascade per switch: the most
    specific rule that says something wins, a model-and-language rule over a
    model-only or language-only one, either over a rule for everything.

    Attributes:
        model: A catalog id, or ``None`` for every model.
        language: A BCP 47 tag the request's language must equal or extend
            (``zh`` covers ``zh-TW``), or ``None`` for every language.
    """

    model: str | None = None
    language: str | None = None
    normalize_text: bool | None = None
    expand_numbers: bool | None = None
    convert_script: bool | None = None
    taiwan_readings: bool | None = None

    def covers(self, model: str, language: str) -> bool:
        if self.model is not None and self.model != model:
            return False
        if self.language is None:
            return True
        tag, want = language.lower(), self.language.lower()
        return tag == want or tag.startswith(want + "-")

    @property
    def specificity(self) -> tuple[int, int]:
        return (self.model is not None, len(self.language or ""))


@dataclass(frozen=True)
class Preferences:
    """How the app behaves, as the user last set it.

    Attributes:
        num_threads: ONNX Runtime threads per synthesis; 0 lets ORT decide.
            More is not faster — the decode loop is Python-bound, and past a
            couple of threads the runtime spends its time synchronising them.
        execution_provider: `auto`, `cpu` or `cuda`. `cuda` refuses to fall
            back, because a host with a GPU quietly on its CPU is the failure
            nobody notices.
        max_loaded_models: How many engines may stay resident at once.
        idle_unload_seconds: Drop a model this long after its last request;
            0 keeps it until something evicts it. The next reply pays the
            load again, which is what makes this a choice: on a card shared
            with another workload the memory is worth more than the seconds.
        max_synthesis_seconds: Refuse a request whose estimated render would
            take longer than this on the measured speed of the chosen model;
            0 accepts any length. What it stops is a reply so long it renders
            past the client's own timeout — the audio finishes into a socket
            nobody is reading, having held the model for minutes.
        default_model: Model used when a request names none.
        default_voice: Voice used when a request names none. Empty takes the
            first the model offers.
        temperature: Sampling temperature for engines that have one.
        preload: Load the default model at startup rather than on first use.
        text_rules: What the text switches default to, per model and
            language, for a request that leaves them out.
    """

    num_threads: int = 2
    execution_provider: ExecutionProvider = "auto"
    max_loaded_models: int = 1
    idle_unload_seconds: int = 0
    max_synthesis_seconds: int = 0
    default_model: str = "hojo-40m"
    default_voice: str = "hojo_zh_f_01"
    temperature: float = 0.8
    preload: bool = True
    text_rules: tuple[TextRule, ...] = ()

    def text_defaults(self, model: str, language: str) -> TextRule:
        """The switches the rules settle for this model and language.

        Per switch, the most specific rule that says something; among rules
        of equal specificity the later one.
        """
        answer: dict[str, bool | None] = dict.fromkeys(SWITCHES)
        covering = sorted(
            (r for r in self.text_rules if r.covers(model, language)),
            key=lambda r: r.specificity,
        )
        for rule in covering:
            for name in SWITCHES:
                value = getattr(rule, name)
                if value is not None:
                    answer[name] = value
        return TextRule(None, None, **answer)

    def merged(self, changes: dict[str, Any]) -> Preferences:
        """Return a copy with `changes` applied, each one validated.

        A field that fails validation keeps its current value rather than
        taking a default: a bad number in one box must not silently reset the
        model someone chose in another.
        """
        return self.validated(changes)[0]

    def validated(self, changes: dict[str, Any]) -> tuple[Preferences, list[str]]:
        """Like `merged`, and also which fields were refused."""
        clean: dict[str, Any] = {}
        ignored: list[str] = []
        for key, value in changes.items():
            if key not in _VALIDATORS:
                continue
            try:
                clean[key] = _VALIDATORS[key](value)
            except (TypeError, ValueError):
                _LOGGER.warning("ignoring %s=%r: not a usable value", key, value)
                ignored.append(key)
        return replace(self, **clean), ignored

    def rebuild_needed(self, other: Preferences) -> bool:
        """Whether moving to `other` invalidates the engines already loaded.

        These two are bound when a session is created; the rest are read
        again on the next request, so changing them needs nothing dropped.
        A smaller resident bound evicts down to it, which is not a rebuild.
        """
        return (
            self.num_threads != other.num_threads
            or self.execution_provider != other.execution_provider
        )


def _threads(value: Any) -> int:
    number = int(value)
    if not 0 <= number <= 16:
        raise ValueError("threads out of range")
    return number


def _loaded(value: Any) -> int:
    # The ceiling is the catalog size rather than a number: a limit below it
    # makes some pair of models unable to be resident together, and the two
    # drifted apart the first time the catalog grew.
    number = int(value)
    if not 1 <= number <= len(CATALOG):
        raise ValueError("resident models out of range")
    return number


def _idle(value: Any) -> int:
    number = int(value)
    if not 0 <= number <= 86400:
        raise ValueError("idle unload out of range")
    return number


def _synthesis_seconds(value: Any) -> int:
    number = int(value)
    if not 0 <= number <= 3600:
        raise ValueError("max synthesis seconds out of range")
    return number


def _provider(value: Any) -> ExecutionProvider:
    text = str(value).strip().lower()
    if text not in EXECUTION_PROVIDERS:
        raise ValueError("unknown execution provider")
    return cast(ExecutionProvider, text)


def _model(value: Any) -> str:
    text = str(value).strip()
    if text not in BY_ID:
        raise ValueError("unknown model")
    return text


def _temperature(value: Any) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("temperature out of range")
    return number


def _flag(value: Any) -> bool:
    """Read a boolean, including the strings a hand-edited file may hold."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"not a boolean: {value!r}")


def _flag_or_none(value: Any) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _flag(value)


def _text_rule(value: Any) -> TextRule:
    if not isinstance(value, dict):
        raise TypeError("a text rule is an object")
    known = {f.name for f in fields(TextRule)}
    if unknown := set(value) - known:
        raise ValueError(f"unknown text rule fields: {sorted(unknown)}")
    model = value.get("model")
    model = _model(model) if model not in (None, "") else None
    language = str(value.get("language") or "").strip().replace("_", "-") or None
    if language is not None and len(language) > 32:
        raise ValueError("language tag too long")
    return TextRule(
        model,
        language,
        **{name: _flag_or_none(value.get(name)) for name in SWITCHES},
    )


def _text_rules(value: Any) -> tuple[TextRule, ...]:
    if not isinstance(value, list):
        raise TypeError("text rules are a list")
    return tuple(_text_rule(rule) for rule in value)


_VALIDATORS: dict[str, Any] = {
    "num_threads": _threads,
    "execution_provider": _provider,
    "max_loaded_models": _loaded,
    "idle_unload_seconds": _idle,
    "max_synthesis_seconds": _synthesis_seconds,
    "default_model": _model,
    # Deliberately unvalidated: a voice only exists once its model is
    # downloaded, and refusing one that is not there yet would make the field
    # impossible to set before the first download. A voice the model does not
    # offer is reported at synthesis, where the model is loaded and knows.
    "default_voice": lambda value: str(value).strip(),
    "temperature": _temperature,
    "preload": _flag,
    "text_rules": _text_rules,
}


def load(data_dir: Path) -> Preferences:
    """Read stored settings, falling back to the defaults above.

    A file that cannot be read is logged and ignored rather than fatal: the
    app starting with defaults is recoverable from the UI, and refusing to
    start is not.
    """
    path = data_dir / FILE_NAME
    if not path.is_file():
        return Preferences()
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        # ValueError covers a malformed file and one that is not UTF-8.
        _LOGGER.error("settings unreadable, starting with defaults: %s", err)
        return Preferences()
    return Preferences().merged(stored if isinstance(stored, dict) else {})


def save(data_dir: Path, preferences: Preferences) -> None:
    """Write settings, replacing the file atomically.

    Raises:
        OSError: The directory is not writable, which the caller reports
            rather than pretending the change was kept.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / FILE_NAME
    write_json(path, asdict(preferences), indent=2, sort_keys=True)
