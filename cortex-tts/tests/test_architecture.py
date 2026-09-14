"""The boundary between the speech library and the Home Assistant app.

`cortex_speech` is the library: models, engines, text preparation, references.
`cortex_tts` is the app that serves it to Home Assistant. The dependency runs
one way, and the app talks to the library through its facade rather than
reaching into submodules.

A boundary that is only written down erodes. These tests are what make it cost
something to cross, and they are cheap: they read the AST, they load nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
LIBRARY = "cortex_speech"
APP = "cortex_tts"

# The app is allowed to import the Supervisor-facing modules it owns, and the
# library is allowed to import anything that is not the app.
FORBIDDEN_IN_LIBRARY = {"fastapi", "starlette", "aiohttp", "uvicorn", APP}


def _modules(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


def _imported_names(tree: ast.AST, path: Path) -> list[str]:
    """Return every module this file imports, as absolute dotted names."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Resolve a relative import to the absolute module it means.
                parts = path.relative_to(SRC).with_suffix("").parts
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                base = parts[: len(parts) - node.level + 1]
                names.append(".".join([*base, node.module or ""]).rstrip("."))
            else:
                names.append(node.module or "")
    return [n for n in names if n]


def _parsed(package: str) -> list[tuple[Path, list[str]]]:
    out = []
    for path in _modules(package):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        out.append((path, _imported_names(tree, path)))
    return out


def test_library_does_not_import_the_app() -> None:
    """The speech library must not know the Home Assistant app exists."""
    offenders = [
        f"{path.relative_to(SRC)} imports {name}"
        for path, names in _parsed(LIBRARY)
        for name in names
        if name == APP or name.startswith(f"{APP}.")
    ]
    assert not offenders, "library depends on the app:\n  " + "\n  ".join(offenders)


def test_library_has_no_web_or_supervisor_dependencies() -> None:
    """The library stays runnable without a web framework or a Supervisor.

    `notifications.py` exists because this rule does: firing an event on the
    Home Assistant bus is the app's answer to a library-level fact.
    """
    offenders = [
        f"{path.relative_to(SRC)} imports {name}"
        for path, names in _parsed(LIBRARY)
        for name in names
        if name.split(".")[0] in FORBIDDEN_IN_LIBRARY
    ]
    assert not offenders, "library reaches for the app's world:\n  " + "\n  ".join(
        offenders
    )


def test_app_uses_the_library_facade_only() -> None:
    """The app imports `cortex_speech`, never `cortex_speech.something`.

    Reaching into a submodule turns the library's internal layout into the
    app's problem, which is the coupling this boundary exists to prevent.
    """
    offenders = [
        f"{path.relative_to(SRC)} imports {name}"
        for path, names in _parsed(APP)
        for name in names
        if name.startswith(f"{LIBRARY}.")
    ]
    assert not offenders, (
        "app reached past the facade — import the name from `cortex_speech` "
        "instead, adding it to `__all__` if it is missing:\n  " + "\n  ".join(offenders)
    )


def test_facade_exports_resolve() -> None:
    """Every name the facade promises actually exists.

    A stale `__all__` entry is only found at import time by whoever imports it,
    which in production is the app at startup.
    """
    import cortex_speech

    missing = [n for n in cortex_speech.__all__ if not hasattr(cortex_speech, n)]
    assert not missing, f"__all__ names nothing: {missing}"


@pytest.mark.parametrize("package", [LIBRARY, APP])
def test_every_module_parses(package: str) -> None:
    """Guard against a half-applied rename leaving a file unparseable.

    `_parsed` is what does the parsing; asserting on `_modules` alone would
    pass on a package full of syntax errors.
    """
    assert _parsed(package), f"no modules found for {package}"


# The surface the facade promises: a service, the config it takes, and the value
# types that cross the boundary. Listed rather than derived, so widening it is
# a deliberate edit to this list and shows up in review — the import rule above
# cannot see a flat re-export, which is how the surface grew to 35 names
# unnoticed.
FACADE = {
    # the service and its configuration
    "SpeechService",
    "SpeechConfig",
    # catalog value types
    "CATALOG",
    "BY_ID",
    "ModelSpec",
    "ModelState",
    # synthesis value types
    "Voice",
    "Synthesis",
    "Delivery",
    "EngineRegistry",
    "StreamingEngine",
    # references
    "Reference",
    "ReferenceStore",
    "MAX_REFERENCE_SECONDS",
    # downloads
    "DownloadManager",
    "DownloadProgress",
    # text path, which the API exposes directly as /api/preview
    "TextOptions",
    "NormalizeOptions",
    "prepare",
    "prepared_text",
    # audio encoding for the HTTP layer
    "AudioFormat",
    "CONTENT_TYPES",
    "encode",
    "estimated_audio_seconds",
    "decode_reference",
    "level",
    "StreamGain",
    "STREAM_ENCODERS",
    "MP3_BITRATE",
    "wav_header",
    # which ONNX Runtime provider the engines ask for, and what they got
    "EXECUTION_PROVIDERS",
    "ExecutionProvider",
    "ProviderUnavailableError",
    # notifications
    "notify_models_changed",
    "subscribe_models_changed",
    # errors
    "EngineError",
    "NoAudioError",
    "UnknownVoiceError",
    "write_json",
    "UnsupportedLanguageError",
    "UnknownModelError",
    "ModelNotReadyError",
    "OutOfMemoryError",
    "ReferenceError",
}


def test_the_facade_exports_exactly_what_it_promises() -> None:
    """The surface is pinned, because an import rule cannot police its width.

    `test_app_uses_the_library_facade_only` stops the app reaching into a
    submodule, and a flat re-export satisfies it perfectly — so the library can
    hand out anything it likes and the boundary still reads as enforced.
    """
    import cortex_speech

    actual = set(cortex_speech.__all__)
    assert actual == FACADE, (
        "facade surface changed — added: "
        f"{sorted(actual - FACADE)}, removed: {sorted(FACADE - actual)}. "
        "Widening it is a decision, so update this list on purpose."
    )


def test_the_app_never_names_the_library_s_directory_layout() -> None:
    """Where a bundle lives is the library's business.

    The app owns a `data_dir` setting — Home Assistant decides where `/data` is
    — and hands it over once when it builds `SpeechConfig`. What it must not do
    is keep using it to call the library, because that is the same fact living
    in two places and drifting independently.

    So the rule is about the combination, not the name: `settings.data_dir`
    passed as an argument. `config.py` may define it and `app.py` may translate
    it at the boundary — as may `__main__.py`, which has to read the stored
    settings before anything numeric is imported.
    """
    boundary = {"config.py", "app.py", "__main__.py"}
    offenders = [
        f"{path.relative_to(SRC)}:{line_no} {line.strip()}"
        for path in _modules(APP)
        if path.name not in boundary
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "settings.data_dir" in line and not line.lstrip().startswith("#")
    ]
    assert not offenders, "app reaches into the library's layout:\n  " + "\n  ".join(
        offenders
    )
