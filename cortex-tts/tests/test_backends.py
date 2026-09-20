"""A backend whose extra is not installed says so, as a 503 with the extra's name."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from cortex_speech.engine import backends
from cortex_speech.engine.backends import BackendUnavailableError, BuildContext
from cortex_tts.api.routes import engine_errors


def _context() -> BuildContext:
    return BuildContext.__new__(BuildContext)


def test_a_missing_import_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def builder(context: BuildContext):
        raise ImportError("No module named 'torch'", name="torch")

    monkeypatch.setitem(backends._BUILDERS, "omnivoice", builder)  # noqa: SLF001
    with pytest.raises(BackendUnavailableError) as raised:
        backends.build("omnivoice", _context())
    assert "torch" in str(raised.value)
    assert "`omnivoice` extra" in str(raised.value)


def test_it_reaches_the_wire_as_backend_missing() -> None:
    with pytest.raises(HTTPException) as raised, engine_errors():
        raise BackendUnavailableError("backend 'omnivoice' needs torch")
    assert raised.value.status_code == 503
    assert raised.value.detail["code"] == "BACKEND_MISSING"


class TestOneOnnxRuntimeBuild:
    def test_both_installed_refuses_to_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cortex_speech import providers

        monkeypatch.setattr(
            providers, "installed_builds", lambda: ["onnxruntime", "onnxruntime-gpu"]
        )
        with pytest.raises(RuntimeError) as raised:
            providers.check_installation()
        assert "--group cuda" in str(raised.value)

    def test_one_build_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cortex_speech import providers

        monkeypatch.setattr(providers, "installed_builds", lambda: ["onnxruntime-gpu"])
        providers.check_installation()
