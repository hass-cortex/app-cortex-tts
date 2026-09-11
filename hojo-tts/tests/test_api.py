"""API tests that do not need model weights.

Synthesis itself is exercised against real bundles by hand; what is worth
pinning here is everything around it — auth, the shape of the model list, and
the errors a caller sees when a model is missing or the text is unspeakable.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hojo_tts.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client against an empty data directory with auth enabled."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("PRELOAD", "false")
    with TestClient(create_app()) as test_client:
        yield test_client


AUTH = {"Authorization": "Bearer test-key"}


class TestHealth:
    """The probe Home Assistant and the deploy loop rely on."""

    def test_health_needs_no_auth(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_reports_nothing_loaded_on_a_cold_start(
        self, client: TestClient
    ) -> None:
        assert client.get("/health").json()["loaded_models"] == 0


class TestAuth:
    """A configured key is required on /api, and only there."""

    def test_missing_key_is_rejected(self, client: TestClient) -> None:
        response = client.get("/api/models")
        assert response.status_code == 401
        assert response.json()["code"] == "AUTH_REQUIRED"

    def test_wrong_key_is_rejected(self, client: TestClient) -> None:
        response = client.get("/api/models", headers={"Authorization": "Bearer nope"})
        assert response.status_code == 401

    def test_x_api_key_header_is_accepted(self, client: TestClient) -> None:
        assert (
            client.get("/api/models", headers={"X-API-Key": "test-key"}).status_code
            == 200
        )

    def test_ingress_requests_skip_the_key(self, client: TestClient) -> None:
        # Behind ingress the Supervisor has already authenticated the user.
        response = client.get(
            "/api/models", headers={"X-Ingress-Path": "/api/hassio_ingress/x"}
        )
        assert response.status_code == 200


class TestDefaults:
    """What the UI reads to open its pickers where the addon points."""

    def test_defaults_report_the_shipped_names(self, client: TestClient) -> None:
        body = client.get("/api/defaults", headers=AUTH).json()
        assert body == {"model": "hojo-40m", "voice": "hojo_zh_f_01"}

    def test_defaults_follow_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("PRELOAD", "false")
        monkeypatch.setenv("DEFAULT_MODEL", "hojo-80m-clone")
        monkeypatch.setenv("DEFAULT_VOICE", "ref_lounge")
        with TestClient(create_app()) as configured:
            body = configured.get("/api/defaults", headers=AUTH).json()
        assert body == {"model": "hojo-80m-clone", "voice": "ref_lounge"}

    def test_defaults_need_auth(self, client: TestClient) -> None:
        assert client.get("/api/defaults").status_code == 401


class TestModels:
    """The catalog view a fresh install sees."""

    def test_catalog_is_listed_with_nothing_downloaded(
        self, client: TestClient
    ) -> None:
        models = client.get("/api/models", headers=AUTH).json()
        assert {m["id"] for m in models} == {"hojo-40m", "hojo-80m-clone"}
        assert all(m["downloaded"] is False for m in models)
        assert all(m["loaded"] is False for m in models)

    def test_missing_files_are_reported(self, client: TestClient) -> None:
        models = client.get("/api/models", headers=AUTH).json()
        assert all(m["missing_files"] for m in models)

    def test_unknown_model_is_404(self, client: TestClient) -> None:
        response = client.post("/api/models/nope/load", headers=AUTH)
        assert response.status_code == 404
        assert response.json()["code"] == "UNKNOWN_MODEL"

    def test_loading_an_undownloaded_model_is_a_conflict(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/models/hojo-40m/load", headers=AUTH)
        assert response.status_code == 409
        assert response.json()["code"] == "MODEL_NOT_READY"


class TestVoices:
    """Voices come only from models that are actually on disk."""

    def test_no_downloaded_models_means_no_voices(self, client: TestClient) -> None:
        assert client.get("/api/voices", headers=AUTH).json() == []

    def test_unknown_model_filter_is_404(self, client: TestClient) -> None:
        response = client.get("/api/voices", params={"model": "nope"}, headers=AUTH)
        assert response.status_code == 404


class TestPreview:
    """The dry run that shows what the model will be asked to say."""

    def test_preview_applies_the_full_pipeline(self, client: TestClient) -> None:
        response = client.post(
            "/api/preview",
            headers=AUTH,
            json={"text": "現在室內溫度是 26.5°C。"},
        )
        assert response.status_code == 200
        assert response.json()["prepared"] == "现在室内温度是摄氏二十六点五度。"

    def test_preview_respects_the_switches(self, client: TestClient) -> None:
        response = client.post(
            "/api/preview",
            headers=AUTH,
            json={
                "text": "溫度 26.5°C",
                "normalize_text": False,
                "convert_script": False,
            },
        )
        assert response.json()["prepared"] == "溫度 26.5°C。"


class TestSpeakErrors:
    """What a caller sees when synthesis cannot happen."""

    def test_unspeakable_text_is_rejected_before_the_model(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/speak", headers=AUTH, json={"text": "。。。"})
        assert response.status_code == 400
        assert response.json()["code"] == "EMPTY_TEXT"

    def test_empty_text_fails_validation(self, client: TestClient) -> None:
        assert (
            client.post("/api/speak", headers=AUTH, json={"text": ""}).status_code
            == 422
        )

    def test_undownloaded_model_is_a_conflict(self, client: TestClient) -> None:
        response = client.post(
            "/api/speak", headers=AUTH, json={"text": "測試", "model": "hojo-40m"}
        )
        assert response.status_code == 409
        assert response.json()["code"] == "MODEL_NOT_READY"


class TestReferences:
    """Cloned-voice storage and its validation."""

    def test_empty_store_lists_nothing(self, client: TestClient) -> None:
        assert client.get("/api/references", headers=AUTH).json() == []

    def test_unreadable_audio_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/references",
            headers=AUTH,
            data={"name": "x", "transcript": "測試"},
            files={"audio": ("x.wav", b"not audio", "audio/wav")},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "BAD_REFERENCE"

    def test_deleting_an_unknown_reference_is_404(self, client: TestClient) -> None:
        response = client.delete("/api/references/nope", headers=AUTH)
        assert response.status_code == 404

    def test_editing_an_unknown_reference_is_404(self, client: TestClient) -> None:
        response = client.patch(
            "/api/references/nope", headers=AUTH, json={"transcript": "測試"}
        )
        assert response.status_code == 404

    def test_an_empty_transcript_is_rejected(self, client: TestClient) -> None:
        response = client.patch(
            "/api/references/nope", headers=AUTH, json={"transcript": ""}
        )
        assert response.status_code == 422


class TestTemperatureOption:
    """The sampling temperature is configurable per app and per request."""

    def test_default_comes_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hojo_tts import config

        monkeypatch.setenv("TEMPERATURE", "0")
        assert config.load().temperature == 0.0

    def test_a_missing_value_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hojo_tts import config

        monkeypatch.delenv("TEMPERATURE", raising=False)
        assert config.load().temperature == 0.8

    def test_a_malformed_value_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hojo_tts import config

        monkeypatch.setenv("TEMPERATURE", "hot")
        assert config.load().temperature == 0.8

    def test_a_request_may_override_it(self, client: TestClient) -> None:
        # Accepted by the schema; synthesis itself needs a downloaded model.
        response = client.post(
            "/api/speak",
            headers=AUTH,
            json={"text": "測試", "model": "hojo-40m", "temperature": 0},
        )
        assert response.status_code == 409

    def test_an_out_of_range_temperature_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/speak", headers=AUTH, json={"text": "測試", "temperature": 5}
        )
        assert response.status_code == 422
