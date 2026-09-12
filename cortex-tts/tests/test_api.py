"""API tests that do not need model weights.

Synthesis itself is exercised against real bundles by hand; what is worth
pinning here is everything around it — auth, the shape of the model list, and
the errors a caller sees when a model is missing or the text is unspeakable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cortex_speech import CATALOG
from cortex_tts import preferences
from cortex_tts.app import create_app
from cortex_tts.preferences import FILE_NAME

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

    def test_ingress_requests_skip_the_key(self, ingress_client: TestClient) -> None:
        # Behind ingress the Supervisor has already authenticated the user.
        response = ingress_client.get(
            "/api/models", headers={"X-Ingress-Path": "/api/hassio_ingress/x"}
        )
        assert response.status_code == 200

    def test_ingress_header_from_another_peer_is_not_ingress(
        self, client: TestClient
    ) -> None:
        """The header is free to forge; the Supervisor's address is not."""
        response = client.get(
            "/api/models", headers={"X-Ingress-Path": "/api/hassio_ingress/x"}
        )
        assert response.status_code == 401

    def test_supervisor_peer_without_the_header_still_needs_the_key(
        self, ingress_client: TestClient
    ) -> None:
        assert ingress_client.get("/api/models").status_code == 401

    def test_non_ascii_token_is_rejected_not_a_server_error(
        self, client: TestClient
    ) -> None:
        response = client.get(
            "/api/models", headers={"Authorization": "Bearer kéy".encode("latin-1")}
        )
        assert response.status_code == 401


class TestDefaults:
    """What the UI reads to open its pickers where the addon points."""

    def test_defaults_report_the_shipped_names(self, client: TestClient) -> None:
        body = client.get("/api/defaults", headers=AUTH).json()
        assert body == {"model": "hojo-40m", "voice": "hojo_zh_f_01"}

    def test_defaults_follow_what_was_saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """They are stored settings now, not addon options."""
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
        monkeypatch.setenv("API_KEY", "test-key")
        (tmp_path / FILE_NAME).write_text(
            json.dumps(
                {
                    "preload": False,
                    "default_model": "hojo-80m-clone",
                    "default_voice": "ref_lounge",
                }
            )
        )
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
        # Derived from the catalog rather than pinned to today's contents: the
        # endpoint's job is to reflect it, and adding a model is not a break.
        assert {m["id"] for m in models} == {spec.id for spec in CATALOG}
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

    def test_default_comes_from_what_was_saved(self, tmp_path: Path) -> None:
        (tmp_path / FILE_NAME).write_text(json.dumps({"temperature": 0}))
        assert preferences.load(tmp_path).temperature == 0.0

    def test_no_saved_value_falls_back(self, tmp_path: Path) -> None:
        assert preferences.load(tmp_path).temperature == 0.8

    def test_a_malformed_value_falls_back(self, tmp_path: Path) -> None:
        """One bad field must not take the others down with it."""
        (tmp_path / FILE_NAME).write_text(
            json.dumps({"temperature": "hot", "default_model": "moss-nano"})
        )
        saved = preferences.load(tmp_path)
        assert saved.temperature == 0.8
        assert saved.default_model == "moss-nano"

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


class TestSettingsEndpoint:
    """Settings are changed here now, not in the addon's options form.

    The addon keeps only what must be settled before the process starts. Every
    other change used to cost a restart, and a change to the *list of choices*
    cost a rebuild — which on this host once took the service offline when the
    build failed on a transient network error.
    """

    def test_it_reports_what_is_in_force(self, client: TestClient) -> None:
        body = client.get("/api/settings", headers=AUTH).json()
        assert body["default_model"] == "hojo-40m"
        assert body["execution_provider"] == "auto"

    def test_a_change_is_readable_immediately(self, client: TestClient) -> None:
        client.put("/api/settings", headers=AUTH, json={"default_model": "moss-nano"})
        assert (
            client.get("/api/settings", headers=AUTH).json()["default_model"]
            == "moss-nano"
        )

    def test_it_reaches_the_defaults_the_ui_reads(self, client: TestClient) -> None:
        """`/api/defaults` is what opens the pickers; a stale one sends the
        user to a voice they did not choose."""
        client.put("/api/settings", headers=AUTH, json={"default_voice": "Yuewen"})
        assert client.get("/api/defaults", headers=AUTH).json()["voice"] == "Yuewen"

    def test_it_survives_a_restart(self, client: TestClient, tmp_path: Path) -> None:
        client.put("/api/settings", headers=AUTH, json={"temperature": 0.0})
        assert preferences.load(tmp_path).temperature == 0.0

    def test_omitted_fields_keep_their_value(self, client: TestClient) -> None:
        """The UI sends one box, not the whole form."""
        client.put("/api/settings", headers=AUTH, json={"default_model": "moss-nano"})
        client.put("/api/settings", headers=AUTH, json={"temperature": 0.5})
        body = client.get("/api/settings", headers=AUTH).json()
        assert body == {**body, "default_model": "moss-nano", "temperature": 0.5}

    def test_a_bad_value_does_not_discard_a_good_one(self, client: TestClient) -> None:
        response = client.put(
            "/api/settings",
            headers=AUTH,
            json={"num_threads": 99, "default_model": "moss-nano"},
        )
        assert response.status_code == 200
        assert response.json()["settings"]["num_threads"] == 2
        assert response.json()["settings"]["default_model"] == "moss-nano"

    def test_a_session_bound_change_says_so(self, client: TestClient) -> None:
        """`reloaded` is what the UI turns into "takes effect on the next
        reply" rather than claiming it is already speaking differently."""
        body = client.put(
            "/api/settings", headers=AUTH, json={"execution_provider": "cpu"}
        ).json()
        assert body["reloaded"] is False, "nothing was loaded to drop"
        assert body["settings"]["execution_provider"] == "cpu"

    def test_it_needs_auth(self, client: TestClient) -> None:
        assert client.get("/api/settings").status_code == 401
        assert client.put("/api/settings", json={}).status_code == 401


class TestErrorShape:
    """Every error is `{"code", "message"}`, whoever raised it."""

    def test_validation_errors_use_the_shape(self, client: TestClient) -> None:
        response = client.post("/api/speak", headers=AUTH, json={"text": ""})
        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "VALIDATION"
        assert "text" in body["message"]

    def test_unknown_routes_use_the_shape(self, client: TestClient) -> None:
        response = client.get("/api/nope", headers=AUTH)
        assert response.status_code == 404
        assert set(response.json()) == {"code", "message"}


class TestHealthContract:
    def test_health_carries_the_api_version(self, client: TestClient) -> None:
        assert client.get("/health").json()["api_version"] == 1


class TestSettingsReporting:
    def test_a_refused_field_is_named(self, client: TestClient) -> None:
        response = client.put(
            "/api/settings",
            headers=AUTH,
            json={"execution_provider": "tpu", "temperature": 0.5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ignored"] == ["execution_provider"]
        assert body["settings"]["execution_provider"] == "auto"
        assert body["settings"]["temperature"] == 0.5


class TestDeletingAModel:
    def test_refused_while_its_download_runs(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = client.app.state.cortex  # type: ignore[attr-defined]
        monkeypatch.setattr(state.downloads, "is_running", lambda model_id: True)
        response = client.delete("/api/models/hojo-40m", headers=AUTH)
        assert response.status_code == 409
        assert response.json()["code"] == "DOWNLOAD_RUNNING"

    def test_an_absent_bundle_deletes_cleanly(self, client: TestClient) -> None:
        response = client.delete("/api/models/hojo-40m", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["downloaded"] is False


class TestReferenceValidation:
    def test_an_unpronounceable_transcript_is_refused_and_nothing_is_stored(
        self, client: TestClient, reference_wav: bytes, tmp_path: Path
    ) -> None:
        response = client.post(
            "/api/references",
            headers=AUTH,
            data={"name": "dots", "transcript": "。。。"},
            files={"audio": ("ref.wav", reference_wav, "audio/wav")},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "BAD_REFERENCE"
        assert not list((tmp_path / "references").glob("*.wav"))

    def test_a_long_latin_transcript_keeps_its_spaces(
        self, client: TestClient, reference_wav: bytes
    ) -> None:
        transcript = ("This is a sentence about the reference recording. " * 6).strip()
        response = client.post(
            "/api/references",
            headers=AUTH,
            data={"name": "long", "transcript": transcript, "language": "en"},
            files={"audio": ("ref.wav", reference_wav, "audio/wav")},
        )
        assert response.status_code == 201, response.text
        assert "recording.This" not in response.json()["transcript"]
