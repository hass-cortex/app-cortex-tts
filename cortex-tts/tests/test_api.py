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

from cortex_speech import CATALOG, RenderSample
from cortex_tts import preferences
from cortex_tts.api import routes
from cortex_tts.app import create_app
from cortex_tts.preferences import FILE_NAME
from cortex_tts.stats import FILE_NAME as STATS_FILE
from cortex_tts.stats import StatsStore, split_key

INGRESS = {"X-Ingress-Path": "/api/hassio_ingress/x"}
"""What the Supervisor adds; `is_ingress` wants it and the peer both."""

AUTH = {"Authorization": "Bearer test-key"}


def _refused_live(client: TestClient, headers: dict[str, str], **start: object) -> dict:
    """The error frame a live reply is refused with, before any audio exists.

    `check_delivery` runs on the opening frame, so a field the model cannot
    honour is answered here rather than after a download or a render.
    """
    with client, client.websocket_connect("/api/speak/live", headers=headers) as ws:
        ws.send_json({"type": "start", **start})
        frame = ws.receive_json()
    assert frame["type"] == "error", frame
    return frame


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
                    "default_model": "moss-nano",
                    "default_voice": "ref_lounge",
                }
            )
        )
        with TestClient(create_app()) as configured:
            body = configured.get("/api/defaults", headers=AUTH).json()
        assert body == {"model": "moss-nano", "voice": "ref_lounge"}

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

    def _preview_segments(self, client: TestClient, text: str, model: str) -> int:
        response = client.post(
            "/api/preview", headers=AUTH, json={"text": text, "model": model}
        )
        assert response.status_code == 200
        return len(response.json()["segments"])

    def test_preview_cuts_the_text_the_way_the_chosen_model_will(
        self, client: TestClient
    ) -> None:
        """The whole point of the preview is that it is not an approximation.

        How much text one synthesis takes is the model's own, so a preview
        that used one figure for all of them would show a reply cut into
        pieces the engine would never be handed.
        """
        text = "".join(
            f"這是第{n}句話，用來把回覆拉長到超過任何一個模型的上限。"
            for n in range(40)
        )
        counts = {
            model: self._preview_segments(client, text, model)
            for model in ("moss-nano", "hojo-40m")
        }
        # The order of their ceilings: 30 s, 41 s. A larger ceiling is fewer
        # pieces of the same reply.
        assert counts["moss-nano"] > counts["hojo-40m"], counts

    def test_a_model_with_no_established_ceiling_is_not_cut_for_one(
        self, client: TestClient
    ) -> None:
        """OmniVoice declares neither bound, so nothing splits it by length.

        A figure that is neither the model's nor this host's would be wrong on
        some machine (ADR 0007); a live reply is bounded by its sentences.
        """
        text = "".join(
            f"這是第{n}句話，用來把回覆拉長到超過任何一個模型的上限。"
            for n in range(40)
        )
        assert self._preview_segments(client, text, "omnivoice") == 1

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

    def test_preview_lists_the_taiwan_readings_it_applied(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/preview", headers=AUTH, json={"text": "垃圾車來了，請把垃圾拿出去。"}
        )
        body = response.json()
        assert body["prepared"] == "乐色车来了，请把乐色拿出去。"
        assert body["readings"] == [
            {"word": "垃圾车", "standin": "乐色车"},
            {"word": "垃圾", "standin": "乐色"},
        ]

    def test_the_language_picks_the_locale_and_is_reported(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/preview",
            headers=AUTH,
            json={"text": "Es sind 26.5°C.", "language": "de-DE"},
        )
        body = response.json()
        assert body["prepared"] == "Es sind sechsundzwanzig Komma fünf Grad Celsius."
        assert body["language"] == "de-DE"
        # A language without Chinese rewrites lists none, rather than "off";
        # the default model here is Hojo, which cannot say a digit.
        assert body["passes"] == {"normalize_text": True, "expand_numbers": True}

    def test_the_model_decides_whether_the_generic_locale_runs(
        self, client: TestClient
    ) -> None:
        # Nothing in the catalog reads numerals until measured; the field
        # is on the wire so the UI's column follows whatever is declared.
        response = client.post(
            "/api/preview",
            headers=AUTH,
            json={"text": "Es sind 26.5 Grad.", "language": "de", "model": "hojo-40m"},
        )
        assert response.json()["passes"]["normalize_text"] is True
        models = client.get("/api/models", headers=AUTH).json()
        assert all("reads_numerals" in m for m in models)

    def test_a_mainland_tag_keeps_the_mainland_readings(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/preview",
            headers=AUTH,
            json={"text": "垃圾车来了", "language": "zh-CN"},
        )
        body = response.json()
        assert body["prepared"] == "垃圾车来了。"
        assert body["passes"] == {
            "normalize_text": True,
            "expand_numbers": True,
            "convert_script": True,
            "taiwan_readings": False,
        }

    def test_untagged_traditional_text_is_read_as_taiwan_s(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/preview", headers=AUTH, json={"text": "垃圾車來了"}
        )
        body = response.json()
        assert body["language"] == "zh-Hant"
        assert body["prepared"] == "乐色车来了。"

    def test_both_endpoints_leave_the_chinese_switches_to_the_language(self) -> None:
        # The opening frame of a reply and /api/preview must default alike, or
        # a zh-CN call would get Taiwan readings on one and not the other.
        from cortex_tts.api.schemas import LiveStart, PreviewRequest

        for request in (LiveStart(), PreviewRequest(text="x")):
            assert request.convert_script is None
            assert request.taiwan_readings is None

    def test_the_model_decides_whether_a_bare_number_is_read(
        self, client: TestClient
    ) -> None:
        # 110 is an emergency number; read as a quantity it would mislead —
        # unless the model cannot say a digit at all, when digits are noise.
        hojo = client.post(
            "/api/preview", headers=AUTH, json={"text": "撥打 110", "model": "hojo-40m"}
        )
        assert hojo.json()["prepared"] == "拨打一百一十。"
        assert hojo.json()["passes"]["expand_numbers"] is True
        moss = client.post(
            "/api/preview",
            headers=AUTH,
            json={"text": "撥打 110", "model": "moss-nano"},
        )
        assert moss.json()["prepared"] == "拨打 110。"
        assert moss.json()["passes"]["expand_numbers"] is False

    def test_a_settings_rule_answers_what_the_request_left_out(
        self, client: TestClient
    ) -> None:
        saved = client.put(
            "/api/settings",
            headers=AUTH,
            json={
                "text_rules": [
                    {"model": "hojo-40m", "expand_numbers": False},
                    {"language": "zh-TW", "taiwan_readings": False},
                ]
            },
        )
        assert saved.status_code == 200
        assert saved.json()["ignored"] == []
        assert len(saved.json()["settings"]["text_rules"]) == 2
        assert len(client.get("/api/settings", headers=AUTH).json()["text_rules"]) == 2

        ruled = client.post(
            "/api/preview",
            headers=AUTH,
            json={"text": "撥打 110，垃圾車", "model": "hojo-40m", "language": "zh-TW"},
        ).json()
        assert ruled["prepared"] == "拨打 110，垃圾车。"
        assert ruled["passes"]["expand_numbers"] is False
        assert ruled["passes"]["taiwan_readings"] is False
        # The request still has the last word.
        asked = client.post(
            "/api/preview",
            headers=AUTH,
            json={
                "text": "撥打 110，垃圾車",
                "model": "hojo-40m",
                "language": "zh-TW",
                "expand_numbers": True,
                "taiwan_readings": True,
            },
        ).json()
        assert asked["prepared"] == "拨打一百一十，乐色车。"
        # A rule is matched on the language the text resolves to, so a
        # sniffed Traditional text falls under the zh-TW rule as well.
        sniffed = client.post(
            "/api/preview", headers=AUTH, json={"text": "垃圾車", "model": "moss-nano"}
        ).json()
        assert sniffed["language"] == "zh-Hant"
        assert sniffed["passes"]["taiwan_readings"] is True

    def test_the_request_overrides_the_model_on_bare_numbers(
        self, client: TestClient
    ) -> None:
        told_off = client.post(
            "/api/preview",
            headers=AUTH,
            json={"text": "撥打 110", "model": "hojo-40m", "expand_numbers": False},
        )
        assert told_off.json()["prepared"] == "拨打 110。"
        told_on = client.post(
            "/api/preview",
            headers=AUTH,
            json={"text": "撥打 110", "model": "moss-nano", "expand_numbers": True},
        )
        assert told_on.json()["prepared"] == "拨打一百一十。"
        assert told_on.json()["passes"]["expand_numbers"] is True

    def test_preview_can_leave_the_readings_alone(self, client: TestClient) -> None:
        response = client.post(
            "/api/preview",
            headers=AUTH,
            json={"text": "垃圾車來了", "taiwan_readings": False},
        )
        assert response.json()["prepared"] == "垃圾车来了。"
        assert response.json()["readings"] == []


class TestSpeakErrors:
    """What a caller sees when synthesis cannot happen."""

    def test_unspeakable_text_is_rejected_before_the_model(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/audio/speech", headers=AUTH, json={"input": "。。。"}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "EMPTY_TEXT"

    def test_empty_text_fails_validation(self, client: TestClient) -> None:
        assert (
            client.post(
                "/v1/audio/speech", headers=AUTH, json={"input": ""}
            ).status_code
            == 422
        )


class TestACloneOnlyModel:
    """A stored reference makes a voice exist before the model is downloaded.

    Listing voices reads the reference store, not the bundle, so resolving the
    voice succeeds and the load is what fails.
    """

    @pytest.fixture
    def with_reference(self, client: TestClient, reference_wav: bytes) -> TestClient:
        response = client.post(
            "/api/references",
            headers=AUTH,
            data={"name": "Tester", "transcript": "這是一段測試錄音。"},
            files={"audio": ("ref.wav", reference_wav, "audio/wav")},
        )
        assert response.status_code == 201, response.text
        return client

    def test_the_load_is_a_conflict(self, with_reference: TestClient) -> None:
        response = with_reference.post(
            "/v1/audio/speech",
            headers=AUTH,
            json={"input": "你好。", "model": "moss-nano"},
        )
        assert response.status_code == 409
        assert response.json()["code"] == "MODEL_NOT_READY"


class TestTheCardsFigure:
    """What a model card reads, end to end: the median, and the verdict it gives."""

    @staticmethod
    def _measured(tmp_path: Path) -> None:
        store = StatsStore(tmp_path / STATS_FILE)
        for rtf in (0.5, 0.6, 0.55):
            store.record(
                "hojo-40m",
                "builtin",
                "hojo_zh_f_01",
                RenderSample(4.0, rtf * 4.0, "cpu"),
            )

    def test_an_unmeasured_model_reports_nothing(self, client: TestClient) -> None:
        body = client.get("/api/models", headers=AUTH).json()
        assert next(m for m in body if m["id"] == "hojo-40m")["rtf"] == []

    def test_the_figure_is_the_median_and_its_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._measured(tmp_path)
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
        monkeypatch.setenv("API_KEY", "test-key")
        with TestClient(create_app()) as fresh:
            body = fresh.get("/api/models", headers=AUTH).json()
        [entry] = next(m for m in body if m["id"] == "hojo-40m")["rtf"]
        assert entry == {
            "kind": "builtin",
            "voice": "hojo_zh_f_01",
            "rtf": 0.55,
            "samples": 3,
            "provider": "cpu",
            "threshold": 0.8,
            "verdict": "streaming",
        }

    def test_a_voice_spoken_once_is_listed_without_a_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = StatsStore(tmp_path / STATS_FILE)
        store.record(
            "hojo-40m", "builtin", "hojo_zh_f_02", RenderSample(4.0, 2.0, "cpu")
        )
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
        monkeypatch.setenv("API_KEY", "test-key")
        with TestClient(create_app()) as fresh:
            body = fresh.get("/api/models", headers=AUTH).json()
        [entry] = next(m for m in body if m["id"] == "hojo-40m")["rtf"]
        assert (entry["voice"], entry["samples"], entry["verdict"]) == (
            "hojo_zh_f_02",
            1,
            None,
        )

    def test_the_verdict_follows_the_hosts_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Move the line under the measured 0.55 and the same figure is
        buffered; the card says which line it was judged against."""
        self._measured(tmp_path)
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
        monkeypatch.setenv("API_KEY", "test-key")
        with TestClient(create_app()) as fresh:
            fresh.put("/api/settings", headers=AUTH, json={"stream_rtf": 0.5})
            body = fresh.get("/api/models", headers=AUTH).json()
        [entry] = next(m for m in body if m["id"] == "hojo-40m")["rtf"]
        assert (entry["threshold"], entry["verdict"]) == (0.5, "buffered")


class TestWhatAMeasurementCounts:
    """A load is not what a request costs.

    Measured on OmniVoice: a cold first request recorded 14.3 s against a
    steady 4.1 s, and that one sample took the fitted spread from 0.06 s to
    3.16 s — which every later reply waits for, because the spread is added to
    the opening hold. The live path has always made the model resident before
    its clock starts; the file path now does the same.
    """

    def test_the_model_is_resident_before_the_clock_starts(self) -> None:
        source = (
            Path(routes.__file__)
            .read_text(encoding="utf-8")
            .split("async def _synthesize")[1]
        )
        acquire = source.index("registry.acquire")
        clock = source.index("started = time.perf_counter()")
        assert acquire < clock, "a load would be folded into the recorded sample"


class TestWhatACostIsMeasuredPer:
    """Every voice is measured on its own, clones and built-in voices alike."""

    def test_a_key_names_its_kind_and_voice(self) -> None:
        assert split_key("reference:ya-ping") == ("reference", "ya-ping")
        assert split_key("builtin:hojo_zh_f_01") == ("builtin", "hojo_zh_f_01")


class TestResettingStats:
    """RTF is measured per host and goes stale when the host changes; a reset
    lets the next replies re-measure. The endpoints answer even when nothing
    was ever measured, so the UI can offer them without first checking."""

    def test_resetting_one_model_returns_it(self, client: TestClient) -> None:
        response = client.delete("/api/models/hojo-40m/stats", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["id"] == "hojo-40m"
        assert response.json()["rtf"] == []

    def test_resetting_an_unknown_model_is_404(self, client: TestClient) -> None:
        response = client.delete("/api/models/gpt-9/stats", headers=AUTH)
        assert response.status_code == 404

    def test_resetting_all_returns_the_catalog(self, client: TestClient) -> None:
        response = client.delete("/api/stats", headers=AUTH)
        assert response.status_code == 200
        assert {m["id"] for m in response.json()} == {m.id for m in CATALOG}
        assert all(m["rtf"] == [] for m in response.json())

    def test_a_reset_needs_the_key(self, client: TestClient) -> None:
        assert client.delete("/api/stats").status_code == 401

    def test_undownloaded_model_is_a_conflict(self, client: TestClient) -> None:
        response = client.post(
            "/v1/audio/speech",
            headers=AUTH,
            json={"input": "測試", "model": "hojo-40m"},
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

    def test_an_edit_that_changes_nothing_is_rejected(self, client: TestClient) -> None:
        response = client.patch("/api/references/nope", headers=AUTH, json={})
        assert response.status_code == 422

    def test_a_made_up_gender_is_rejected(self, client: TestClient) -> None:
        response = client.patch(
            "/api/references/nope", headers=AUTH, json={"gender": "robot"}
        )
        assert response.status_code == 422

    def test_the_gender_label_can_be_corrected_alone(
        self, client: TestClient, reference_wav: bytes
    ) -> None:
        """Uploads made through the API without a label all read unknown."""
        added = client.post(
            "/api/references",
            headers=AUTH,
            data={"name": "Anna", "transcript": "你好。"},
            files={"audio": ("ref.wav", reference_wav, "audio/wav")},
        ).json()
        assert added["gender"] == "unknown"

        response = client.patch(
            f"/api/references/{added['id']}", headers=AUTH, json={"gender": "female"}
        )
        assert response.status_code == 200
        assert response.json()["gender"] == "female"
        assert response.json()["raw_transcript"] == "你好。"
        listed = client.get("/api/references", headers=AUTH).json()
        assert [r["gender"] for r in listed] == ["female"]

    def test_a_reference_s_language_is_a_whole_tag_and_editable(
        self, client: TestClient, reference_wav: bytes
    ) -> None:
        """The tag is stored whole, and moving it rewrites nothing.

        A reference's language says what the recording was spoken in. It
        picks the text pipeline for whatever the voice is later asked to
        say — but the transcript is a record of the audio, so re-tagging it
        must not re-transcribe it.
        """
        added = client.post(
            "/api/references",
            headers=AUTH,
            data={"name": "Ya", "transcript": "今天要到垃圾。", "language": "zh"},
            files={"audio": ("ref.wav", reference_wav, "audio/wav")},
        ).json()
        assert added["language"] == "zh"
        assert added["transcript"] == "今天要到垃圾。"
        moved = client.patch(
            f"/api/references/{added['id']}", headers=AUTH, json={"language": "zh_TW"}
        )
        assert moved.status_code == 200
        assert moved.json()["language"] == "zh-TW"
        assert moved.json()["transcript"] == added["transcript"]
        listed = client.get("/api/references", headers=AUTH).json()
        assert next(r for r in listed if r["id"] == added["id"])["language"] == "zh-TW"

    def test_renaming_a_reference_does_not_move_its_id(
        self, client: TestClient, reference_wav: bytes
    ) -> None:
        """The id is what a stored pipeline holds, so a rename must not touch it."""
        added = client.post(
            "/api/references",
            headers=AUTH,
            data={"name": "Anna Su", "transcript": "你好。"},
            files={"audio": ("ref.wav", reference_wav, "audio/wav")},
        ).json()
        assert added["id"] == "anna-su"

        renamed = client.patch(
            f"/api/references/{added['id']}", headers=AUTH, json={"name": "蘇小姐"}
        )
        assert renamed.status_code == 200
        assert renamed.json()["name"] == "蘇小姐"
        assert renamed.json()["id"] == "anna-su"
        # And nothing else moved with it.
        assert renamed.json()["raw_transcript"] == "你好。"
        listed = client.get("/api/references", headers=AUTH).json()
        assert [(r["id"], r["name"]) for r in listed] == [("anna-su", "蘇小姐")]

    def test_a_name_erased_falls_back_to_the_id(
        self, client: TestClient, reference_wav: bytes
    ) -> None:
        """The same rule the upload has: no name of its own means the id."""
        added = client.post(
            "/api/references",
            headers=AUTH,
            data={"name": "Anna Su", "transcript": "你好。"},
            files={"audio": ("ref.wav", reference_wav, "audio/wav")},
        ).json()
        renamed = client.patch(
            f"/api/references/{added['id']}", headers=AUTH, json={"name": "   "}
        )
        assert renamed.status_code == 200
        assert renamed.json()["name"] == "anna-su"


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
        frame = _refused_live(client, AUTH, model="hojo-40m", temperature=0)
        assert frame["code"] == "MODEL_NOT_READY"

    def test_an_out_of_range_temperature_is_rejected(self, client: TestClient) -> None:
        """Outside 0-1 the opening frame does not parse, and says which field."""
        frame = _refused_live(client, AUTH, temperature=5)
        assert "temperature" in frame["message"]


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

    def test_idle_unload_is_a_setting(self, client: TestClient) -> None:
        saved = client.put(
            "/api/settings", headers=AUTH, json={"idle_unload_seconds": 300}
        ).json()
        assert saved["settings"]["idle_unload_seconds"] == 300
        assert not saved["reloaded"], "nothing resident needs rebuilding for it"
        assert (
            client.get("/api/settings", headers=AUTH).json()["idle_unload_seconds"]
            == 300
        )

    def test_the_streaming_threshold_is_a_setting_with_a_range(
        self, client: TestClient
    ) -> None:
        saved = client.put(
            "/api/settings", headers=AUTH, json={"stream_rtf": 1.2}
        ).json()
        assert saved["settings"]["stream_rtf"] == 1.2
        assert not saved["reloaded"], "the pacer reads it per reply"
        refused = client.put(
            "/api/settings", headers=AUTH, json={"stream_rtf": 3.5}
        ).json()
        assert refused["ignored"] == ["stream_rtf"]
        assert refused["settings"]["stream_rtf"] == 1.2

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
        response = client.post("/v1/audio/speech", headers=AUTH, json={"input": ""})
        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "VALIDATION"
        assert "input" in body["message"]

    def test_unknown_routes_use_the_shape(self, client: TestClient) -> None:
        response = client.get("/api/nope", headers=AUTH)
        assert response.status_code == 404
        assert set(response.json()) == {"code", "message"}


class TestHealthContract:
    def test_health_carries_the_api_version(self, client: TestClient) -> None:
        assert client.get("/health").json()["api_version"] == 5


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


class TestAKnobTheModelDoesNotHave:
    """A field a model cannot honour is refused, never silently dropped.

    The same reasoning as `NO_TEMPERATURE`: accepting an instruction on a
    model that reads none would be indistinguishable from having worked, and
    the caller would go looking for the fault in the audio. Answered from the
    catalog, so it does not wait behind a download.

    `language` is not such a field: it says what the text is, which the
    pipeline uses on every model, so it is never refused (see TestPreview).
    """

    def test_a_temperature_on_a_model_without_one_is_refused(
        self, client: TestClient
    ) -> None:
        """MOSS fuses sampling into its graph; there is no knob to turn."""
        frame = _refused_live(client, AUTH, model="moss-nano", temperature=0.5)
        assert frame["code"] == "NO_TEMPERATURE"

    def test_an_instruction_on_a_model_without_one_is_refused(
        self, ingress_client: TestClient
    ) -> None:
        frame = _refused_live(
            ingress_client,
            INGRESS,
            model="moss-nano",
            instruct="speak slowly, in a warm tone",
        )
        assert frame["code"] == "NO_STYLE_INSTRUCTION"

    def test_the_cloning_checkpoint_refuses_an_instruction(
        self, ingress_client: TestClient
    ) -> None:
        """Upstream's own feature gate: instruct is a CustomVoice feature and
        the Base checkpoint raises on it."""
        frame = _refused_live(
            ingress_client,
            INGRESS,
            model="moss-nano",
            instruct="speak slowly, in a warm tone",
        )
        assert frame["code"] == "NO_STYLE_INSTRUCTION"

    def test_the_refusal_beats_not_downloaded(self, ingress_client: TestClient) -> None:
        """No model is on disk in the tests, so a 409 here would mean the
        check moved behind the download and stopped being answerable."""
        frame = _refused_live(
            ingress_client, INGRESS, model="omnivoice", instruct="speak slowly"
        )
        assert frame["code"] == "NO_STYLE_INSTRUCTION"

    def test_the_capabilities_are_reported(self, ingress_client: TestClient) -> None:
        """The UI shows a field only where one means something."""
        models = {
            m["id"]: m
            for m in ingress_client.get("/api/models", headers=INGRESS).json()
        }
        assert models["omnivoice"]["language_choice"] is True
        assert models["hojo-40m"]["language_choice"] is False
        assert models["omnivoice"]["style_instruction"] is False
