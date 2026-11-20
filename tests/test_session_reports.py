"""Persistence and report regressions, independent of provider API calls."""
import pytest
from services.db_manager import DatabaseManager
import app as app_module

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "db_manager", DatabaseManager(str(tmp_path / "sessions.db")))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()

def test_profile_session_and_report_roundtrip(client):
    profile = client.post("/api/users", json={"name": "Demo Student", "role": "Student", "age": 22})
    assert profile.status_code == 200
    user_id = profile.json["user"]["id"]
    payload = {"session_id": "demo_session", "user_id": user_id,
               "locked_emotion": "Neutral", "mental_health_score": 40,
               "primary_concern": "Study planning", "transcript": [
                   {"role": "user", "content": "Help me plan my revision."}],
               "summary_insight": "Discussed revision planning."}
    assert client.post("/api/sessions/save", json=payload).json["success"]
    saved = client.get("/api/sessions/demo_session").json["session"]
    assert saved["transcript"] == payload["transcript"]
    assert saved["summary_insight"] == payload["summary_insight"]
    history = client.get(f"/api/users/{user_id}/sessions").json["sessions"]
    assert len(history) == 1
    report = client.get("/session/demo_session")
    assert report.status_code == 200
    assert b"Demo Student" in report.data
    # The report gives a summary for a counsellor, not the transcript (stored with the session).
    assert b"Discussed revision planning." in report.data
    assert b"Help me plan my revision." not in report.data
    assert client.get("/session/unknown").status_code == 404


def test_summary_distinguishes_the_student_from_samvya(client, monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    provider = Mock()
    summary = "The student shared that college feels busy. Samvya suggested prioritizing one task."
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=summary))]
    )
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(app_module, "Groq", lambda **kwargs: provider)
    result = client.post("/api/sessions/summarize", json={
        "transcript": [
            {"role": "user", "content": "College feels busy."},
            {"role": "bot", "content": "Try prioritizing one task."},
        ],
        "user_name": "Samvya",
    })

    assert result.json["summary"] == summary
    system_message, transcript_message = provider.chat.completions.create.call_args.kwargs["messages"]
    assert system_message["role"] == "system"
    assert "Samvya is the AI assistant" in system_message["content"]
    assert json.loads(transcript_message["content"]) == [
        {"speaker": "student", "message": "College feels busy."},
        {"speaker": "Samvya (assistant)", "message": "Try prioritizing one task."},
    ]


def test_report_escapes_saved_user_content(client):
    user = client.post("/api/users", json={"name": "<script>alert(1)</script>"}).json["user"]
    client.post("/api/sessions/save", json={"session_id": "escape_test", "user_id": user["id"],
                "locked_emotion": "Neutral", "summary_insight": "<script>alert(2)</script>"})
    report = client.get("/session/escape_test").get_data(as_text=True)
    assert "<script>alert(" not in report
    assert "&lt;script&gt;" in report

def test_missing_camera_model_is_explicit(client, monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    assert client.get("/api/health").json["emotion_model_available"] is False
    assert client.post("/detect", json={"image": "invalid"}).status_code == 503

def test_home_debug_and_missing_key_chat(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/debug/state?session_id=demo").json["session_id"] == "demo"
    response = client.post("/chat", json={"message": "Hello", "session_id": "demo", "emotion": "Neutral"})
    assert response.status_code == 200
    assert response.json["debug_state"]["session_id"] == "demo"
    assert "API key" in response.json["response"]


@pytest.mark.parametrize("message", [
    "I want to end my life",
    "मैं जीना नहीं चाहता",
    "मला जगायचं नाही",
    "What is the crisis helpline number?",
])
def test_indian_support_available_without_provider(client, message):
    response = client.post("/chat", json={"message": message}).json["response"]
    assert "14416" in response
    assert "112" in response
    assert "API key" not in response


def test_generated_foreign_numbers_are_replaced_before_audio(client, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    provider = Mock()
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="Call ९८८ or 116 123 for help. [[INDIA_SUPPORT]]"
        ))]
    )
    audio = Mock()
    audio.text_to_speech.convert.return_value = [b"example audio"]
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setattr(app_module, "Groq", lambda **kwargs: provider)
    monkeypatch.setattr(app_module, "ElevenLabs", lambda **kwargs: audio)

    reply = client.post("/chat", json={"message": "मला खूप त्रास होतोय"}).json
    assert "भारतात" in reply["response"]
    assert "14416" in reply["response"]
    assert "९८८" not in reply["response"]
    assert "116 123" not in reply["response"]
    # Spoken the same, with helpline numbers read digit by digit.
    spoken = reply["response"].replace("14416", "1 4 4 1 6").replace("112", "1 1 2")
    assert audio.text_to_speech.convert.call_args.kwargs["text"] == spoken


@pytest.mark.parametrize("with_answer", [False, True])
def test_chat_answers_the_latest_marathi_message_after_a_referral(client, monkeypatch, with_answer):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from services.india_support import SUPPORT_CONTACTS

    question = "माझ्याबरोबर कोण बोलत नाही, तुम्ही बोला ना माझ्याबरोबर."
    answer = "कोणी बोलत नाही असं वाटणं कठीण आहे. आज काय घडलं ते सांगाल का?"
    old_referral = SUPPORT_CONTACTS["mr"]
    replies = [f"{answer} {old_referral}"] if with_answer else [old_referral, answer]
    provider = Mock()
    provider.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
        for text in replies
    ]
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(app_module, "Groq", lambda **kwargs: provider)

    response = client.post("/chat", json={
        "message": question,
        "history": [
            {"role": "user", "content": "मला आज एकटं वाटतंय."},
            {"role": "assistant", "content": old_referral},
        ],
        "defer_audio": True,
    })

    assert response.json["response"] == answer
    calls = provider.chat.completions.create.call_args_list
    assert len(calls) == len(replies)
    for call in calls:
        messages = call.kwargs["messages"]
        assert messages[-1] == {"role": "user", "content": question}
        assert all(old_referral not in message["content"] for message in messages)


@pytest.mark.parametrize("request_ticket", [False, True])
def test_voice_reply_is_generated_only_when_requested(client, monkeypatch, request_ticket):
    from types import SimpleNamespace
    from unittest.mock import Mock

    provider = Mock()
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Let's plan one small step."))]
    )
    speech = Mock(return_value=b"example audio")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(app_module, "Groq", lambda **kwargs: provider)
    monkeypatch.setattr(app_module, "synthesize_speech", speech)

    reply = client.post("/chat", json={
        "message": "Can you help me prepare for my exam?",
        "session_id": "voice_reply",
        "defer_audio": request_ticket,
    }).json

    if request_ticket:
        assert reply["audio_base64"] is None
        speech.assert_not_called()
        ticket = reply["speech_ticket"]
        assert client.post("/chat/speech", json={"ticket": ticket + "invalid"}).status_code == 400
        speech.assert_not_called()
        audio = client.post("/chat/speech", json={"ticket": ticket})
        assert audio.status_code == 200
        assert audio.mimetype == "audio/mpeg"
        assert audio.data == b"example audio"
    else:
        assert reply["speech_ticket"] is None
        assert reply["audio_base64"]
    speech.assert_called_once_with(reply["response"], "EXAVITQu4vr4xnSDxMaL")   # female voice by default


def test_old_foreign_referrals_are_replaced_in_saved_reports(client):
    user = client.post("/api/users", json={"name": "Demo Student"}).json["user"]
    old_referral = "For crisis support, call 988 or 116 123."
    app_module.db_manager.save_session({
        "session_id": "old_referral", "user_id": user["id"],
        "summary_insight": old_referral,
        "transcript": [{"role": "assistant", "content": old_referral}],
    })
    report = client.get("/session/old_referral").get_data(as_text=True)
    assert "14416" in report
    assert old_referral not in report


def test_text_only_check_in_saves_without_a_camera_score(client):
    response = client.post("/api/sessions/save", json={
        "session_id": "text_only",
        "locked_emotion": None,
        "mental_health_score": None,
        "transcript": [{"role": "user", "content": "Help me plan my revision."}],
    })
    assert response.status_code == 200
    assert response.json["success"]
    report = client.get("/session/text_only").get_data(as_text=True)
    assert "Not captured" in report
    assert "Help me plan my revision." not in report
    assert "did not include a camera check-in" in report


@pytest.mark.parametrize(("route", "payload"), [
    ("/api/users", {"name": None}),
    ("/api/users", {"name": "Demo Student", "age": []}),
    ("/api/users", {"name": "Demo Student", "age": 121}),
    ("/detect", {"image": 42}),
    ("/chat", {"message": 12}),
    ("/chat", {"message": "Hello", "history": [None]}),
    ("/api/sessions/save", {"session_id": "bad", "locked_emotion": 42}),
    ("/api/sessions/save", {"session_id": "bad", "user_id": {}}),
    ("/api/observation", []),
])
def test_malformed_requests_are_reported_without_server_errors(client, route, payload):
    response = client.post(route, json=payload)
    assert response.status_code == 400
    assert response.json.get("error")


def test_missing_transcription_provider_does_not_create_a_chat_message(client, monkeypatch):
    from io import BytesIO

    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    response = client.post("/transcribe", data={"audio": (BytesIO(b"a" * 500), "voice.webm")})
    assert response.status_code == 503
    assert "transcription" not in response.json


def test_camera_smoothing_is_isolated_by_session(client, monkeypatch):
    import base64
    from types import SimpleNamespace
    import cv2
    import numpy as np
    import torch

    _, encoded = cv2.imencode(".jpg", np.zeros((80, 80, 3), dtype=np.uint8))
    image = base64.b64encode(encoded.tobytes()).decode()
    monkeypatch.setattr(app_module, "model", object())
    monkeypatch.setattr(app_module, "expression_session", None)   # the stubbed Swin output alone
    monkeypatch.setattr(app_module, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(app_module, "face_cascade", SimpleNamespace(detectMultiScale=lambda *a, **k: [(0, 0, 80, 80)]))
    frames = iter([0, 3, 0])
    monkeypatch.setattr(app_module, "run_model", lambda _: torch.nn.functional.one_hot(torch.tensor([next(frames)]), 7).float() * 10)

    def classify(session):
        return client.post("/detect", json={"image": image, "session_id": session}).json

    assert classify("camera-test-a")["emotion"] == "Angry"
    assert classify("camera-test-b")["emotion"] == "Happy"
    assert classify("camera-test-a")["confidence"] > 99


def test_chat_allows_practical_time_ranges_without_allowing_phone_numbers():
    from services.india_support import has_generated_contact

    assert not has_generated_contact("Spend 15–20 minutes on your first task.")
    assert not has_generated_contact("06:30–07:00 अभ्यास करा. मग १०‑१५ मिनिटं विश्रांती घ्या.")
    assert not has_generated_contact("०६:३०–०७:०० अभ्यास करा.")
    assert not has_generated_contact("Call your friend at 6:30 and talk for 10–15 minutes.")
    assert has_generated_contact("Dial 988 for help.")
    assert has_generated_contact("Call 116 123.")
