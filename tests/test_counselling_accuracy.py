"""Regressions for safety routing, reply handling and provider limits found in browser testing."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from groq import RateLimitError

import app as app_module
from services.db_manager import DatabaseManager
from services.live_transcription import EndUpgradedConnections
from services.nlp_processor import NLPProcessorService
from services.reply_text import plain_reply

nlp = NLPProcessorService()


def routed_intent(message):
    return nlp.apply_multimodal_fallback(nlp.process(message), {}, None, 0.0).intent


@pytest.mark.parametrize("message", [
    "I feel suicidal",
    "I don't know, I want to die",
    "I never thought I'd want to die but here I am",
    "I’m thinking of taking my own life",
    "I've been cutting myself again",
    "I wish I was dead",
    "Life is not worth living",
    "nobody would care if I was gone",
    "I keep thinking about jumping from the terrace",
    "I swallowed a whole strip of tablets",
    "I've started giving away my things, I won't need them",
    "mujhe marna hai",
    "main jeena nahi chahta",
])
def test_self_harm_messages_reach_the_safety_path(message):
    assert routed_intent(message) == "safety_concern"


@pytest.mark.parametrize("message", [
    "I don't want to die without seeing Ladakh",
    "I'm not suicidal, just tired of exams",
    "This assignment is killing me",
    "I'm cutting myself some slack today",
    "I cut myself off from Instagram for a week",
    "I walked 5 kms today",
    "We watched a movie about suicide prevention in class",
    "I need to take my pills on time",
])
def test_everyday_messages_do_not_reach_the_safety_path(message):
    assert routed_intent(message) != "safety_concern"


@pytest.mark.parametrize("message, intent", [
    ("I'm feeling overwhelmed by my studies.", "academic_pressure"),
    ("I've been feeling low lately.", "general_emotional_distress"),
    ("I feel really lonely in my hostel, nobody talks to me.", "social_isolation"),
    ("I can’t keep up with the syllabus", "academic_pressure"),
    ("I'm not sure, I failed my exam", "academic_performance"),
    ("I'm worried about placements and my career", "career_uncertainty"),
    ("How should I deal with my roommate?", "problem_solving"),
    ("My grandfather died last month and I miss him", "grief_or_loss"),
    ("Yaar exams ki wajah se bahut stress hai, kuch samajh nahi aa raha", "academic_pressure"),
])
def test_common_student_messages_get_their_intent(message, intent):
    assert routed_intent(message) == intent


def test_replies_are_plain_text():
    reply = "Right now:\n\n- **Pulse:** 80.9 BPM  \n- SpO₂: 98.7 %\n\n### Next\n1. *Breathe* slowly."
    assert plain_reply(reply) == "Right now:\n\nPulse: 80.9 BPM\nSpO₂: 98.7 %\n\nNext\n1. Breathe slowly."
    assert plain_reply("2 * 3 = 6 - fine") == "2 * 3 = 6 - fine"


def rate_limited(retry_after="0"):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers={"retry-after": retry_after}, request=request)
    return RateLimitError("Rate limit reached", response=response, body=None)


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture
def chat_client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "db_manager", DatabaseManager(str(tmp_path / "chat.db")))
    monkeypatch.setenv("GROQ_API_KEY", "test")
    provider = Mock()
    monkeypatch.setattr(app_module, "Groq", lambda **kwargs: provider)
    monkeypatch.setattr(app_module.time, "sleep", lambda seconds: None)
    return app_module.app.test_client(), provider


def send(client, message, session_id="accuracy"):
    return client.post("/chat", json={"message": message, "session_id": session_id, "defer_audio": True})


def test_short_rate_limit_is_retried_and_reply_is_plain(chat_client):
    client, provider = chat_client
    provider.chat.completions.create.side_effect = [rate_limited("1"), completion("**Breathe** with me.")]
    response = send(client, "Can we try a grounding exercise?")
    assert response.status_code == 200
    assert response.json["response"] == "Breathe with me."


def test_provider_failure_is_an_error_not_a_counsellor_reply(chat_client):
    client, provider = chat_client
    # Main model, both fallback models, then one short wait and the main model again.
    provider.chat.completions.create.side_effect = [rate_limited()] * 4
    response = send(client, "I have an exam tomorrow.")
    assert response.status_code == 503
    assert "response" not in response.json
    assert "Wait a few seconds" in response.json["error"]

    provider.chat.completions.create.side_effect = [rate_limited("30")] * 3
    assert send(client, "Still there?").status_code == 503
    assert provider.chat.completions.create.call_count == 7


def test_rate_limited_main_model_falls_back_to_another_model(chat_client):
    client, provider = chat_client
    provider.chat.completions.create.side_effect = [rate_limited("30"), completion("Let's take it one step at a time.")]
    response = send(client, "I have an exam tomorrow.")
    assert response.status_code == 200
    assert response.json["response"] == "Let's take it one step at a time."
    first, second = provider.chat.completions.create.call_args_list
    assert first.kwargs["model"] == app_module.GROQ_MODEL
    assert second.kwargs["model"] == app_module.GROQ_FALLBACK_MODELS[0]


def test_fallback_models_get_only_options_they_accept():
    request = {"model": app_module.GROQ_MODEL, "messages": [], "reasoning_effort": "low"}
    assert app_module.model_request(request, "openai/gpt-oss-20b")["reasoning_effort"] == "low"
    other = app_module.model_request(request, "qwen/qwen3.8-27b")
    assert "reasoning_effort" not in other and other["reasoning_format"] == "hidden"
    assert request["model"] == app_module.GROQ_MODEL


def test_support_marker_anywhere_adds_verified_contacts(chat_client):
    client, provider = chat_client
    provider.chat.completions.create.return_value = completion(
        "I'm really glad you told me. [[INDIA_SUPPORT]] Can you stay with someone tonight?")
    reply = send(client, "I feel like ending it").json["response"]
    assert "[[INDIA_SUPPORT]]" not in reply
    assert "Tele MANAS at 14416" in reply


def test_safety_messages_skip_the_model(chat_client):
    client, provider = chat_client
    reply = send(client, "I feel suicidal").json["response"]
    assert "Are you safe right now?" in reply and "14416" in reply
    provider.chat.completions.create.assert_not_called()


def test_safety_reply_is_spoken_in_the_chosen_voice(chat_client):
    client, provider = chat_client
    for voice in ("female", "male"):
        response = client.post("/chat", json={"message": "I feel suicidal", "session_id": "voice-" + voice,
                                              "defer_audio": True, "counsellor_voice": voice}).json
        assert "Are you safe right now?" in response["response"]
        assert app_module.speech_request(response["speech_ticket"]) == (response["response"], voice)
    reply = client.post("/chat", json={"message": "मला जगायचं नाही", "session_id": "voice-mr", "defer_audio": True}).json
    assert "14416" in reply["response"] and reply["speech_ticket"]
    provider.chat.completions.create.assert_not_called()


def test_save_returns_safety_flags_for_the_report(chat_client):
    client, _ = chat_client
    transcript = [{"role": "user", "content": "Honestly I am just a burden to my family"},
                  {"role": "assistant", "content": "I'm here with you."}]
    saved = client.post("/api/sessions/save", json={"session_id": "flagged", "transcript": transcript}).json
    review = saved["safety_review"]
    assert review["flagged"] and review["urgent"]
    assert review["categories"][0]["id"] == "indirect_self_harm"
    calm = client.post("/api/sessions/save", json={"session_id": "calm", "transcript": [
        {"role": "user", "content": "Exams are stressful"}]}).json
    assert calm["safety_review"]["flagged"] is False


def test_public_deployment_accepts_any_host(monkeypatch):
    monkeypatch.setattr(app_module, "ALLOWED_HOSTS", app_module.ALLOWED_HOSTS | {"*"})
    client = app_module.app.test_client()
    assert client.get("/api/health", headers={"Host": "samvya-ai.vercel.app"}).status_code == 200


def test_requests_for_other_hosts_are_refused():
    client = app_module.app.test_client()
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health", headers={"Host": "127.0.0.1:5002"}).status_code == 200
    assert client.get("/api/users", headers={"Host": "rebind.example:5002"}).status_code == 400


def test_speech_retries_once_when_elevenlabs_is_busy(monkeypatch):
    busy = type("ApiError", (Exception,), {"status_code": 429})
    calls = []

    def convert(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise busy()
        return iter([b"mp3"])

    client = SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setattr(app_module, "ElevenLabs", lambda **kwargs: client)
    monkeypatch.setattr(app_module.time, "sleep", lambda seconds: None)
    assert app_module.synthesize_speech("Hello") == b"mp3"
    assert len(calls) == 2
    # Both slots are free again afterwards.
    assert app_module.speech_slots.acquire(blocking=False) and app_module.speech_slots.acquire(blocking=False)
    app_module.speech_slots.release()
    app_module.speech_slots.release()


def test_closed_websocket_gets_no_trailing_http_response():
    def flask_sock_like(environ, start_response):
        start_response("200 OK", [])
        return [b""]

    upgrade = {"PATH_INFO": "/transcribe/live", "werkzeug.socket": object(), "HTTP_UPGRADE": "websocket"}
    with pytest.raises(ConnectionError):
        EndUpgradedConnections(flask_sock_like)(upgrade, lambda *args: None)
    assert EndUpgradedConnections(flask_sock_like)({"PATH_INFO": "/"}, lambda *args: None) == [b""]


def test_report_formats_sensor_vectors():
    assert app_module.format_axes([-1.09, 3.3, -0.001]) == "-1.09 / 3.30 / 0.00"


def test_reply_script_follows_the_student():
    from services.reply_text import wrong_script
    hinglish = "Yaar exams ki wajah se bahut stress hai, kuch samajh nahi aa raha"
    assert wrong_script(hinglish, "Samajh sakti hoon. क्या तुम ठीक हो?")
    assert not wrong_script(hinglish, "Samajh sakti hoon, thoda break lo.")
    assert wrong_script("मुझे बहुत तनाव है", "I understand, that sounds hard.")
    assert not wrong_script("मुझे बहुत तनाव है", "मैं समझ सकती हूँ।")
    # English questions about Hindi words are not Hinglish.
    assert not wrong_script("How do you say thank you in Hindi?", "You can say धन्यवाद.")
    # An English message answered in Hindi (seen after earlier Hindi turns) is regenerated.
    assert wrong_script("What should I do before the exam morning?", "सुबह के लिए एक छोटी सी रूटीन रखो।")
    assert not wrong_script("What does shanti mean?", "शांति means peace or calm.")
    assert not wrong_script("Thank you, that helps.", "You're welcome. Good luck with your exams!")


def test_hinglish_reply_in_devanagari_is_regenerated(chat_client):
    client, provider = chat_client
    provider.chat.completions.create.side_effect = [
        completion("Samajh sakti hoon. क्या तुम ठीक हो?"),
        completion("Samajh sakti hoon. Kya tum theek ho?"),
    ]
    reply = send(client, "Yaar mujhe bahut tension hai exams ki").json["response"]
    assert reply == "Samajh sakti hoon. Kya tum theek ho?"
    assert "English letters" in provider.chat.completions.create.call_args.kwargs["messages"][0]["content"]


def test_emotion_labels_follow_the_checkpoint_class_order():
    # The Swin checkpoint was trained with torchvision ImageFolder, which numbers the class
    # folders alphabetically. Checked on labelled FANE images: this order gets 125/140 right,
    # the earlier order (... Sad, Surprise, Neutral) 73/140, with Neutral reported as Sad.
    assert app_module.EMOTIONS == ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    assert app_module.EMOTIONS == sorted(app_module.EMOTIONS)


def test_voice_pauses_when_the_elevenlabs_quota_is_used_up(monkeypatch):
    # Seen on 2026-11-19: HTTP 401 quota_exceeded for every reply once 10,000 characters were used.
    class QuotaError(Exception):
        status_code = 401
        body = {"detail": {"status": "quota_exceeded", "message": "This request exceeds your quota of 10000."}}

    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.text_to_speech = self

        def convert(self, **kwargs):
            calls.append(kwargs)
            raise QuotaError()

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setattr(app_module, "ElevenLabs", Client)
    monkeypatch.setattr(app_module, "speech_quota_until", 0.0)
    assert app_module.synthesize_speech("Hello") is None
    assert app_module.synthesize_speech("Hello again") is None
    assert len(calls) == 1                      # the second reply does not ask again
    # The reply is still spoken by the next voice (offline Piper here); only if every voice
    # fails does the page fall back to the browser's own voice.
    import services.speech_synthesis as speech
    monkeypatch.setattr(speech, "piper_speech", lambda text, language, gender="female": b"RIFF" + b"0" * 2000)
    client = app_module.app.test_client()
    spoken = client.post("/chat/speech", json={"ticket": app_module.speech_tickets.dumps("Hello")})
    assert spoken.status_code == 200 and spoken.headers["X-Speech-Provider"] == "piper"
    assert len(calls) == 1
    monkeypatch.setattr(speech, "piper_speech", lambda text, language, gender="female": None)
    response = client.post("/chat/speech", json={"ticket": app_module.speech_tickets.dumps("Hello")})
    assert response.status_code == 503 and response.json["fallback"] == "browser"


def test_marathi_and_hindi_are_told_apart():
    from services.reply_text import devanagari_language, wrong_language
    marathi = "मला परीक्षेची खूप भीती वाटते आहे, मी काय करू?"
    hindi = "मुझे नींद नहीं आ रही, क्या करूँ?"
    assert devanagari_language(marathi) == "mr" and devanagari_language(hindi) == "hi"
    assert devanagari_language("ठीक") is None and devanagari_language("hello") is None
    # Seen in testing: this Marathi message was answered in Hindi.
    assert wrong_language(marathi, "मैं समझ सकता हूँ कि परीक्षा का डर कितना भारी हो सकता है।")
    assert not wrong_language(marathi, "मी समजू शकतो, परीक्षेची भीती खूप जड वाटू शकते.")
    assert not wrong_language("ठीक", "मैं समझ सकती हूँ।")   # unclear message: no check


def test_marathi_message_answered_in_hindi_is_regenerated(chat_client):
    client, provider = chat_client
    provider.chat.completions.create.side_effect = [
        completion("मैं समझ सकती हूँ कि परीक्षा का डर कितना भारी हो सकता है।"),
        completion("मी समजू शकते, परीक्षेची भीती खूप जड वाटू शकते."),
    ]
    reply = send(client, "मला परीक्षेची खूप भीती वाटते आहे, मी काय करू?").json["response"]
    assert reply == "मी समजू शकते, परीक्षेची भीती खूप जड वाटू शकते."
    first, second = provider.chat.completions.create.call_args_list
    assert "reply in Marathi (मराठी), not Hindi" in first.kwargs["messages"][0]["content"]
    assert "Write the whole reply in Marathi, not Hindi." in second.kwargs["messages"][0]["content"]
