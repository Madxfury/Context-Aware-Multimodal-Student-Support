"""Spoken replies: language detection and the provider order with free fallbacks."""
import pytest

import app as app_module
import services.speech_synthesis as speech
from services.india_support import SAFETY_CHECK_INS, SUPPORT_CONTACTS

MARATHI = "तुम्ही हे सांगितलं याचं मला बरं वाटलं. चला, आपण एक छोटा श्वसनाचा व्यायाम करूया."
HINDI = "मुझे सुनकर अच्छा लगा कि आपने बात की। चलिए एक छोटी सी साँस लेने की कसरत करते हैं।"


@pytest.mark.parametrize("text, language", [
    (MARATHI, "mr"), (HINDI, "hi"),
    (SUPPORT_CONTACTS["mr"], "mr"), (SAFETY_CHECK_INS["mr"], "mr"),
    (SUPPORT_CONTACTS["hi"], "hi"), (SAFETY_CHECK_INS["hi"], "hi"),
    ("ठीक आहे, सांगा.", "mr"), ("ठीक है, बताइए।", "hi"),
    ("I hear you. Let's try a slow breath together.", "en"),
    ("Main samajh sakta hoon, exams ka pressure bahut hota hai.", "en"),   # Hindi in English letters
])
def test_reply_language(text, language):
    assert speech.reply_language(text) == language


def test_hindi_and_marathi_use_native_voices_first_and_english_uses_elevenlabs_first(monkeypatch):
    monkeypatch.setattr(speech, "edge_speech", lambda text, language, gender="female": f"edge-{language}-{gender}".encode())
    eleven = lambda text, voice_id: f"elevenlabs-{voice_id}".encode()
    assert speech.synthesize(MARATHI, eleven) == (b"edge-mr-female", "audio/mpeg", "edge")
    assert speech.synthesize(HINDI, eleven, "male") == (b"edge-hi-male", "audio/mpeg", "edge")
    assert speech.synthesize("Hello there.", eleven) == (b"elevenlabs-EXAVITQu4vr4xnSDxMaL", "audio/mpeg", "elevenlabs")
    assert speech.synthesize("Hello there.", eleven, "male") == (b"elevenlabs-nPczCjzI2devNBz1zQrb", "audio/mpeg", "elevenlabs")
    # ElevenLabs unavailable (for example, its quota is used up): the free neural voice speaks.
    assert speech.synthesize("Hello there.", lambda text, voice_id: None) == (b"edge-en-female", "audio/mpeg", "edge")
    assert speech.synthesize("Hello", eleven, "robot")[0] == b"elevenlabs-EXAVITQu4vr4xnSDxMaL"   # unknown -> female
    monkeypatch.setenv("SAMVYA_TTS_ORDER", "piper")
    monkeypatch.setattr(speech, "piper_speech", lambda text, language, gender="female": b"RIFFwav")
    assert speech.synthesize(MARATHI, eleven) == (b"RIFFwav", "audio/wav", "piper")


def test_voices_exist_for_both_genders_in_all_three_languages():
    for gender in speech.GENDERS:
        assert set(speech.EDGE_VOICES[gender]) == set(speech.PIPER_VOICES[gender]) == {"en", "hi", "mr"}
        assert set(speech.SARVAM_VOICES[gender]) == {"en", "hi", "mr"}
    assert speech.EDGE_VOICES["male"]["mr"] == "mr-IN-ManoharNeural" and speech.EDGE_VOICES["female"]["hi"] == "hi-IN-SwaraNeural"


def test_streaming_starts_with_the_first_chunk_and_falls_back(monkeypatch):
    monkeypatch.setattr(speech, "_edge_chunks", lambda text, language, gender="female": iter([b"ID3", b"more"]))
    chunks, mimetype, provider = speech.stream(MARATHI, lambda text, voice_id: None, "male")
    assert provider == "edge" and mimetype == "audio/mpeg" and b"".join(chunks) == b"ID3more"
    monkeypatch.setattr(speech, "_edge_chunks", lambda text, language, gender="female": None)
    monkeypatch.setattr(speech, "piper_speech", lambda text, language, gender="female": b"RIFF" + b"0" * 2000)
    chunks, mimetype, provider = speech.stream(MARATHI, lambda text, voice_id: None)
    assert provider == "piper" and mimetype == "audio/wav"


def test_offline_piper_voice_speaks_marathi_when_online_voices_fail():
    if not all(speech.os.path.isfile(speech.os.path.join(speech.PIPER_DIR, name + ".onnx"))
               for voices in speech.PIPER_VOICES.values() for name in voices.values()):
        pytest.skip("Piper voices not installed")
    audio, mimetype, provider = speech.synthesize(MARATHI, lambda text, voice_id: None)
    assert provider == "piper" and mimetype == "audio/wav" and audio[:4] == b"RIFF" and len(audio) > 50_000


def test_speech_route_returns_the_providers_audio_type(monkeypatch):
    monkeypatch.setattr(speech, "synthesize", lambda text, eleven, gender="female": (b"RIFFwav", "audio/wav", "piper"))
    ticket = app_module.speech_tickets.dumps("नमस्कार")
    response = app_module.app.test_client().post("/chat/speech", json={"ticket": ticket})
    assert response.status_code == 200 and response.mimetype == "audio/wav"
    assert response.headers["X-Speech-Provider"] == "piper"


def test_chat_uses_the_chosen_counsellor_voice(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from services.db_manager import DatabaseManager
    monkeypatch.setattr(app_module, "db_manager", DatabaseManager(str(tmp_path / "voice.db")))
    monkeypatch.setenv("GROQ_API_KEY", "test")
    provider = Mock()
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="मी समजू शकतो."))])
    monkeypatch.setattr(app_module, "Groq", lambda **_: provider)
    client = app_module.app.test_client()
    reply = client.post("/chat", json={"message": "मला भीती वाटते", "session_id": "voice", "defer_audio": True,
                                       "counsellor_voice": "male"}).json
    prompt = provider.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "You speak as a male counsellor" in prompt and "मी समजू शकतो" in prompt
    assert app_module.speech_request(reply["speech_ticket"]) == ("मी समजू शकतो.", "male")
    monkeypatch.setattr(speech, "stream", lambda text, eleven, gender="female": (iter([b"ID3", gender.encode()]), "audio/mpeg", "edge"))
    streamed = client.get("/chat/speech/stream", query_string={"ticket": reply["speech_ticket"]})
    assert streamed.status_code == 200 and streamed.data == b"ID3male" and streamed.headers["X-Speech-Provider"] == "edge"
    assert client.get("/chat/speech/stream", query_string={"ticket": "tampered"}).status_code == 400


# Captured at import, before tests/conftest.py replaces them for every test.
REAL_SARVAM_SPEECH, REAL_SARVAM_CHUNKS = speech.sarvam_speech, speech._sarvam_chunks


class FakeSarvam:
    """Stands in for the SarvamAI client; records requests and never uses the network."""
    def __init__(self, error=None):
        self.requests, self.error = [], error
        self.text_to_speech = self

    def convert(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return type("Response", (), {"audios": [speech.base64.b64encode(b"ID3sarvam").decode()]})()

    def convert_stream(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return iter([b"", b"ID3", b"stream"])


def test_sarvam_voice_comes_first_in_every_language(monkeypatch):
    fake = FakeSarvam()
    monkeypatch.setattr(speech, "_sarvam_client", lambda: fake)
    monkeypatch.setattr(speech, "sarvam_speech", REAL_SARVAM_SPEECH)
    monkeypatch.setattr(speech, "_sarvam_chunks", REAL_SARVAM_CHUNKS)
    eleven = lambda text, voice_id: b"elevenlabs"
    assert speech.synthesize("Hello there.", eleven) == (b"ID3sarvam", "audio/mpeg", "sarvam")
    assert speech.synthesize(MARATHI, eleven, "male")[2] == "sarvam"
    chunks, mimetype, provider = speech.stream(HINDI, eleven)
    assert provider == "sarvam" and mimetype == "audio/mpeg" and b"".join(chunks) == b"ID3stream"
    languages = [(r["target_language_code"], r["speaker"]) for r in fake.requests]
    assert languages == [("en-IN", "pooja"), ("mr-IN", "aditya"), ("hi-IN", "pooja")]
    assert speech.SARVAM_VOICES["female"]["mr"] == "rupali"
    assert {r["model"] for r in fake.requests} == {"bulbul:v3"}


def test_refused_sarvam_account_pauses_and_the_next_voice_speaks(monkeypatch):
    from sarvamai.core.api_error import ApiError
    fake = FakeSarvam(ApiError(status_code=403, body="insufficient credits"))
    clients = []
    monkeypatch.setenv("SARVAM_API_KEY", "test")
    import sarvamai
    monkeypatch.setattr(sarvamai, "SarvamAI", lambda **kwargs: clients.append(kwargs) or fake)
    monkeypatch.setattr(speech, "sarvam_speech", REAL_SARVAM_SPEECH)
    monkeypatch.setattr(speech, "_sarvam_chunks", REAL_SARVAM_CHUNKS)
    monkeypatch.setattr(speech, "edge_speech", lambda text, language, gender="female": b"edge")
    assert speech.synthesize(HINDI, lambda text, voice_id: None) == (b"edge", "audio/mpeg", "edge")
    assert speech.synthesize(HINDI, lambda text, voice_id: None)[2] == "edge"
    assert len(clients) == 1                  # paused after the refusal: not asked again


def test_sarvam_without_a_key_is_skipped(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "")
    assert REAL_SARVAM_SPEECH("Hello", "en") is None and REAL_SARVAM_CHUNKS("Hello", "en") is None
