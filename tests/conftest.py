import os
import sys

import pytest

# Tests must not open the real USB wearable: reading it makes results depend on
# whether a board is plugged in, and opening the port restarts the board.
# app.py reads this when it is imported, and python-dotenv does not override it.
os.environ["SAMVYA_WEARABLE_PORT"] = "off"

# Tests must never reach the paid providers with the real keys in .env: tests that did used
# real ElevenLabs characters and Groq tokens. A test that needs a provider sets a fake key
# with monkeypatch and replaces the client. python-dotenv does not override these values.
for key in ("GROQ_API_KEY", "ELEVENLABS_API_KEY", "SARVAM_API_KEY"):
    os.environ[key] = ""


@pytest.fixture(autouse=True)
def _reset_voice_pause():
    """A test that simulates a used-up voice quota must not pause voice for later tests."""
    yield
    if "app" in sys.modules:
        sys.modules["app"].speech_quota_until = 0.0


@pytest.fixture(autouse=True)
def _no_online_voice(monkeypatch):
    """Tests never call Microsoft's online voices; a test that checks them replaces this."""
    import services.speech_synthesis as speech
    monkeypatch.setattr(speech, "edge_speech", lambda text, language, gender="female": None)
    monkeypatch.setattr(speech, "_edge_chunks", lambda text, language, gender="female": None)
