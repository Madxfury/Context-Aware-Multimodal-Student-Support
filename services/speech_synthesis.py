"""Spoken replies in English, Hindi and Marathi, in a female or male counsellor voice.

Providers, tried in order for the reply's language:
- ElevenLabs (paid; the free plan allows 10,000 characters a month).
- Microsoft neural voices through the open-source edge-tts library (free, online): native
  Indian voices for English, Hindi and Marathi. edge-tts uses the Read Aloud service of the
  Edge browser, which is not an official API and may change.
- Piper (open source, offline, on this computer) with Indian Hindi and Marathi voices.
The browser's own voice is the last resort (static/js/main.js).

Pronunciation was checked by transcribing test sentences with Sarvam speech-to-text: the
Microsoft voices had no character errors in any of the three languages; Piper had none in
English and Marathi and 7.7% in Hindi (one inserted word). So Hindi and Marathi replies use
the native Microsoft voices first, and English replies use ElevenLabs first.

The Microsoft voices stream: playback starts with the first audio chunk instead of after the
whole reply (about 4-5 s for 500 characters, and longer for Marathi).
"""

import asyncio
import io
import os
import queue
import re
import wave
from threading import Lock, Thread

from services.reply_text import devanagari_language

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENDERS = ("female", "male")
EDGE_VOICES = {
    "female": {"en": "en-IN-NeerjaExpressiveNeural", "hi": "hi-IN-SwaraNeural", "mr": "mr-IN-AarohiNeural"},
    "male": {"en": "en-IN-PrabhatNeural", "hi": "hi-IN-MadhurNeural", "mr": "mr-IN-ManoharNeural"},
}
# Marathi was found slow to listen to at the default pace; a little faster sounds natural.
EDGE_RATE = {"en": "+0%", "hi": "+0%", "mr": "+10%"}
# Piper has no male Marathi voice, so the female one is used for both.
PIPER_VOICES = {
    "female": {"en": "en_US-lessac-medium", "hi": "hi_IN-priyamvada-medium", "mr": "mr_IN-google-medium"},
    "male": {"en": "en_US-ryan-medium", "hi": "hi_IN-rohan-medium", "mr": "mr_IN-google-medium"},
}
ELEVENLABS_VOICES = {"female": "EXAVITQu4vr4xnSDxMaL",   # Sarah: mature, reassuring
                     "male": "nPczCjzI2devNBz1zQrb"}     # Brian: deep, resonant and comforting
PIPER_DIR = os.environ.get("PIPER_VOICE_DIR", os.path.join(BASE_DIR, "Models", "piper"))
DEFAULT_ORDER = {"en": ("elevenlabs", "edge", "piper"), "hi": ("edge", "elevenlabs", "piper"), "mr": ("edge", "elevenlabs", "piper")}
EDGE_FIRST_AUDIO_TIMEOUT = 10.0   # seconds to the first audio chunk before trying the next voice
EDGE_TOTAL_TIMEOUT = 60.0
MAX_SPEECH_CHARS = 2500

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
def reply_language(text):
    """'hi', 'mr' or 'en' (English letters, including Hindi or Marathi written in English letters)."""
    devanagari = len(DEVANAGARI.findall(text))
    if devanagari <= len(LATIN.findall(text)):
        return "en"
    return devanagari_language(text) or "hi"


def clean_gender(value):
    return value if value in GENDERS else "female"


def provider_order(language):
    custom = [name.strip() for name in os.environ.get("SAMVYA_TTS_ORDER", "").split(",") if name.strip()]
    return tuple(custom) if custom else DEFAULT_ORDER[language]


def _edge_chunks(text, language, gender):
    """Microsoft neural voice audio as an iterator of MP3 chunks, or None if it does not start."""
    import edge_tts
    voice, rate = EDGE_VOICES[gender][language], EDGE_RATE[language]
    chunks, done = queue.Queue(), object()

    def worker():
        async def run():
            async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
                if chunk["type"] == "audio" and chunk["data"]:
                    chunks.put(chunk["data"])
        try:
            asyncio.run(asyncio.wait_for(run(), EDGE_TOTAL_TIMEOUT))
        except Exception as error:   # network or service failure
            chunks.put(error)
        finally:
            chunks.put(done)

    Thread(target=worker, name="samvya-edge-tts", daemon=True).start()
    try:
        first = chunks.get(timeout=EDGE_FIRST_AUDIO_TIMEOUT)
    except queue.Empty:
        return None
    if first is done or isinstance(first, Exception):
        return None

    def rest():
        yield first
        while True:
            try:
                item = chunks.get(timeout=EDGE_TOTAL_TIMEOUT)
            except queue.Empty:
                return
            if item is done or isinstance(item, Exception):
                return
            yield item
    return rest()


def edge_speech(text, language, gender="female"):
    """Complete MP3 from a Microsoft neural voice, or None."""
    chunks = _edge_chunks(text, language, gender)
    if chunks is None:
        return None
    audio = b"".join(chunks)
    return audio or None


_piper_voices = {}
_piper_lock = Lock()


def piper_speech(text, language, gender="female"):
    """WAV bytes from an offline Piper voice, or None when the voice is not installed."""
    path = os.path.join(PIPER_DIR, PIPER_VOICES[gender][language] + ".onnx")
    if not os.path.isfile(path) or not os.path.isfile(path + ".json"):
        return None
    try:
        with _piper_lock:
            if path not in _piper_voices:
                from piper import PiperVoice
                _piper_voices[path] = PiperVoice.load(path)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                _piper_voices[path].synthesize_wav(text, wav_file)
        audio = buffer.getvalue()
    except Exception:
        return None
    return audio if len(audio) > 1000 else None


def _prepare(text):
    text = (text or "").strip()[:MAX_SPEECH_CHARS]
    return text, (reply_language(text) if text else None)


def synthesize(text, elevenlabs, gender="female"):
    """(audio bytes, mimetype, provider) for a reply, or (None, None, None).

    `elevenlabs(text, voice_id)` returns MP3 bytes or None (unconfigured, quota used up or failed).
    """
    gender = clean_gender(gender)
    text, language = _prepare(text)
    if not text:
        return None, None, None
    for provider in provider_order(language):
        if provider == "elevenlabs":
            audio, mimetype = elevenlabs(text, ELEVENLABS_VOICES[gender]), "audio/mpeg"
        elif provider == "edge":
            audio, mimetype = edge_speech(text, language, gender), "audio/mpeg"
        elif provider == "piper":
            audio, mimetype = piper_speech(text, language, gender), "audio/wav"
        else:
            continue
        if audio:
            return audio, mimetype, provider
    return None, None, None


def stream(text, elevenlabs, gender="female"):
    """Like synthesize, but the Microsoft voice is returned as a chunk iterator so playback
    can start early. Returns (iterator of bytes, mimetype, provider) or (None, None, None)."""
    gender = clean_gender(gender)
    text, language = _prepare(text)
    if not text:
        return None, None, None
    for provider in provider_order(language):
        if provider == "edge":
            # Resolved through the module so a test can replace edge_speech or _edge_chunks.
            chunks = _edge_chunks(text, language, gender)
            if chunks is not None:
                return chunks, "audio/mpeg", "edge"
            continue
        if provider == "elevenlabs":
            audio, mimetype = elevenlabs(text, ELEVENLABS_VOICES[gender]), "audio/mpeg"
        elif provider == "piper":
            audio, mimetype = piper_speech(text, language, gender), "audio/wav"
        else:
            continue
        if audio:
            return iter([audio]), mimetype, provider
    return None, None, None
