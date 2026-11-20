"""Spoken replies in English, Hindi and Marathi, in a female or male counsellor voice.

Providers, tried in order for the reply's language:
- Sarvam Bulbul v3 (paid per character, with the same key as speech-to-text): natural Indian
  voices for English, Hindi and Marathi. Its audio streams, starting in about 0.2 s.
- ElevenLabs (paid; the free plan allows 10,000 characters a month).
- Microsoft neural voices through the open-source edge-tts library (free, online): native
  Indian voices for English, Hindi and Marathi. edge-tts uses the Read Aloud service of the
  Edge browser, which is not an official API and may change.
- Piper (open source, offline, on this computer) with Indian Hindi and Marathi voices.
The browser's own voice is the last resort (static/js/main.js).

Pronunciation was checked by transcribing test sentences with Sarvam speech-to-text: the
Microsoft voices had no character errors in any of the three languages; Piper had none in
English and Marathi and 7.7% in Hindi (one inserted word). So Hindi and Marathi replies use
the native Microsoft voices first, and English replies use ElevenLabs first. Since
2026-11-21 Sarvam's voices come first in all three languages, because they sound the most human.

The Microsoft voices stream: playback starts with the first audio chunk instead of after the
whole reply (about 4-5 s for 500 characters, and longer for Marathi).
"""

import asyncio
import base64
import io
import os
import queue
import re
import time
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
# Bulbul v3 voices for each counsellor gender and language. The male voice was chosen by
# listening. The female voices were chosen on 2026-11-21 from all 14 female voices: every one
# was transcribed back without errors, and these had the widest pitch movement (the least
# flat, robotic delivery) at a natural speaking pace in their language.
SARVAM_VOICES = {
    "female": {"en": "pooja", "hi": "pooja", "mr": "rupali"},
    "male": {"en": "aditya", "hi": "aditya", "mr": "aditya"},
}
SARVAM_LANGUAGES = {"en": "en-IN", "hi": "hi-IN", "mr": "mr-IN"}
SARVAM_MODEL = "bulbul:v3"
SARVAM_TIMEOUT = 10.0
# After the account refuses (a bad key or no credit left), use the next voice for a while
# instead of adding a failed request to every reply.
SARVAM_PAUSE = 1800
sarvam_paused_until = 0.0
PIPER_DIR = os.environ.get("PIPER_VOICE_DIR", os.path.join(BASE_DIR, "Models", "piper"))
DEFAULT_ORDER = {"en": ("sarvam", "elevenlabs", "edge", "piper"),
                 "hi": ("sarvam", "edge", "elevenlabs", "piper"),
                 "mr": ("sarvam", "edge", "elevenlabs", "piper")}
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


def _sarvam_client():
    key = os.environ.get("SARVAM_API_KEY")
    if not key or key == "your_sarvam_api_key_here" or time.monotonic() < sarvam_paused_until:
        return None
    from sarvamai import SarvamAI
    return SarvamAI(api_subscription_key=key, timeout=SARVAM_TIMEOUT)


def _sarvam_failed(error):
    global sarvam_paused_until
    if getattr(error, "status_code", None) in (401, 402, 403):
        sarvam_paused_until = time.monotonic() + SARVAM_PAUSE


def _sarvam_request(text, language, gender):
    return {"text": text, "target_language_code": SARVAM_LANGUAGES[language], "speaker": SARVAM_VOICES[gender][language],
            "model": SARVAM_MODEL, "output_audio_codec": "mp3"}


def sarvam_speech(text, language, gender="female"):
    """Complete MP3 from a Sarvam voice, or None (no key, paused or failed)."""
    client = _sarvam_client()
    if client is None:
        return None
    try:
        response = client.text_to_speech.convert(**_sarvam_request(text, language, gender))
        audio = b"".join(base64.b64decode(part) for part in response.audios or [])
    except Exception as error:
        _sarvam_failed(error)
        return None
    return audio or None


def _sarvam_chunks(text, language, gender="female"):
    """Sarvam MP3 audio as an iterator of chunks, or None if it does not start."""
    client = _sarvam_client()
    if client is None:
        return None
    try:
        chunks = iter(client.text_to_speech.convert_stream(**_sarvam_request(text, language, gender)))
        first = next(chunk for chunk in chunks if chunk)
    except Exception as error:      # includes StopIteration: no audio at all
        _sarvam_failed(error)
        return None

    def rest():
        yield first
        try:
            for chunk in chunks:
                if chunk:
                    yield chunk
        except Exception:
            return
    return rest()


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


# Emergency (112) and Tele MANAS (14416) numbers are read digit by digit, not as "one
# hundred and twelve"; spaced digits do this in every voice and language.
HELPLINE_NUMBERS = re.compile(r"(?<!\d)(112|14416)(?!\d)")


def _prepare(text):
    text = (text or "").strip()[:MAX_SPEECH_CHARS]
    text = HELPLINE_NUMBERS.sub(lambda match: " ".join(match.group(1)), text)
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
        if provider == "sarvam":
            audio, mimetype = sarvam_speech(text, language, gender), "audio/mpeg"
        elif provider == "elevenlabs":
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
    """Like synthesize, but the Sarvam and Microsoft voices are returned as chunk iterators so playback
    can start early. Returns (iterator of bytes, mimetype, provider) or (None, None, None)."""
    gender = clean_gender(gender)
    text, language = _prepare(text)
    if not text:
        return None, None, None
    for provider in provider_order(language):
        if provider == "sarvam":
            chunks = _sarvam_chunks(text, language, gender)
            if chunks is not None:
                return chunks, "audio/mpeg", "sarvam"
            continue
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
