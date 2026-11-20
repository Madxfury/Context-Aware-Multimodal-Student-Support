"""Relay microphone PCM and transcript events without exposing the provider key."""

import base64
import json
import os
from time import monotonic

from flask import current_app, request
from flask_sock import Sock
from simple_websocket import ConnectionClosed
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect


sock = Sock()
WEBSOCKET_PATHS = {"/transcribe/live"}


class EndUpgradedConnections:
    """Stop the development server writing an HTTP response into a closed WebSocket.

    flask-sock returns a 200 response once the handler finishes. Werkzeug's development
    server then writes it after the close frame, and browsers report "Invalid frame
    header". Werkzeug treats ConnectionError as a finished connection and writes nothing.
    """

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        statuses = []

        def record_status(status, headers, exc_info=None):
            statuses.append(status)
            return start_response(status, headers, exc_info)

        result = self.app(environ, record_status)
        if (
            environ.get("PATH_INFO") in WEBSOCKET_PATHS
            and "werkzeug.socket" in environ
            and environ.get("HTTP_UPGRADE", "").lower() == "websocket"
            and statuses and statuses[-1].startswith("200")
        ):
            if hasattr(result, "close"):
                result.close()
            raise ConnectionError("WebSocket closed")
        return result
SARVAM_URL = (
    "wss://api.sarvam.ai/speech-to-text-realtime/ws"
    "?model=saaras:v4&language_code=auto&encoding=linear16"
    "&sample_rate=16000&stream_type=fast&endpointing=vad&silence_duration_ms=1800"
)
MAX_RECORDING_BYTES = 60 * 16000 * 2


@sock.route("/transcribe/live")
def live_transcribe(browser):
    def send(event_type, **fields):
        browser.send(json.dumps({"type": event_type, **fields}))

    if request.headers.get("Origin") != request.host_url.rstrip("/"):
        send("unavailable")
        return
    key = os.environ.get("SARVAM_API_KEY")
    if not key or key == "your_sarvam_api_key_here":
        send("unavailable")
        return

    try:
        with connect(
            SARVAM_URL,
            additional_headers={"api-subscription-key": key},
            open_timeout=5,
            close_timeout=0.5,
            max_size=65536,
        ) as provider:
            remaining_bytes = MAX_RECORDING_BYTES
            deadline = monotonic() + 65
            ending = False
            ended_utterance = None
            while monotonic() < deadline:
                frame = browser.receive(timeout=0.02)
                if frame is not None:
                    if frame == "end" and not ending:
                        provider.send(json.dumps({"event": "end"}))
                        ending = True
                        deadline = monotonic() + 4
                    elif (
                        not ending and isinstance(frame, bytes)
                        and 0 < len(frame) <= min(6400, remaining_bytes)
                        and len(frame) % 2 == 0
                    ):
                        remaining_bytes -= len(frame)
                        provider.send(json.dumps({
                            "event": "audio_input",
                            "audio": base64.b64encode(frame).decode("ascii"),
                        }))
                    else:
                        break

                try:
                    event = json.loads(provider.recv(timeout=0.01))
                except TimeoutError:
                    continue
                event_type = event.get("event")
                if event_type == "session.begin":
                    send("ready")
                elif event_type == "vad.speech_start":
                    ended_utterance = None
                elif event_type == "vad.speech_end":
                    ended_utterance = event.get("utterance_idx")
                elif event_type in {"transcript.partial", "transcript.final"}:
                    transcript = event.get("text")
                    if isinstance(transcript, str):
                        end_of_turn = (
                            event_type == "transcript.final" and not ending
                            and ended_utterance is not None and ended_utterance == event.get("utterance_idx")
                        )
                        send(event_type.split(".")[1], text=transcript[:4000], end_of_turn=end_of_turn)
                elif event_type == "session.end":
                    send("done")
                    return
                elif event_type == "error" and event.get("is_fatal", True):
                    break
            send("unavailable")
    except ConnectionClosed:
        return
    except (OSError, TimeoutError, WebSocketException, ValueError) as error:
        current_app.logger.warning("Live transcription unavailable: %s", type(error).__name__)
        try:
            send("unavailable")
        except ConnectionClosed:
            pass
