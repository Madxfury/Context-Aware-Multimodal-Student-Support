"""Camera facial cues: clear frames only, 'unclear' below the confidence threshold, check-in baseline."""
import base64
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

import app as app_module
from services.camera_cues import (CLEAR_CONFIDENCE, EMOTIONS, MIN_CHECKIN_FRAMES, CameraCues,
                                  clean_cues, clean_probabilities)
from services.db_manager import DatabaseManager


def probs(emotion, confidence):
    rest = (1 - confidence) / (len(EMOTIONS) - 1)
    return [confidence if name == emotion else rest for name in EMOTIONS]


def test_a_label_needs_enough_clear_frames_and_confidence():
    cues = CameraCues()
    for _ in range(MIN_CHECKIN_FRAMES - 1):
        cues.record("s", "checkin", probs("Neutral", 0.95), now=1.0)
    assert cues.checkin("s")["clear"] is False          # one frame short
    cues.record("s", "checkin", probs("Neutral", 0.95), now=1.0)
    result = cues.checkin("s")
    assert result["clear"] is True and result["emotion"] == "Neutral" and result["confidence"] == pytest.approx(95.0)

    unsure = CameraCues()
    for _ in range(20):
        unsure.record("s", "checkin", probs("Sad", CLEAR_CONFIDENCE - 0.05), now=1.0)
    result = unsure.checkin("s")
    assert result["clear"] is False and result["emotion"] is None
    assert unsure.evidence("s") is None
    assert "unclear" in unsure.context("s")


def test_mixed_frames_average_to_unclear():
    cues = CameraCues()
    for k in range(12):
        cues.record("s", "checkin", probs("Neutral" if k % 2 else "Sad", 0.9), now=1.0)
    assert cues.checkin("s")["clear"] is False   # 0.5 average for each label


def test_conversation_cue_uses_the_last_two_minutes_and_the_check_in_baseline():
    cues = CameraCues()
    for _ in range(10):
        cues.record("s", "checkin", probs("Neutral", 0.9), {"smile": 0.05, "brow_down": 0.1, "eye_openness": 0.9}, now=0.0)
    # Old conversation frames fall outside the two-minute window.
    for _ in range(6):
        cues.record("s", "chat", probs("Sad", 0.9), {"smile": 0.0}, now=10.0)
    for _ in range(6):
        cues.record("s", "chat", probs("Happy", 0.92), {"smile": 0.6, "brow_down": 0.1, "eye_openness": 0.9}, now=200.0)
    recent = cues.recent("s", now=210.0)
    assert recent["clear"] and recent["emotion"] == "Happy" and recent["frames"] == 6
    assert recent["changes"] == ["smiling more"]
    evidence = cues.evidence("s", now=210.0)
    assert evidence["source"] == "conversation" and evidence["emotion"] == "Happy"
    # Without recent clear frames the check-in is the evidence.
    assert cues.evidence("s", now=1000.0)["source"] == "check-in"
    text = cues.context("s", now=210.0)
    assert "check-in: mostly Neutral" in text and "last 2 minutes: mostly Happy" in text and "smiling more" in text
    summary = cues.summary("s")
    assert summary["checkin"]["emotion"] == "Neutral" and summary["conversation"]["frames"] == 12
    assert summary["clear_confidence"] == 80


def test_inputs_are_validated():
    assert clean_probabilities([0.5] * 7) == pytest.approx([1 / 7] * 7)
    assert clean_probabilities([0.1] * 6) is None
    assert clean_probabilities([float("nan")] + [0.1] * 6) is None
    assert clean_probabilities([-1] + [0.2] * 6) is None
    assert clean_cues({"smile": 1.7, "brow_down": -0.2, "unknown": 0.5, "frown": "x", "jaw_open": True}) == {"smile": 1.0, "brow_down": 0.0}
    cues = CameraCues()
    assert not cues.record("s", "elsewhere", probs("Happy", 0.9))
    assert cues.summary("s") is None and cues.context("s") == "not captured (no camera)"


@pytest.fixture
def camera_app(monkeypatch, tmp_path):
    """The real /detect route with a stub model whose output we control."""
    monkeypatch.setattr(app_module, "camera_cues", CameraCues())
    monkeypatch.setattr(app_module, "expression_session", None)   # Swin-FANE alone; the ensemble is tested separately
    monkeypatch.setattr(app_module, "model", object())
    monkeypatch.setattr(app_module, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(app_module, "db_manager", DatabaseManager(str(tmp_path / "camera.db")))
    haar = Mock(side_effect=AssertionError("a face crop must not be searched for a face again"))
    monkeypatch.setattr(app_module, "face_cascade", SimpleNamespace(detectMultiScale=haar))
    outputs = []
    monkeypatch.setattr(app_module, "run_model", lambda _: torch.log(torch.tensor([outputs.pop(0)])))
    _, encoded = cv2.imencode(".jpg", np.full((120, 100, 3), 128, dtype=np.uint8))
    image = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()
    client = app_module.app.test_client()

    def send(session, emotion, confidence, phase="checkin", cues=None):
        outputs.append(probs(emotion, confidence))
        return client.post("/detect", json={"image": image, "session_id": session, "face_crop": True,
                                            "phase": phase, "cues": cues or {"smile": 0.1}}).json
    return client, send


def test_detect_reports_unclear_until_the_reading_is_confident(camera_app):
    client, send = camera_app
    for _ in range(MIN_CHECKIN_FRAMES):
        result = send("cam-unclear", "Sad", 0.6)
    assert result["face_detected"] and result["clear"] is False and result["clear_frames"] == MIN_CHECKIN_FRAMES
    assert result["final_ms"] is None and result["risk"] is None and result["face_box"] is None
    assert "cam-unclear" not in app_module.session_latest_emotions   # no camera evidence for the counsellor

    for _ in range(MIN_CHECKIN_FRAMES):
        result = send("cam-clear", "Neutral", 0.95)
    assert result["clear"] is True and result["emotion"] == "Neutral" and result["final_ms"] is not None
    assert app_module.session_latest_emotions["cam-clear"]["emotion"] == "Neutral"


def test_chat_uses_the_server_camera_cue_not_the_browser_label(camera_app, monkeypatch):
    client, send = camera_app
    for _ in range(MIN_CHECKIN_FRAMES):
        send("cam-chat", "Sad", 0.55)                       # unclear check-in
    provider = Mock()
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Thanks for sharing that."))])
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "Groq", lambda **_: provider)
    response = client.post("/chat", json={"message": "Hello", "session_id": "cam-chat", "emotion": "Sad", "defer_audio": True})
    assert response.status_code == 200
    prompt = provider.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Camera facial cue (private): check-in: unclear" in prompt
    assert "prominently detected physical emotion" not in prompt
    # A browser label alone never selects the emotion-based intervention.
    assert app_module.session_latest_intervention["cam-chat"]["intervention"] != "emotional_validation"


def test_saved_report_uses_the_server_check_in_result(camera_app):
    client, send = camera_app
    for _ in range(MIN_CHECKIN_FRAMES):
        send("cam-report", "Sad", 0.6)
    for _ in range(6):
        send("cam-report", "Happy", 0.9, phase="chat", cues={"smile": 0.7})
    saved = client.post("/api/sessions/save", json={
        "session_id": "cam-report", "locked_emotion": "Sad", "mental_health_score": 80.0,
        "risk_level": "high", "transcript": [{"role": "user", "content": "Hi"}]}).json
    assert saved["success"]
    session = app_module.db_manager.get_session("cam-report")
    assert session["locked_emotion"] == "Unclear" and session["mental_health_score"] is None and session["risk_level"] is None
    assert session["camera_summary"]["conversation"]["frames"] == 6
    report = client.get("/session/cam-report").get_data(as_text=True)
    assert "Unclear" in report and "a reading needs 80%" in report
    assert "Camera during the conversation" in report and "mostly Happy expression" in report
