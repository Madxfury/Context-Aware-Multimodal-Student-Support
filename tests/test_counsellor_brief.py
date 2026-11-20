"""Counsellor brief in the report, and the two-model camera reading."""
import base64
import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

import app as app_module
from services.camera_cues import ENSEMBLE_CLEAR_CONFIDENCE, EMOTIONS, CameraCues
from services.counsellor_brief import parse_brief, safety_review
from services.db_manager import DatabaseManager

TRANSCRIPT = [
    {"role": "assistant", "content": "Hi. What's been on your mind?"},
    {"role": "user", "content": "I have my semester exams next week and I can't sleep properly."},
    {"role": "assistant", "content": "That sounds tiring. What usually helps you wind down?"},
    {"role": "user", "content": "Talking to my sister helps, but I feel overwhelmed by the syllabus."},
]
BRIEF = {
    "overview": "The student is preparing for semester exams next week and is sleeping poorly. They feel overwhelmed by the syllabus.",
    "presenting_concerns": ["Semester exams next week.", "Poor sleep."],
    "student_described_feelings": ["Says they feel overwhelmed by the syllabus."],
    "context": ["Exams next week."],
    "coping_and_strengths": ["Talking to their sister helps."],
    "support_offered": ["Samvya asked what helps them wind down."],
    "suggested_follow_up": ["How many hours of sleep are they getting?", "Which subjects feel most overwhelming?"],
    "key_statements": ["I can't sleep properly", "I feel overwhelmed by the syllabus", "I am failing everything"],
    "unexpected": "dropped",
}


def test_brief_keeps_only_real_quotes_and_known_fields():
    reply = "Here is the brief:\n" + json.dumps({**BRIEF, "context": ["Call 9876543210 for help.", "Exams next week."]})
    brief = parse_brief(reply, TRANSCRIPT)
    assert brief["overview"].startswith("The student is preparing")
    # The made-up quote is dropped; real ones are kept even with different spacing or case.
    assert brief["key_statements"] == ["I can't sleep properly", "I feel overwhelmed by the syllabus"]
    assert "unexpected" not in brief
    assert all("9876543210" not in item for item in brief["context"])
    assert parse_brief("Sorry, I cannot help.", TRANSCRIPT) is None
    assert parse_brief(json.dumps({**BRIEF, "overview": ""}), TRANSCRIPT) is None
    long = parse_brief(json.dumps({**BRIEF, "presenting_concerns": [f"Concern {k}." for k in range(20)]}), TRANSCRIPT)
    assert len(long["presenting_concerns"]) == 6


def test_safety_is_decided_by_the_app_rules_not_the_model():
    risky = TRANSCRIPT + [
        {"role": "user", "content": "Honestly I want to die."},
        {"role": "assistant", "content": "India: For immediate danger, call 112. Tele MANAS at 14416."},
    ]
    review = safety_review(risky, app_module.routed_to_safety)
    assert review == {"flagged": True, "messages": 1, "statements": ["Honestly I want to die."], "support_shown": True}
    assert safety_review(TRANSCRIPT, app_module.routed_to_safety)["flagged"] is False


@pytest.fixture
def brief_app(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "db_manager", DatabaseManager(str(tmp_path / "brief.db")))
    monkeypatch.setattr(app_module, "session_briefs", app_module.OrderedDict())
    monkeypatch.setenv("GROQ_API_KEY", "test")
    provider = Mock()
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(BRIEF)))])
    monkeypatch.setattr(app_module, "Groq", lambda **_: provider)
    return app_module.app.test_client(), provider


def test_finishing_a_check_in_saves_the_brief_and_the_report_shows_it(brief_app):
    client, provider = brief_app
    result = client.post("/api/sessions/summarize", json={"session_id": "brief-1", "transcript": TRANSCRIPT}).json
    assert result["summary"] == BRIEF["overview"] and result["brief"]["safety"]["flagged"] is False
    system = provider.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "human counsellor" in system and "Do not diagnose" in system
    client.post("/api/sessions/save", json={"session_id": "brief-1", "transcript": TRANSCRIPT, "summary_insight": result["summary"]})
    saved = app_module.db_manager.get_session("brief-1")
    assert saved["counsellor_brief"]["student_messages"] == 2
    report = client.get("/session/brief-1").get_data(as_text=True)
    for text in ("Counsellor summary", "Main concerns", "Poor sleep.", "Suggested follow-up for the counsellor",
                 "Talking to their sister helps.", "I feel overwhelmed by the syllabus", "No safety concern was flagged"):
        assert text in report
    assert "Talking to my sister helps, but" not in report      # no transcript in the report
    assert "I am failing everything" not in report              # invented quote removed
    calls = provider.chat.completions.create.call_count
    client.get("/session/brief-1")
    assert provider.chat.completions.create.call_count == calls  # stored, not generated again


def test_reports_without_a_brief_get_one_on_first_view(brief_app):
    client, provider = brief_app
    risky = TRANSCRIPT + [{"role": "user", "content": "Sometimes I think everyone would be better off without me."}]
    app_module.db_manager.save_session({"session_id": "old-report", "transcript": risky, "summary_insight": "Old recap."})
    report = client.get("/session/old-report").get_data(as_text=True)
    assert provider.chat.completions.create.call_count == 1
    assert "Safety concern raised." in report and "better off without me" in report
    assert app_module.db_manager.get_session("old-report")["counsellor_brief"]["safety"]["flagged"] is True


def test_brief_falls_back_when_the_provider_fails_or_judges_sensors(brief_app, monkeypatch):
    client, provider = brief_app
    provider.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=json.dumps({**BRIEF, "overview": "The student's skin conductance is higher, a sign of stress."})))])
    assert app_module.generate_brief(TRANSCRIPT) is None
    provider.chat.completions.create.side_effect = RuntimeError("provider down")
    assert app_module.generate_brief(TRANSCRIPT) is None
    app_module.db_manager.save_session({"session_id": "no-brief", "transcript": TRANSCRIPT, "summary_insight": "Short recap."})
    report = client.get("/session/no-brief").get_data(as_text=True)
    assert "Short recap." in report and "could not be generated" in report


def test_camera_reading_averages_both_expression_models(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "camera_cues", CameraCues(ENSEMBLE_CLEAR_CONFIDENCE))
    monkeypatch.setattr(app_module, "model", object())
    monkeypatch.setattr(app_module, "DEVICE", torch.device("cpu"))
    swin = [0.02, 0.02, 0.02, 0.02, 0.5, 0.4, 0.02]        # Neutral 0.50, Sad 0.40
    second = [0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0]           # Neutral 0.90
    monkeypatch.setattr(app_module, "run_model", lambda _: torch.log(torch.tensor([swin])))
    monkeypatch.setattr(app_module, "expression_session", object())
    monkeypatch.setattr(app_module, "expression_probabilities", lambda _: second)
    _, encoded = cv2.imencode(".jpg", np.full((120, 100, 3), 128, dtype=np.uint8))
    image = base64.b64encode(encoded.tobytes()).decode()
    client = app_module.app.test_client()
    for _ in range(8):
        result = client.post("/detect", json={"image": image, "session_id": "ensemble", "face_crop": True}).json
    assert result["clear"] is True and result["emotion"] == "Neutral"
    assert result["confidence"] == pytest.approx(70.0, abs=0.1)   # (0.5 + 0.9) / 2, above the 60% ensemble bar


def test_expression_model_gives_seven_probabilities_when_installed():
    if app_module.expression_session is None:
        pytest.skip("HSEmotion model not installed")
    face = np.full((200, 170, 3), 140, dtype=np.uint8)
    probs = app_module.expression_probabilities(face)
    assert len(probs) == len(EMOTIONS) and sum(probs) == pytest.approx(1.0, abs=1e-5)


def test_conversation_timeline_is_minute_by_minute():
    cues = CameraCues(ENSEMBLE_CLEAR_CONFIDENCE)
    happy = [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]
    mixed = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
    for k in range(4):
        cues.record("t", "chat", happy, now=10.0 + k)
    for k in range(4):
        cues.record("t", "chat", mixed, now=75.0 + k)
    cues.record("t", "chat", happy, now=130.0)
    timeline = cues.summary("t")["conversation"]["timeline"]
    assert [(slot["minute"], slot["emotion"], slot["frames"]) for slot in timeline] == [(1, "Happy", 4), (2, None, 4), (3, None, 1)]
