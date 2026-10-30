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
    assert b"Help me plan my revision." in report.data
    assert client.get("/session/unknown").status_code == 404

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
