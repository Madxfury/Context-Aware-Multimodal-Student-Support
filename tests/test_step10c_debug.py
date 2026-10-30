import pytest
from unittest.mock import patch
from app import app, session_timelines, session_latest_emotions, session_latest_intervention, get_debug_state, global_simulators
from services.physiological_stress import PhysiologicalStressService

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_debug_state_initialization(client):
    """Test that a new session debug state initializes correctly"""
    session_id = "test_init_session"
    state = get_debug_state(session_id)
    
    assert state["session_id"] == session_id
    assert state["camera"]["emotion"] == "Neutral"
    assert state["calibration"]["status"] == "WAITING"
    assert state["css"]["s_t"] is None
    assert state["intervention"] is None
    assert state["groq_context"] is None
    assert state["wearable"] is None

def test_debug_timeline_logging(client):
    """Test the timeline logging logic across endpoints"""
    session_id = "test_timeline_session"
    
    # 1. Trigger wearable
    response2 = client.get(f'/api/simulator/stream?session_id={session_id}&emotion=Angry')
    
    # Check that events were logged
    events = session_timelines[session_id]
    categories = [e["category"] for e in events]
    assert "WEARABLE" in categories

def test_debug_state_wearable_mock_injection(client):
    """Test that the wearable mock data is safely populated into debug state"""
    session_id = "test_mock_wearable_session"
    
    # Call the stream endpoint
    response = client.get(f'/api/simulator/stream?session_id={session_id}&emotion=Fear')
    data = response.get_json()
    
    assert "debug_state" in data
    debug_state = data["debug_state"]
    
    assert debug_state["wearable"] is not None
    assert debug_state["wearable"]["scenario"] == "high_stress"
    assert debug_state["wearable"]["hr"] > 0
    assert debug_state["wearable"]["gsr_adc"] > 0

def test_debug_state_chat_injection(client):
    """Test that the chat endpoint correctly returns the latest CSS and Groq Context"""
    session_id = "test_chat_session"
    
    response = client.post('/chat', json={
        "session_id": session_id,
        "emotion": "Sad",
        "message": "I feel very overwhelmed right now.",
        "sensors": {"hr": 95, "temp": 37.2, "gsr_adc": 3000}
    })
    
    data = response.get_json()
    assert "debug_state" in data
    
    debug_state = data["debug_state"]
    assert debug_state["intervention"] is not None
    assert debug_state["groq_context"] is not None
    assert debug_state["groq_context"]["user_message"] == "I feel very overwhelmed right now."
