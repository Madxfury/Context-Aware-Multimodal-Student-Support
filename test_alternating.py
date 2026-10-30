import time
import requests

print("Posting chat message...")
res = requests.post("http://localhost:5000/chat", json={
    "message": "I am so stressed about my final exams.",
    "session_id": "default_session"
})
print(res.json().get('response')[:50])

print("Polling debug state...")
for _ in range(15):
    res = requests.get("http://localhost:5000/api/debug/state?session_id=default_session")
    state = res.json()
    intent = state.get("css", {}).get("intent")
    print(f"[{time.strftime('%H:%M:%S')}] Intent: {intent}")
    time.sleep(1)
