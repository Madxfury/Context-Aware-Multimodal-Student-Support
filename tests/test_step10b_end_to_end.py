import unittest
import json
from app import app, global_simulators, session_stress_histories

class TestStep10BEndToEnd(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        global_simulators.clear()
        session_stress_histories.clear()
        # Reset the physiological service state
        from app import physiological_service
        physiological_service.sessions.clear()

    def test_end_to_end_simulator_chat_flow(self):
        session_id = "e2e_test_session"
        emotion = "Fear" # Maps to high_stress scenario
        
        # 1. Frontend polls /api/simulator/stream 10 times (Calibration Phase)
        # Each call generates 60s of data. Since PhysiologicalStressService throws away 
        # the first 30s of subsequent windows assuming overlap, we need 9 windows to hit 300s.
        latest_v2_sequence = None
        for i in range(10):
            res = self.client.get(f'/api/simulator/stream?session_id={session_id}&emotion={emotion}')
            self.assertEqual(res.status_code, 200)
            data = json.loads(res.data)
            
            self.assertIn("hr", data)
            self.assertIn("gsr_adc", data)
            self.assertIn("temp", data)
            self.assertIn("v2_sequence", data)
            self.assertTrue(data["is_simulated"])
            latest_v2_sequence = data["v2_sequence"]
        
        # After 10 calls, calibration is done and CSS history should have predictions
        css_history = session_stress_histories.get(session_id, [])
        self.assertGreater(len(css_history), 0, "Calibration failed or CSS history not populated.")
        
        # 2. Frontend sends chat message with the latest v2_sequence
        chat_payload = {
            "message": "I'm really worried about my exams.",
            "emotion": emotion,
            "history": [],
            "sensors": {
                "v2_sequence": latest_v2_sequence
            },
            "session_id": session_id
        }
        
        chat_res = self.client.post('/chat', data=json.dumps(chat_payload), content_type='application/json')
        self.assertEqual(chat_res.status_code, 200)
        
        chat_data = json.loads(chat_res.data)
        self.assertIn("response", chat_data)

if __name__ == '__main__':
    unittest.main()
