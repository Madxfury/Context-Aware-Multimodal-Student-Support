import unittest
import numpy as np
from schemas.observation import PhysiologicalData, StressData, Observation, VisualData, FatigueData, MotionData, AudioData
from services.physiological_stress import PhysiologicalStressService
from services.css_manager import CSSManager
from services.intervention_selector import RuleBasedInterventionSelector
from schemas.css import ContextualStudentState

class TestStep9Pipeline(unittest.TestCase):
    def setUp(self):
        self.physio_service = PhysiologicalStressService()
        self.css_manager = CSSManager()
        self.intervention_selector = RuleBasedInterventionSelector()
        self.session_stress_histories = {}
        
    def simulate_chat_pipeline(self, session_id, v2_sequence=None):
        physio_data = PhysiologicalData(session_id=session_id, v2_sequence=v2_sequence)
        stress_prediction = self.physio_service.predict(physio_data)
        
        mock_observation = Observation(
            physiological=physio_data,
            visual=VisualData(
                emotion="Neutral",
                confidence=100.0,
                emotion_probabilities={"Neutral": 1.0, "Angry": 0.0, "Disgust": 0.0, "Fear": 0.0, "Happy": 0.0, "Sad": 0.0, "Surprise": 0.0},
                mental_health_score=50.0 
            ),
            fatigue=FatigueData(),
            motion=MotionData(),
            audio=AudioData(intent="neutral", sentiment="neutral"),
            stress=stress_prediction
        )
        
        prev_stress_history = self.session_stress_histories.get(session_id, [])
        css = self.css_manager.build_state(observation=mock_observation, dialogue_history=[], previous_stress_history=prev_stress_history)
        self.session_stress_histories[session_id] = css.stress_history
        
        intervention = self.intervention_selector.select(css)
        return css, intervention

    def test_end_to_end_pipeline(self):
        session_id = "test_pipeline_session"
        np.random.seed(42)
        
        # 1. Calibration Phase
        for i in range(9):
            mock_seq = np.random.randn(12, 14).tolist()
            css, intervention = self.simulate_chat_pipeline(session_id, v2_sequence=mock_seq)
            
            # During calibration S_t should be None
            self.assertIsNone(css.current_stress)
            self.assertIsNone(css.stress_change)
            self.assertEqual(len(css.stress_history), 0)
            
        self.assertTrue(self.physio_service.sessions[session_id]["calibrated"])
        
        # 2. First Prediction
        mock_stress_seq_1 = np.random.randn(12, 14).tolist()
        css1, intervention1 = self.simulate_chat_pipeline(session_id, v2_sequence=mock_stress_seq_1)
        
        self.assertIsNotNone(css1.current_stress)
        self.assertIsNone(css1.stress_change) # No previous stress to compare to
        self.assertEqual(len(css1.stress_history), 1)
        self.assertEqual(css1.stress_history[-1], css1.current_stress)
        
        # 3. Consecutive Predictions & Delta S_t
        mock_stress_seq_2 = np.random.randn(12, 14).tolist()
        css2, intervention2 = self.simulate_chat_pipeline(session_id, v2_sequence=mock_stress_seq_2)
        
        self.assertIsNotNone(css2.current_stress)
        self.assertIsNotNone(css2.stress_change)
        self.assertEqual(css2.stress_change, css2.current_stress - css1.current_stress)
        self.assertEqual(len(css2.stress_history), 2)
        self.assertEqual(css2.stress_history[-1], css2.current_stress)
        
        # 4. Intervention Selection
        self.assertIsNotNone(intervention2)
        self.assertTrue(hasattr(intervention2, 'intervention'))
        
    def test_backward_compatibility_no_sequence(self):
        session_id = "test_backward_compat"
        css, intervention = self.simulate_chat_pipeline(session_id, v2_sequence=None)
        
        self.assertIsNone(css.current_stress)
        self.assertIsNone(css.stress_change)
        self.assertEqual(len(css.stress_history), 0)
        
if __name__ == '__main__':
    unittest.main()
