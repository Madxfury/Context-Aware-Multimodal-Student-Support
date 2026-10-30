import unittest
import numpy as np
from schemas.observation import PhysiologicalData, StressData
from services.physiological_stress import PhysiologicalStressService

class TestStep8Inference(unittest.TestCase):
    def test_step8_inference_integration(self):
        # 1. Initialize the service
        service = PhysiologicalStressService()
        self.assertTrue(service.model_available, "Model should be successfully loaded")
        
        session_id = "test_user_session"
        
        # 2. Calibration Fixture
        np.random.seed(42)
        
        for i in range(9):
            mock_seq = np.random.randn(12, 14).tolist() # 12x14 sequence
            data = PhysiologicalData(session_id=session_id, v2_sequence=mock_seq)
            
            result = service.predict(data)
            
            # During calibration, it should return None for stress values
            self.assertIsNone(result.stress_level)
            self.assertIsNone(result.confidence)
            self.assertTrue(result.model_available)
            
        # By now, the session should be calibrated and the buffer should be empty
        self.assertTrue(service.sessions[session_id]["calibrated"])
        self.assertEqual(len(service.sessions[session_id]["buffer"]), 0)
        self.assertIsNotNone(service.sessions[session_id]["mu"])
        self.assertIsNotNone(service.sessions[session_id]["sigma"])
        
        # 3. Inference Fixture
        # Now send a held-out synthetic stress window
        mock_stress_seq = np.random.randn(12, 14).tolist()
        stress_data = PhysiologicalData(session_id=session_id, v2_sequence=mock_stress_seq)
        
        result1 = service.predict(stress_data)
        
        self.assertIsNotNone(result1.stress_level)
        self.assertIsInstance(result1.stress_level, float)
        self.assertTrue(0.0 <= result1.stress_level <= 1.0)
        
        self.assertIsNotNone(result1.confidence)
        self.assertIsInstance(result1.confidence, float)
        self.assertTrue(0.0 <= result1.confidence <= 1.0)
        
        self.assertIn(result1.predicted_class, ["Baseline", "Stress"])
        self.assertTrue(result1.model_available)
        
        # 4. Deterministic check
        result2 = service.predict(stress_data)
        self.assertEqual(result1.stress_level, result2.stress_level)
        self.assertEqual(result1.confidence, result2.confidence)
        self.assertEqual(result1.predicted_class, result2.predicted_class)

    def test_step8_invalid_shapes(self):
        service = PhysiologicalStressService()
        session_id = "test_user_session_invalid"
        
        # Send a sequence with an invalid shape (e.g. 10x14)
        invalid_seq = np.random.randn(10, 14).tolist()
        data = PhysiologicalData(session_id=session_id, v2_sequence=invalid_seq)
        
        result = service.predict(data)
        self.assertIsNone(result.stress_level)
        self.assertIsNone(result.confidence)

if __name__ == '__main__':
    unittest.main()
