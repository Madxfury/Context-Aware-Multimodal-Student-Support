import unittest
from schemas.observation import PhysiologicalData, StressData
from services.physiological_stress import PhysiologicalStressService

class TestPhysiologicalStress(unittest.TestCase):

    def setUp(self):
        self.service = PhysiologicalStressService()
        
    def test_service_initialization(self):
        """3. Physiological stress service initializes correctly."""
        self.assertIsNotNone(self.service)
        
    def test_model_availability(self):
        """4. Model availability correctly reports True after integration."""
        self.assertTrue(self.service.model_available)
        
    def test_predict_with_valid_data(self):
        """1. Valid physiological data can be represented.
           5. No stress prediction is produced when no model exists.
           6. stress_level remains None.
           7. confidence remains None."""
        data = PhysiologicalData(heart_rate=85.0, spo2=98.0, temperature=36.6)
        stress = self.service.predict(data)
        
        self.assertIsInstance(stress, StressData)
        self.assertIsNone(stress.stress_level)
        self.assertIsNone(stress.confidence)

    def test_predict_with_missing_data(self):
        """2. Missing physiological values remain None."""
        data = PhysiologicalData() # All fields default to None
        self.assertIsNone(data.heart_rate)
        self.assertIsNone(data.spo2)
        
        stress = self.service.predict(data)
        self.assertIsNone(stress.stress_level)
        self.assertIsNone(stress.confidence)

if __name__ == '__main__':
    unittest.main()
