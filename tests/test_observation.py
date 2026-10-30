import unittest
from pydantic import ValidationError
from schemas.observation import Observation, PhysiologicalData, VisualData, FatigueData, MotionData, AudioData, StressData
from services.mental_health_score import calculate_mental_health_score, get_risk_level

class TestObservation(unittest.TestCase):

    def setUp(self):
        self.valid_probs = {
            "Angry": 0.05,
            "Disgust": 0.05,
            "Fear": 0.10,
            "Happy": 0.10,
            "Sad": 0.60,
            "Surprise": 0.05,
            "Neutral": 0.05
        }
        
    def test_valid_observation(self):
        """Test instantiation with valid fields including optional/null handling"""
        obs = Observation(
            physiological=PhysiologicalData(), # all defaults to None
            visual=VisualData(
                emotion="Sad",
                confidence=87.5,
                emotion_probabilities=self.valid_probs,
                mental_health_score=71.2
            ),
            fatigue=FatigueData(blink_rate=15.0, fatigue_score=0.4),
            motion=MotionData(),
            audio=AudioData(),
            stress=StressData() # fields default to None
        )
        self.assertIsNotNone(obs)
        self.assertIsNone(obs.physiological.heart_rate)
        self.assertIsNone(obs.stress.stress_level)
        self.assertEqual(obs.visual.emotion, "Sad")

    def test_missing_emotion(self):
        """Test exactly seven emotions - missing one"""
        probs = self.valid_probs.copy()
        del probs["Angry"]
        
        with self.assertRaises(ValidationError) as context:
            VisualData(
                emotion="Sad",
                confidence=87.5,
                emotion_probabilities=probs,
                mental_health_score=71.2
            )
        self.assertIn("Missing emotion probabilities", str(context.exception))

    def test_extra_emotion(self):
        """Test exactly seven emotions - extra one"""
        probs = self.valid_probs.copy()
        probs["Confused"] = 0.0
        
        with self.assertRaises(ValidationError) as context:
            VisualData(
                emotion="Sad",
                confidence=87.5,
                emotion_probabilities=probs,
                mental_health_score=71.2
            )
        self.assertIn("Unexpected emotion probabilities", str(context.exception))

    def test_probability_out_of_bounds(self):
        """Test probabilities must be in [0,1]"""
        probs = self.valid_probs.copy()
        probs["Sad"] = 1.2
        probs["Happy"] = -0.2
        
        with self.assertRaises(ValidationError) as context:
            VisualData(
                emotion="Sad",
                confidence=87.5,
                emotion_probabilities=probs,
                mental_health_score=71.2
            )
        self.assertIn("must be between 0 and 1", str(context.exception))

    def test_probability_sum_validation(self):
        """Test probabilities must sum to 1.0"""
        probs = self.valid_probs.copy()
        probs["Sad"] = 0.9 # sum becomes 1.3
        
        with self.assertRaises(ValidationError) as context:
            VisualData(
                emotion="Sad",
                confidence=87.5,
                emotion_probabilities=probs,
                mental_health_score=71.2
            )
        self.assertIn("must sum to 1.0", str(context.exception))

    def test_calculate_mental_health_score(self):
        """Test mental health score calculation logic"""
        # Risk weights: Happy=0, Neutral=0.3, Surprise=0.35, Angry=0.75, Disgust=0.75, Sad=0.8, Fear=0.85
        probs = {
            "Angry": 0.1,    # 0.1 * 0.75 = 0.075
            "Disgust": 0.0,  # 0
            "Fear": 0.2,     # 0.2 * 0.85 = 0.17
            "Happy": 0.3,    # 0.3 * 0 = 0
            "Sad": 0.4,      # 0.4 * 0.80 = 0.32
            "Surprise": 0.0, # 0
            "Neutral": 0.0   # 0
        }
        # Expected: 0.075 + 0.17 + 0 + 0.32 = 0.565 -> * 100 = 56.5
        score = calculate_mental_health_score(probs)
        self.assertAlmostEqual(score, 56.5)

    def test_get_risk_level(self):
        """Test the 55.0 threshold"""
        self.assertEqual(get_risk_level(54.9), "normal")
        self.assertEqual(get_risk_level(55.0), "high")
        self.assertEqual(get_risk_level(90.0), "high")
        
if __name__ == '__main__':
    unittest.main()
