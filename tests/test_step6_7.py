import unittest
from schemas.observation import Observation, PhysiologicalData, VisualData, FatigueData, MotionData, AudioData, StressData
from schemas.css import ContextualStudentState
from services.nlp_processor import NLPProcessorService
from services.css_manager import CSSManager

class TestStep6And7(unittest.TestCase):

    def setUp(self):
        self.nlp = NLPProcessorService()
        self.css = CSSManager(max_dialogue_turns=3, max_stress_history=5)
        
        # Valid dummy observation for CSS building
        self.valid_obs = Observation(
            physiological=PhysiologicalData(),
            visual=VisualData(
                emotion="Sad",
                confidence=90.0,
                emotion_probabilities={"Sad": 1.0, "Happy": 0.0, "Fear": 0.0, "Disgust": 0.0, "Angry": 0.0, "Surprise": 0.0, "Neutral": 0.0},
                mental_health_score=80.0
            ),
            fatigue=FatigueData(),
            motion=MotionData(),
            audio=AudioData(transcript="I am sad", language="en", nlp_result=None),
            stress=StressData(stress_level=None, confidence=None)
        )

    # ==========================
    # STEP 6: NLP TESTS
    # ==========================
    def test_nlp_intent_none_when_unavailable(self):
        """3. NLP service returns unknown intent when no match."""
        res = self.nlp.process("test")
        self.assertEqual(res.intent, "unknown")
        
    def test_nlp_sentiment_none_when_unavailable(self):
        """4. NLP service returns neutral sentiment when no match."""
        res = self.nlp.process("test")
        self.assertEqual(res.sentiment, "neutral")
        
    def test_nlp_model_available_false(self):
        """5. NLP model_available is False."""
        self.assertFalse(self.nlp.model_available)

    # ==========================
    # STEP 7: CSS TESTS
    # ==========================
    def test_valid_css_constructed(self):
        """1. Valid CSS can be constructed.
           2. CSS accepts Observation data."""
        state = self.css.build_state(self.valid_obs, dialogue_history=[])
        self.assertIsInstance(state, ContextualStudentState)
        self.assertEqual(state.current_emotion, "Sad")

    def test_mental_health_score_preserved(self):
        """3. Mental Health Score is preserved."""
        state = self.css.build_state(self.valid_obs, dialogue_history=[])
        self.assertEqual(state.mental_health_score, 80.0)

    def test_current_stress_remains_none(self):
        """4. Current stress remains None when no physiological model exists."""
        state = self.css.build_state(self.valid_obs, dialogue_history=[])
        self.assertIsNone(state.current_stress)

    def test_stress_confidence_none(self):
        """5. Stress confidence remains None."""
        state = self.css.build_state(self.valid_obs, dialogue_history=[])
        self.assertIsNone(state.stress_confidence)

    def test_stress_change_none_when_unavailable(self):
        """6. Stress change is None when stress is unavailable."""
        state = self.css.build_state(self.valid_obs, dialogue_history=[])
        self.assertIsNone(state.stress_change)

    def test_stress_change_calculated(self):
        """7. Stress change is correctly calculated when two REAL stress values are supplied."""
        obs = self.valid_obs.model_copy()
        obs.stress.stress_level = 75.0
        state = self.css.build_state(obs, dialogue_history=[], previous_stress_history=[70.0])
        self.assertEqual(state.stress_change, 5.0)

    def test_stress_history_bounded(self):
        """8. Stress history is bounded."""
        obs = self.valid_obs.model_copy()
        obs.stress.stress_level = 100.0
        # max is 5
        history = [10.0, 20.0, 30.0, 40.0, 50.0] 
        state = self.css.build_state(obs, dialogue_history=[], previous_stress_history=history)
        self.assertEqual(len(state.stress_history), 5)
        self.assertEqual(state.stress_history[-1], 100.0)
        self.assertEqual(state.stress_history[0], 20.0)

    def test_stress_history_no_fabricated(self):
        """9. Stress history does not contain fabricated None values."""
        state = self.css.build_state(self.valid_obs, dialogue_history=[], previous_stress_history=[])
        self.assertEqual(len(state.stress_history), 0)

    def test_dialogue_history_bounded(self):
        """10. Dialogue history is bounded."""
        dialogue = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"}
        ]
        # max is 3
        state = self.css.build_state(self.valid_obs, dialogue_history=dialogue)
        self.assertEqual(len(state.dialogue_context), 3)
        self.assertEqual(state.dialogue_context[0]["content"], "2")

    def test_academic_context_optional(self):
        """11. Academic context remains optional."""
        state = self.css.build_state(self.valid_obs, dialogue_history=[])
        self.assertIsNone(state.academic_context)
        
        state_with = self.css.build_state(self.valid_obs, dialogue_history=[], academic_context="Exam tomorrow")
        self.assertEqual(state_with.academic_context, "Exam tomorrow")

    def test_previous_intervention_optional(self):
        """12. Previous intervention remains optional."""
        state = self.css.build_state(self.valid_obs, dialogue_history=[])
        self.assertIsNone(state.previous_intervention)

    def test_css_does_not_modify_mhs(self):
        """13. CSS does not calculate or modify Mental Health Score."""
        # Check that it blindly accepts the score rather than overriding it
        obs = self.valid_obs.model_copy()
        obs.visual.mental_health_score = 12.34
        state = self.css.build_state(obs, dialogue_history=[])
        self.assertEqual(state.mental_health_score, 12.34)

if __name__ == '__main__':
    unittest.main()
