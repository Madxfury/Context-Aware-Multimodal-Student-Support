import unittest
from schemas.css import ContextualStudentState
from schemas.nlp import NLPResult
from services.intervention_selector import RuleBasedInterventionSelector
from app import app
from services.mental_health_score import calculate_mental_health_score
from services.nlp_processor import NLPProcessorService
from services.css_manager import CSSManager

class TestStep8(unittest.TestCase):

    def setUp(self):
        self.selector = RuleBasedInterventionSelector()
        self.base_css = ContextualStudentState(
            current_stress=None,
            stress_confidence=None,
            stress_change=None,
            stress_history=[],
            mental_health_score=50.0,
            mental_health_risk="normal",
            current_emotion="Neutral",
            emotion_probabilities={"Neutral": 1.0, "Happy": 0.0, "Sad": 0.0, "Fear": 0.0, "Angry": 0.0, "Disgust": 0.0, "Surprise": 0.0},
            current_nlp=None,
            academic_context=None,
            dialogue_context=[],
            previous_intervention=None
        )
        self.client = app.test_client()

    def test_default_selection(self):
        """1. Default selection -> supportive_check_in"""
        res = self.selector.select(self.base_css)
        self.assertEqual(res.intervention, "supportive_check_in")

    def test_strong_emotional_signal(self):
        """2. Strong emotional signal -> emotional_validation"""
        css = self.base_css.model_copy()
        css.current_emotion = "Sad"
        res = self.selector.select(css)
        self.assertEqual(res.intervention, "emotional_validation")

    def test_study_intent(self):
        """3. Explicit study intent -> study_planning"""
        css = self.base_css.model_copy()
        css.current_nlp = NLPResult(intent="study_planning")
        res = self.selector.select(css)
        self.assertEqual(res.intervention, "study_planning")

    def test_problem_solving_intent(self):
        """4. Explicit problem-solving intent -> problem_solving"""
        css = self.base_css.model_copy()
        css.current_nlp = NLPResult(intent="problem_solving")
        res = self.selector.select(css)
        self.assertEqual(res.intervention, "problem_solving")

    def test_elevated_stress(self):
        """5. Elevated physiological stress -> grounding_breathing"""
        css = self.base_css.model_copy()
        css.current_stress = 80.0
        res = self.selector.select(css)
        self.assertEqual(res.intervention, "grounding_breathing")

    def test_increasing_stress(self):
        """6. Increasing physiological stress -> grounding_breathing"""
        css = self.base_css.model_copy()
        css.current_stress = 50.0
        css.stress_change = 5.0
        res = self.selector.select(css)
        self.assertEqual(res.intervention, "grounding_breathing")

    def test_explicit_safety_signal(self):
        """7. Explicit safety signal -> safety_support"""
        css = self.base_css.model_copy()
        css.current_nlp = NLPResult(intent="safety_concern")
        res = self.selector.select(css)
        self.assertEqual(res.intervention, "safety_support")

    def test_safety_precedence(self):
        """8. Safety takes precedence over all other interventions."""
        css = self.base_css.model_copy()
        css.current_nlp = NLPResult(intent="safety_concern")
        css.current_stress = 100.0  # Would otherwise be grounding_breathing
        css.current_emotion = "Sad" # Would otherwise be emotional_validation
        res = self.selector.select(css)
        self.assertEqual(res.intervention, "safety_support")

    def test_missing_stress_does_not_trigger(self):
        """9. Missing physiological stress does NOT trigger physiological-stress rules."""
        # High MHS but no S_t
        css = self.base_css.model_copy()
        css.current_stress = None
        css.mental_health_score = 90.0
        res = self.selector.select(css)
        self.assertEqual(res.intervention, "supportive_check_in")

    def test_mhs_not_substituted(self):
        """10. Missing physiological stress does NOT cause MHS to become S_t."""
        css = self.base_css.model_copy()
        css.current_stress = None
        css.mental_health_score = 100.0
        # If the selector substituted MHS for stress, it would trigger grounding_breathing
        res = self.selector.select(css)
        self.assertNotEqual(res.intervention, "grounding_breathing")

    def test_selection_fields(self):
        """11. Every selection contains: reason, priority, safety_required, selector_version"""
        res = self.selector.select(self.base_css)
        self.assertTrue(hasattr(res, "reason"))
        self.assertTrue(hasattr(res, "priority"))
        self.assertTrue(hasattr(res, "safety_required"))
        self.assertTrue(hasattr(res, "selector_version"))
        self.assertEqual(res.selector_version, "rule-based-v1")

    def test_deterministic(self):
        """12. Selector is deterministic: identical CSS input -> identical intervention."""
        css = self.base_css.model_copy()
        css.current_nlp = NLPResult(intent="problem_solving")
        res1 = self.selector.select(css)
        res2 = self.selector.select(css)
        self.assertEqual(res1.intervention, res2.intervention)

    def test_chat_endpoint_works(self):
        """13. Existing /chat endpoint still works."""
        r = self.client.post("/chat", json={
            "message": "hello",
            "emotion": "Happy",
            "history": [],
            "sensors": {"hr": 80, "spo2": 98, "temp": 36.5}
        })
        self.assertEqual(r.status_code, 200)

    def test_detect_endpoint_works(self):
        """14. Existing /detect endpoint still works."""
        r = self.client.post("/detect", json={})
        self.assertIn(r.status_code, [200, 400]) # 400 means no image passed, but endpoint is up

    def test_observation_endpoint_works(self):
        """15. Existing /api/observation endpoint still works."""
        valid_data = {
            "physiological": {"heart_rate": 85.0},
            "visual": {
                "emotion": "Sad",
                "confidence": 87.0,
                "emotion_probabilities": {
                    "Angry": 0.02, "Disgust": 0.01, "Fear": 0.08, "Happy": 0.01,
                    "Sad": 0.87, "Surprise": 0.01, "Neutral": 0.00
                },
                "mental_health_score": 71.2
            },
            "fatigue": {}, "motion": {}, "audio": {}, "stress": {}
        }
        r = self.client.post("/api/observation", json=valid_data)
        self.assertEqual(r.status_code, 200)

    def test_mhs_unchanged(self):
        """16. Mental Health Score calculation remains unchanged."""
        probs = {"Angry": 0.0, "Disgust": 0.0, "Fear": 0.0, "Happy": 1.0, "Sad": 0.0, "Surprise": 0.0, "Neutral": 0.0}
        score = calculate_mental_health_score(probs)
        self.assertEqual(score, 0.0) # Happy is 0% risk

    def test_nlp_stub_unchanged(self):
        """17. Existing NLP stub behavior remains unchanged."""
        nlp = NLPProcessorService()
        res = nlp.process("test", "en")
        self.assertEqual(res.intent, "unknown")

    def test_css_bounded_history_unchanged(self):
        """18. Existing CSS bounded history behavior remains unchanged."""
        manager = CSSManager(max_stress_history=2)
        history = manager._bound_stress_history([1.0, 2.0, 3.0], 4.0)
        self.assertEqual(history, [3.0, 4.0])

if __name__ == '__main__':
    unittest.main()
