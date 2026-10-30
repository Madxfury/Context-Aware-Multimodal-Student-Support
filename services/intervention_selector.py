from schemas.css import ContextualStudentState
from schemas.intervention import InterventionSelection

class RuleBasedInterventionSelector:
    """
    IMPLEMENTED:
    - CSS-driven intervention selection
    - deterministic rules
    - auditable intervention choice
    - intervention passed separately to LLM
    
    NOT IMPLEMENTED:
    - learned intervention policy
    - probabilistic P(i | CSS_t, M_t)
    - experimental intervention evaluation
    - Gemma-2B offline fallback
    - LoRA
    - federated intervention learning
    """
    
    def __init__(self):
        self.version = "rule-based-v1"
        self.high_stress_threshold = 60.0

    def select(self, css: ContextualStudentState) -> InterventionSelection:
        """
        Deterministic, rule-based intervention selection from Contextual Student State.
        Rules evaluated in strict priority order.
        """
        # 1. SAFETY PATH (Highest Priority)
        # Check explicit safety signals in NLP
        if css.current_nlp and css.current_nlp.intent == "safety_concern":
            return InterventionSelection(
                intervention="safety_support",
                reason="Explicit safety/self-harm signal detected in NLP intent.",
                priority="high",
                safety_required=True,
                selector_version=self.version
            )
            
        # 2. STRESS TRAJECTORY
        # Check purely physiological stress (S_t and ΔS_t). Never use MS_t here.
        if css.current_stress is not None:
            elevated = css.current_stress > self.high_stress_threshold
            increasing = css.stress_change is not None and css.stress_change > 0
            
            if elevated and increasing:
                return InterventionSelection(
                    intervention="grounding_breathing",
                    reason="Physiological stress is elevated and increasing.",
                    priority="high",
                    safety_required=False,
                    selector_version=self.version
                )
            elif elevated:
                return InterventionSelection(
                    intervention="grounding_breathing",
                    reason="Physiological stress is elevated.",
                    priority="medium",
                    safety_required=False,
                    selector_version=self.version
                )
            elif increasing:
                return InterventionSelection(
                    intervention="grounding_breathing",
                    reason="Physiological stress is increasing.",
                    priority="medium",
                    safety_required=False,
                    selector_version=self.version
                )

        # 3. ACADEMIC/STUDY INTENT
        if css.current_nlp and css.current_nlp.intent == "study_planning":
            return InterventionSelection(
                intervention="study_planning",
                reason="Explicit study or academic planning intent detected.",
                priority="medium",
                safety_required=False,
                selector_version=self.version
            )

        # 4. PROBLEM SOLVING
        if css.current_nlp and css.current_nlp.intent == "problem_solving":
            return InterventionSelection(
                intervention="problem_solving",
                reason="Explicit problem-solving intent detected.",
                priority="medium",
                safety_required=False,
                selector_version=self.version
            )

        # 5. EMOTIONAL SUPPORT
        # Do not use S_t here. Use purely facial/visual emotion signals.
        strong_emotions = {"Sad", "Fear", "Angry", "Disgust"}
        if css.current_emotion in strong_emotions:
            return InterventionSelection(
                intervention="emotional_validation",
                reason=f"Strong emotion signal ({css.current_emotion}) detected without explicit safety signal.",
                priority="medium",
                safety_required=False,
                selector_version=self.version
            )

        # 6. DEFAULT
        return InterventionSelection(
            intervention="supportive_check_in",
            reason="No high-priority conditions met; defaulting to supportive check-in.",
            priority="low",
            safety_required=False,
            selector_version=self.version
        )
