from typing import List, Dict, Optional
from schemas.observation import Observation
from schemas.css import ContextualStudentState
from schemas.nlp import NLPResult, NLPHistoryEntry

class CSSManager:
    """
    Contextual Student State Manager.
    Builds the CSS from Observations and maintains bounded memory.
    """
    def __init__(self, max_dialogue_turns: int = 10, max_stress_history: int = 20):
        self.max_dialogue_turns = max_dialogue_turns
        self.max_stress_history = max_stress_history

    def _bound_dialogue(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Keeps only the most recent N turns of the conversation."""
        if not history:
            return []
        # Return the last max_dialogue_turns elements
        return history[-self.max_dialogue_turns:]

    def _bound_stress_history(self, history: List[float], current_stress: Optional[float]) -> List[float]:
        """Maintains bounded stress history array containing only real values."""
        new_history = list(history)
        if current_stress is not None:
            new_history.append(current_stress)
        return new_history[-self.max_stress_history:]

    def _calculate_stress_change(self, current_stress: Optional[float], history: List[float]) -> Optional[float]:
        """
        Calculates ΔS_t = S_t - S_(t-1).
        Only works if both current stress and previous stress exist.
        """
        if current_stress is not None and history:
            return current_stress - history[-1]
        return None

    def _bound_nlp_history(self, history: List[NLPHistoryEntry], current_nlp: Optional[NLPResult], transcript: str) -> List[NLPHistoryEntry]:
        """Maintains bounded NLP history array containing only meaningful states."""
        new_history = list(history)
        if current_nlp and current_nlp.intent != "unknown" and transcript.strip():
            entry = NLPHistoryEntry(
                message=transcript,
                intent=current_nlp.intent,
                secondary_intents=current_nlp.secondary_intents,
                sentiment=current_nlp.sentiment,
                intent_confidence=current_nlp.intent_confidence,
                sentiment_confidence=current_nlp.sentiment_confidence
            )
            new_history.append(entry)
        return new_history[-20:] # Bounded to 20

    def build_state(self, 
                    observation: Observation, 
                    dialogue_history: List[Dict[str, str]], 
                    academic_context: Optional[str] = None,
                    previous_stress_history: List[float] = None,
                    previous_nlp_history: List[NLPHistoryEntry] = None,
                    previous_intervention: Optional[str] = None) -> ContextualStudentState:
        """
        Constructs the Contextual Student State deterministically.
        """
        if previous_stress_history is None:
            previous_stress_history = []
        if previous_nlp_history is None:
            previous_nlp_history = []

        current_stress = observation.stress.stress_level
        stress_conf = observation.stress.confidence
        
        # Calculate derived stress state
        stress_change = self._calculate_stress_change(current_stress, previous_stress_history)
        bounded_stress_history = self._bound_stress_history(previous_stress_history, current_stress)
        bounded_dialogue = self._bound_dialogue(dialogue_history)
        
        nlp_result = observation.audio.nlp_result if observation.audio else None
        transcript = observation.audio.transcript if observation.audio and observation.audio.transcript else ""
        
        bounded_nlp_history = self._bound_nlp_history(previous_nlp_history, nlp_result, transcript)

        state = ContextualStudentState(
            current_stress=current_stress,
            stress_confidence=stress_conf,
            stress_change=stress_change,
            stress_history=bounded_stress_history,
            
            mental_health_score=observation.visual.mental_health_score,
            mental_health_risk=observation.visual.mental_health_score >= 55.0 and "high" or "normal",
            current_emotion=observation.visual.emotion,
            emotion_probabilities=observation.visual.emotion_probabilities,
            
            current_nlp=nlp_result,
            nlp_history=bounded_nlp_history,
            
            academic_context=academic_context,
            dialogue_context=bounded_dialogue,
            
            previous_intervention=previous_intervention
        )
        return state
