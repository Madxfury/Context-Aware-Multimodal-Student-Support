from pydantic import BaseModel
from typing import Optional, List, Dict
from schemas.observation import AudioData
from schemas.nlp import NLPResult, NLPHistoryEntry

class ContextualStudentState(BaseModel):
    """
    Contextual Student State (CSS) schema.
    Represents the explicit contextual state used to make adaptive responses.
    """
    # Stress State (S_t) - strictly physiological
    current_stress: Optional[float] = None
    stress_confidence: Optional[float] = None
    stress_change: Optional[float] = None  # ΔS_t
    stress_history: List[float] = []       # H_t
    
    # Emotion & Mental Health (MS_t) - strictly facial-emotion-derived
    mental_health_score: float
    mental_health_risk: str
    current_emotion: str
    emotion_probabilities: Dict[str, float]
    
    # NLP & Speech Data
    current_nlp: Optional[NLPResult] = None
    nlp_history: List[NLPHistoryEntry] = []
    
    # Context & Memory
    academic_context: Optional[str] = None
    dialogue_context: List[Dict[str, str]] = []
    
    # Intervention
    previous_intervention: Optional[str] = None
