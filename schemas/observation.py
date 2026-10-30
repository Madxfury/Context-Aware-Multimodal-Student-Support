from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict
from datetime import datetime, timezone

EXPECTED_EMOTIONS = {"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"}

class PhysiologicalData(BaseModel):
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None
    gsr: Optional[float] = None
    motion: Optional[float] = None
    v2_sequence: Optional[list] = None
    session_id: str = "default_session"

class VisualData(BaseModel):
    emotion: str
    confidence: float = Field(ge=0.0, le=100.0)
    emotion_probabilities: Dict[str, float]
    mental_health_score: float = Field(ge=0.0, le=100.0)

    @field_validator('emotion_probabilities')
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        # Normalize keys to Title Case for safety, but check exactly against EXPECTED_EMOTIONS
        normalized_v = {k.title(): val for k, val in v.items()}
        
        # Check exactly 7 emotions
        missing = EXPECTED_EMOTIONS - set(normalized_v.keys())
        extra = set(normalized_v.keys()) - EXPECTED_EMOTIONS
        
        if missing:
            raise ValueError(f"Missing emotion probabilities: {missing}")
        if extra:
            raise ValueError(f"Unexpected emotion probabilities: {extra}")
            
        # Check range
        for emo, prob in normalized_v.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability for {emo} must be between 0 and 1, got {prob}")
                
        # Check sum to ~1.0
        prob_sum = sum(normalized_v.values())
        if abs(prob_sum - 1.0) > 1e-3:
            raise ValueError(f"Emotion probabilities must sum to 1.0, got {prob_sum}")
            
        return normalized_v

class FatigueData(BaseModel):
    blink_rate: Optional[float] = None
    fatigue_score: Optional[float] = None

class MotionData(BaseModel):
    head_motion: Optional[float] = None

from schemas.nlp import NLPResult

class AudioData(BaseModel):
    transcript: Optional[str] = None
    language: Optional[str] = None
    nlp_result: Optional[NLPResult] = None

class StressData(BaseModel):
    stress_level: Optional[float] = None
    confidence: Optional[float] = None
    predicted_class: Optional[str] = None
    model_available: bool = False

class Observation(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    physiological: PhysiologicalData
    visual: VisualData
    fatigue: FatigueData
    motion: MotionData
    audio: AudioData
    stress: StressData
