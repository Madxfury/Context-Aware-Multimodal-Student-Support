from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone

class NLPResult(BaseModel):
    """
    Result from the deterministic rule-based NLP Classifier.
    """
    intent: Optional[str] = None
    secondary_intents: List[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    intent_confidence: float = 0.0
    sentiment_confidence: float = 0.0
    matched_intent_rules: List[str] = Field(default_factory=list)
    matched_sentiment_rules: List[str] = Field(default_factory=list)
    multimodal_fallback_used: bool = False
    model_available: bool = True

class NLPHistoryEntry(BaseModel):
    """
    Historical snapshot of an NLP evaluation for a single student message.
    """
    message: str
    intent: Optional[str]
    secondary_intents: List[str] = Field(default_factory=list)
    sentiment: Optional[str]
    intent_confidence: float
    sentiment_confidence: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
