from pydantic import BaseModel, validator, field_validator
from typing import Literal

# Controlled intervention identifiers
InterventionType = Literal[
    "supportive_check_in",
    "emotional_validation",
    "grounding_breathing",
    "study_planning",
    "problem_solving",
    "safety_support"
]

PriorityType = Literal["low", "medium", "high"]

class InterventionSelection(BaseModel):
    intervention: InterventionType
    reason: str
    priority: PriorityType
    safety_required: bool
    selector_version: str
