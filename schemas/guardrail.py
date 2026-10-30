from pydantic import BaseModel, Field
from typing import Optional, List

class GuardrailResult(BaseModel):
    allowed: bool = Field(description="True if allowed to proceed without blocking.")
    action: str = Field(description="allow, redirect, or block")
    reason: Optional[str] = Field(None, description="Detailed explanation for the action.")
    category: Optional[str] = Field(None, description="none, prompt_injection, role_manipulation, system_prompt_extraction, instruction_override, unrelated_task, malicious_request, unsafe_content, credential_theft, malware, phishing, cyber_abuse, other")
    confidence: float = Field(0.0, description="Confidence of the guardrail classification, 0.0 to 1.0.")
    matched_rules: List[str] = Field(default_factory=list, description="List of matched patterns or rules that triggered this result.")
