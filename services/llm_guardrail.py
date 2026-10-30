import re
from typing import List, Dict, Tuple
from schemas.guardrail import GuardrailResult

class LLMGuardrailService:
    def __init__(self):
        # 1. System Prompt Extraction Patterns
        self.extraction_patterns = [
            r"(?i)\b(reveal|show|print|output|display)\b.*?\b(system prompt|developer instructions|hidden instructions|guardrails|developer message|system message)\b",
            r"(?i)\bwhat\s+(are|were)\s+your\s+(hidden\s+instructions|developer\s+prompt|system\s+prompt)\b"
        ]

        # 2. Role Manipulation Patterns
        self.role_manipulation_patterns = [
            r"(?i)\b(act\s+as|you\s+are\s+now)\b.*?\b(hacker|unrestricted ai|programmer|developer|DAN)\b",
            r"(?i)\bignore\s+(counselling|your\s+role)\b.*?\b(become|act\s+as)\b",
            r"(?i)\b(forget|disregard)\b.*?\b(that\s+you\s+are\s+a\s+counsellor)\b"
        ]

        # 3. Instruction Override Patterns (Prompt Injection)
        # Avoid false positives like "teacher told me to ignore"
        self.override_patterns = [
            r"(?i)^(ignore|forget|disregard)\b.*?\b(previous\s+instructions|system\s+prompt|the\s+above)\b",
            r"(?i)\bfollow\s+these\s+new\s+instructions\b",
            r"(?i)\boverride\s+your\s+rules\b",
            r"(?i)^(developer\s+message|system\s+message|user\s+instruction)\s*:"
        ]

        # 4. Malicious / Unsafe Requests
        self.malicious_patterns = [
            r"(?i)\b(write|create|generate)\b.*?\b(malware|ransomware|virus|phishing\s+site)\b",
            r"(?i)\b(give|tell|show|steal)\b.*?\b(password|credentials|api\s+key)\b",
            r"(?i)\b(hack\s+into|unauthorized\s+access)\b"
        ]

        # 5. Unrelated Hijacking (Redirect)
        # We look for explicit commands to perform non-counselling technical tasks.
        # Avoid false positives like "stressed about my Python exam" by looking for imperative verbs combined with tech tasks.
        self.hijack_patterns = [
            r"(?i)\b(give\s+me|write\s+me|generate|show\s+me)\b.*?\b(python\s+code|c\+\+\s+code|javascript\s+code|basic\s+code)\b",
            r"(?i)\b(solve\s+this\s+math\s+problem|do\s+my\s+homework)\b"
        ]
        
        # 6. Exceptions (allow)
        # If these are present, they strongly imply a legitimate context
        self.exception_patterns = [
            r"(?i)\b(stressed|anxious|worried|sad|upset|angry)\b.*?\b(exam|assignment|class|teacher|homework|project)\b"
        ]

    def check_input(self, message: str) -> GuardrailResult:
        if not message or not message.strip():
            return GuardrailResult(allowed=True, action="allow", category="none", confidence=0.0)

        normalized = " ".join(message.split())
        
        # Check Malicious first (BLOCK)
        for pattern in self.malicious_patterns:
            if re.search(pattern, normalized):
                return GuardrailResult(
                    allowed=False, action="block", category="malicious_request", 
                    confidence=0.9, matched_rules=[pattern], reason="Malicious activity request detected."
                )

        # Check Extraction (BLOCK)
        for pattern in self.extraction_patterns:
            if re.search(pattern, normalized):
                return GuardrailResult(
                    allowed=False, action="block", category="system_prompt_extraction", 
                    confidence=0.9, matched_rules=[pattern], reason="System prompt extraction attempt detected."
                )

        # Check Role Manipulation (BLOCK)
        for pattern in self.role_manipulation_patterns:
            if re.search(pattern, normalized):
                return GuardrailResult(
                    allowed=False, action="block", category="role_manipulation", 
                    confidence=0.9, matched_rules=[pattern], reason="Role manipulation attempt detected."
                )

        # Check Override/Injection (BLOCK)
        for pattern in self.override_patterns:
            # We look for explicit commands typically at the start or phrased directly at the AI.
            # Using regex matching with anchors or direct imperatives.
            if re.search(pattern, normalized):
                return GuardrailResult(
                    allowed=False, action="block", category="instruction_override", 
                    confidence=0.8, matched_rules=[pattern], reason="Instruction override detected."
                )
                
        # Check Unrelated Hijacking (REDIRECT)
        for pattern in self.hijack_patterns:
            if re.search(pattern, normalized):
                # Before redirecting, ensure it's not a legitimate mixed request that just happens to match broadly.
                # However, explicit "give me python code" is usually a redirect even if mixed.
                return GuardrailResult(
                    allowed=False, action="redirect", category="unrelated_task", 
                    confidence=0.8, matched_rules=[pattern], reason="Unrelated task request detected."
                )

        # Default ALLOW
        return GuardrailResult(
            allowed=True, action="allow", category="none", 
            confidence=1.0, matched_rules=[]
        )

    def check_output(self, response: str) -> GuardrailResult:
        if not response:
            return GuardrailResult(allowed=True, action="allow", category="none", confidence=0.0)
            
        normalized = " ".join(response.split())
        
        # 1. System Prompt Leakage
        leakage_patterns = [
            r"(?i)\b(system\s+message|developer\s+instruction)\b.*?(you\s+are\s+an\s+empathetic\s+ai|strict)",
            r"(?i)\b(api_key|api\s+key)\b"
        ]
        
        # 2. Malicious Content Generation
        malicious_outputs = [
            r"(?i)\b(import\s+os\s*;?\s*os\.system|subprocess\.Popen)\b",  # Basic RCE signatures in python outputs
            r"(?i)\b(def\s+ransomware|payload)\b"
        ]
        
        # 3. Explicit claims of physiological diagnoses from sensors
        sensor_claims = [
            r"(?i)\b(your\s+(heart|heart\s*rate|pulse)\s+(is|seems\s+to\s+be)\s+racing)\b",
            r"(?i)\b(your\s+(physiological|stress|sensor)\s+(data|score|level)\s+shows)\b",
            r"(?i)\b(I\s+detected\s+that\s+you\s+are\s+physiologically\s+stressed)\b",
            r"(?i)\b(your\s+mental\s+health\s+score\s+is)\b",
            r"(?i)\b(your\s+heart\s+rate\s+is)\b"
        ]
        
        for pattern in leakage_patterns:
            if re.search(pattern, normalized):
                return GuardrailResult(
                    allowed=False, action="block", category="system_prompt_extraction", 
                    confidence=0.9, matched_rules=[pattern], reason="System prompt leakage detected."
                )
                
        for pattern in malicious_outputs:
            if re.search(pattern, normalized):
                return GuardrailResult(
                    allowed=False, action="block", category="malicious_request", 
                    confidence=0.9, matched_rules=[pattern], reason="Malicious payload generation detected."
                )
                
        for pattern in sensor_claims:
            if re.search(pattern, normalized):
                return GuardrailResult(
                    allowed=False, action="block", category="unsafe_content", 
                    confidence=0.9, matched_rules=[pattern], reason="Explicit physiological symptom assertion detected."
                )

        return GuardrailResult(allowed=True, action="allow", category="none", confidence=1.0)
