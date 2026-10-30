import re
from typing import Optional, Dict, Tuple, List
from schemas.nlp import NLPResult
from services.nlp_rules import INTENT_RULES, SENTIMENT_RULES, COMBINATION_RULES, NEGATION_WORDS, WEIGHTS

class NLPProcessorService:
    """
    Deterministic Rule-Based NLP Processor.
    Classifies student intents and sentiments using weighted keywords, 
    phrases, and combinations with negation handling.
    """
    def __init__(self):
        self.model_available = False # True external ML model not used
        
    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and remove punctuation except for basic apostrophes
        text = text.lower()
        text = re.sub(r'[^\w\s\']', ' ', text)
        return text.split()

    def _is_negated(self, tokens: List[str], match_index: int, window: int = 3) -> bool:
        """
        Check if any negation word appears within `window` tokens BEFORE the match_index.
        """
        start = max(0, match_index - window)
        preceding_tokens = tokens[start:match_index]
        for token in preceding_tokens:
            if token in NEGATION_WORDS:
                return True
        return False

    def _score_rules(self, text: str, rules_dict: Dict) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """
        Scores intents or sentiments based on the rules_dict.
        Returns (scores, matched_rules).
        """
        scores = {k: 0.0 for k in rules_dict.keys()}
        matched = {k: [] for k in rules_dict.keys()}
        tokens = self._tokenize(text)
        
        for category, rules in rules_dict.items():
            # Keywords
            for word_list, weight_key in [
                (rules.get("keywords", []), "keyword"),
                (rules.get("strong_keywords", []), "strong_keyword")
            ]:
                for word in word_list:
                    # Check each occurrence of the word in tokens
                    for i, token in enumerate(tokens):
                        if token == word:
                            if not self._is_negated(tokens, i):
                                scores[category] += WEIGHTS[weight_key]
                                matched[category].append(f"kw:{word}")

            # Phrases
            for phrase_list, weight_key in [
                (rules.get("phrases", []), "phrase"),
                (rules.get("strong_phrases", []), "strong_phrase")
            ]:
                for phrase in phrase_list:
                    # Find all occurrences of the phrase in the string
                    # Use regex with word boundaries to ensure we don't match substrings of larger words
                    pattern = r'\b' + re.escape(phrase) + r'\b'
                    matches = list(re.finditer(pattern, text.lower()))
                    for match in matches:
                        # Map string index to token index approximately to check negation
                        # Count spaces before the match to estimate token index
                        preceding_text = text[:match.start()]
                        approx_token_idx = len(self._tokenize(preceding_text))
                        
                        if not self._is_negated(tokens, approx_token_idx):
                            scores[category] += WEIGHTS[weight_key]
                            matched[category].append(f"ph:{phrase}")

        return scores, matched

    def _apply_combinations(self, text: str, intent_scores: Dict[str, float], matched_rules: Dict[str, List[str]]):
        """
        Boosts intent scores based on combination rules.
        """
        text_lower = text.lower()
        for rule in COMBINATION_RULES:
            # Check if at least one word from each sub-list is present
            all_groups_present = True
            for group in rule["words"]:
                group_present = False
                for word in group:
                    if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                        group_present = True
                        break
                if not group_present:
                    all_groups_present = False
                    break
            
            if all_groups_present:
                intent_scores[rule["intent"]] += rule["boost"]
                matched_rules[rule["intent"]].append(f"combo:{rule['words']}")

    def process(self, transcript: Optional[str], language: Optional[str] = "en") -> NLPResult:
        if not transcript or not transcript.strip():
            return NLPResult(intent="unknown", sentiment="unknown")

        text = transcript.strip()
        
        # 1. Score Intents
        intent_scores, intent_matches = self._score_rules(text, INTENT_RULES)
        self._apply_combinations(text, intent_scores, intent_matches)
        
        # 2. Score Sentiments
        sentiment_scores, sentiment_matches = self._score_rules(text, SENTIMENT_RULES)
        
        # 3. Determine Safety Concern (Highest Priority)
        if intent_scores.get("safety_concern", 0) > 0:
            return NLPResult(
                intent="safety_concern",
                intent_confidence=1.0,
                sentiment="sad", # Default fallback sentiment for safety
                sentiment_confidence=1.0,
                matched_intent_rules=intent_matches["safety_concern"],
                matched_sentiment_rules=[],
                multimodal_fallback_used=False
            )
            
        # 4. Resolve top intents and sentiments
        # Normalize and cap scores at 1.0
        for k in intent_scores: intent_scores[k] = min(intent_scores[k], 1.0)
        for k in sentiment_scores: sentiment_scores[k] = min(sentiment_scores[k], 1.0)
        
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_sentiments = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_intent = sorted_intents[0][0] if sorted_intents[0][1] > 0 else "unknown"
        intent_conf = sorted_intents[0][1]
        
        # Secondary intents: score > 0 and not the primary
        secondary_intents = [k for k, v in sorted_intents[1:] if v > 0][:2]
        
        primary_sentiment = sorted_sentiments[0][0] if sorted_sentiments[0][1] > 0 else "neutral"
        sentiment_conf = sorted_sentiments[0][1]
        
        # Remove intent thresholding here; we will rely on apply_multimodal_fallback or external check to threshold it,
        # but for testing purposes, if we just call process() we need it to threshold.
        # Actually, let's just do it here IF the confidence is extremely low, like < 0.10.
        if intent_conf < 0.10:
            primary_intent = "unknown"
            secondary_intents = []
            
        if sentiment_conf < 0.10 and primary_sentiment != "neutral":
            primary_sentiment = "neutral"
            
        return NLPResult(
            intent=primary_intent,
            secondary_intents=secondary_intents,
            sentiment=primary_sentiment,
            intent_confidence=intent_conf,
            sentiment_confidence=sentiment_conf,
            matched_intent_rules=intent_matches.get(primary_intent, []),
            matched_sentiment_rules=sentiment_matches.get(primary_sentiment, []),
            multimodal_fallback_used=False
        )

    def apply_multimodal_fallback(self, nlp_result: NLPResult, emotion_probs: Dict[str, float], s_t: Optional[float], ms_t: float) -> NLPResult:
        """
        Multimodal fallback may ONLY support an already text-supported classification.
        Emotion/stress/MS_t must NEVER independently create an intent.
        Activates when text confidence is below strong ( < 0.70) but has SOME evidence (> 0).
        """
        if nlp_result.intent == "safety_concern":
            return nlp_result
            
        # Fallback only if textual intent confidence has some evidence but isn't strong
        if 0 < nlp_result.intent_confidence < 0.70 and nlp_result.intent != "unknown":
            distress_intents = {"academic_performance", "academic_pressure", "general_emotional_distress", "social_isolation", "career_uncertainty", "family_conflict", "relationship_conflict", "financial_concern"}
            if nlp_result.intent in distress_intents:
                if s_t is not None and s_t > 0.7:
                    nlp_result.intent_confidence = min(1.0, nlp_result.intent_confidence + 0.35)
                    nlp_result.multimodal_fallback_used = True
                elif emotion_probs.get("Sad", 0) > 0.6 or emotion_probs.get("Fear", 0) > 0.6:
                    nlp_result.intent_confidence = min(1.0, nlp_result.intent_confidence + 0.30)
                    nlp_result.multimodal_fallback_used = True
                    
        # Apply final thresholding here so that if fallback didn't boost it enough, it becomes unknown
        if nlp_result.intent_confidence < 0.40:
            nlp_result.intent = "unknown"
            nlp_result.secondary_intents = []
            
        return nlp_result
