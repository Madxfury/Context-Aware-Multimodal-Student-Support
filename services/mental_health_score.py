RISK_WEIGHTS = {
    'Happy': 0.00,
    'Neutral': 0.30,
    'Surprise': 0.35,
    'Angry': 0.75,
    'Disgust': 0.75,
    'Sad': 0.80,
    'Fear': 0.85
}

MENTAL_HEALTH_THRESHOLD = 55.0

def calculate_mental_health_score(probabilities: dict) -> float:
    """
    Calculates the Mental Health Score strictly based on the 7 emotion probabilities,
    as defined by the paper's formulation.
    
    MS_t = 100 * sum(r_e * p_t^(e))
    """
    score = 0.0
    for emotion, prob in probabilities.items():
        # Case insensitive match just in case, but prefer Title Case
        weight = RISK_WEIGHTS.get(emotion.title(), 0.0)
        score += weight * prob
        
    final_score = 100.0 * score
    return float(round(final_score, 1))

def get_risk_level(score: float) -> str:
    """
    Returns the risk level string based on the 55.0 threshold.
    """
    if score >= MENTAL_HEALTH_THRESHOLD:
        return "high"
    return "normal"
