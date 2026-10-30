"""
Centralized NLP rule configuration for deterministic intent and sentiment classification.
"""

# Base weights for text scoring
WEIGHTS = {
    "keyword": 0.10,
    "strong_keyword": 0.15,
    "phrase": 0.25,
    "strong_phrase": 0.35
}

NEGATION_WORDS = {
    "not", "don't", "dont", "doesn't", "doesnt", "never", 
    "no", "isn't", "isnt", "wasn't", "wasnt", "can't", "cant"
}

INTENT_RULES = {
    "academic_performance": {
        "keywords": ["exam", "examination", "marks", "grades", "grade", "midterm", "semester", "result", "score", "performance", "backlog", "gpa", "cgpa"],
        "strong_keywords": ["failed", "fail"],
        "phrases": ["did badly", "bad performance", "poor performance", "low marks", "scored badly", "poor marks", "mid semester", "performed badly"],
        "strong_phrases": ["failed my exam", "bad exam result", "my grades dropped"]
    },
    "academic_pressure": {
        "keywords": ["deadline", "assignment", "assignments", "project", "submission", "syllabus", "workload", "overloaded", "pressure"],
        "strong_keywords": [],
        "phrases": ["exam tomorrow", "too much studying", "too much work", "pressure from college"],
        "strong_phrases": ["too many assignments", "can't keep up", "falling behind"]
    },
    "career_uncertainty": {
        "keywords": ["career", "job", "placement", "internship", "future", "profession", "package", "salary", "interview"],
        "strong_keywords": [],
        "phrases": ["career path"],
        "strong_phrases": ["don't know what i want", "what should i do after college", "worried about my career", "don't know what career to choose"]
    },
    "relationship_conflict": {
        "keywords": ["friend", "friends", "boyfriend", "girlfriend", "partner", "relationship", "breakup"],
        "strong_keywords": [],
        "phrases": ["my friend", "my best friend"],
        "strong_phrases": ["fight with friend", "argument with friend", "relationship problems"]
    },
    "family_conflict": {
        "keywords": ["parent", "parents", "mother", "father", "mom", "dad", "family"],
        "strong_keywords": [],
        "phrases": ["my parents", "my family"],
        "strong_phrases": ["parents are angry", "family pressure", "fight with my parents"]
    },
    "financial_concern": {
        "keywords": ["money", "financial", "finance", "fees", "fee", "debt", "expenses", "rent", "loan", "expensive", "affordability"],
        "strong_keywords": [],
        "phrases": [],
        "strong_phrases": ["can't afford", "money problems", "financial pressure"]
    },
    "social_isolation": {
        "keywords": ["lonely", "alone", "isolated", "nobody", "excluded", "ignored"],
        "strong_keywords": [],
        "phrases": ["no friends", "left out"],
        "strong_phrases": ["feel alone", "feel lonely", "nobody understands me", "no one talks to me"]
    },
    "general_emotional_distress": {
        "keywords": ["terrible", "horrible", "overwhelmed", "exhausted", "empty", "hopeless", "miserable", "struggling", "suffering"],
        "strong_keywords": [],
        "phrases": [],
        "strong_phrases": ["don't know what to do", "can't handle this", "everything feels too much"]
    },
    "study_planning": {
        "keywords": ["revise", "revision", "prepare", "preparation"],
        "strong_keywords": [],
        "phrases": ["study plan", "study schedule", "timetable", "help me study"],
        "strong_phrases": ["how should i study", "how do i prepare", "make a study plan", "how should i revise"]
    },
    "problem_solving": {
        "keywords": [],
        "strong_keywords": [],
        "phrases": ["what should i do", "what can i do"],
        "strong_phrases": ["how can i fix", "help me figure out", "how do i handle", "how should i deal with"]
    },
    "casual_conversation": {
        "keywords": ["hello", "hi", "hey", "thanks"],
        "strong_keywords": [],
        "phrases": ["thank you", "good morning", "good night", "what's up"],
        "strong_phrases": ["how are you"]
    },
    "grief_or_loss": {
        "keywords": ["grieving"],
        "strong_keywords": [],
        "phrases": ["dog died", "pet died", "lost someone", "lost my pet"],
        "strong_phrases": ["someone passed away", "death in my family", "i lost someone", "missing someone who died"]
    },
    "safety_concern": {
        "keywords": ["suicide"],
        "strong_keywords": [],
        "phrases": [],
        "strong_phrases": ["hurt myself", "harm myself", "kill myself", "end my life", "want to die", "don't want to live", "better off dead"]
    }
}

SENTIMENT_RULES = {
    "positive": {
        "keywords": ["happy", "good", "great", "excited", "relieved", "proud", "confident", "thankful", "grateful", "amazing"],
        "strong_keywords": [],
        "phrases": ["feeling better", "things are going well"],
        "strong_phrases": []
    },
    "frustrated": {
        "keywords": ["frustrated", "annoyed", "irritated"],
        "strong_keywords": [],
        "phrases": ["fed up", "sick of", "tired of", "so done"],
        "strong_phrases": ["can't stand", "nothing is working"]
    },
    "angry": {
        "keywords": ["angry", "furious", "mad", "rage", "hate"],
        "strong_keywords": [],
        "phrases": ["pissed off"],
        "strong_phrases": ["really angry", "extremely angry"]
    },
    "sad": {
        "keywords": ["sad", "crying", "hurt", "empty", "miserable", "lonely", "depressed"],
        "strong_keywords": [],
        "phrases": ["feel terrible", "feel down"],
        "strong_phrases": []
    },
    "anxious": {
        "keywords": ["worried", "nervous", "anxious", "panic", "overthinking", "uncertainty"],
        "strong_keywords": [],
        "phrases": ["what if"],
        "strong_phrases": ["can't stop worrying"]
    },
    "fearful": {
        "keywords": ["scared", "afraid", "terrified", "fear"],
        "strong_keywords": [],
        "phrases": ["really scared", "i'm afraid"],
        "strong_phrases": []
    },
    "overwhelmed": {
        "keywords": ["overwhelmed", "exhausted", "overloaded"],
        "strong_keywords": [],
        "phrases": ["too much", "burnt out"],
        "strong_phrases": ["can't handle this", "everything is too much"]
    }
}

# Contextual combination boosts
COMBINATION_RULES = [
    {
        "words": [["exam", "examination", "mid semester"], ["failed", "fail", "badly", "poor"]],
        "intent": "academic_performance",
        "boost": 0.30
    },
    {
        "words": [["exam", "examination"], ["worried", "anxious", "nervous"]],
        "intent": "academic_pressure",
        "boost": 0.30
    },
    {
        "words": [["parent", "parents", "mom", "dad"], ["angry", "mad", "fight"]],
        "intent": "family_conflict",
        "boost": 0.30
    },
    {
        "words": [["friend", "friends", "boyfriend", "girlfriend"], ["fight", "argument", "breakup"]],
        "intent": "relationship_conflict",
        "boost": 0.30
    },
    {
        "words": [["career", "job"], ["future", "uncertainty", "worried"]],
        "intent": "career_uncertainty",
        "boost": 0.25
    },
    {
        "words": [["study", "exam", "prepare"], ["plan", "schedule", "timetable"]],
        "intent": "study_planning",
        "boost": 0.25
    },
    {
        "words": [["what should i do", "what can i do"], ["overwhelmed", "exhausted", "struggling", "terrible"]],
        "intent": "problem_solving",
        "boost": 0.25
    }
]
