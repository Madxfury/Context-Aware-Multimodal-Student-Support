"""
Centralized NLP rule configuration for deterministic intent and sentiment classification.
"""

# Base weights for text scoring
WEIGHTS = {
    "keyword": 0.10,
    "strong_keyword": 0.15,
    "phrase": 0.25,
    # A single strong phrase is enough evidence on its own (the final threshold is 0.40).
    "strong_phrase": 0.40
}

# Removed before scoring so idioms do not trigger a rule ("cut myself some slack").
EXCLUDED_PHRASES = (
    "cut myself some slack", "cutting myself some slack",
    "cut myself off", "cutting myself off",
    "suicide prevention", "suicide awareness",
)

# Safety phrases are only negated by a word directly before them ("I don't want to die"),
# so "I never thought I'd want to die" still reaches the safety path.
NEGATION_WINDOWS = {"safety_concern": 1}

NEGATION_WORDS = {
    "not", "don't", "dont", "doesn't", "doesnt", "never", 
    "no", "isn't", "isnt", "wasn't", "wasnt", "can't", "cant"
}

INTENT_RULES = {
    "academic_performance": {
        "keywords": ["exam", "exams", "examination", "marks", "grades", "grade", "midterm", "midterms", "semester", "result", "results", "score", "performance", "backlog", "backlogs", "gpa", "cgpa", "kt", "atkt", "percentage"],
        "strong_keywords": ["failed", "fail"],
        "phrases": ["did badly", "bad performance", "poor performance", "low marks", "scored badly", "poor marks", "mid semester", "performed badly",
                    "cgpa dropped", "gpa dropped", "marks dropped", "grades dropped", "low cgpa", "low gpa", "got a backlog", "got a kt"],
        "strong_phrases": ["failed my exam", "bad exam result", "my grades dropped"]
    },
    "academic_pressure": {
        "keywords": ["deadline", "deadlines", "assignment", "assignments", "project", "projects", "submission", "submissions", "syllabus", "workload", "overloaded", "pressure", "stress", "tension", "studies", "coursework", "practicals", "viva"],
        "strong_keywords": [],
        "phrases": ["exam tomorrow", "exams tomorrow", "exam next week", "exams next week", "too much studying", "too much work", "pressure from college",
                    "exam pressure", "study pressure", "too much to study"],
        "strong_phrases": ["too many assignments", "can't keep up", "cannot keep up", "falling behind",
                           "overwhelmed by my studies", "overwhelmed with my studies", "overwhelmed with studies",
                           "stressed about my exam", "stressed about my exams", "stressed about exams",
                           "can't focus on my studies", "can't concentrate on my studies"]
    },
    "career_uncertainty": {
        "keywords": ["career", "careers", "job", "jobs", "placement", "placements", "internship", "internships", "future", "profession", "package", "salary", "interview", "interviews"],
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
        "keywords": ["lonely", "loneliness", "alone", "isolated", "nobody", "excluded", "ignored"],
        "strong_keywords": [],
        "phrases": ["no friends", "left out", "no one to talk to", "nobody to talk to"],
        "strong_phrases": ["feel alone", "feel lonely", "feeling lonely", "feeling alone", "feel isolated", "so lonely", "really lonely",
                           "nobody understands me", "no one talks to me", "nobody talks to me",
                           "don't have any friends", "dont have any friends", "have no friends"]
    },
    "general_emotional_distress": {
        "keywords": ["terrible", "horrible", "overwhelmed", "exhausted", "empty", "hopeless", "miserable", "struggling", "suffering", "worthless", "numb", "drained", "depressed"],
        "strong_keywords": [],
        "phrases": ["so stressed", "really stressed", "stressed out", "burnt out", "burned out", "falling apart", "breaking down", "mentally exhausted"],
        "strong_phrases": ["don't know what to do", "can't handle this", "everything feels too much", "everything is too much",
                           "so overwhelmed", "feel overwhelmed", "feeling overwhelmed", "feel hopeless", "feeling hopeless",
                           "feel worthless", "feeling worthless", "feel empty", "feeling empty", "feeling low", "feel low",
                           "feeling down", "feel down", "can't cope", "cannot cope"]
    },
    "study_planning": {
        "keywords": ["revise", "revision", "prepare", "preparation"],
        "strong_keywords": [],
        "phrases": ["study plan", "study schedule", "timetable", "help me study", "study strategy", "how to study", "how to prepare"],
        "strong_phrases": ["how should i study", "how do i study", "how do i prepare", "make a study plan", "how should i revise", "plan my studies"]
    },
    "problem_solving": {
        "keywords": [],
        "strong_keywords": [],
        "phrases": ["what should i do", "what can i do", "what do i do", "any advice", "what would you suggest"],
        "strong_phrases": ["how can i fix", "help me figure out", "how do i handle", "how should i deal with", "how do i deal with",
                           "how can i deal with", "how to deal with", "how do i cope", "how can i cope", "help me decide"]
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
        "phrases": ["dog died", "cat died", "pet died", "lost someone", "lost my pet", "passed away"],
        "strong_phrases": ["someone passed away", "death in my family", "i lost someone", "missing someone who died"]
    },
    "safety_concern": {
        # Romanised Hindi and Marathi are included because students often type Hinglish.
        "keywords": ["suicide", "suicidal", "khudkushi", "aatmahatya", "atmahatya"],
        "strong_keywords": [],
        "phrases": [],
        "strong_phrases": [
            "hurt myself", "hurting myself", "harm myself", "harming myself", "self harm", "self-harm",
            "self harming", "self-harming", "selfharm", "cut myself", "cutting myself", "hang myself",
            "kill myself", "end my life", "ending my life", "end my own life", "take my own life",
            "taking my own life", "take my life", "end it all",
            "want to die", "wanna die", "wanting to die", "want to be dead", "rather be dead", "better off dead",
            "better off without me", "don't want to live", "dont want to live", "don't want to be alive",
            "no reason to live", "nothing to live for", "not worth living", "no point in living", "no point living",
            "wish i was dead", "wish i were dead", "wish i was never born", "wish i had never been born",
            "if i was gone", "if i were gone", "if i disappeared", "disappear forever", "goodbye letter", "goodbye note",
            "want everything to end", "want it all to end", "want my life to end",
            "giving away my things", "giving away my stuff", "giving away my belongings", "gave away my things",
            "marna hai", "marna chahta", "marna chahti", "mar jana chahta", "mar jaana chahta",
            "mar jana chahti", "mar jaana chahti", "jeena nahi chahta", "jeena nahi chahti",
            "jina nahi chahta", "jina nahi chahti", "zinda nahi rehna", "khud ko maar", "khud ko khatam",
            "marayche aahe", "marayche ahe", "jagaycha nahi", "jagayche nahi",
        ]
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
    },
    {
        "words": [["studies", "study", "studying", "exam", "exams", "assignment", "assignments", "syllabus", "coursework", "deadline", "deadlines"],
                  ["overwhelmed", "stressed", "stress", "tension", "pressure", "burnt out", "burned out", "behind", "too much"]],
        "intent": "academic_pressure",
        "boost": 0.30
    },
    {
        "words": [["died", "passed away", "death", "funeral"],
                  ["grandfather", "grandmother", "grandpa", "grandma", "father", "mother", "dad", "mom", "papa", "mummy",
                   "brother", "sister", "friend", "uncle", "aunt", "nana", "nani", "dada", "dadi", "dog", "cat", "pet"]],
        "intent": "grief_or_loss",
        "boost": 0.40
    },
    # Method plus intent: these reach the safety path without a self-harm phrase.
    {
        "words": [["jump off", "jump from", "jumping off", "jumping from"],
                  ["roof", "terrace", "building", "bridge", "balcony", "cliff", "floor", "window"]],
        "intent": "safety_concern",
        "boost": 1.0
    },
    {
        "words": [["pills", "tablets", "medicine", "medicines", "meds"],
                  ["overdose", "overdosed", "end it", "end my life", "die", "whole bottle", "whole strip", "all of them", "too many", "a lot of"]],
        "intent": "safety_concern",
        "boost": 1.0
    }
]
