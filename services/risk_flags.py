"""Statements a counsellor must see, flagged in the check-in report by Samvya's own rules.

The flags never depend on the language model, so they appear even when the AI summary could
not be made. Suicide and self-harm use the same rules that route a chat message to the safety
reply (passed in as `is_self_harm`). The other categories are phrase lists in English, Hinglish
(Hindi in English letters), Hindi and Marathi. A flag means "read this statement", not a
diagnosis; the rules can miss indirect wording and can flag a statement out of context.
"""

import re

MAX_STATEMENTS = 3
MAX_STATEMENT_CHARS = 240

# (id, label for the report, urgent, phrases). English and Hinglish phrases match whole words,
# case-insensitively; Devanagari phrases match anywhere in the message.
CATEGORIES = (
    ("self_harm", "Suicide or self-harm", True, (
        "suicide note", "suicide letter", "bought a rope", "tie a rope", "noose", "slit my wrist", "slit my wrists",
        "cut my wrist", "cut my wrists", "jump in front of a train", "poison myself", "drink poison", "rat poison",
        "phenyl pee", "zeher kha", "zehar kha", "fansi laga", "faasi laga", "pankhe se latak",
        "ज़हर खा", "जहर खा", "फांसी", "फाँसी", "विष प्या", "विष पिऊन", "गळफास",
    )),
    ("indirect_self_harm", "Possible indirect signs of suicidal thinking", True, (
        "never wake up", "don't want to wake up", "dont want to wake up", "sleep forever", "no one would miss me", "nobody would miss me",
        "no one will miss me", "nobody will miss me", "no one would care if i", "nobody would care if i",
        "i am a burden", "i'm a burden", "im a burden", "burden to everyone", "burden on everyone",
        "burden to my family", "burden on my family", "can't go on like this", "can't go on anymore",
        "cant go on anymore", "cannot go on anymore",
        "tired of living", "tired of life", "tired of being alive", "life is pointless", "life is meaningless",
        "no way out", "give up on life", "giving up on life", "gave up on life", "final goodbye", "say goodbye to everyone",
        "won't be around much longer", "won't be here much longer", "end everything",
        "jeene ka mann nahi", "jeene ka man nahi", "jeena bekar", "sab khatam karna chahta", "sab khatam karna chahti",
        "sab khatam kar dunga", "sab khatam kar dungi",
        "kisi ko fark nahi padega", "main bojh hoon", "mai bojh hu", "bojh hoon", "jagnyat arth nahi",
        "जीने का मन नहीं", "सब खत्म करना चाहत", "सब ख़त्म करना चाहत", "सब खत्म कर दूँग", "सब खत्म कर दूंग", "बोझ हूँ", "बोझ हूं", "किसी को फर्क नहीं पड़ेगा",
        "किसी को फ़र्क नहीं पड़ेगा", "हमेशा के लिए सो", "ज़िंदगी से थक", "जिंदगी से थक",
        "ओझं आहे", "ओझे आहे", "कोणाला फरक पडणार नाही", "जगण्यात अर्थ नाही", "सगळं संपवून", "सगळं संपवायचं",
        "कायमचं झोप", "आयुष्याचा कंटाळा",
    )),
    ("harm_to_others", "Thoughts of harming someone else", True, (
        "kill him", "kill her", "kill them", "kill someone", "kill somebody", "kill everyone",
        "murder", "hurt him", "hurt her", "hurt them", "hurt someone", "hurt somebody", "stab", "shoot him",
        "shoot her", "shoot them", "shoot everyone", "bring a knife", "bring a gun", "get a gun",
        "maar dalunga", "maar daalunga", "maar dalungi", "maar daalungi", "jaan se maar", "khoon kar",
        "मार डालूँगा", "मार डालूंगा", "मार डालूँगी", "मार डालूंगी", "जान से मार", "खून कर", "जीवे मार", "ठार मार",
    )),
    ("abuse_or_violence", "Abuse, violence or harassment", False, (
        "abused", "abusing me", "abuses me", "abuse me", "hits me", "hitting me", "beat me up", "beats me up",
        "beating me", "slaps me", "slapped me", "molested", "molesting", "molest", "sexually harassed", "sexual harassment", "harassing me",
        "harasses me", "harassed", "raped", "rape", "touched me inappropriately", "touches me inappropriately",
        "ragging", "ragged", "domestic violence", "threatening me", "threatens me", "blackmailing me", "blackmail",
        "stalking me", "stalks me", "bully me", "bullying me", "bullied",
        "mujhe marte", "mujhe maarte", "mujhe peet", "pitai karte", "chhedte", "chhed-chhad", "chhedchhad",
        "मुझे मारते", "मुझे पीटते", "पिटाई", "छेड़", "छेड", "शोषण", "उत्पीड़न", "रैगिंग", "धमकी", "ब्लैकमेल",
        "मारहाण", "छळ", "रॅगिंग", "छेडछाड", "धमकी देत",
    )),
)
# A word directly before an English or Hinglish phrase that negates it: "I would never hurt someone".
NEGATIONS = {"never", "not", "don't", "dont", "won't", "wont", "wouldn't", "wouldnt", "nahi", "nahin", "mat"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _pattern(phrase):
    return re.compile(r"(?<![\w'])" + re.escape(phrase) + r"(?![\w'])", re.IGNORECASE)


_COMPILED = tuple(
    (cid, label, urgent, tuple((phrase, None if DEVANAGARI.search(phrase) else _pattern(phrase)) for phrase in phrases))
    for cid, label, urgent, phrases in CATEGORIES
)


def _normalise(text):
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'")).strip()


def _negated(text, start):
    before = re.findall(r"[\w']+", text[:start].lower())[-2:]
    return any(word in NEGATIONS for word in before)


def matches(text):
    """Category ids whose phrases appear in one message."""
    text = _normalise(text)
    found = []
    for cid, _, _, phrases in _COMPILED:
        for phrase, pattern in phrases:
            if pattern is None:
                if phrase in text:
                    found.append(cid)
                    break
            else:
                hit = next((m for m in pattern.finditer(text) if not _negated(text, m.start())), None)
                if hit:
                    found.append(cid)
                    break
    return found


def review(transcript, is_self_harm=None):
    """Report flags for the student's messages in a check-in transcript."""
    by_category = {cid: [] for cid, *_ in CATEGORIES}
    flagged_messages = 0
    for message in transcript or []:
        if not isinstance(message, dict) or message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        text = message["content"].strip()
        if not text:
            continue
        found = matches(text)
        if is_self_harm is not None and "self_harm" not in found and is_self_harm(text):
            found.insert(0, "self_harm")
        if "self_harm" in found and "indirect_self_harm" in found:
            found.remove("indirect_self_harm")      # report the clearer category once
        for cid in found:
            by_category[cid].append(text[:MAX_STATEMENT_CHARS])
        flagged_messages += bool(found)
    categories = [
        {"id": cid, "label": label, "urgent": urgent, "messages": len(by_category[cid]),
         "statements": by_category[cid][:MAX_STATEMENTS]}
        for cid, label, urgent, _ in CATEGORIES if by_category[cid]
    ]
    support_shown = any(
        isinstance(m, dict) and m.get("role") != "user" and "14416" in str(m.get("content", "")) for m in transcript or []
    )
    return {
        "flagged": bool(categories),
        "urgent": any(category["urgent"] for category in categories),
        "messages": flagged_messages,
        "categories": categories,
        "support_shown": support_shown,
    }
