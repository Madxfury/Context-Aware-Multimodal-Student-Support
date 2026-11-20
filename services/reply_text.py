"""Turn model replies into the plain text the chat bubbles and speech output expect."""

import re

# Chat bubbles show text as-is and replies are also spoken, so markdown markers would be
# shown or read out literally. Line breaks are kept: the bubbles preserve them.
_RULES = (
    (re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL), r"\1"),
    (re.compile(r"`([^`\n]+)`"), r"\1"),
    (re.compile(r"\[([^\]\n]+)\]\([^)\s]+\)"), r"\1"),
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),
    (re.compile(r"__(.+?)__", re.DOTALL), r"\1"),
    (re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])"), r"\1"),
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE), ""),
    (re.compile(r"^[ \t]*(?:[-*+•][ \t]+)", re.MULTILINE), ""),
    (re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE), ""),
    (re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE), ""),
    (re.compile(r"[ \t]+$", re.MULTILINE), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
)


def plain_reply(text):
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Common Hindi/Marathi words that mark a message typed in English letters (Hinglish).
_ROMAN_HINDI_WORDS = {
    "hai", "hain", "nahi", "nahin", "mujhe", "mera", "meri", "mere", "kya", "kyu", "kyun", "bahut", "yaar",
    "raha", "rahi", "rahe", "kuch", "aur", "lekin", "matlab", "samajh", "karna", "karu", "kaise", "kab",
    "mein", "hoon", "tha", "thi", "lagta", "lagti", "ghar", "padhai", "accha", "acha",
    "aahe", "kasa", "majha", "majhi",
}


def wrong_script(message, reply):
    """True when the reply's script does not match how the student wrote.

    Hinglish (Hindi in English letters) should get English letters back, and a message
    in Devanagari should get Devanagari back.
    """
    if DEVANAGARI.search(message):
        return not DEVANAGARI.search(reply)
    words = re.findall(r"[a-z]+", message.lower())
    hinglish = sum(word in _ROMAN_HINDI_WORDS for word in words) >= 2
    return hinglish and bool(DEVANAGARI.search(reply))


# Common words that differ between Marathi and Hindi (words used in both, such as का, हो and
# आता, are left out). The letter ळ is common in Marathi and rare in Hindi.
MARATHI_WORDS = {"आहे", "आहेत", "आहात", "आहेस", "नाही", "नाहीत", "मी", "मला", "माझं", "माझी", "माझ्या", "तू",
                 "तुला", "तुझ्या", "तुम्ही", "तुम्हाला", "तुमच्या", "तुमचं", "आम्ही", "आपण", "आणि", "पण", "तर",
                 "काय", "कसं", "कसे", "काही", "नको", "नकोस", "करा", "करू", "करूया", "करून", "केलं", "झालं", "होईल",
                 "होतं", "वाटतं", "वाटलं", "वाटत", "वाटते", "सांगा", "सांगितलं", "म्हणून", "बरं", "असं", "थोडं", "हा",
                 "होतोय", "खूप"}
HINDI_WORDS = {"है", "हैं", "नहीं", "मैं", "मुझे", "आप", "आपको", "आपके", "आपकी", "और", "की", "के", "में",
               "को", "क्या", "यह", "वह", "हूँ", "हूं", "भी", "लेकिन", "अब", "थोड़ा", "बताइए", "करते", "चलिए",
               "रहा", "रही", "सकते", "सकता", "सकती", "लिए", "साथ", "बहुत", "करूँ", "करूं"}


def devanagari_language(text):
    """'mr' or 'hi' for Devanagari text with clear Marathi or Hindi words, else None."""
    words = re.findall(r"[\u0900-\u097F]+", text)
    if not words:
        return None
    marathi = sum(word in MARATHI_WORDS for word in words) + text.count("ळ")
    hindi = sum(word in HINDI_WORDS for word in words)
    if marathi == hindi:
        return None
    return "mr" if marathi > hindi else "hi"


def wrong_language(message, reply):
    """True when the student wrote Marathi and the reply is Hindi, or the other way round.
    Both use Devanagari, so wrong_script cannot tell them apart."""
    asked = devanagari_language(message)
    answered = devanagari_language(reply)
    return asked is not None and answered is not None and asked != answered
