"""Application-owned referrals for Samvya's India deployment."""

import re
from unicodedata import category, decimal, normalize


SUPPORT_MARKER = "[[INDIA_SUPPORT]]"

# Tele MANAS is maintained by MoHFW. 112 is the national emergency number (MHA ERSS).
SUPPORT_CONTACTS = {
    "en": "India: For immediate danger, call 112. For free, 24/7 mental health support, call Tele MANAS at 14416.",
    "hi": "भारत में तत्काल खतरे की स्थिति में 112 पर कॉल करें। मुफ़्त मानसिक स्वास्थ्य सहायता के लिए किसी भी समय टेली मानस के 14416 पर कॉल करें।",
    "mr": "भारतात तातडीचा धोका असल्यास 112 वर कॉल करा. मोफत मानसिक आरोग्य सहाय्यासाठी कोणत्याही वेळी टेली मानसच्या 14416 या क्रमांकावर कॉल करा.",
}

SAFETY_CHECK_INS = {
    "en": "I’m sorry things feel so hard. Are you safe right now? If you can, tell someone you trust nearby and ask them to stay with you.",
    "hi": "मुझे दुख है कि आप इतनी तकलीफ़ में हैं। क्या आप अभी सुरक्षित हैं? हो सके तो किसी भरोसेमंद व्यक्ति को बताइए और उनसे अपने साथ रहने को कहिए।",
    "mr": "तुम्हाला इतका त्रास होत आहे, याचं मला वाईट वाटतं. तुम्ही सध्या सुरक्षित आहात का? शक्य असल्यास, जवळच्या विश्वासू व्यक्तीला सांगा आणि त्यांना तुमच्यासोबत राहायला सांगा.",
}

REGIONAL_SAFETY_PHRASES = (
    "आत्महत्या", "खुद को मार", "खुद को नुकसान", "जीना नहीं चाहता", "जीना नहीं चाहती",
    "मरना चाहता", "मरना चाहती", "स्वतःला इजा", "स्वतःला संपव", "जीव द्यावासा",
    "जीव द्यायचा", "जगायचं नाही", "जगायचे नाही", "मला मरायचं", "मला मरायचे",
)
REFERRAL_TERMS = re.compile(
    r"\b(?:help[ -]?lines?|hotlines?|crisis lines?|lifelines?|emergency (?:(?:phone )?numbers?|services)|"
    r"suicide (?:phone )?numbers?|tele[ -]?manas)\b|"
    r"हेल्प\s*लाइन|(?:आपत्कालीन|आपातकालीन) (?:क्रमांक|नंबर|सेवा)|मदतीचा (?:क्रमांक|नंबर)|टेली मानस",
    re.IGNORECASE,
)
CALL_TERMS = re.compile(
    r"\b(?:call|dial|text|phone)\b|कॉल|फ़ोन|फोन|क्रमांक|नंबर", re.IGNORECASE
)
CONTACT_LINK = re.compile(r"(?:https?://|www\.|\b(?:tel|sms|mailto):)", re.IGNORECASE)
PHONE_LIKE = re.compile(r"\d(?:[\s().+*\-‐‑–—]*\d){2,}")

TIMED_RANGE = re.compile(
    r"(?<!\d)\d{1,2}\s*[-‐‑–—]\s*\d{1,2}\s*"
    r"(?:minutes?|mins?|seconds?|secs?|hours?|मिनिट[\u0900-\u097f]*|मिनट|तास|घंटे)(?!\w)",
    re.IGNORECASE,
)


CLOCK_TIME = re.compile(r"(?<![\d:])(?:[01]?\d|2[0-3]):[0-5]\d(?![\d:])")
SENSOR_VALUE = re.compile(
    r"(?<![\d.])[-+]?\d+(?:[.,]\d+)?\s*(?:ADC(?:\s+counts)?|mV|millivolts?|BPM|beats?\s+(?:per|a|/)\s*min(?:ute)?|"
    r"°\s*C|degrees?\s+(?:C|Celsius)|m/s2|rad/s|%|percent)(?!\w)",
    re.IGNORECASE,
)
SENTENCE_BREAK = re.compile(r"((?<=[.!?।॥])\s+|\n+)")


def support_language(text):
    if re.search(r"मला|माझ्या|तुम्हाला|तुम्ही|आहे|करा|क्रमांक|नाही|स्वतःला", text):
        return "mr"
    return "hi" if re.search(r"[\u0900-\u097f]", text) else "en"


def support_message(text, *, safety_check_in=False):
    language = support_language(text)
    contacts = SUPPORT_CONTACTS[language]
    if safety_check_in:
        return f"{SAFETY_CHECK_INS[language]}\n\n{contacts}"
    return contacts


def has_generated_contact(text):
    normalized = "".join(
        str(decimal(char)) if char.isdecimal() else char
        for char in normalize("NFKC", text)
        if category(char) != "Cf"
    )
    if REFERRAL_TERMS.search(normalized) or CONTACT_LINK.search(normalized):
        return True
    normalized = TIMED_RANGE.sub(" ", CLOCK_TIME.sub(" ", normalized))
    if not CALL_TERMS.search(normalized):
        normalized = SENSOR_VALUE.sub(" ", normalized)
    return bool(PHONE_LIKE.search(normalized))


def without_generated_contacts(text):
    if not has_generated_contact(text):
        return text
    parts = SENTENCE_BREAK.split(text)
    kept = "".join(
        part + separator
        for part, separator in zip(parts[::2], parts[1::2] + [""])
        if not has_generated_contact(part)
    ).strip()
    # Check the joined text as well for contact details split over sentence boundaries.
    return kept if not has_generated_contact(kept) else ""


def has_referral(text):
    return bool(REFERRAL_TERMS.search(text) or CONTACT_LINK.search(text)
                or (CALL_TERMS.search(text) and PHONE_LIKE.search(text)))


def refresh_saved_referral(text):
    """Replace outdated assistant referrals on read while preserving the saved record."""
    if not isinstance(text, str):
        return text
    remaining = text
    for contacts in SUPPORT_CONTACTS.values():
        remaining = remaining.replace(contacts, "")
    if has_referral(remaining):
        return support_message(remaining)
    return text
