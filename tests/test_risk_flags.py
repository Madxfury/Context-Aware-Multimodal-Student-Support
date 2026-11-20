"""Report flags for suicide, self-harm and other statements a counsellor must see."""
import pytest

import app as app_module
from services.risk_flags import matches, review


@pytest.mark.parametrize("message, category", [
    ("I wrote a suicide note last night", "self_harm"),
    ("Maine socha hai zeher kha lu", "self_harm"),
    ("मैं ज़हर खा लूँगा", "self_harm"),
    ("मला गळफास घ्यावासा वाटतो", "self_harm"),
    ("Sometimes I wish I could sleep forever and never wake up", "indirect_self_harm"),
    ("No one would miss me if I was gone", "indirect_self_harm"),
    ("I'm a burden to my family", "indirect_self_harm"),
    ("I can't go on like this", "indirect_self_harm"),
    ("Main bojh hoon sab pe", "indirect_self_harm"),
    ("मुझे लगता है मैं सब पर बोझ हूँ", "indirect_self_harm"),
    ("मी सगळ्यांवर ओझं आहे", "indirect_self_harm"),
    ("I want to kill him for what he did", "harm_to_others"),
    ("Main use jaan se maar dunga", "harm_to_others"),
    ("मैं उसे मार डालूँगा", "harm_to_others"),
    ("My father hits me when he is drunk", "abuse_or_violence"),
    ("Seniors are ragging me in the hostel", "abuse_or_violence"),
    ("मेरे साथ रैगिंग होती है", "abuse_or_violence"),
    ("मला घरी छळ सहन करावा लागतो", "abuse_or_violence"),
])
def test_extreme_statements_are_flagged(message, category):
    assert category in matches(message)


@pytest.mark.parametrize("message", [
    "It hit me that exams are close",
    "I could not wake up on time for class",
    "I can't go on the trip this weekend",
    "Maine syllabus sab khatam kar liya",
    "I would never hurt someone",
    "This exam is killing me",
    "I feel stressed about my presentation",
    "मुझे परीक्षा की चिंता है",
    "मला परीक्षेची भीती वाटते",
])
def test_everyday_statements_are_not_flagged(message):
    assert matches(message) == []


def test_review_uses_the_chat_safety_rules_and_groups_categories():
    transcript = [
        {"role": "user", "content": "I feel suicidal"},
        {"role": "assistant", "content": "Are you safe right now? ... call Tele MANAS at 14416."},
        {"role": "user", "content": "My roommate harasses me every night"},
        {"role": "user", "content": "Thanks for listening"},
    ]
    result = review(transcript, app_module.routed_to_safety)
    assert result["flagged"] and result["urgent"] and result["messages"] == 2 and result["support_shown"]
    assert [c["id"] for c in result["categories"]] == ["self_harm", "abuse_or_violence"]
    assert result["categories"][0]["statements"] == ["I feel suicidal"]
    assert review([{"role": "user", "content": "Exams are stressful"}], app_module.routed_to_safety)["flagged"] is False
    assert review([], app_module.routed_to_safety) == {"flagged": False, "urgent": False, "messages": 0,
                                                       "categories": [], "support_shown": False}
