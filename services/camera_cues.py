"""Camera facial cues for each check-in.

The browser sends only frames it judged clear (face found, facing the camera, close enough,
lit, still and not mid-speech; see static/js/face-tracker.js). Their emotion probabilities are
averaged, and an expression label is reported only when the average top probability reaches the
clear threshold; otherwise the cue is "unclear" instead of a guess. Facial movement scores
(smiling, lowered or raised brows, eye openness) are compared with the student's own check-in
baseline.

Thresholds, from never-seen FANE faces and the project author's own posed webcam frames:
- The Swin-FANE model alone (CLEAR_CONFIDENCE = 0.80): right 81% of the time at this
  confidence on FANE (70% of faces), but on webcam faces under ~200 px it drifts to "Angry".
- Swin-FANE averaged with HSEmotion enet_b2_7 (ENSEMBLE_CLEAR_CONFIDENCE = 0.60): right 82% of
  the time at this confidence on FANE and 91-95% on the webcam frames at every face size.

These describe facial expression only. They are not a measure of how the student feels.
"""

from collections import OrderedDict
from math import isfinite
from threading import Lock
from time import monotonic

EMOTIONS = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
CUES = ("smile", "frown", "brow_down", "brow_raise", "eye_openness", "jaw_open")
CLEAR_CONFIDENCE = 0.80           # Swin-FANE alone
ENSEMBLE_CLEAR_CONFIDENCE = 0.60  # Swin-FANE + HSEmotion average
MIN_MINUTE_FRAMES = 3             # clear frames needed to label a minute of the conversation
MIN_CHECKIN_FRAMES = 8        # fewer clear frames than this cannot give a check-in label
MIN_CHAT_FRAMES = 5           # per CHAT_WINDOW during the conversation
CHAT_WINDOW = 120             # seconds of conversation the current cue describes
CUE_CHANGE = 0.15             # movement-score change (0-1 scale) described as a change
MAX_FRAMES = 1800             # per phase and session (about 90 minutes at one frame per 3 s)
SESSION_LIMIT = 128
# Described movements. Jaw opening follows speech and frowning is rare in the training faces,
# so they are recorded but not described.
CUE_WORDS = {
    "smile": ("smiling more", "smiling less"),
    "brow_down": ("brows lowered more", "brows lowered less"),
    "brow_raise": ("brows raised more", "brows raised less"),
    "eye_openness": ("eyes more open", "eyes more closed"),
}


def clean_probabilities(values):
    """Seven finite, non-negative probabilities in EMOTIONS order, normalised; else None."""
    if not isinstance(values, (list, tuple)) or len(values) != len(EMOTIONS):
        return None
    if not all(isinstance(v, (int, float)) and isfinite(v) and v >= 0 for v in values):
        return None
    total = float(sum(values))
    return [float(v) / total for v in values] if total > 0 else None


def clean_cues(values):
    """Known movement scores clipped to 0-1; unknown or invalid entries are dropped."""
    if not isinstance(values, dict):
        return None
    cues = {}
    for name in CUES:
        value = values.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
            cues[name] = min(1.0, max(0.0, float(value)))
    return cues or None


def _label(frames, minimum, threshold):
    count = len(frames)
    if not count:
        return {"frames": 0, "clear": False, "emotion": None, "confidence": None, "probabilities": None}
    mean = [sum(frame[1][k] for frame in frames) / count for k in range(len(EMOTIONS))]
    best = max(range(len(EMOTIONS)), key=lambda k: mean[k])
    clear = count >= minimum and mean[best] >= threshold
    return {
        "frames": count,
        "clear": clear,
        "emotion": EMOTIONS[best] if clear else None,
        "confidence": round(100 * mean[best], 1),
        "probabilities": dict(zip(EMOTIONS, (round(value, 4) for value in mean))),
    }


def _cue_means(frames):
    totals, counts = {}, {}
    for frame in frames:
        for name, value in (frame[2] or {}).items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    return {name: round(totals[name] / counts[name], 3) for name in totals}


def _changes(baseline, current):
    words = []
    for name, (more, less) in CUE_WORDS.items():
        if name in baseline and name in current:
            delta = current[name] - baseline[name]
            if delta >= CUE_CHANGE:
                words.append(more)
            elif delta <= -CUE_CHANGE:
                words.append(less)
    return words


class CameraCues:
    def __init__(self, clear_confidence=CLEAR_CONFIDENCE):
        self.clear_confidence = clear_confidence
        self.lock = Lock()
        self.sessions = OrderedDict()

    def record(self, session_id, phase, probabilities, cues=None, now=None):
        """Store one clear frame. Returns False when the frame is invalid."""
        probabilities = clean_probabilities(probabilities)
        if phase not in ("checkin", "chat") or probabilities is None:
            return False
        now = monotonic() if now is None else now
        with self.lock:
            entry = self.sessions.setdefault(session_id, {"checkin": [], "chat": []})
            self.sessions.move_to_end(session_id)
            while len(self.sessions) > SESSION_LIMIT:
                self.sessions.popitem(last=False)
            frames = entry[phase]
            frames.append((now, probabilities, clean_cues(cues)))
            if len(frames) > MAX_FRAMES:
                del frames[: len(frames) - MAX_FRAMES]
        return True

    def _frames(self, session_id):
        entry = self.sessions.get(session_id) or {"checkin": [], "chat": []}
        return list(entry["checkin"]), list(entry["chat"])

    def checkin(self, session_id):
        with self.lock:
            checkin, _ = self._frames(session_id)
        result = _label(checkin, MIN_CHECKIN_FRAMES, self.clear_confidence)
        result["baseline"] = _cue_means(checkin)
        return result

    def recent(self, session_id, now=None):
        now = monotonic() if now is None else now
        with self.lock:
            checkin, chat = self._frames(session_id)
        window = [frame for frame in chat if now - frame[0] <= CHAT_WINDOW]
        result = _label(window, MIN_CHAT_FRAMES, self.clear_confidence)
        result["changes"] = _changes(_cue_means(checkin), _cue_means(window)) if len(window) >= MIN_CHAT_FRAMES else []
        return result

    def evidence(self, session_id, now=None):
        """The most recent clear expression cue, or None: the last CHAT_WINDOW of the
        conversation if it is clear, otherwise a clear check-in."""
        recent = self.recent(session_id, now)
        if recent["clear"]:
            return {**recent, "source": "conversation"}
        checkin = self.checkin(session_id)
        if checkin["clear"]:
            return {**checkin, "source": "check-in"}
        return None

    def context(self, session_id, now=None):
        """One line for the counsellor's private context."""
        checkin = self.checkin(session_id)
        recent = self.recent(session_id, now)
        parts = []
        if not checkin["frames"] and not recent["frames"]:
            return "not captured (no camera)"
        if checkin["clear"]:
            parts.append(f"check-in: mostly {checkin['emotion']} expression "
                         f"({checkin['confidence']:.0f}% model confidence over {checkin['frames']} clear frames)")
        elif checkin["frames"]:
            parts.append("check-in: unclear (the expression model was not confident)")
        if recent["frames"]:
            if recent["clear"]:
                text = f"last {CHAT_WINDOW // 60} minutes: mostly {recent['emotion']} expression ({recent['confidence']:.0f}%)"
            else:
                text = f"last {CHAT_WINDOW // 60} minutes: unclear"
            if recent["changes"]:
                text += "; compared with the check-in: " + ", ".join(recent["changes"])
            parts.append(text)
        return "; ".join(parts)

    def summary(self, session_id):
        """Report summary, or None when no camera frames were recorded."""
        with self.lock:
            checkin, chat = self._frames(session_id)
        if not checkin and not chat:
            return None
        result = {"clear_confidence": round(100 * self.clear_confidence),
                  "checkin": _label(checkin, MIN_CHECKIN_FRAMES, self.clear_confidence)}
        if chat:
            conversation = _label(chat, MIN_CHAT_FRAMES, self.clear_confidence)
            conversation["minutes"] = round((chat[-1][0] - chat[0][0]) / 60, 1)
            conversation["changes"] = _changes(_cue_means(checkin), _cue_means(chat)) if len(chat) >= MIN_CHAT_FRAMES else []
            # Minute by minute from the first conversation frame.
            buckets = {}
            for frame in chat:
                buckets.setdefault(int((frame[0] - chat[0][0]) // 60), []).append(frame)
            conversation["timeline"] = []
            for minute in sorted(buckets):
                label = _label(buckets[minute], MIN_MINUTE_FRAMES, self.clear_confidence)
                conversation["timeline"].append({"minute": minute + 1, "frames": label["frames"],
                                                 "emotion": label["emotion"], "confidence": label["confidence"]})
            result["conversation"] = conversation
        return result
