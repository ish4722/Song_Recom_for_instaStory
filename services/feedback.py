import json
import os
import threading
from collections import defaultdict

from config import FEEDBACK_PATH

_LOCK = threading.Lock()


def _load():
    if not os.path.exists(FEEDBACK_PATH):
        return {}
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def record_feedback(user_id, artist, track, feedback):
    if feedback not in {"like", "dislike"}:
        raise ValueError("feedback must be like or dislike")
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with _LOCK:
        data = _load()
        user = data.setdefault(user_id, [])
        user.append({"artist": artist, "track": track, "feedback": feedback})
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def get_feedback(user_id):
    return _load().get(user_id, [])


def artist_preferences(user_id):
    scores = defaultdict(float)
    for item in get_feedback(user_id):
        scores[item["artist"]] += 1.0 if item["feedback"] == "like" else -1.0
    return dict(scores)
