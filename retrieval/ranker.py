import re

from config import IMAGE_WEIGHT, MOOD_WEIGHT, PREFERENCE_WEIGHT, TEXT_WEIGHT

MOOD_TERMS = {
    "happy": {"happy", "joy", "joyful", "fun", "bright", "celebration", "smile", "cheerful", "uplifting"},
    "sad": {"sad", "lonely", "rain", "cry", "tears", "melancholy", "heartbreak", "grief"},
    "energetic": {"energy", "dance", "party", "running", "sport", "fast", "excited", "vibrant"},
    "calm": {"calm", "peaceful", "quiet", "relaxed", "serene", "soft", "sunset"},
    "romantic": {"love", "romantic", "couple", "kiss", "date", "heart", "affection"},
    "nostalgic": {"old", "memory", "nostalgia", "childhood", "vintage", "retro", "remember"},
    "uplifting": {"hope", "positive", "inspiring", "victory", "freedom", "smile", "uplifting"},
    "melancholic": {"melancholy", "lonely", "rain", "night", "sad", "reflective"},
    "dramatic": {"dramatic", "storm", "intense", "powerful", "cinematic", "dark"},
    "peaceful": {"peaceful", "nature", "quiet", "serene", "relaxed", "calm"},
}


def _tokens(text):
    return set(re.findall(r"[a-zA-Z]+", (text or "").lower()))


def mood_score(query, song_text, mood):
    terms = MOOD_TERMS.get((mood or "").lower(), set())
    if not terms:
        return 0.0
    query_hits = len(_tokens(query) & terms)
    song_hits = len(_tokens(song_text) & terms)
    return min(1.0, 0.5 * bool(query_hits) + 0.5 * min(song_hits / max(1, len(terms)), 1.0))


def preference_score(artist, preferences):
    raw = preferences.get(artist, 0.0)
    return max(-1.0, min(1.0, raw / 3.0))


def rerank(candidates, query_description, mood=None, preferences=None):
    preferences = preferences or {}
    ranked = []
    for item in candidates:
        song = item["song"]
        semantic = float(item.get("semantic_score", 0.0))
        image = float(item.get("image_score", 0.0))
        mood_value = mood_score(query_description, song.get("description", ""), mood)
        pref_value = preference_score(song.get("artist", ""), preferences)
        final = TEXT_WEIGHT * semantic + IMAGE_WEIGHT * image + MOOD_WEIGHT * mood_value + PREFERENCE_WEIGHT * pref_value
        ranked.append({**item, "mood_score": mood_value, "preference_score": pref_value, "score": final})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def explain(song, result):
    reasons = []
    if result.get("semantic_score", 0) >= 0.65:
        reasons.append("strong semantic match to the image context")
    if result.get("image_score", 0) >= 0.65:
        reasons.append("strong visual-to-music embedding match")
    if result.get("mood_score", 0) > 0:
        reasons.append("compatible mood cues")
    if result.get("preference_score", 0) > 0:
        reasons.append("matches your previous liked artists")
    if not reasons:
        reasons.append("best available semantic match")
    return "Recommended because of " + ", ".join(reasons) + "."
