import re
from typing import Any, Dict, Optional

from llm_extractor import extract_with_primary_llm

WATER_GLASS_LITERS = 0.25
STEPS_PER_KM = 1312
STEPS_PER_WALKING_MINUTE = 100

POSITIVE_WORDS = {
    "good", "great", "happy", "energized", "energetic", "positive", "fine", "relaxed", "awesome"
}
NEGATIVE_WORDS = {
    "bad", "sad", "tired", "exhausted", "stressed", "angry", "down", "low", "anxious"
}
NEUTRAL_WORDS = {
    "neutral", "okay", "ok", "average", "normal"
}

WORKOUT_MAP = {
    "none": "None",
    "rest": "None",
    "light": "Light",
    "walk": "Light",
    "yoga": "Light",
    "moderate": "Moderate",
    "jog": "Moderate",
    "run": "Moderate",
    "gym": "Moderate",
    "heavy": "Heavy",
    "intense": "Heavy",
    "hiit": "Heavy",
    "weights": "Heavy",
}

FOOD_MAP = {
    "light": "Light",
    "salad": "Light",
    "snack": "Light",
    "moderate": "Moderate",
    "normal": "Moderate",
    "balanced": "Moderate",
    "heavy": "Heavy",
    "pizza": "Heavy",
    "burger": "Heavy",
    "fried": "Heavy",
}


EXPECTED_KEYS = {
    "steps", "sleep_hours", "water_intake", "food_category", "mood", "workout_intensity", "weight"
}


def _extract_steps(text: str) -> Optional[int]:
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(k)?\s*(steps?|walked|stp)")
    match = pattern.search(text)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):
        value *= 1000
    return int(value)


def _extract_distance_km(text: str) -> Optional[float]:
    km_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(km|kilometers|kilometres)")
    km_match = km_pattern.search(text)
    if km_match:
        return float(km_match.group(1))

    mile_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(mile|miles)")
    mile_match = mile_pattern.search(text)
    if mile_match:
        return float(mile_match.group(1)) * 1.60934

    return None


def _extract_walking_minutes(text: str) -> Optional[float]:
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(minutes|min|mins)\s*(walk|walking)")
    match = pattern.search(text)
    if match:
        return float(match.group(1))
    return None


def _extract_sleep_hours(text: str) -> Optional[float]:
    pattern = re.compile(r"(slept|sleep)\s*(around|about)?\s*(\d+(?:\.\d+)?)\s*(hours|hrs|h)")
    match = pattern.search(text)
    if match:
        return float(match.group(3))
    pattern2 = re.compile(r"(\d+(?:\.\d+)?)\s*(hours|hrs|h)\s*(of)?\s*sleep")
    match2 = pattern2.search(text)
    if match2:
        return float(match2.group(1))
    pattern3 = re.compile(r"(around|about)\s*(\d+(?:\.\d+)?)\s*(hours|hrs|h)")
    match3 = pattern3.search(text)
    if match3:
        return float(match3.group(2))
    pattern4 = re.compile(r"^(\d+(?:\.\d+)?)\s*(hours|hrs|h)$")
    match4 = pattern4.search(text.strip())
    if match4:
        return float(match4.group(1))
    return None


def _extract_water(text: str) -> Optional[float]:
    liters_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(liters|liter|l)")
    match = liters_pattern.search(text)
    if match:
        return float(match.group(1))

    glasses_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(glasses|glass)")
    match2 = glasses_pattern.search(text)
    if match2:
        return float(match2.group(1)) * WATER_GLASS_LITERS
    return None


def _extract_weight(text: str) -> Optional[float]:
    kg_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|kilograms|kilos)")
    match = kg_pattern.search(text)
    if match:
        return float(match.group(1))
    lb_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(lb|lbs|pounds)")
    match2 = lb_pattern.search(text)
    if match2:
        pounds = float(match2.group(1))
        return pounds * 0.453592
    return None


def _extract_mood(text: str) -> Optional[str]:
    # Handle negated mood phrases before keyword matching.
    negated_positive = re.search(r"\b(not|don't|do not|never)\s+(feeling\s+)?(good|great|happy|fine|awesome|positive)\b", text)
    if negated_positive:
        return "negative"

    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if tokens & POSITIVE_WORDS:
        return "positive"
    if tokens & NEGATIVE_WORDS:
        return "negative"
    if tokens & NEUTRAL_WORDS:
        return "neutral"
    return None


def _extract_workout_intensity(text: str) -> Optional[str]:
    # Handle common low-activity phrases before keyword search.
    if re.search(r"\b(didn['’]?t|did['’]?nt|didnt|did not|no)\s+(do\s+)?(much\s+)?(workout|exercise|activity)\b", text):
        return "None"
    if re.search(r"\b(not\s+much|barely|very\s+little)\s+(workout|exercise|activity)\b", text):
        return "None"

    # Prefer the strongest explicit intensity found in the message.
    rank = {"None": 0, "Light": 1, "Moderate": 2, "Heavy": 3}
    chosen = None
    chosen_rank = -1
    for key, value in WORKOUT_MAP.items():
        if key in text and rank[value] > chosen_rank:
            chosen = value
            chosen_rank = rank[value]
    return chosen


def _extract_food_category(text: str) -> Optional[str]:
    # Handle free-form food volume phrases.
    if re.search(r"\b(a\s+lot\s+of|too\s+much|overeating|overeat|binge)\s+(food|ate|eating)\b", text):
        return "Heavy"
    if re.search(r"\b(very\s+light|small|little)\s+(meal|food|ate)\b", text):
        return "Light"

    for key, value in FOOD_MAP.items():
        if key in text:
            return value
    return None


def _normalize_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    try:
        if key == "steps":
            return int(float(value))
        if key in {"sleep_hours", "water_intake", "weight"}:
            return float(value)
        if key == "food_category":
            normalized = str(value).strip().capitalize()
            return normalized if normalized in {"Light", "Moderate", "Heavy"} else None
        if key == "mood":
            normalized = str(value).strip().lower()
            return normalized if normalized in {"positive", "neutral", "negative"} else None
        if key == "workout_intensity":
            normalized = str(value).strip().capitalize()
            return normalized if normalized in {"None", "Light", "Moderate", "Heavy"} else None
    except Exception:  # noqa: BLE001
        return None
    return None


def _rule_based_extract(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    steps = _extract_steps(lowered)
    if steps is None:
        distance_km = _extract_distance_km(lowered)
        if distance_km is not None:
            steps = int(distance_km * STEPS_PER_KM)
    if steps is None:
        walking_minutes = _extract_walking_minutes(lowered)
        if walking_minutes is not None:
            steps = int(walking_minutes * STEPS_PER_WALKING_MINUTE)

    return {
        "steps": steps,
        "sleep_hours": _extract_sleep_hours(lowered),
        "water_intake": _extract_water(lowered),
        "food_category": _extract_food_category(lowered),
        "mood": _extract_mood(lowered),
        "workout_intensity": _extract_workout_intensity(lowered),
        "weight": _extract_weight(lowered),
    }


def extract_health_data(text: str, use_llm: bool = True) -> Dict[str, Any]:
    combined = {k: None for k in EXPECTED_KEYS}

    llm_data = None
    if use_llm:
        llm_data, _ = extract_with_primary_llm(text)

    if isinstance(llm_data, dict):
        for key in EXPECTED_KEYS:
            combined[key] = _normalize_value(key, llm_data.get(key))

    rule_data = _rule_based_extract(text)
    for key, value in rule_data.items():
        # Rule mood extraction is more reliable for negation phrases (e.g., "not happy").
        if key == "mood" and value is not None:
            combined[key] = value
            continue
        if combined[key] is None and value is not None:
            combined[key] = value

    return combined


def summarize_extraction(data: Dict[str, Any]) -> str:
    known = {k: v for k, v in data.items() if v is not None}
    if not known:
        return "Tell me about your day: steps, sleep, water, food, mood, workout, and weight."
    return f"Current extracted info: {known}"


def missing_fields_prompt(data: Dict[str, Any]) -> str:
    missing = [k for k, v in data.items() if v is None]
    if not missing:
        return ""
    prompts = {
        "steps": "If you do not know steps, share approximate walking distance (e.g., 5 km) or walking minutes.",
        "sleep_hours": "How many hours did you sleep?",
        "water_intake": "How much water did you drink (liters or glasses)?",
        "food_category": "How would you rate your food intake (Light/Moderate/Heavy)?",
        "mood": "How are you feeling today (positive/neutral/negative)?",
        "workout_intensity": "How intense was your workout (None/Light/Moderate/Heavy)?",
        "weight": "What is your current weight in kg (or lbs)?",
    }
    return prompts[missing[0]]
