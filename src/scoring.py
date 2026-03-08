
def _clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def compute_health_score(data):
    weighted_sum = 0.0
    active_weight = 0.0
    breakdown = {
        "activity_score": None,
        "sleep_score": None,
        "hydration_score": None,
    }

    # Activity score: steps + workout intensity (50%)
    steps = data.get("steps")
    intensity = data.get("workout_intensity")
    if steps is not None or intensity is not None:
        intensity_bonus = {"None": 0, "Light": 5, "Moderate": 12, "Heavy": 20}.get(intensity, 0)
        steps_score = 0 if steps is None else _clamp((steps / 8000) * 80)
        activity_score = _clamp(steps_score + intensity_bonus)
        breakdown["activity_score"] = activity_score
        weighted_sum += activity_score * 0.50
        active_weight += 0.50

    # Sleep score (30%)
    sleep_hours = data.get("sleep_hours")
    if sleep_hours is not None:
        if sleep_hours < 4:
            sleep_score = 20
        elif sleep_hours < 6:
            sleep_score = 45
        elif sleep_hours < 7:
            sleep_score = 65
        elif sleep_hours <= 9:
            sleep_score = 90
        else:
            sleep_score = 70
        breakdown["sleep_score"] = sleep_score
        weighted_sum += sleep_score * 0.30
        active_weight += 0.30

    # Hydration score (20%)
    water = data.get("water_intake")
    if water is not None:
        hydration_score = _clamp((water / 2.3) * 100)
        breakdown["hydration_score"] = hydration_score
        weighted_sum += hydration_score * 0.20
        active_weight += 0.20

    if active_weight == 0:
        return None, breakdown

    normalized_score = weighted_sum / active_weight
    return _clamp(normalized_score), breakdown


def detect_risks(data):
    risks = []
    if (
        data.get("sleep_hours") is not None
        and data.get("workout_intensity") is not None
        and data["sleep_hours"] < 6
        and data["workout_intensity"] in {"Moderate", "Heavy"}
    ):
        risks.append("Low sleep with high activity may increase fatigue risk.")
    if (
        data.get("water_intake") is not None
        and data.get("workout_intensity") is not None
        and data["water_intake"] < 1.5
        and data["workout_intensity"] == "Heavy"
    ):
        risks.append("Low hydration with heavy workout increases dehydration risk.")
    return risks
