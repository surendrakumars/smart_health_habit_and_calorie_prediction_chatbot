from datetime import datetime, timedelta


def _weekly_summary(history):
    if not history:
        return None
    last_week = datetime.now() - timedelta(days=7)
    recent = [h for h in history if datetime.strptime(h["date"], "%Y-%m-%d") >= last_week]
    if not recent:
        return None

    steps_vals = [h["steps"] for h in recent if h.get("steps") is not None]
    sleep_vals = [h["sleep_hours"] for h in recent if h.get("sleep_hours") is not None]
    water_vals = [h["water_intake"] for h in recent if h.get("water_intake") is not None]
    score_vals = [h["health_score"] for h in recent if h.get("health_score") is not None]

    avg_steps = (sum(steps_vals) / len(steps_vals)) if steps_vals else None
    avg_sleep = (sum(sleep_vals) / len(sleep_vals)) if sleep_vals else None
    avg_water = (sum(water_vals) / len(water_vals)) if water_vals else None
    avg_score = (sum(score_vals) / len(score_vals)) if score_vals else None

    return {
        "avg_steps": avg_steps,
        "avg_sleep": avg_sleep,
        "avg_water": avg_water,
        "avg_score": avg_score,
        "days": len(recent),
    }


def build_recommendations(
    data,
    predicted_calories,
    bmr,
    net_balance,
    health_score,
    score_breakdown,
    risks,
    history=None,
):
    recs = []

    # Immediate corrective feedback
    if data.get("steps") is not None and data["steps"] < 8000:
        recs.append(f"Add about {8000 - data['steps']} steps to reach the daily goal.")
    if data.get("water_intake") is not None and data["water_intake"] < 2.3:
        delta = 2.3 - data["water_intake"]
        recs.append(f"Increase water intake by about {delta:.1f} liters.")

    # Recovery and lifestyle advice
    if data.get("sleep_hours") is not None and data["sleep_hours"] < 7:
        recs.append("Aim for 7-9 hours of sleep tonight to support recovery.")
    if (
        data.get("workout_intensity") == "Heavy"
        and data.get("sleep_hours") is not None
        and data["sleep_hours"] < 6
    ):
        recs.append("Consider a lighter workout or an active recovery session tomorrow.")
    if data.get("food_category") == "Heavy" and data.get("workout_intensity") in {"None", "Light"}:
        recs.append("You reported heavy food and low activity; add a 20-30 minute brisk walk today.")

    # Nutritional context
    if net_balance is not None and net_balance > 400:
        recs.append("Your net calorie balance is high today; consider lighter meals tomorrow.")
    elif net_balance is not None and net_balance < -400:
        recs.append("Your net calorie balance is low; add a healthy snack to refuel.")

    # Risks
    for risk in risks:
        recs.append(f"Risk check: {risk}")

    # Long-term insights
    summary = _weekly_summary(history or [])
    if summary:
        parts = []
        if summary["avg_steps"] is not None:
            parts.append(f"avg steps {summary['avg_steps']:.0f}")
        if summary["avg_sleep"] is not None:
            parts.append(f"avg sleep {summary['avg_sleep']:.1f}h")
        if summary["avg_water"] is not None:
            parts.append(f"avg water {summary['avg_water']:.1f}L")
        if parts:
            recs.append(f"Weekly trend: {', '.join(parts)} over {summary['days']} day(s).")
        if summary["avg_score"] is not None and summary["avg_score"] < 65:
            recs.append("Early warning: overall health score is trending low this week.")

    if not recs:
        recs.append("Great job maintaining a balanced routine today.")

    return recs
