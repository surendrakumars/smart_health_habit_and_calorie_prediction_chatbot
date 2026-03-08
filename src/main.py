import json
import os
from datetime import datetime
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from ml import (
    estimate_bmr,
    estimate_intake,
    get_available_models,
    load_or_train_model,
    predict_calories,
    train_model_from_dataset,
)
from nlp import extract_health_data
from recommender import build_recommendations
from scoring import compute_health_score, detect_risks

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")

DATA_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "session_log.json")

DEFAULT_USER_DATA = {
    "steps": None,
    "sleep_hours": None,
    "water_intake": None,
    "food_category": None,
    "mood": None,
    "workout_intensity": None,
    "weight": None,
}

MODEL_BUNDLE, TRAIN_METRICS = load_or_train_model()


def load_history() -> list[dict[str, Any]]:
    if not os.path.exists(DATA_LOG_PATH):
        return []
    try:
        with open(DATA_LOG_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:  # noqa: BLE001
        return []


def append_history(record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(DATA_LOG_PATH), exist_ok=True)
    history = load_history()
    history.append(record)
    with open(DATA_LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def _round_or_none(value: Any, digits: int = 2) -> Any:
    if value is None:
        return None
    return round(float(value), digits)


def _build_bot_reply(extracted: Dict[str, Any], summary: Dict[str, Any]) -> str:
    details = []
    if extracted.get("steps") is not None:
        details.append(f"{int(extracted['steps'])} steps")
    if extracted.get("sleep_hours") is not None:
        details.append(f"{extracted['sleep_hours']} hours of sleep")
    if extracted.get("water_intake") is not None:
        details.append(f"{extracted['water_intake']} L water")
    if extracted.get("workout_intensity") is not None:
        details.append(f"{extracted['workout_intensity'].lower()} workout intensity")
    if extracted.get("food_category") is not None:
        details.append(f"{extracted['food_category'].lower()} food intake")
    if extracted.get("weight") is not None:
        details.append(f"{extracted['weight']} kg body weight")
    if extracted.get("mood") is not None:
        details.append(f"{extracted['mood']} mood")
    concrete_count = sum(
        1
        for key in ["steps", "sleep_hours", "water_intake", "workout_intensity", "food_category", "weight"]
        if extracted.get(key) is not None
    )

    immediate_recs = []
    trend_recs = []
    for rec in summary.get("recommendations", []):
        if rec.strip().lower() == "great job maintaining a balanced routine today.":
            continue
        if rec.lower().startswith("weekly trend") or rec.lower().startswith("early warning"):
            trend_recs.append(rec)
        else:
            immediate_recs.append(rec)

    lines = ["Here is your health update for today."]
    if details:
        lines.append("From your message, I used: " + ", ".join(details) + ".")
    lines.append(f"Estimated calories burned today: {summary['predicted_calories']} kcal.")
    if summary.get("health_score") is not None:
        lines.append(f"Current health score: {summary['health_score']}/100.")
    if summary.get("net_balance") is not None:
        lines.append(f"Estimated net calorie balance: {summary['net_balance']} kcal.")
    if summary.get("risks"):
        lines.append("Risk note: " + " ".join(summary["risks"]))

    if immediate_recs:
        lines.append("What to do next:")
        for idx, rec in enumerate(immediate_recs[:3], start=1):
            lines.append(f"{idx}. {rec}")
    elif extracted.get("mood") == "negative":
        lines.append("You seem low today. Try a short walk, hydrate, and aim for better sleep tonight.")
    elif not summary.get("risks"):
        lines.append("You are doing well today. Keep this consistency and maintain the same routine tomorrow.")
    if trend_recs and concrete_count >= 2:
        lines.append("Long-term insight: " + " ".join(trend_recs[:1]))

    return "\n".join(lines)


@app.get("/")
def index():
    return render_template(
        "index.html",
        training_metrics=TRAIN_METRICS,
        available_models=get_available_models(),
    )


@app.post("/api/reset")
def reset_state():
    return jsonify({"message": "Session reset. Share your latest health update."})


@app.post("/api/retrain")
def retrain_model():
    global MODEL_BUNDLE, TRAIN_METRICS
    payload = request.get_json(silent=True) or {}
    selected_model_raw = payload.get("selected_model")
    selected_model = str(selected_model_raw).strip() if selected_model_raw is not None else ""
    if selected_model.lower() == "auto":
        selected_model = ""

    try:
        MODEL_BUNDLE, TRAIN_METRICS = train_model_from_dataset(
            selected_model=selected_model or None
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "message": "Model retrained from dataset.",
            "metrics": TRAIN_METRICS,
        }
    )


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    user_data = DEFAULT_USER_DATA.copy()
    extracted = extract_health_data(message, use_llm=True)

    for key, value in extracted.items():
        if value is not None:
            user_data[key] = value

    predicted_calories = predict_calories(MODEL_BUNDLE, user_data)
    bmr = estimate_bmr(user_data["weight"]) if user_data.get("weight") is not None else None
    intake = estimate_intake(user_data["food_category"]) if user_data.get("food_category") is not None else None
    net_balance = (intake - (bmr + predicted_calories)) if (intake is not None and bmr is not None) else None

    health_score, score_breakdown = compute_health_score(user_data)
    risks = detect_risks(user_data)

    recommendations = build_recommendations(
        data=user_data,
        predicted_calories=predicted_calories,
        bmr=bmr,
        net_balance=net_balance,
        health_score=health_score,
        score_breakdown=score_breakdown,
        risks=risks,
        history=load_history(),
    )

    record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        **user_data,
        "predicted_calories": float(predicted_calories),
        "bmr": None if bmr is None else float(bmr),
        "intake": None if intake is None else float(intake),
        "net_balance": None if net_balance is None else float(net_balance),
        "health_score": None if health_score is None else float(health_score),
    }
    append_history(record)

    summary_payload = {
        "predicted_calories": _round_or_none(predicted_calories),
        "bmr": _round_or_none(bmr),
        "intake": _round_or_none(intake),
        "net_balance": _round_or_none(net_balance),
        "health_score": _round_or_none(health_score),
        "score_breakdown": score_breakdown,
        "risks": risks,
        "recommendations": recommendations,
    }

    return jsonify(
        {
            "status": "complete",
            "extracted": user_data,
            "summary": summary_payload,
            "bot_reply": _build_bot_reply(user_data, summary_payload),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
