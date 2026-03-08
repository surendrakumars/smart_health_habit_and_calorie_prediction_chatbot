import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "calorie_model.pkl")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Sleep_health_and_lifestyle_dataset.csv")

INTENSITY_MAP = {
    "None": 0,
    "Light": 1,
    "Moderate": 2,
    "Heavy": 3,
}


BMI_WEIGHT_PROXY = {
    "underweight": 55.0,
    "normal": 70.0,
    "normal weight": 70.0,
    "overweight": 82.0,
    "obese": 95.0,
}


def _normalize_col(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


def _derive_workout_intensity(activity_level: float) -> str:
    if activity_level < 20:
        return "None"
    if activity_level < 40:
        return "Light"
    if activity_level < 70:
        return "Moderate"
    return "Heavy"


def _prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [_normalize_col(c) for c in frame.columns]

    frame["steps"] = pd.to_numeric(frame.get("daily_steps"), errors="coerce")
    frame["sleep_hours"] = pd.to_numeric(frame.get("sleep_duration"), errors="coerce")

    activity_raw = pd.to_numeric(frame.get("physical_activity_level"), errors="coerce")
    frame["workout_intensity"] = activity_raw.apply(
        lambda v: _derive_workout_intensity(float(v)) if pd.notna(v) else "Light"
    )

    bmi_series = frame.get("bmi_category", pd.Series(["normal"] * len(frame)))
    frame["weight"] = bmi_series.astype(str).str.lower().map(BMI_WEIGHT_PROXY).fillna(70.0)

    calorie_col = None
    for candidate in ["calories_burned", "calories", "daily_calories_burned"]:
        if candidate in frame.columns:
            calorie_col = candidate
            break

    if calorie_col is not None:
        frame["target_calories"] = pd.to_numeric(frame[calorie_col], errors="coerce")
    else:
        heart_rate = pd.to_numeric(frame.get("heart_rate"), errors="coerce").fillna(75.0)
        stress = pd.to_numeric(frame.get("stress_level"), errors="coerce").fillna(5.0)
        intensity_code = frame["workout_intensity"].map(INTENSITY_MAP).fillna(1)
        frame["target_calories"] = (
            frame["steps"].fillna(5000) * 0.04
            + activity_raw.fillna(40) * 6
            + heart_rate * 1.5
            + stress * 5
            + intensity_code * 65
        )

    frame = frame[["steps", "sleep_hours", "workout_intensity", "weight", "target_calories"]]
    frame = frame.dropna(subset=["target_calories"])
    return frame


def _build_preprocessor() -> ColumnTransformer:
    numeric_features = ["steps", "sleep_hours", "weight"]
    categorical_features = ["workout_intensity"]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

def _build_pipeline(model: Any) -> Pipeline:
    return Pipeline(steps=[("preprocess", _build_preprocessor()), ("model", model)])


def _candidate_models() -> Dict[str, Any]:
    return {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }


def get_available_models() -> list[str]:
    return list(_candidate_models().keys())


def _is_better(metrics_a: Dict[str, float], metrics_b: Dict[str, float], r2_tol: float = 1e-3) -> bool:
    # Prefer higher R2. If R2 values are very close, prefer lower MAE.
    if metrics_a["r2"] > metrics_b["r2"] + r2_tol:
        return True
    if abs(metrics_a["r2"] - metrics_b["r2"]) <= r2_tol:
        return metrics_a["mae"] < metrics_b["mae"]
    return False


def train_model_from_dataset(
    dataset_path: str = DATASET_PATH, selected_model: str | None = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    df = pd.read_csv(dataset_path)
    frame = _prepare_training_frame(df)

    X = frame[["steps", "sleep_hours", "workout_intensity", "weight"]]
    y = frame["target_calories"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_pipeline = None
    best_model_name = None
    best_metrics = None
    all_metrics: Dict[str, Dict[str, float]] = {}

    candidates = _candidate_models()
    if selected_model is not None:
        if selected_model not in candidates:
            raise ValueError(f"Unsupported model: {selected_model}")
        candidates = {selected_model: candidates[selected_model]}

    for model_name, model in candidates.items():
        pipeline = _build_pipeline(model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        model_metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
        }
        all_metrics[model_name] = model_metrics

        if best_metrics is None or _is_better(model_metrics, best_metrics):
            best_pipeline = pipeline
            best_model_name = model_name
            best_metrics = model_metrics

    metrics = {
        "selection_mode": "manual" if selected_model is not None else "auto",
        "selected_model": str(best_model_name),
        "r2": float(best_metrics["r2"]),
        "mae": float(best_metrics["mae"]),
        "rows": float(len(frame)),
        "model_comparison": all_metrics,
    }

    bundle = {
        "pipeline": best_pipeline,
        "version": 4,
        "selected_model": best_model_name,
        "training_metrics": metrics,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle, metrics


def load_or_train_model(force_retrain: bool = False) -> Tuple[Dict[str, Any], Dict[str, float]]:
    if not force_retrain and os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            loaded_metrics = bundle.get("training_metrics")
            if isinstance(loaded_metrics, dict):
                return bundle, loaded_metrics
            # Older model format without persisted metrics; retrain once to populate them.
            return train_model_from_dataset()
        except Exception:
            pass
    return train_model_from_dataset()


def predict_calories(model_bundle: Dict[str, Any], data: Dict[str, Any]) -> float:
    pipeline = model_bundle["pipeline"]
    frame = pd.DataFrame(
        [
            {
                "steps": np.nan if data.get("steps") is None else float(data["steps"]),
                "sleep_hours": np.nan if data.get("sleep_hours") is None else float(data["sleep_hours"]),
                "workout_intensity": data.get("workout_intensity") if data.get("workout_intensity") is not None else np.nan,
                "weight": np.nan if data.get("weight") is None else float(data["weight"]),
            }
        ]
    )
    prediction = pipeline.predict(frame)[0]
    return float(prediction)


def estimate_bmr(weight_kg: float) -> float:
    return 24.0 * weight_kg


def estimate_intake(food_category: str) -> float:
    if food_category == "Light":
        return 1500.0
    if food_category == "Heavy":
        return 2800.0
    return 2200.0
