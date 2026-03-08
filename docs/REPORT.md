# Smart Health Habit & Calorie Prediction Chatbot

## Architecture

- `src/main.py`: Flask web app with chat, reset, and retrain endpoints.
- `src/nlp.py`: Hybrid extraction pipeline (LLM first, rule-based fallback).
- `src/llm_extractor.py`: Local LLM extraction via Ollama API.
- `src/ml.py`: Dataset-based model training and calorie prediction.
- `src/scoring.py`: Health score and risk detection.
- `src/recommender.py`: Personalized recommendations and weekly trends.

## LLM Extraction Logic

The chat message is processed with two stages:

1. LLM extraction (`Ollama`):
- Prompted to return strict JSON with keys:
  - `steps`, `sleep_hours`, `water_intake`, `food_category`, `mood`, `workout_intensity`, `weight`
- Unit normalization requested in prompt (`k steps`, `glasses -> liters`, `lbs -> kg`).

2. Rule-based fallback:
- Regex and keyword spotting fills any missing fields.
- This keeps the system robust if local LLM is unavailable.

## Dataset Training and Prediction

Dataset used:
- `data/Sleep_health_and_lifestyle_dataset.csv`

Training features:
- `steps` (from `Daily Steps`)
- `sleep_hours` (from `Sleep Duration`)
- `workout_intensity` (derived from `Physical Activity Level`)
- `weight` (estimated from `BMI Category` proxy)

Target:
- If a calorie column exists (`calories_burned`, `calories`, `daily_calories_burned`), it is used.
- Otherwise, a transparent derived calorie target is generated from activity, heart rate, stress, and steps.

Model:
- Candidate models:
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
- User can choose training mode from UI:
  - `Auto Select` (compare both and pick best)
  - `RandomForestRegressor` (force)
  - `GradientBoostingRegressor` (force)
- Both models are trained on the same split.
- Selection rule:
  - Choose higher `R2`
  - If `R2` is tied, choose lower `MAE`
- The selected best model is persisted in `models/calorie_model.pkl`.
- Saved to `models/calorie_model.pkl`.

### Model Comparison Conclusion

- For this project, the best model is selected automatically from the two candidates at train time.
- Latest retrain result on this dataset:
  - `RandomForestRegressor`: `R2 = 0.992726`, `MAE = 10.3355`
  - `GradientBoostingRegressor`: `R2 = 0.992928`, `MAE = 10.3673`
- Final choice for this project: `RandomForestRegressor`.
- Reason: the `R2` difference is very small (treated as near-tie), while Random Forest gives lower error (`MAE`), which is more practical for calorie estimation.
- This conclusion is data-driven and reproducible through the retrain endpoint (`POST /api/retrain`), which reports the selected model and metrics.

## Health Score and Risks

Score range: `0-100`

Weights:
- Activity: `50%`
- Sleep: `30%`
- Hydration: `20%`

Risk rules:
- Low sleep + high workout intensity -> fatigue risk.
- Low hydration + heavy workout -> dehydration risk.

## Website Endpoints

- `GET /`: chat UI
- `POST /api/chat`: process health message
- `POST /api/reset`: clear session data
- `POST /api/retrain`: retrain model from dataset

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

Open:
- `http://localhost:5000`

Optional for LLM extraction:
```bash
ollama serve
ollama pull llama3.1:8b
```
