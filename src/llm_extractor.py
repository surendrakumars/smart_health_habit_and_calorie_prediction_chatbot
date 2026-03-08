import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import requests

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_URL = os.getenv("HF_API_URL", f"https://api-inference.huggingface.co/models/{HF_MODEL}")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def _build_extraction_prompt(chat_text: str) -> str:
    return (
        "Extract health fields from this user message and return strict JSON only. "
        "Use null for missing values.\\n"
        "Schema keys: steps (int), sleep_hours (float), water_intake (float liters), "
        "food_category (Light|Moderate|Heavy), mood (positive|neutral|negative), "
        "workout_intensity (None|Light|Moderate|Heavy), weight (float kg).\\n"
        "Rules: convert 6k steps to 6000, convert glasses of water to liters with 1 glass=0.25 liters, "
        "convert lbs to kg with 1 lb=0.453592.\\n"
        f"User message: {chat_text}"
    )


def _parse_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    text = raw_text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            pass

    obj = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj:
        try:
            parsed = json.loads(obj.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            return None

    return None


def extract_with_hf(chat_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not HF_API_TOKEN:
        return None, "HF_API_TOKEN is not set"

    prompt = _build_extraction_prompt(chat_text)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 220,
            "temperature": 0.0,
            "return_full_text": False,
        },
    }
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        body = response.json()

        if isinstance(body, list) and body and isinstance(body[0], dict):
            generated_text = body[0].get("generated_text", "")
        elif isinstance(body, dict):
            generated_text = (
                body.get("generated_text")
                or body.get("output_text")
                or body.get("response")
                or ""
            )
        else:
            generated_text = ""

        parsed = _parse_json_object(str(generated_text))
        if not isinstance(parsed, dict):
            return None, "HF output was not parseable JSON"
        return parsed, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def extract_with_ollama(chat_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    prompt = _build_extraction_prompt(chat_text)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        raw = body.get("response", "{}")
        parsed = _parse_json_object(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            return None, "LLM output was not a JSON object"
        return parsed, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def extract_with_primary_llm(chat_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    hf_data, hf_error = extract_with_hf(chat_text)
    if isinstance(hf_data, dict):
        return hf_data, None

    ollama_data, ollama_error = extract_with_ollama(chat_text)
    if isinstance(ollama_data, dict):
        return ollama_data, None

    return None, f"HF failed: {hf_error}; Ollama failed: {ollama_error}"
