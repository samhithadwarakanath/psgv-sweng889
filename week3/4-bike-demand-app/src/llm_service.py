import os

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

REQUEST_TIMEOUT = 75


class LLMServiceError(Exception):
    """Raised when the local LLM service cannot produce an explanation."""


def _clean_response(payload: dict) -> str:
    text = payload["response"].strip()
    lines = text.splitlines()
    if payload.get("done_reason") == "length" and len(lines) > 1:
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _build_prompt(predicted_demand, date, time, temperature, is_holiday) -> str:
    holiday_text = "a holiday" if is_holiday else "a regular day"
    return (
        "You are a bike-share operations assistant. "
        f"On {date} at {time}, the temperature is {temperature}°C and it is {holiday_text}. "
        f"A machine learning model predicts {predicted_demand} bike rentals for that hour. "
        "In one short sentence (no more than 35 words), explain why demand might look like this "
        "and recommend one action for the bike-share operator."
    )


def get_explanation(predicted_demand, date, time, temperature, is_holiday) -> str:
    prompt = _build_prompt(predicted_demand, date, time, temperature, is_holiday)

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 70},
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise LLMServiceError(
            f"Could not connect to the LLM service at {OLLAMA_URL}."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMServiceError("The LLM service took too long to respond.") from exc
    except requests.exceptions.RequestException as exc:
        raise LLMServiceError(f"The LLM request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMServiceError(
            f"The LLM service returned an error (status {response.status_code})."
        )

    try:
        return _clean_response(response.json())
    except (ValueError, KeyError) as exc:
        raise LLMServiceError("Received an unexpected response from the LLM service.") from exc
