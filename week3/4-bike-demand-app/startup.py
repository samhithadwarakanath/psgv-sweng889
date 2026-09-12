import os
import time

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def wait_for_ollama(max_attempts: int = 60, delay_seconds: int = 2) -> None:
    print(f"Waiting for Ollama at {OLLAMA_URL}...")
    for _ in range(max_attempts):
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama is up.")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay_seconds)
    print("Warning: Ollama did not become ready in time; continuing anyway.")


def pull_model() -> None:
    print(f"Pulling model: {OLLAMA_MODEL} (this may take a while on first run)...")
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"model": OLLAMA_MODEL},
            stream=True,
            timeout=(10, 600),
        )
        if response.status_code != 200:
            print(f"Warning: failed to pull model: {response.text}")
            return
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
        print("Model is ready.")
    except requests.exceptions.RequestException as exc:
        print(f"Warning: could not pull model: {exc}")


if __name__ == "__main__":
    wait_for_ollama()
    pull_model()
