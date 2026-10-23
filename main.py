from __future__ import annotations

import argparse
import os
import sys

import requests


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
).rstrip("/")

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5-coder:3b",
)

SYSTEM_PROMPT = """
You are a software engineering assistant.

Rules:
- Answer the user's task directly.
- Be technically accurate.
- State assumptions when necessary.
- Do not claim to have executed code or tests.
- You currently have no tools and cannot inspect files.
"""


def call_model(task: str) -> str:
    """Send a task to the local LLM and return its response."""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        "stream": False,
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=(5, 180),
    )
    response.raise_for_status()

    data = response.json()
    message = data.get("message")

    if not isinstance(message, dict):
        raise ValueError("Ollama returned no valid message object.")

    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned empty or invalid content.")

    return content.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A minimal local LLM harness."
    )
    parser.add_argument(
        "task",
        nargs="+",
        help="The task to send to the local model.",
    )
    args = parser.parse_args()

    task = " ".join(args.task)

    try:
        answer = call_model(task)
    except requests.RequestException as exc:
        print(f"LLM request failed: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        print(f"Invalid model response: {exc}", file=sys.stderr)
        return 1

    print(f"\nModel: {MODEL}\n")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())