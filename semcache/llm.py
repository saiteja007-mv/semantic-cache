"""OpenRouter chat completion — non-streamed full response (for caching)."""
from __future__ import annotations

import os

from ._openrouter import get_api_key, make_client

DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"


def get_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL).strip()


def complete(prompt, history=None, api_key=None, model=None, temperature=0.2) -> str:
    api_key = api_key or get_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. See the README.")
    client = make_client(api_key)
    messages = []
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model or get_model(), messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content or ""
