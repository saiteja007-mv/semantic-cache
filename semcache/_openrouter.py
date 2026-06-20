"""Shared OpenRouter access: key resolution + a configured OpenAI client."""
from __future__ import annotations

import os

BASE_URL = "https://openrouter.ai/api/v1"
_HEADERS = {
    "HTTP-Referer": "https://github.com/saiteja007-mv/semantic-cache",
    "X-Title": "Semantic Cache for LLM Responses",
}


def get_api_key() -> str | None:
    """Resolve the OpenRouter key from env, then Streamlit secrets if available."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    try:
        import streamlit as st

        if "OPENROUTER_API_KEY" in st.secrets:
            return str(st.secrets["OPENROUTER_API_KEY"]).strip()
    except Exception:  # noqa: BLE001 - streamlit absent / no secrets file
        pass
    return None


def make_client(api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=api_key, default_headers=_HEADERS)
