"""OpenRouter embeddings — batched, L2-normalized float32 vectors.

Uses a raw HTTP POST rather than the OpenAI SDK: OpenRouter's embeddings
response is not fully SDK-shaped, so the SDK parser raises "No embedding data
received". The raw path (same as the sibling RAG project) is reliable.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from ._openrouter import BASE_URL, _HEADERS, get_api_key

DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBED_DIM = 2048  # llama-nemotron-embed-vl-1b-v2


def get_embed_model() -> str:
    return os.environ.get("OPENROUTER_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip()


def _post(inputs, api_key: str, model: str) -> list[list[float]]:
    body = json.dumps({"model": model, "input": list(inputs)}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **_HEADERS,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "ignore")
        raise RuntimeError(f"OpenRouter embeddings HTTP {exc.code}: {detail}") from exc
    # Preserve request order (OpenAI-compatible responses carry an index).
    rows = sorted(data["data"], key=lambda d: d.get("index", 0))
    return [r["embedding"] for r in rows]


def embed_texts(texts, api_key=None, model=None) -> np.ndarray:
    api_key = api_key or get_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. See the README.")
    vecs = _post(texts, api_key, model or get_embed_model())
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_one(text: str, **kw) -> np.ndarray:
    return embed_texts([text], **kw)[0]
