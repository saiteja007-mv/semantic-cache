"""Deploy this app to Hugging Face Spaces (Docker SDK) and print the live URL.

Requires:
  * a Hugging Face token (cached via `huggingface-cli login`, or HF_TOKEN env)
  * OPENROUTER_API_KEY (read from env or .streamlit/secrets.toml) — set as a
    Space secret so the deployed app can call OpenRouter.

    python scripts/deploy_hf.py [--space-name semantic-cache]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Semantic Cache for LLMs
emoji: ⚡
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
short_description: Embedding-keyed LLM response cache — semantic hits skip the call
---

# Semantic Cache for LLM Responses

Caches LLM answers by prompt **embedding**: a semantically similar prompt returns
the cached response instantly, skipping the LLM call (saving latency and token
cost). Hybrid exact-hash + cosine-similarity lookup with a tunable threshold,
LRU eviction, and live hit-rate / savings metrics. Runs on free OpenRouter models.

Source: https://github.com/saiteja007-mv/semantic-cache
"""

IGNORE = [
    ".venv/*",
    ".git/*",
    ".github/*",
    "__pycache__/*",
    "*/__pycache__/*",
    "*.pyc",
    ".streamlit/secrets.toml",
    ".streamlit/secrets.toml.example",
    ".env",
    "tests/*",
    "docs/*",
    ".boot.log",
    "README.md",  # replaced by the Space README with YAML front-matter
]


def _load_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    sec = ROOT / ".streamlit" / "secrets.toml"
    if sec.exists():
        import tomllib

        return str(tomllib.load(sec.open("rb")).get("OPENROUTER_API_KEY", "")).strip() or None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-name", default="semantic-cache")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    user = who["name"]
    repo_id = f"{user}/{args.space_name}"
    print(f"[deploy] HF user: {user} -> space {repo_id}")

    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    key = _load_key()
    if key:
        api.add_space_secret(repo_id=repo_id, key="OPENROUTER_API_KEY", value=key)
        print("[deploy] set Space secret OPENROUTER_API_KEY")
    else:
        print("[deploy] WARNING: no OPENROUTER_API_KEY found — set it in Space settings.")

    with tempfile.TemporaryDirectory() as td:
        readme = Path(td) / "README.md"
        readme.write_text(SPACE_README, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
        )

    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Deploy semantic-cache",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\n[deploy] DONE. Live (build takes ~1-2 min):\n  {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
