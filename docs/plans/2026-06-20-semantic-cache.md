# Semantic Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an embedding-keyed semantic cache that fronts an OpenRouter LLM call, returning cached responses for semantically-similar prompts, with a live Streamlit demo.

**Architecture:** A `semcache` package: `_openrouter.py` (shared key/client), `embed.py` (normalized embeddings), `llm.py` (chat completion), `cache.py` (the `SemanticCache` + `ask()` orchestrator). A Streamlit `app.py` renders HIT/MISS, similarity, and saved-cost metrics. Core logic is TDD'd offline with injected `embed_fn`/`llm_fn`; OpenRouter clients are thin and verified via `scripts/smoke.py`.

**Tech Stack:** Python 3.12, OpenAI SDK → OpenRouter, numpy, Streamlit, pytest, Docker (HF Spaces).

## Global Constraints

- Python 3.12; per-project `.venv`. **No torch / no local ML libs** — fully API-backed.
- OpenRouter via OpenAI SDK, `base_url=https://openrouter.ai/api/v1`.
- Models, `:free` suffix **required** (404 otherwise):
  - Embeddings: `nvidia/llama-nemotron-embed-vl-1b-v2:free` (2048-dim)
  - Chat: `nvidia/nemotron-3-nano-30b-a3b:free`
- Key resolution: `OPENROUTER_API_KEY` env → `.streamlit/secrets.toml`. Overrides: `OPENROUTER_MODEL`, `OPENROUTER_EMBED_MODEL`.
- Embeddings **L2-normalized** so cosine == dot product.
- Secrets gitignored; never commit `.streamlit/secrets.toml` or `.env`.
- Default cache: `threshold=0.85`, `max_size=256`, `ttl_s=None`.
- GitHub: `saiteja007-mv` · HF: `SaitejaMothukuri`. Repo name `semantic-cache`.
- Commit author: Saiteja mothukuri <saiteja.motukuri@gmail.com>.

---

### Task 1: Scaffold project + shared OpenRouter client

**Files:**
- Create: `requirements.txt`, `.gitignore`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`, `semcache/__init__.py`, `semcache/_openrouter.py`
- Test: (none — verified by import + venv)

**Interfaces:**
- Produces: `semcache._openrouter.get_api_key() -> str | None`, `make_client(api_key: str) -> OpenAI`

- [ ] **Step 1: requirements.txt**

```
# Semantic Cache for LLM responses — fully cloud (OpenRouter), no local ML deps.
streamlit>=1.33
openai>=1.30
numpy>=1.26
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 2: .gitignore**

```
.venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
.env
.boot.log
assets/.gitkeep
```

- [ ] **Step 3: .streamlit/config.toml**

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
```

- [ ] **Step 4: .streamlit/secrets.toml.example**

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```

- [ ] **Step 5: semcache/_openrouter.py**

```python
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
```

- [ ] **Step 6: semcache/__init__.py**

```python
"""Embedding-keyed semantic cache for LLM responses."""
from .cache import AskResult, CacheEntry, CacheStats, LookupResult, SemanticCache, ask

__all__ = [
    "SemanticCache",
    "CacheEntry",
    "CacheStats",
    "LookupResult",
    "AskResult",
    "ask",
]
```

- [ ] **Step 7: Create venv + install**

Run:
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -q -r requirements.txt
```
Expected: installs cleanly (no torch).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .gitignore .streamlit semcache/__init__.py semcache/_openrouter.py
git commit -m "chore: scaffold semantic-cache project + shared OpenRouter client"
```
Note: `__init__.py` imports `.cache` which lands in Task 3 — commit is fine (not imported until then); or stage `__init__.py` in Task 3. Keep `__init__.py` here but expect import to resolve after Task 3.

---

### Task 2: Cache core — `SemanticCache` lookup/put/eviction/TTL (TDD)

**Files:**
- Create: `semcache/cache.py` (partial — dataclasses + `SemanticCache`), `tests/test_cache.py`, `tests/__init__.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `semcache.embed.embed_one` (Task 4) as the **default** `embed_fn`, but tests inject a fake; to avoid an import cycle/network at import time, default is bound lazily (see code).
- Produces:
  - `CacheEntry(prompt, embedding, response, created_at, last_used, hits=0)`
  - `LookupResult(entry, similarity, exact)`
  - `CacheStats(...)` with `.hit_rate`
  - `SemanticCache(threshold=0.85, max_size=256, ttl_s=None, embed_fn=None, cost_per_1k_tokens=0.0, avg_llm_latency_s=1.5)` with `.lookup(prompt) -> LookupResult|None`, `.put(prompt, response)`, `.entries()`, `.clear()`, `__len__`, `.stats`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cache.py
import numpy as np
import pytest

from semcache.cache import SemanticCache

VOCAB = ["capital", "france", "germany", "weather", "tokyo", "python"]


def fake_embed(text: str) -> np.ndarray:
    """Deterministic bag-of-words embedding, L2-normalized (offline)."""
    toks = set(text.lower().replace("?", "").replace("'s", "").split())
    v = np.array([1.0 if w in toks else 0.0 for w in VOCAB], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def make_cache(**kw):
    kw.setdefault("threshold", 0.85)
    kw.setdefault("embed_fn", fake_embed)
    return SemanticCache(**kw)


def test_exact_hit_is_case_insensitive():
    c = make_cache()
    c.put("Capital of France", "Paris")
    r = c.lookup("capital of france")
    assert r is not None and r.exact and r.similarity == 1.0
    assert r.entry.response == "Paris"


def test_semantic_hit_above_threshold():
    c = make_cache()
    c.put("capital of france", "Paris")
    r = c.lookup("france capital")  # same BoW tokens, different string
    assert r is not None and not r.exact and r.similarity >= 0.85
    assert r.entry.response == "Paris"


def test_miss_below_threshold_returns_none():
    c = make_cache()
    c.put("capital of france", "Paris")
    assert c.lookup("capital of germany") is None  # cosine 0.5 < 0.85


def test_threshold_boundary():
    c_low = make_cache(threshold=0.80)
    c_low.put("capital of france", "Paris")
    # {capital,france,weather} vs {capital,france} -> 2/(sqrt3*sqrt2) ~= 0.816
    assert c_low.lookup("capital france weather") is not None
    c_high = make_cache(threshold=0.85)
    c_high.put("capital of france", "Paris")
    assert c_high.lookup("capital france weather") is None


def test_lru_eviction():
    c = make_cache(max_size=2)
    c.put("capital of france", "Paris")
    c.put("weather tokyo", "Rainy")
    c.put("python list", "Use list()")
    assert len(c) == 2  # oldest-used evicted


def test_ttl_expiry():
    c = make_cache(ttl_s=10)
    c.put("capital of france", "Paris")
    c.entries()[0].created_at -= 100  # force-expire
    assert c.lookup("capital of france") is None
    assert len(c) == 0
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_cache.py -q`
Expected: FAIL (`ModuleNotFoundError: semcache.cache` / `ImportError`).

- [ ] **Step 3: Implement `semcache/cache.py` (dataclasses + SemanticCache)**

```python
"""Embedding-keyed semantic cache: lookup, put, eviction, TTL, stats."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class CacheEntry:
    prompt: str
    embedding: np.ndarray
    response: str
    created_at: float
    last_used: float
    hits: int = 0


@dataclass
class LookupResult:
    entry: CacheEntry
    similarity: float
    exact: bool


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    llm_calls_saved: int = 0
    latency_saved_s: float = 0.0
    tokens_saved_est: int = 0
    cost_saved_est: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


def _norm_key(prompt: str) -> str:
    return prompt.strip().lower()


def _default_embed(text: str) -> np.ndarray:
    # Bound lazily to avoid importing the network client at module import.
    from .embed import embed_one

    return embed_one(text)


class SemanticCache:
    def __init__(
        self,
        threshold: float = 0.85,
        max_size: int = 256,
        ttl_s: Optional[float] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        cost_per_1k_tokens: float = 0.0,
        avg_llm_latency_s: float = 1.5,
    ):
        self.threshold = threshold
        self.max_size = max_size
        self.ttl_s = ttl_s
        self.embed_fn = embed_fn or _default_embed
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.avg_llm_latency_s = avg_llm_latency_s
        self._entries: list[CacheEntry] = []
        self._exact: dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[CacheEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._exact.clear()

    @staticmethod
    def _now() -> float:
        return time.time()

    def _purge_expired(self) -> None:
        if self.ttl_s is None:
            return
        cutoff = self._now() - self.ttl_s
        keep = [e for e in self._entries if e.created_at >= cutoff]
        if len(keep) != len(self._entries):
            self._entries = keep
            self._exact = {_norm_key(e.prompt): e for e in keep}

    def lookup(self, prompt: str) -> Optional[LookupResult]:
        self._purge_expired()
        key = _norm_key(prompt)
        exact = self._exact.get(key)
        if exact is not None:
            exact.hits += 1
            exact.last_used = self._now()
            return LookupResult(exact, 1.0, exact=True)
        if not self._entries:
            return None
        q = self.embed_fn(prompt)
        candidates = [e for e in self._entries if e.embedding.shape == q.shape]
        if not candidates:
            return None
        mat = np.vstack([e.embedding for e in candidates])
        sims = mat @ q
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        if best >= self.threshold:
            entry = candidates[idx]
            entry.hits += 1
            entry.last_used = self._now()
            return LookupResult(entry, best, exact=False)
        return None

    def put(self, prompt: str, response: str) -> None:
        emb = self.embed_fn(prompt)
        now = self._now()
        entry = CacheEntry(prompt, emb, response, created_at=now, last_used=now)
        self._entries.append(entry)
        self._exact[_norm_key(prompt)] = entry
        if len(self._entries) > self.max_size:
            lru = min(self._entries, key=lambda e: e.last_used)
            self._entries.remove(lru)
            self._exact.pop(_norm_key(lru.prompt), None)
```

- [ ] **Step 4: Run tests — verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_cache.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add semcache/cache.py tests/
git commit -m "feat: semantic cache core (lookup/put/eviction/ttl) + tests"
```

---

### Task 3: `ask()` orchestrator + `AskResult` + stats (TDD)

**Files:**
- Modify: `semcache/cache.py` (append `AskResult` + `ask()`)
- Test: `tests/test_cache.py` (append)

**Interfaces:**
- Consumes: `SemanticCache`, `semcache.llm.complete` (Task 5) as default `llm_fn` (lazy-bound; tests inject a fake).
- Produces: `AskResult(prompt, response, hit, exact, similarity, matched_prompt, latency_s)`; `ask(prompt, cache, *, llm_fn=None) -> AskResult`.

- [ ] **Step 1: Append failing tests**

```python
# append to tests/test_cache.py
from semcache.cache import ask


class CountingLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return f"answer:{prompt}"


def test_ask_miss_calls_llm_and_stores():
    c = make_cache()
    llm = CountingLLM()
    r = ask("capital of france", c, llm_fn=llm)
    assert llm.calls == 1 and not r.hit and r.response == "answer:capital of france"
    assert len(c) == 1 and c.stats.misses == 1 and c.stats.lookups == 1


def test_ask_hit_skips_llm_and_counts_savings():
    c = make_cache()
    llm = CountingLLM()
    ask("capital of france", c, llm_fn=llm)        # miss -> 1 call
    r = ask("france capital", c, llm_fn=llm)        # semantic hit -> no call
    assert llm.calls == 1 and r.hit and not r.exact
    assert c.stats.hits == 1 and c.stats.llm_calls_saved == 1
    assert c.stats.tokens_saved_est > 0
    assert 0.0 < c.stats.hit_rate <= 1.0


def test_ask_cost_estimate_uses_rate():
    c = make_cache(cost_per_1k_tokens=10.0)
    llm = CountingLLM()
    ask("capital of france", c, llm_fn=llm)
    ask("capital of france", c, llm_fn=llm)         # exact hit
    assert c.stats.cost_saved_est > 0
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python -m pytest tests/test_cache.py -q`
Expected: FAIL (`ImportError: cannot import name 'ask'`).

- [ ] **Step 3: Append implementation to `semcache/cache.py`**

```python
@dataclass
class AskResult:
    prompt: str
    response: str
    hit: bool
    exact: bool
    similarity: float
    matched_prompt: Optional[str]
    latency_s: float


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 chars/token heuristic


def _default_complete(prompt: str) -> str:
    from .llm import complete

    return complete(prompt)


def ask(
    prompt: str,
    cache: SemanticCache,
    *,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> AskResult:
    llm_fn = llm_fn or _default_complete
    prompt = prompt.strip()
    cache.stats.lookups += 1
    t0 = time.perf_counter()
    hit = cache.lookup(prompt)
    if hit is not None:
        latency = time.perf_counter() - t0
        cache.stats.hits += 1
        cache.stats.llm_calls_saved += 1
        cache.stats.latency_saved_s += max(0.0, cache.avg_llm_latency_s - latency)
        toks = _est_tokens(hit.entry.response)
        cache.stats.tokens_saved_est += toks
        cache.stats.cost_saved_est += toks / 1000.0 * cache.cost_per_1k_tokens
        return AskResult(
            prompt, hit.entry.response, hit=True, exact=hit.exact,
            similarity=hit.similarity, matched_prompt=hit.entry.prompt,
            latency_s=latency,
        )
    response = llm_fn(prompt)
    cache.put(prompt, response)
    latency = time.perf_counter() - t0
    cache.stats.misses += 1
    return AskResult(
        prompt, response, hit=False, exact=False, similarity=0.0,
        matched_prompt=None, latency_s=latency,
    )
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_cache.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add semcache/cache.py tests/test_cache.py
git commit -m "feat: ask() orchestrator with hit/miss stats and savings"
```

---

### Task 4: `embed.py` — OpenRouter embeddings (normalized)

**Files:**
- Create: `semcache/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `_openrouter.get_api_key`, `make_client`.
- Produces: `embed_texts(texts, api_key=None, model=None) -> np.ndarray (n,d) float32 L2-normalized`; `embed_one(text, **kw) -> np.ndarray (d,)`; `get_embed_model() -> str`.

- [ ] **Step 1: Failing test (mock the client — no network)**

```python
# tests/test_embed.py
import numpy as np

import semcache.embed as embed


class _Resp:
    def __init__(self, vecs):
        self.data = [type("D", (), {"embedding": v}) for v in vecs]


def test_embed_texts_normalizes(monkeypatch):
    class FakeClient:
        class embeddings:
            @staticmethod
            def create(model, input):
                return _Resp([[3.0, 4.0]] * len(input))  # norm 5

    monkeypatch.setattr(embed, "get_api_key", lambda: "k")
    monkeypatch.setattr(embed, "make_client", lambda key: FakeClient())
    out = embed.embed_texts(["a", "b"])
    assert out.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)


def test_embed_one_returns_1d(monkeypatch):
    class FakeClient:
        class embeddings:
            @staticmethod
            def create(model, input):
                return _Resp([[1.0, 0.0]])

    monkeypatch.setattr(embed, "get_api_key", lambda: "k")
    monkeypatch.setattr(embed, "make_client", lambda key: FakeClient())
    v = embed.embed_one("hi")
    assert v.shape == (2,)
```

- [ ] **Step 2: Run — verify fail** (`.venv/Scripts/python -m pytest tests/test_embed.py -q` → ImportError)

- [ ] **Step 3: Implement `semcache/embed.py`**

```python
"""OpenRouter embeddings — batched, L2-normalized float32 vectors."""
from __future__ import annotations

import os

import numpy as np

from ._openrouter import get_api_key, make_client

DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


def get_embed_model() -> str:
    return os.environ.get("OPENROUTER_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip()


def embed_texts(texts, api_key=None, model=None) -> np.ndarray:
    api_key = api_key or get_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. See the README.")
    client = make_client(api_key)
    resp = client.embeddings.create(model=model or get_embed_model(), input=list(texts))
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_one(text: str, **kw) -> np.ndarray:
    return embed_texts([text], **kw)[0]
```

- [ ] **Step 4: Run — verify pass** (2 passed)

- [ ] **Step 5: Commit**

```bash
git add semcache/embed.py tests/test_embed.py
git commit -m "feat: OpenRouter embeddings client (normalized) + tests"
```

---

### Task 5: `llm.py` — OpenRouter chat completion

**Files:**
- Create: `semcache/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `complete(prompt, history=None, api_key=None, model=None, temperature=0.2) -> str`; `get_model() -> str`.

- [ ] **Step 1: Failing test (mock client)**

```python
# tests/test_llm.py
import semcache.llm as llm


class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _Resp:
    def __init__(self, content):
        self.choices = [_Msg(content)]


def test_complete_returns_text(monkeypatch):
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature):
                    return _Resp("Paris")

    monkeypatch.setattr(llm, "get_api_key", lambda: "k")
    monkeypatch.setattr(llm, "make_client", lambda key: FakeClient())
    assert llm.complete("capital of france?") == "Paris"


def test_complete_raises_without_key(monkeypatch):
    monkeypatch.setattr(llm, "get_api_key", lambda: None)
    try:
        llm.complete("x")
    except RuntimeError as e:
        assert "OPENROUTER_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
```

- [ ] **Step 2: Run — verify fail**

- [ ] **Step 3: Implement `semcache/llm.py`**

```python
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
```

- [ ] **Step 4: Run — verify pass** (2 passed). Then full suite: `.venv/Scripts/python -m pytest -q` → 13 passed.

- [ ] **Step 5: Commit**

```bash
git add semcache/llm.py tests/test_llm.py
git commit -m "feat: OpenRouter chat completion client + tests"
```

---

### Task 6: Streamlit demo `app.py`

**Files:**
- Create: `app.py`
- Test: manual (`streamlit run app.py`) + Task 7 smoke

**Interfaces:**
- Consumes: `semcache.SemanticCache`, `semcache.ask`, `semcache._openrouter.get_api_key`.

- [ ] **Step 1: Implement `app.py`**

```python
"""Streamlit demo: semantic cache in front of an OpenRouter LLM."""
from __future__ import annotations

import streamlit as st

from semcache import SemanticCache, ask
from semcache._openrouter import get_api_key

st.set_page_config(page_title="Semantic Cache for LLMs", page_icon="⚡", layout="wide")

EXAMPLES = [
    "What is the capital of France?",
    "Tell me France's capital city",          # paraphrase of #1 -> semantic HIT
    "How do I reverse a list in Python?",
    "What's the Python way to reverse a list?",  # paraphrase of #3 -> semantic HIT
    "What is the boiling point of water?",
    "Explain reciprocal rank fusion in one sentence.",
]


def _cache() -> SemanticCache:
    if "cache" not in st.session_state:
        st.session_state.cache = SemanticCache(
            threshold=0.85, cost_per_1k_tokens=0.50, avg_llm_latency_s=1.5
        )
    return st.session_state.cache


cache = _cache()

st.title("⚡ Semantic Cache for LLM Responses")
st.caption(
    "Caches answers by prompt **embedding** — a semantically similar prompt returns "
    "instantly, skipping the LLM call. Reuses free OpenRouter models."
)

with st.sidebar:
    st.header("Cache settings")
    cache.threshold = st.slider("Similarity threshold", 0.50, 0.99, cache.threshold, 0.01)
    cache.max_size = st.number_input("Max cache size", 8, 4096, cache.max_size, 8)
    cache.cost_per_1k_tokens = st.number_input(
        "$ / 1k tokens (for savings est.)", 0.0, 100.0, cache.cost_per_1k_tokens, 0.05
    )
    st.caption("Models are free-tier; $ is illustrative from this rate.")
    if st.button("Seed example prompts"):
        for q in EXAMPLES:
            ask(q, cache)
        st.success(f"Seeded {len(EXAMPLES)} prompts.")
    if st.button("Clear cache"):
        cache.clear()
        cache.stats.__init__()
        st.session_state.pop("history", None)

if not get_api_key():
    st.warning("Set `OPENROUTER_API_KEY` (env or `.streamlit/secrets.toml`). See README.")

s = cache.stats
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hit rate", f"{s.hit_rate*100:.0f}%")
c2.metric("Lookups", s.lookups)
c3.metric("LLM calls saved", s.llm_calls_saved)
c4.metric("Latency saved", f"{s.latency_saved_s:.1f}s")
c5.metric("Est. $ saved", f"${s.cost_saved_est:.4f}")

prompt = st.chat_input("Ask something (then ask it again in different words)…")
if prompt:
    try:
        r = ask(prompt, cache)
    except Exception as e:  # noqa: BLE001
        st.error(f"LLM/embedding error: {e}")
    else:
        hist = st.session_state.setdefault("history", [])
        hist.append(r)

for r in reversed(st.session_state.get("history", [])):
    with st.chat_message("user"):
        st.write(r.prompt)
    with st.chat_message("assistant"):
        if r.hit:
            kind = "exact" if r.exact else f"semantic · sim {r.similarity:.3f}"
            st.success(f"⚡ CACHE HIT ({kind}) · {r.latency_s*1000:.0f} ms")
            if r.matched_prompt and not r.exact:
                st.caption(f"matched cached prompt: “{r.matched_prompt}”")
        else:
            st.info(f"🌐 MISS → called LLM · {r.latency_s:.2f} s")
        st.write(r.response)

with st.expander(f"Cache contents ({len(cache)} entries)"):
    rows = [
        {"prompt": e.prompt, "hits": e.hits, "chars": len(e.response)}
        for e in cache.entries()
    ]
    st.table(rows) if rows else st.write("empty")
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit demo with HIT/MISS, similarity, savings metrics"
```

---

### Task 7: `scripts/smoke.py` — headless integration check

**Files:**
- Create: `scripts/smoke.py`

**Interfaces:** Consumes `semcache.SemanticCache`, `ask`. Requires a real key (hits OpenRouter).

- [ ] **Step 1: Implement**

```python
"""Headless pipeline check: seed prompts, fire a paraphrase, assert a semantic HIT."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semcache import SemanticCache, ask  # noqa: E402


def main() -> int:
    cache = SemanticCache(threshold=0.80, cost_per_1k_tokens=0.5)
    miss = ask("What is the capital of France?", cache)
    print(f"[1] hit={miss.hit} (expect False) latency={miss.latency_s:.2f}s")
    hit = ask("Tell me France's capital city", cache)
    print(f"[2] hit={hit.hit} exact={hit.exact} sim={hit.similarity:.3f}")
    s = cache.stats
    print(
        f"stats: lookups={s.lookups} hits={s.hits} misses={s.misses} "
        f"hit_rate={s.hit_rate*100:.0f}% latency_saved={s.latency_saved_s:.2f}s"
    )
    if not hit.hit:
        print("FAIL: expected a semantic cache hit on the paraphrase.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run (needs key)**

Run: `.venv/Scripts/python scripts/smoke.py`
Expected: `[2] hit=True ...` and `OK`. (If no key, it raises a clear RuntimeError — acceptable; note in summary.)

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke.py
git commit -m "test: headless smoke check (semantic hit on paraphrase)"
```

---

### Task 8: `scripts/deploy_hf.py` — one-command HF deploy (for later)

**Files:**
- Create: `scripts/deploy_hf.py`

- [ ] **Step 1: Adapt from P1** — copy `../hybrid-search-rag/scripts/deploy_hf.py` and change only:
  - `--space-name` default → `semantic-cache`
  - `SPACE_README` front-matter: `title: Semantic Cache for LLMs`, `emoji: ⚡`, `colorFrom: yellow`, `colorTo: orange`, `short_description: Embedding-keyed LLM response cache`, body + `Source: https://github.com/saiteja007-mv/semantic-cache`
  - `IGNORE`: keep `.venv/*`, `.git/*`, `__pycache__`, `.streamlit/secrets.toml`, `.env`, `tests/*`, `docs/*`, `README.md`
  - secret key copied: `OPENROUTER_API_KEY` (same logic as P1)

- [ ] **Step 2: Commit**

```bash
git add scripts/deploy_hf.py
git commit -m "chore: one-command HF Spaces deploy script (deferred use)"
```

---

### Task 9: Dockerfile + README + demo asset

**Files:**
- Create: `Dockerfile`, `README.md`, `assets/.gitkeep`

- [ ] **Step 1: Dockerfile** (HF docker SDK, Streamlit on :7860)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
```

- [ ] **Step 2: README.md** — P1-style. Include:
  - Centered `<h1>⚡ Semantic Cache for LLM Responses</h1>` + one-line pitch.
  - Badges: Python 3.12, Streamlit, OpenRouter (free), `Cache: BM25`→replace with `Similarity: cosine`, MIT. (No live-demo badge yet — deferred.)
  - ASCII diagram:
    ```
    prompt ─▶ exact-hash? ─▶ embed ─▶ cosine vs cache ─▶ ≥thr? ─▶ HIT (cached)
                                                          └─no─▶ LLM ─▶ store ─▶ MISS
    ```
  - "Why semantic caching" table (exact-string cache misses paraphrases; semantic cache catches them).
  - Model table (embeddings/chat — same as Global Constraints).
  - How-it-works table mapping stage → file (`embed.py`/`llm.py`/`cache.py`/`app.py`).
  - Run-locally (venv + `streamlit run app.py`), key setup, `pytest`, `python scripts/smoke.py`.
  - Deploy section: `python scripts/deploy_hf.py` (note it's the one-command path).
  - Project layout tree. Honest notes: in-memory only; brute-force cosine (FAISS = scale path); $ is illustrative on free tier.
  - License: MIT.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile README.md assets/.gitkeep
git commit -m "docs: README (badges, diagram, how-it-works) + Dockerfile"
```

---

### Task 10: Final verification + push

- [ ] **Step 1:** Full test suite — `.venv/Scripts/python -m pytest -q` → all pass (expect 13).
- [ ] **Step 2:** Smoke (if key present) — `.venv/Scripts/python scripts/smoke.py` → `OK`.
- [ ] **Step 3:** Manual UI sanity — `streamlit run app.py`, seed examples, ask a paraphrase, confirm semantic HIT + metrics move. Capture `assets/demo.png`.
- [ ] **Step 4:** Create GitHub repo + push:

```bash
gh repo create saiteja007-mv/semantic-cache --public --source=. --remote=origin --push
```
Expected: repo live at `https://github.com/saiteja007-mv/semantic-cache`.

- [ ] **Step 5 (optional, deferred):** Live deploy — `$env:HF_TOKEN=(gc C:\Users\saite\.cache\huggingface\token); .venv/Scripts/python scripts/deploy_hf.py`.

---

## Self-Review

**Spec coverage:** purpose/cache→Tasks 2–3; embeddings→4; chat→5; demo+threshold/metrics→6; smoke→7; deploy script→8; Dockerfile/README→9; error handling (missing key, dim mismatch, empty prompt)→embedded in Tasks 2/4/5/6; tests→2–5; delivery (repo, README, deferred HF)→9–10. No gaps.

**Placeholders:** none — all code steps carry full code; Tasks 8–9 reference an existing real P1 file with explicit, enumerated deltas (not "similar to").

**Type consistency:** `embed_fn`/`llm_fn` default to `None` (lazy-bound) everywhere; `AskResult`/`LookupResult`/`CacheStats` field names match between `cache.py` and `app.py`/`smoke.py`; `ask(prompt, cache, *, llm_fn=...)` signature consistent across Task 3 tests, `app.py`, `smoke.py`. Test count: 6 (Task 2) + 3 (Task 3) + 2 (Task 4) + 2 (Task 5) = 13.
