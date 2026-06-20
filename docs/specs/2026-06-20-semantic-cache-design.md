# Semantic Cache for LLM Responses — Design

> Date: 2026-06-20 · Status: approved · Author: Sai Teja Mothukuri

## Purpose

A drop-in **semantic cache** that sits in front of an LLM chat call. Instead of
keying the cache on the exact prompt string, it keys on the prompt's **embedding**:
a new prompt that is *semantically similar* to a previously-seen prompt returns the
cached response instantly (a **cache HIT**), skipping the LLM call. This saves
latency and token cost whenever users ask the same thing in different words.

Portfolio goal: demonstrate clean AI/ML engineering — embeddings, vector
similarity, cache policy (threshold, eviction, TTL), and an honest, measurable
value proposition — with a live, clickable demo. Scope is the **simplest tier
that still demos** (no formal eval/benchmark tab in this version).

## Non-Goals

- No approximate-nearest-neighbor index (FAISS/HNSW). Brute-force numpy cosine is
  enough at demo scale; the README names ANN as the scale path.
- No persistent/disk-backed cache. In-memory, process-lifetime only.
- No eval/benchmark harness (hit-rate-vs-threshold curve, false-hit precision).
  Deferred — would be the natural next iteration.
- No live cloud hosting in this version. A one-command HF deploy script is included
  for later; default delivery is the GitHub repo + README + demo screenshot.
- No streaming responses (full responses are cached, so generation is non-streamed).

## Conventions (inherited from P1 hybrid-search-rag)

- OpenRouter via the OpenAI SDK (`base_url=https://openrouter.ai/api/v1`).
- Free models, `:free` suffix **required**:
  - Embeddings: `nvidia/llama-nemotron-embed-vl-1b-v2:free` (2048-dim)
  - Chat: `nvidia/nemotron-3-nano-30b-a3b:free`
- API key resolution: `OPENROUTER_API_KEY` env → `.streamlit/secrets.toml` fallback.
- Model override via `OPENROUTER_MODEL` / `OPENROUTER_EMBED_MODEL`.
- Per-project `.venv`. No torch / no local ML libraries — fully API-backed.
- Docker SDK for HF Spaces (Streamlit on port 7860, CORS/XSRF disabled).
- Secrets gitignored; `.streamlit/secrets.toml` never committed.

## Architecture

```
semantic-cache/
├── app.py                       # Streamlit demo UI
├── semcache/
│   ├── __init__.py              # exports SemanticCache, CacheStats, ask()
│   ├── embed.py                 # OpenRouter embeddings (batched, L2-normalized)
│   ├── llm.py                   # OpenRouter chat (non-streamed complete())
│   └── cache.py                 # SemanticCache + CacheStats + cached ask()
├── scripts/
│   ├── smoke.py                 # headless: seed prompts, fire paraphrases, assert HITs
│   └── deploy_hf.py             # one-command HF Spaces deploy (for later)
├── tests/
│   └── test_cache.py            # cache logic with mocked embeddings (offline)
├── Dockerfile                   # HF Spaces (docker SDK), Streamlit on :7860
├── requirements.txt             # streamlit, openai, numpy, python-dotenv, pytest
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
└── README.md                    # badges, ASCII diagram, how-it-works table
```

### Units and responsibilities

**`semcache/embed.py`** — `embed_texts(texts: list[str]) -> np.ndarray`.
Calls OpenRouter embeddings, returns an `(n, d)` float32 array, **L2-normalized**
(so cosine similarity == dot product). Resolves key/model like P1. `embed_one(text)`
convenience wrapper.

**`semcache/llm.py`** — `complete(prompt, history=None, api_key=None, model=None,
temperature=0.2) -> str`. Non-streamed OpenRouter chat completion returning the
full assistant message text. Same key/header pattern as P1.

**`semcache/cache.py`** — the core.
- `@dataclass CacheEntry`: `prompt: str`, `embedding: np.ndarray` (normalized),
  `response: str`, `created_at: float`, `hits: int`, `last_used: float`.
- `@dataclass CacheStats`: `lookups`, `hits`, `misses`, `llm_calls_saved`,
  `latency_saved_s`, `tokens_saved_est`, `cost_saved_est`. Property `hit_rate`.
- `class SemanticCache`:
  - `__init__(threshold=0.85, max_size=256, ttl_s=None, embed_fn=embed_one,
    cost_per_1k_tokens=0.0, avg_llm_latency_s=1.5)`. `embed_fn` is injectable so
    tests run offline.
  - `lookup(prompt) -> LookupResult | None`: exact-hash fast-path first (dict on
    normalized prompt string → instant exact HIT). Else embed prompt, compute
    cosine vs all live entries (vectorized numpy), take the argmax; if
    `best_sim >= threshold` return `LookupResult(entry, best_sim, exact=False)`,
    else `None`. Expired entries (TTL) are skipped/purged. Updates `hits`,
    `last_used` on the matched entry.
  - `put(prompt, response) -> None`: embed, append `CacheEntry`, evict LRU if over
    `max_size` (drop smallest `last_used`).
  - `entries() -> list[CacheEntry]`, `clear()`, `__len__`.
- `LookupResult`: `entry`, `similarity`, `exact`.
- Module-level `ask(prompt, cache, *, llm_fn=complete) -> AskResult`: orchestrator.
  Times the call. `lookup` → if hit, build `AskResult(response, hit=True,
  similarity, matched_prompt, latency_s, exact)` and increment saved-counters
  (latency_saved += avg_llm_latency_s; tokens/cost est from response length).
  → if miss, call `llm_fn`, `put`, return `AskResult(hit=False, ...)`.
- `AskResult` dataclass carries everything the UI needs to render one turn.

**`app.py`** — Streamlit.
- Cache lives in `st.session_state` (per-browser-session, multi-user safe like P1).
- Input box → `ask(...)`; render the answer + a **HIT (exact) / HIT (semantic) /
  MISS** badge, the nearest cached prompt and its similarity, this-call latency vs
  `avg_llm_latency_s`.
- Running metrics row: **hit-rate, lookups, calls saved, latency saved (s),
  est. $ saved** (with an honest caption that the models are free-tier, so $ is
  computed from a configurable `$/1k tokens` to illustrate the value prop).
- Sidebar: **threshold slider** (re-applied live to the cache), max size,
  `$/1k tokens` input, "Seed example prompts" button (loads ~6 prompts incl.
  paraphrase pairs so HITs are reproducible), "Clear cache".
- Expander: table of current cache entries (prompt, hits, age).

**`scripts/smoke.py`** — offline-friendly only if a key is present; seeds a few
prompts, fires paraphrases, asserts at least one semantic HIT, prints final stats.
Exit non-zero on failure.

**`tests/test_cache.py`** — pure-unit, **no network**: inject a deterministic
`embed_fn` (e.g. bag-of-words → vector, or a small fixed map) and a fake `llm_fn`.
Cover: exact-hash HIT; semantic HIT at sim ≥ threshold; MISS below threshold (→
llm_fn called once, entry stored); LRU eviction past `max_size`; TTL expiry;
stats counters increment correctly.

## Data flow

```
prompt ──▶ exact-hash hit? ──yes──▶ return cached (HIT, exact)
                │ no
                ▼
        embed(prompt) ──▶ cosine vs all entries ──▶ best_sim ≥ threshold?
                                                       │yes──▶ return cached (HIT, semantic)
                                                       │no
                                                       ▼
                                           llm.complete(prompt) ──▶ put(prompt, resp) ──▶ return (MISS)
```

## Error handling

- Missing `OPENROUTER_API_KEY`: `embed`/`llm` raise `RuntimeError` with a clear
  "set your key (see README)" message; `app.py` catches and shows a friendly banner.
- OpenRouter call failure (network/4xx/5xx): surfaced to the UI as an error toast;
  the cache is left unchanged (no poisoning with empty responses).
- Empty/whitespace prompt: ignored (no lookup, no call).
- Embedding dimension mismatch (model change mid-session): guard in `lookup` — if a
  stored entry's dim ≠ query dim, skip it (defensive; logged once).

## Testing & verification

- `pytest tests/ -q` → all cache-logic tests pass offline (mocked embeddings/LLM).
- `python scripts/smoke.py` (with a key) → prints stats, asserts a semantic HIT.
- Manual: `streamlit run app.py`, seed examples, ask a paraphrase, confirm a
  semantic HIT with similarity shown and counters incrementing.

## Delivery

- GitHub repo `github.com/saiteja007-mv/semantic-cache` (public).
- README: badge header, ASCII pipeline diagram, "why semantic caching" table,
  how-it-works table, run-locally + deploy sections (P1 style), demo screenshot.
- Live HF Space: **deferred** — `scripts/deploy_hf.py` makes it one command later.
