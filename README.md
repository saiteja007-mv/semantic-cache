<h1 align="center">⚡ Semantic Cache for LLM Responses</h1>

<p align="center">
  <b>Cache LLM answers by prompt <i>meaning</i>, not exact text.</b><br>
  A semantically similar prompt returns the cached answer instantly — no LLM call, no cost, no wait.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/LLM-OpenRouter%20(free)-6E56CF" alt="OpenRouter">
  <img src="https://img.shields.io/badge/Lookup-cosine%20%2B%20exact--hash-16A34A" alt="Lookup">
  <img src="https://img.shields.io/badge/License-MIT-22C55E" alt="MIT License">
</p>

---

A drop-in **semantic cache** that sits in front of an LLM chat call. Instead of
keying on the exact prompt string, it keys on the prompt's **embedding**: when a
new prompt is close enough (cosine similarity ≥ a tunable threshold) to one it has
already answered, it returns the cached response — skipping the LLM entirely. That
saves **latency** and **token cost** on the paraphrased, repeated questions real
users actually ask. Everything runs on **free OpenRouter models** — no local GPU,
no local ML libraries.

```
prompt ─▶ exact-hash hit? ─yes─▶ HIT (cached, ~0 ms)
            │ no
            ▼
      embed(prompt) ─▶ cosine vs cached embeddings ─▶ best ≥ threshold?
                                                        │ yes ─▶ HIT (cached)
                                                        │ no
                                                        ▼
                                            LLM call ─▶ store ─▶ MISS
```

## 🧠 Why semantic caching?

| Scenario | Exact-string cache | Semantic cache |
|---|---|---|
| Identical prompt repeated | ✅ hit | ✅ hit |
| "Capital of France?" vs "Tell me France's capital" | ❌ miss → pays again | ✅ hit |
| Same intent, different wording / casing / punctuation | ❌ miss | ✅ hit |

An exact cache only helps when the string matches byte-for-byte. A semantic cache
catches **paraphrases**, which is where the real repetition (and cost) lives.

| | Model (OpenRouter, free tier) |
|---|---|
| **Embeddings** | `nvidia/llama-nemotron-embed-vl-1b-v2:free` (2048-dim, L2-normalized) |
| **Chat** | `nvidia/nemotron-3-nano-30b-a3b:free` |

## ✨ Features

- **Two-stage lookup** — O(1) exact-hash fast path, then vectorized cosine over cached embeddings.
- **Tunable threshold** — trade hit-rate vs precision live in the UI.
- **LRU eviction + optional TTL** — bounded memory, optional staleness window.
- **Live savings metrics** — hit-rate, LLM calls saved, latency saved, est. $ saved.
- **Multi-user safe** — the cache lives in `st.session_state`, scoped per browser session.
- **Fully API-backed** — no torch, no local models.

## 🖥️ Run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# add your OpenRouter key (see below), then:
streamlit run app.py        # http://localhost:8501
```

Run the tests (offline — mocked embeddings/LLM) and the headless smoke check:

```powershell
python -m pytest -q          # 13 unit tests, no network
python scripts/smoke.py      # needs a key: seeds + asserts a semantic HIT
```

### 🔑 Where to put your OpenRouter API key

Get a free key at <https://openrouter.ai/keys>. Locally, use any one of:

- **`.streamlit/secrets.toml`** (recommended, gitignored):
  ```toml
  OPENROUTER_API_KEY = "sk-or-v1-..."
  ```
- **`.env`** file: `OPENROUTER_API_KEY=sk-or-v1-...`
- env var: `setx OPENROUTER_API_KEY "sk-or-v1-..."`

Override models with `OPENROUTER_MODEL` / `OPENROUTER_EMBED_MODEL`.

## 🚀 Deploy (Hugging Face Spaces, free)

One command — creates the Space (Docker SDK), sets the key as a Space secret, and
uploads the app:

```powershell
python scripts/deploy_hf.py --space-name semantic-cache
```

(Needs a Hugging Face token via `huggingface-cli login` or `HF_TOKEN`.) The
included `Dockerfile` runs Streamlit on port 7860.

## 🔧 How it works

| Stage | File | What it does |
|---|---|---|
| Embed | `semcache/embed.py` | OpenRouter embeddings, batched, L2-normalized |
| Chat | `semcache/llm.py` | OpenRouter chat completion (non-streamed, full response) |
| Cache | `semcache/cache.py` | `SemanticCache` (exact-hash + cosine, LRU, TTL) + `ask()` orchestrator + stats |
| Client | `semcache/_openrouter.py` | Shared key resolution + configured OpenAI client |
| UI | `app.py` | Streamlit chat, HIT/MISS badge, similarity, live savings metrics |

`ask(prompt, cache)` is the whole API: it looks the prompt up, returns a cached
answer on a hit, or calls the LLM and stores the result on a miss — tracking
savings either way.

## 🗂️ Project layout

```
semantic-cache/
├── app.py                       # Streamlit demo
├── semcache/
│   ├── _openrouter.py           # shared key + client
│   ├── embed.py                 # OpenRouter embeddings (normalized)
│   ├── llm.py                   # OpenRouter chat completion
│   └── cache.py                 # SemanticCache + ask() + stats
├── scripts/
│   ├── smoke.py                 # headless: assert a semantic HIT
│   └── deploy_hf.py             # one-command HF Spaces deploy
├── tests/                       # 13 offline unit tests (mocked embeds/LLM)
├── Dockerfile                   # HF Spaces (docker SDK), Streamlit on :7860
└── requirements.txt             # no torch — fully API-backed
```

## ⚖️ Honest notes / scale path

- **In-memory only** — the cache lives for the process / browser session; not persisted.
- **Brute-force cosine** — fine at demo scale; for large caches swap in an ANN
  index (FAISS / HNSW) behind the same `lookup()` interface.
- **$ saved is illustrative** — the demo models are free-tier, so the dollar figure
  is computed from a configurable `$/1k tokens` rate to show the value proposition.
- **Threshold is a precision/recall dial** — too low risks returning a cached answer
  for a similar-but-different prompt; tune it for your domain.

## 📄 License

MIT
