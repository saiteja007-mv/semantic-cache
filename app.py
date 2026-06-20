"""Streamlit demo: semantic cache in front of an OpenRouter LLM."""
from __future__ import annotations

import streamlit as st

from semcache import SemanticCache, ask
from semcache._openrouter import get_api_key

st.set_page_config(page_title="Semantic Cache for LLMs", page_icon="⚡", layout="wide")

EXAMPLES = [
    "What is the capital of France?",
    "Tell me France's capital city",             # paraphrase of #1 -> semantic HIT
    "How do I reverse a list in Python?",
    "What's the Python way to reverse a list?",   # paraphrase of #3 -> semantic HIT
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
    cache.max_size = int(st.number_input("Max cache size", 8, 4096, cache.max_size, 8))
    cache.cost_per_1k_tokens = st.number_input(
        "$ / 1k tokens (for savings est.)", 0.0, 100.0, cache.cost_per_1k_tokens, 0.05
    )
    st.caption("Models are free-tier; $ is illustrative from this rate.")
    if st.button("Seed example prompts"):
        for q in EXAMPLES:
            try:
                ask(q, cache)
            except Exception as e:  # noqa: BLE001
                st.error(f"Seed failed: {e}")
                break
        else:
            st.success(f"Seeded {len(EXAMPLES)} prompts.")
    if st.button("Clear cache"):
        cache.clear()
        cache.stats = type(cache.stats)()
        st.session_state.pop("history", None)

if not get_api_key():
    st.warning("Set `OPENROUTER_API_KEY` (env or `.streamlit/secrets.toml`). See README.")

s = cache.stats
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hit rate", f"{s.hit_rate * 100:.0f}%")
c2.metric("Lookups", s.lookups)
c3.metric("LLM calls saved", s.llm_calls_saved)
c4.metric("Latency saved", f"{s.latency_saved_s:.1f}s")
c5.metric("Est. $ saved", f"${s.cost_saved_est:.4f}")

prompt = st.chat_input("Ask something (then ask it again in different words)…")
if prompt and prompt.strip():
    try:
        r = ask(prompt, cache)
    except Exception as e:  # noqa: BLE001
        st.error(f"LLM/embedding error: {e}")
    else:
        st.session_state.setdefault("history", []).append(r)

for r in reversed(st.session_state.get("history", [])):
    with st.chat_message("user"):
        st.write(r.prompt)
    with st.chat_message("assistant"):
        if r.hit:
            kind = "exact" if r.exact else f"semantic · sim {r.similarity:.3f}"
            st.success(f"⚡ CACHE HIT ({kind}) · {r.latency_s * 1000:.0f} ms")
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
    if rows:
        st.table(rows)
    else:
        st.write("empty")
