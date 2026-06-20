import numpy as np

from semcache.cache import SemanticCache, ask

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


def test_duplicate_put_updates_in_place_no_extra_embed():
    calls = {"n": 0}

    def counting_embed(text):
        calls["n"] += 1
        return fake_embed(text)

    c = SemanticCache(threshold=0.85, max_size=1, embed_fn=counting_embed)
    c.put("x", "old")          # embeds once
    c.put("x", "new")          # same key -> update in place: no duplicate, no embed
    assert len(c) == 1
    r = c.lookup("x")          # exact hit -> no embed
    assert r is not None and r.exact and r.entry.response == "new"
    assert calls["n"] == 1


def test_exact_hit_survives_when_embeddings_degraded():
    state = {"allow": True}

    def embed(text):
        if not state["allow"]:
            raise RuntimeError("embedding unavailable")
        return fake_embed(text)

    c = SemanticCache(threshold=0.85, max_size=1, embed_fn=embed)
    c.put("x", "old")
    c.put("x", "new")
    state["allow"] = False      # embeddings now down
    r = c.lookup("x")           # must still exact-hit without embedding
    assert r is not None and r.exact and r.entry.response == "new"


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
    ask("capital of france", c, llm_fn=llm)         # miss -> 1 call
    r = ask("france capital", c, llm_fn=llm)         # semantic hit -> no call
    assert llm.calls == 1 and r.hit and not r.exact
    assert c.stats.hits == 1 and c.stats.llm_calls_saved == 1
    assert c.stats.tokens_saved_est > 0
    assert 0.0 < c.stats.hit_rate <= 1.0


def test_ask_cost_estimate_uses_rate():
    c = make_cache(cost_per_1k_tokens=10.0)
    llm = CountingLLM()
    ask("capital of france", c, llm_fn=llm)
    ask("capital of france", c, llm_fn=llm)          # exact hit
    assert c.stats.cost_saved_est > 0
