"""Embedding-keyed semantic cache: lookup, put, eviction, TTL, stats, ask()."""
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


@dataclass
class AskResult:
    prompt: str
    response: str
    hit: bool
    exact: bool
    similarity: float
    matched_prompt: Optional[str]
    latency_s: float


def _norm_key(prompt: str) -> str:
    return prompt.strip().lower()


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 chars/token heuristic


def _default_embed(text: str) -> np.ndarray:
    # Bound lazily to avoid importing the network client at module import.
    from .embed import embed_one

    return embed_one(text)


def _default_complete(prompt: str) -> str:
    from .llm import complete

    return complete(prompt)


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
        key = _norm_key(prompt)
        now = self._now()
        existing = self._exact.get(key)
        if existing is not None:
            # Same prompt re-seen: refresh in place — no duplicate entry, no re-embed.
            existing.response = response
            existing.created_at = now
            existing.last_used = now
            return
        emb = self.embed_fn(prompt)
        entry = CacheEntry(prompt, emb, response, created_at=now, last_used=now)
        self._entries.append(entry)
        self._exact[key] = entry
        if len(self._entries) > self.max_size:
            lru = min(self._entries, key=lambda e: e.last_used)
            self._entries.remove(lru)
            lru_key = _norm_key(lru.prompt)
            # Only drop the exact mapping if it still points to the evicted entry.
            if self._exact.get(lru_key) is lru:
                self._exact.pop(lru_key, None)


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
            prompt,
            hit.entry.response,
            hit=True,
            exact=hit.exact,
            similarity=hit.similarity,
            matched_prompt=hit.entry.prompt,
            latency_s=latency,
        )
    response = llm_fn(prompt)
    cache.put(prompt, response)
    latency = time.perf_counter() - t0
    cache.stats.misses += 1
    return AskResult(
        prompt,
        response,
        hit=False,
        exact=False,
        similarity=0.0,
        matched_prompt=None,
        latency_s=latency,
    )
