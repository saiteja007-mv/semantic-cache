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
