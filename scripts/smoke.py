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
        f"hit_rate={s.hit_rate * 100:.0f}% latency_saved={s.latency_saved_s:.2f}s"
    )
    if not hit.hit:
        print("FAIL: expected a semantic cache hit on the paraphrase.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
