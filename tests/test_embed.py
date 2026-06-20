import numpy as np

import semcache.embed as embed


def test_embed_texts_normalizes(monkeypatch):
    # _post returns raw (un-normalized) vectors; embed_texts must L2-normalize.
    monkeypatch.setattr(embed, "get_api_key", lambda: "k")
    monkeypatch.setattr(embed, "_post", lambda inputs, api_key, model: [[3.0, 4.0]] * len(list(inputs)))
    out = embed.embed_texts(["a", "b"])
    assert out.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)


def test_embed_one_returns_1d(monkeypatch):
    monkeypatch.setattr(embed, "get_api_key", lambda: "k")
    monkeypatch.setattr(embed, "_post", lambda inputs, api_key, model: [[1.0, 0.0]])
    v = embed.embed_one("hi")
    assert v.shape == (2,)


def test_embed_raises_without_key(monkeypatch):
    monkeypatch.setattr(embed, "get_api_key", lambda: None)
    try:
        embed.embed_texts(["x"])
    except RuntimeError as e:
        assert "OPENROUTER_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
