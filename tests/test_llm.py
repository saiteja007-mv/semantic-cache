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
