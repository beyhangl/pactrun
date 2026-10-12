"""Tests for pactrun.cost_model — real tokenizer + pricing with honest tags."""

import builtins

from pactrun import cost_model as cm


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_TIKTOKEN = _have("tiktoken")
HAS_LITELLM = _have("litellm")


def test_openai_input_uses_real_tokenizer():
    text = "The quick brown fox jumps over the lazy dog. " * 5
    n, tag = cm.count_input_tokens("gpt-4o", [{"role": "user", "content": text}])
    assert n > 0
    # tiktoken when installed; honest heuristic fallback otherwise.
    assert tag == (cm.ESTIMATED if HAS_TIKTOKEN else cm.HEURISTIC)


def test_anthropic_and_gemini_are_not_treated_as_openai():
    assert cm._is_openai_model("claude-sonnet-4-6") is False
    assert cm._is_openai_model("gemini-2.5-pro") is False
    assert cm._is_openai_model("gpt-4o") is True
    # Claude must still count (via litellm, never tiktoken which undercounts it).
    n, tag = cm.count_input_tokens("claude-sonnet-4-6", [{"role": "user", "content": "hello there friend"}])
    assert n > 0
    assert tag == (cm.ESTIMATED if HAS_LITELLM else cm.HEURISTIC)


def test_heuristic_fallback_when_libs_absent(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("tiktoken", "litellm"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    n, tag = cm.count_input_tokens("gpt-4o", [{"role": "user", "content": "x" * 40}])
    assert tag == cm.HEURISTIC
    assert n == max(1, 40 // 4)

    cost, ctag = cm.worstcase_cost("gpt-4o", 100, 1000)
    assert ctag == cm.HEURISTIC
    assert cost > 0


def test_worstcase_is_monotone_in_max_output():
    a, _ = cm.worstcase_cost("gpt-4o", 100, 100)
    b, _ = cm.worstcase_cost("gpt-4o", 100, 1000)
    assert b >= a


def test_actual_cost_is_exact_via_litellm():
    cost, tag = cm.actual_cost("gpt-4o", 1000, 500)
    assert cost > 0
    assert tag == (cm.EXACT if HAS_LITELLM else cm.HEURISTIC)


def test_unknown_model_falls_back_conservatively():
    cost, tag = cm.actual_cost("totally-made-up-model-xyz", 1000, 1000)
    assert cost > 0
    assert tag == cm.HEURISTIC


def test_precall_worstcase_weakest_link_tag(monkeypatch):
    # Real tokenizer + real pricing -> estimated.
    _, tag = cm.precall_worstcase("gpt-4o", [{"role": "user", "content": "hi"}], 1000)
    assert tag in (cm.ESTIMATED, cm.HEURISTIC)  # estimated when litellm prices gpt-4o
    # If the tokenizer falls back, the whole thing is heuristic.
    real_import = builtins.__import__

    def block_tok(name, *a, **k):
        if name == "tiktoken":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", block_tok)
    _, tag2 = cm.precall_worstcase("gpt-4o", [{"role": "user", "content": "hi"}], 1000)
    assert tag2 == cm.HEURISTIC


def test_heuristic_counts_tool_definitions(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("tiktoken", "litellm"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "refund", "description": "policy " * 500}}]
    before, _ = cm.count_input_tokens("gpt-4.1", messages)
    after, tag = cm.count_input_tokens("gpt-4.1", messages, tools=tools)
    assert after > before + 500
    assert tag == cm.HEURISTIC
    base_cost, _ = cm.precall_worstcase("gpt-4.1", messages, 100)
    tool_cost, _ = cm.precall_worstcase("gpt-4.1", messages, 100, tools=tools)
    assert tool_cost > base_cost


def test_openai_tokenizer_counts_serialized_tools(monkeypatch):
    import sys
    from types import SimpleNamespace

    encoded = []

    def encode(text):
        encoded.append(text)
        return list(text)

    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(
        encoding_for_model=lambda model: SimpleNamespace(encode=encode),
    ))
    tools = [{"type": "function", "function": {"name": "refund", "parameters": {"type": "object"}}}]
    before, _ = cm.count_input_tokens("gpt-4.1", [{"role": "user", "content": "hello"}])
    after, tag = cm.count_input_tokens("gpt-4.1", [{"role": "user", "content": "hello"}], tools=tools)
    assert after > before
    assert '"refund"' in encoded[-1]
    assert '"parameters"' in encoded[-1]
    assert tag == cm.ESTIMATED


def test_nonopenai_token_counter_receives_tools(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []

    def token_counter(**kwargs):
        calls.append(kwargs)
        return 100 if kwargs.get("tools") else 5

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(token_counter=token_counter))
    tools = [{"name": "refund", "input_schema": {"type": "object"}}]
    assert cm.count_input_tokens("claude-sonnet-4", [], tools=tools) == (100, cm.ESTIMATED)
    assert calls[-1]["tools"] == tools


def test_anthropic_system_text_blocks_count_like_equivalent_text():
    text = "follow this refund policy " * 1000
    plain, _ = cm.count_input_tokens("claude-sonnet-4", [], system=text)
    blocks, _ = cm.count_input_tokens(
        "claude-sonnet-4", [], system=[{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}],
    )
    assert blocks == plain
    assert blocks > 1000


def test_historical_assistant_tool_arguments_are_counted():
    messages = [{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "local-demo", "type": "function", "function": {
            "name": "refund", "arguments": '{"explanation":"' + "policy " * 2000 + '"}',
        }}],
    }]
    n, _ = cm.count_input_tokens("gpt-4.1", messages)
    assert n > 1000
    normalized = cm._normalize_messages(messages, None)
    assert "refund" in normalized[0]["content"]
    assert "explanation" in normalized[0]["content"]


def test_anthropic_tool_use_and_tool_result_text_are_counted():
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "name": "refund", "input": {"reason": "policy " * 1000}}]},
        {"role": "user", "content": [{"type": "tool_result", "content": [{"type": "text", "text": "receipt " * 1000}]}]},
    ]
    n, _ = cm.count_input_tokens("claude-sonnet-4", messages)
    assert n > 1000
