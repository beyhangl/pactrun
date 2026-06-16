"""Tests for pactrun.cost_model — real tokenizer + pricing with honest tags."""

import builtins

from pactrun import cost_model as cm


def test_openai_input_uses_real_tokenizer():
    text = "The quick brown fox jumps over the lazy dog. " * 5
    n, tag = cm.count_input_tokens("gpt-4o", [{"role": "user", "content": text}])
    assert n > 0
    assert tag == cm.ESTIMATED  # a real tiktoken count, not the heuristic


def test_anthropic_and_gemini_are_not_treated_as_openai():
    assert cm._is_openai_model("claude-sonnet-4-6") is False
    assert cm._is_openai_model("gemini-2.5-pro") is False
    assert cm._is_openai_model("gpt-4o") is True
    # Claude must still count (via litellm, never tiktoken which undercounts it).
    n, tag = cm.count_input_tokens("claude-sonnet-4-6", [{"role": "user", "content": "hello there friend"}])
    assert n > 0
    assert tag == cm.ESTIMATED


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
    assert tag == cm.EXACT


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
