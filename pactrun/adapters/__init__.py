"""Framework adapters — auto-instrument agent frameworks with contract enforcement."""

from pactrun.adapters.manual import emit_llm_call, emit_tool_call

__all__ = ["emit_llm_call", "emit_tool_call"]

# Lazy imports for optional adapters
def __getattr__(name: str):
    if name == "OpenAIAdapter":
        from pactrun.adapters.openai import OpenAIAdapter
        return OpenAIAdapter
    if name == "AnthropicAdapter":
        from pactrun.adapters.anthropic import AnthropicAdapter
        return AnthropicAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
