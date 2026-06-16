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
    if name == "PactrunCallbackHandler":
        from pactrun.adapters.langchain import PactrunCallbackHandler
        return PactrunCallbackHandler
    if name == "GeminiAdapter":
        from pactrun.adapters.gemini import GeminiAdapter
        return GeminiAdapter
    if name == "LiteLLMAdapter":
        from pactrun.adapters.litellm import LiteLLMAdapter
        return LiteLLMAdapter
    if name == "GuardedMCPSession":
        from pactrun.adapters.mcp import GuardedMCPSession
        return GuardedMCPSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
