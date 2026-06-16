"""Observability integrations for pactrun (currently OpenTelemetry GenAI)."""


def __getattr__(name):
    if name in ("OTelObserver", "assert_gen_ai_span"):
        from pactrun.observability import otel

        return getattr(otel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
