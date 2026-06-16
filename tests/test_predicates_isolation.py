"""Tests for tenant_response_isolation + the session-metadata seeding it needs."""

from pactrun import Contract, tenant_response_isolation


def test_matching_tenant_passes():
    c = Contract("t").require(tenant_response_isolation(), on_fail="log")
    with c.session(metadata={"tenant": "acme"}) as s:
        s.emit_llm_response(model="m", output="hi", metadata={"tenant": "acme"})
    assert s.is_compliant


def test_mismatched_tenant_blocks():
    c = Contract("t").require(tenant_response_isolation(), on_fail="log")
    with c.session(metadata={"tenant": "acme"}) as s:
        s.emit_llm_response(model="m", output="leak", metadata={"tenant": "globex"})
    assert not s.is_compliant


def test_unbound_run_fails_closed():
    c = Contract("t").require(tenant_response_isolation(), on_fail="log")
    with c.session() as s:  # no tenant set
        s.emit_llm_response(model="m", output="hi")
    assert not s.is_compliant


def test_untagged_response_in_bound_run_passes():
    # A response with no tenant tag in a bound run is fine (nothing to mismatch).
    c = Contract("t").require(tenant_response_isolation(), on_fail="log")
    with c.session(metadata={"tenant": "acme"}) as s:
        s.emit_llm_response(model="m", output="hi")
    assert s.is_compliant


def test_known_tenants_text_scan():
    c = Contract("t").require(
        tenant_response_isolation(known_tenants=["acme", "globex"]), on_fail="log"
    )
    with c.session(metadata={"tenant": "acme"}) as s:
        s.emit_llm_response(model="m", output="Here is globex's secret plan")
    assert not s.is_compliant


def test_callable_tenant_key():
    pred = tenant_response_isolation(tenant_key=lambda state: state.metadata.get("org"))
    c = Contract("t").require(pred, on_fail="log")
    with c.session(metadata={"org": "acme"}) as s:
        s.emit_llm_response(model="m", output="x", metadata={"tenant": "globex"})
    assert not s.is_compliant


def test_custom_response_tag_key():
    pred = tenant_response_isolation(response_tag_key="owner")
    c = Contract("t").require(pred, on_fail="log")
    with c.session(metadata={"tenant": "acme"}) as s:
        s.emit_llm_response(model="m", output="x", metadata={"owner": "globex"})
    assert not s.is_compliant


# ---------------------------------------------------------------------------
# regression: _start seeds state.metadata from session metadata
# ---------------------------------------------------------------------------

def test_session_metadata_seeds_state_metadata():
    c = Contract("t")
    with c.session(metadata={"tenant": "acme", "k": 1}) as s:
        assert s.state.metadata.get("tenant") == "acme"
        assert s.state.metadata.get("k") == 1


def test_explicit_state_metadata_not_overwritten():
    c = Contract("t")
    s = c.session(metadata={"tenant": "from-session"})
    s.state.metadata["tenant"] = "preset"  # set before _start
    with s:
        assert s.state.metadata["tenant"] == "preset"  # existing key wins


def test_registered():
    import pactrun
    assert "tenant_response_isolation" in pactrun.list_predicates()
