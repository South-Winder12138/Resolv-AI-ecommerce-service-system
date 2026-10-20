import json

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from resolv.schemas.conversation import ConversationError
from tests.test_business_model import Stream, final
from tests.test_logistics_contracts import api
from tests.test_logistics_policy import config, policy


class Counter:
    def __init__(self):
        self.schemas = []

    def count(self, messages, tools):
        self.schemas.append(tools)
        return 1000


def chunk(name="query_order", args=None):
    return AIMessageChunk(content="", tool_call_chunks=[{
        "name": name, "id": "tc", "index": 0,
        "args": json.dumps(args or {"order_id": "MOCK001"}),
    }])


def invoke(client, p, audits, counter=None):
    return api("resolv.core.business_model").invoke_business_model(
        client, [HumanMessage(content="查订单 MOCK001 物流")], config=config(),
        counter=counter or Counter(), model_step=p.model_step, attempt=1,
        audit_hook=audits.append, tool_policy=p,
    )


def test_dynamic_tools_and_native_choice_match_counted_schema_and_audit():
    client, audits, counter = Stream([chunk(), final("", reason="tool_calls")]), [], Counter()
    p = policy()
    result = invoke(client, p, audits, counter)
    assert client.bound == [["query_order"]]
    assert client.parameters["tool_choice"] == {"type": "function", "function": {"name": "query_order"}}
    assert [s["function"]["name"] for s in counter.schemas[-1]] == ["query_order"]
    saved = result.audit.model_dump(mode="json")
    assert saved["tool_policy_fingerprint"] == p.fingerprint
    assert saved["tool_policy"]["allowed_tool_names"] == ["query_order"]
    assert saved["returned_tool_names"] == ["query_order"]
    assert result.model_dump(mode="json")["audit"]["tool_policy_fingerprint"] == p.fingerprint


def test_closed_phase_binds_no_tools_and_counts_empty_schema():
    client, audits, counter = Stream([final()]), [], Counter()
    p = policy(step=6)
    invoke(client, p, audits, counter)
    assert client.bound == [] and client.parameters["tool_choice"] == "none"
    assert counter.schemas == [[]] and audits[0].require_tool is False


def test_unopened_provider_tool_is_rejected_and_audited_without_resend():
    client, audits = Stream([chunk("query_logistics", {"tracking_no": "MOCKSF001"}), final("", reason="tool_calls")]), []
    with pytest.raises(ConversationError, match="business_scope_violation"):
        invoke(client, policy(), audits)
    assert len(client.calls) == 1 and audits[-1].error_code == "business_scope_violation"
    assert audits[-1].returned_tool_names == ["query_logistics"]


def test_v2_required_read_rejects_text_only_response():
    audits = []
    with pytest.raises(ConversationError, match="model_tool_call_required"):
        invoke(Stream([final("将查询订单")]), policy(), audits)
    assert audits[-1].require_tool is True


def test_sixth_step_preserves_violating_calls_for_persisted_step_limit():
    client, audits = Stream([chunk(), final("", reason="tool_calls")]), []
    result = invoke(client, policy(step=6), audits)
    assert client.bound == [] and result.message.tool_calls[0]["name"] == "query_order"
    assert audits[-1].tool_policy.phase == "step_limit"
