import importlib

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from tests.test_business_model import Stream
from tests.test_context import Counter


def api():
    try:
        return importlib.import_module("resolv.services.business_summary")
    except ModuleNotFoundError:
        pytest.fail("independent audited summary transport missing")


def test_non_qwen_summary_preserves_responses_profile(monkeypatch):
    from resolv.core.config import SummaryModelSettings

    params = []
    monkeypatch.setattr(
        api(), "ChatOpenAI", lambda **kwargs: params.append(kwargs) or object()
    )
    settings = SummaryModelSettings(
        _env_file=None,
        model_name="gpt-6-sol",
        api_key="secret",
        base_url="https://example.test/v1",
        model_context_window_tokens=256000,
    )
    api().get_summary_client(settings)
    assert params[0]["use_responses_api"] is True and params[0]["max_retries"] == 0
    assert "extra_body" not in params[0]


@pytest.mark.parametrize(
    "usage,raises",
    [
        (None, True),
        ({"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}, False),
    ],
)
def test_summary_usage_is_independent_and_public_only(usage, raises):
    from resolv.core.config import SummaryModelSettings
    from resolv.schemas.conversation import ConversationError

    settings = SummaryModelSettings(
        _env_file=None,
        model_name="gpt-6-sol",
        api_key="secret",
        base_url="https://example.test/v1",
        model_context_window_tokens=256000,
        provider_input_overhead_tokens=5000,
    )
    chunk = AIMessageChunk(
        content='{"user_goal":"查询","confirmed_facts":[],"completed_actions":[],"open_questions":[]}',
        usage_metadata=usage,
        response_metadata={"status": "completed"},
        additional_kwargs={"reasoning_content": "private thought"},
    )
    audits = []
    adapter = api().AuditedSummaryModel(
        Stream([chunk]), Counter(), settings, audit_hook=audits.append
    )
    if raises:
        with pytest.raises(ConversationError, match="summary_usage_unknown"):
            adapter.invoke([HumanMessage(content="历史")])
    else:
        result = adapter.invoke([HumanMessage(content="历史")])
        assert "private thought" not in result.model_dump_json()
        assert audits[-1].actual_input == 100 and audits[-1].purpose == "summary"
