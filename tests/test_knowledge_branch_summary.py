"""The existing summary service stays authoritative; knowledge adds purpose audits."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from resolv.core.config import SummaryModelSettings
from resolv.schemas.knowledge_branch import FrozenJSON
from resolv.services.business_summary import summary_profile
from resolv.services.knowledge_branch_summary import KnowledgeSummaryAdapter
from resolv.services.summarization import SUMMARY_PROMPT, SummaryService
from tests.conversation_fixtures import make_turn
from tests.knowledge_branch_fixtures import branch_profile
from tests.test_context import Counter


def adapter(hook):
    settings = SummaryModelSettings(
        _env_file=None,
        model_name="fixture-summary",
        api_key="offline",
        base_url="https://example.invalid/v1",
        model_context_window_tokens=16000,
    )

    class Model:
        def invoke(self, messages):
            return AIMessage(
                content='{"user_goal":"goal","confirmed_facts":[],"completed_actions":[],"open_questions":[]}',
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
                response_metadata={"finish_reason": "stop"},
            )

    base = SummaryService(Model(), Counter(), settings)
    p = branch_profile()
    cfg = summary_profile(settings)
    model = p.summary_profile.model_copy(
        update={
            "model_name": settings.model_name,
            "api_address_fingerprint": cfg["api_address_fingerprint"],
            "model_api_mode": cfg["model_api_mode"],
            "system_prompt": SUMMARY_PROMPT,
            "model_context_window_tokens": settings.model_context_window_tokens,
            "app_max_input_tokens": settings.input_max_tokens,
            "provider_input_overhead_tokens": settings.provider_input_overhead_tokens,
            "reserved_output_tokens": settings.reserved_output_tokens,
            "safety_margin_tokens": settings.safety_margin_tokens,
            "timeout_seconds": settings.timeout_seconds,
            "enable_thinking": None,
            "top_p": None,
        }
    )
    profile = p.model_copy(
        update={"summary_profile": model, "summary_config": FrozenJSON.from_value(cfg)}
    )
    return KnowledgeSummaryAdapter(base, profile, hook)


def test_summary_reuses_existing_candidate_logic_with_started_and_succeeded_audits():
    audits = []
    wrapped = adapter(audits.append).for_run(SimpleNamespace(), None)
    candidate = wrapped.create_candidate(None, [make_turn()])
    assert candidate.content.user_goal == "goal"
    assert [a.outcome for a in audits] == ["started", "succeeded"]
    assert all(a.purpose == "summary" for a in audits)
    assert audits[-1].input_tokens == 100


def test_summary_audit_storage_failure_cannot_become_summary_fallback():
    def fail(audit):
        raise ValueError("storage hash")

    wrapped = adapter(fail).for_run(SimpleNamespace(), None)
    with pytest.raises(ValueError, match="storage hash"):
        wrapped.create_candidate(None, [make_turn()])


def test_summary_model_profile_cannot_mislabel_actual_provider():
    from resolv.schemas.knowledge_branch import KnowledgeBranchError

    wrapped = adapter(lambda _: None)
    wrapped.profile = wrapped.profile.model_copy(
        update={
            "summary_profile": wrapped.profile.summary_profile.model_copy(
                update={"model_name": "wrong-model"}
            )
        }
    )
    with pytest.raises(
        KnowledgeBranchError, match="knowledge_summary_profile_mismatch"
    ):
        wrapped.for_run(SimpleNamespace(), None)
