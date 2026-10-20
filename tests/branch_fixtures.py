"""Offline, evidence-backed DTO builders; no model accuracy claims."""

from datetime import UTC, datetime

import pytest

from resolv.schemas.understanding import UnderstandingRequestRecord, UnderstandingView
from resolv.understanding_rules import INTENT_BRANCHES_V1
from tests.understanding_fixtures import input_ref, make_resolution


@pytest.fixture
def branch_factory(tmp_path):
    from resolv.db.engine import create_db_engine, create_session_factory
    from resolv.models.conversation import Conversation
    from resolv.models.understanding import ConversationUnderstanding  # noqa: F401

    engine = create_db_engine(f"sqlite:///{tmp_path / 'branch-domain.db'}")
    Conversation.metadata.create_all(engine)
    yield create_session_factory(engine)
    engine.dispose()


def make_view(text: str, intent: str | None, *, ready: bool = True):
    branch = INTENT_BRANCHES_V1.get(intent)
    primary = UnderstandingRequestRecord(
        task_id="task-1",
        understanding_id="u-1",
        conversation_id="c",
        ordinal=0,
        role="primary",
        text=text,
        intent=intent,
        branch=branch,
        evidence_refs=[input_ref(text)],
        bindings=[],
        ambiguities=[],
        unresolved_references=[],
        queue_status="not_queued",
        selected_by_understanding_id=None,
        created_at=datetime.now(UTC),
    )
    return UnderstandingView(
        schema_version=1,
        understanding_id="u-1",
        conversation_id="c",
        status="ready" if ready else "needs_clarification",
        original_text=text,
        resolution=make_resolution(resolved_text=text),
        primary_request=primary,
        branch=branch,
        route_ready=ready,
        pending_requests=[],
        clarification_question=None if ready else "请问您指的是哪笔订单？",
        context_refs=[],
        versions={"mapping": "intent-branches-v1", "schema": 1},
        stage_results={},
        followup_ids={},
        error_code=None,
        retryable=False,
    )


def make_routed_service(
    factory, checkpoints, *, stage_outputs, business_model=None, business_factory=None
):
    from langchain_core.messages import AIMessage

    from resolv.core.config import ContextSettings
    from resolv.services.context import ContextBuilder
    from resolv.services.conversation import ConversationService
    from resolv.services.tool_execution import WriteExecutionService
    from tests.test_context import Counter
    from tests.understanding_fixtures import build_test_service

    understanding = build_test_service(factory, checkpoints)
    understanding.resolver.model.client.replies = [
        value
        if isinstance(value, (AIMessage, Exception))
        else AIMessage(content=value.model_dump_json())
        for value in stage_outputs
    ]
    return ConversationService(
        factory,
        checkpoints,
        understanding=understanding,
        orchestration_version="routed-v2",
        context_builder=ContextBuilder(
            ContextSettings(
                _env_file=None,
                tokenizer_profile="o200k_base",
                model_context_window_tokens=256000,
            ),
            Counter(),
        ),
        model=OfflineUsageModel(business_model) if business_model is not None else None,
        write_service=WriteExecutionService(factory, business_factory)
        if business_factory
        else None,
    )


class OfflineUsageModel:
    """Deterministic transport metadata for offline state/recovery tests only."""

    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def bind_tools(self, tools):
        self.source.bind_tools(tools)
        return self

    def bind(self, **kwargs):
        self.source.bind(**kwargs)
        return self

    def stream(self, messages):
        for chunk in self.source.stream(messages):
            yield chunk.model_copy(
                update={
                    "usage_metadata": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "total_tokens": 110,
                    },
                    "response_metadata": {
                        "finish_reason": "tool_calls" if chunk.tool_calls else "stop"
                    },
                }
            )
