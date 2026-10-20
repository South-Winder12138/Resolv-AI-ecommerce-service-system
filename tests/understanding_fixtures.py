"""Offline builders; these do not infer model semantics."""

from datetime import UTC, datetime

import pytest

from resolv.schemas.understanding import (
    ClassificationResult,
    ContextSnapshot,
    ResolutionResult,
    UnderstandingRequestRecord,
)


def input_ref(text, start=0, source_id="current"):
    return {
        "kind": "input",
        "source_id": source_id,
        "span": {"start": start, "end": start + len(text)},
        "json_path": [],
        "quote": text,
    }


def make_snapshot(text, **overrides):
    return ContextSnapshot(
        **(
            {
                "conversation_id": "c",
                "owner_id": "local",
                "original_text": text,
                "linked_inputs": {},
                "cutoff_seq": 0,
                "summary_version": 0,
                "summary": None,
                "model_sources": [],
                "source_fingerprints": [],
                "omitted_ranges": [],
                "pending_requests": [],
                "parent": None,
                "config": {
                    "model_name": "fake",
                    "api_address_fingerprint": "f" * 64,
                    "tokenizer_profile": "o200k_base",
                    "model_context_window_tokens": 256000,
                    "app_max_input_tokens": 12000,
                    "provider_input_overhead_tokens": 5000,
                    "reserved_output_tokens": 2000,
                    "safety_margin_tokens": 1000,
                    "recent_history_target_tokens": 2000,
                    "resolution_prompt": "resolve",
                    "intent_prompt": "classify",
                    "resolution_version": "resolution-v1",
                    "intent_version": "intent-v1",
                    "resolution_schema": {},
                    "intent_schema": {},
                    "mapping_version": "intent-branches-v1",
                    "state_schema_version": 1,
                    "snapshot_version": 1,
                },
                "snapshot_hash": "",
            }
            | overrides
        )
    )


def make_resolution(**overrides):
    return ResolutionResult(
        **(
            {
                "resolved_text": "查订单",
                "bindings": [],
                "unresolved_references": [],
                "continuation_selection": None,
            }
            | overrides
        )
    )


def make_classification(texts, intents, *, original_text=None, **overrides):
    original_text = original_text or "，".join(texts)
    requests = [
        {
            "text": text,
            "intent": intent,
            "evidence_refs": [input_ref(text, original_text.index(text))],
            "binding_indexes": [],
            "explicit_priority": None,
            "priority_evidence_refs": [],
            "ambiguities": [],
        }
        for text, intent in zip(texts, intents, strict=True)
    ]
    return ClassificationResult(**({"requests": requests} | overrides))


def make_pending_task(task_id, intent, **overrides):
    now = datetime.now(UTC)
    return UnderstandingRequestRecord(
        **(
            {
                "task_id": task_id,
                "understanding_id": "old",
                "conversation_id": "c",
                "ordinal": 1,
                "role": "secondary",
                "text": "退款",
                "intent": intent,
                "branch": "refund_flow",
                "evidence_refs": [],
                "bindings": [],
                "ambiguities": [],
                "unresolved_references": [],
                "queue_status": "pending",
                "selected_by_understanding_id": None,
                "created_at": now,
            }
            | overrides
        )
    )


@pytest.fixture
def understanding_factory(tmp_path):
    from resolv.db.engine import create_db_engine, create_session_factory
    from resolv.models.conversation import Conversation

    try:
        __import__("resolv.models.understanding")
    except ModuleNotFoundError:
        pass  # RED still exercises the missing repository from the test body.
    engine = create_db_engine(f"sqlite:///{tmp_path / 'understanding.db'}")
    Conversation.metadata.create_all(engine)
    factory = create_session_factory(engine)
    with factory.begin() as session:
        session.add(Conversation(conversation_id="c", thread_id="c", title="test"))
    yield factory
    engine.dispose()


def make_audit(stage="resolution", **overrides):
    from resolv.schemas.understanding import StageAudit

    now = datetime.now(UTC)
    return StageAudit(
        **(
            {
                "stage": stage,
                "invocation_id": "call-" + stage,
                "attempt": 1,
                "input_estimate": 6000,
                "input_tokens": None,
                "output_tokens": None,
                "started_at": now,
                "finished_at": now,
                "outcome": "succeeded",
                "verified_refs": [],
            }
            | overrides
        )
    )


class FakeJsonModel:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls, self.bound = [], []

    def bind(self, **kwargs):
        self.bound.append(kwargs)
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def fail_once(point):
    fired = False

    def fail(actual):
        nonlocal fired
        if actual == point and not fired:
            fired = True
            raise RuntimeError("injected technical interruption")

    return fail


def seed_completed(factory, conversation_id="c"):
    from resolv.models.conversation import (
        Conversation,
        ConversationMessage,
        ConversationRun,
    )

    with factory.begin() as session:
        c = session.get(Conversation, conversation_id)
        if c is None:
            c = Conversation(
                conversation_id=conversation_id, thread_id=conversation_id, title="test"
            )
            session.add(c)
            session.flush()
        run_id, q, a = (
            conversation_id + "-run",
            conversation_id + "-q",
            conversation_id + "-a",
        )
        session.add(
            ConversationRun(
                run_id=run_id,
                conversation_id=conversation_id,
                request_id="seed",
                request_fingerprint="f" * 64,
                status="completed",
                question_message_id=q,
                answer_message_id=a,
            )
        )
        session.flush()
        session.add_all(
            [
                ConversationMessage(
                    message_id=q,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    seq=1,
                    role="user",
                    text="订单 MH001",
                ),
                ConversationMessage(
                    message_id=a,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    seq=2,
                    role="assistant",
                    text="已记录订单 MH001 供后续理解。",
                ),
            ]
        )
        c.next_message_seq = 3


def build_test_service(factory, checkpoints=None):
    from langchain_core.messages import AIMessage

    from resolv.core.config import ContextSettings
    from resolv.core.understanding_model import JsonStageModel
    from resolv.core.understanding_prompts import (
        INTENT_PROMPT_V1,
        RESOLUTION_PROMPT_V1,
    )
    from resolv.services.intent_classification import IntentClassifier
    from resolv.services.reference_resolution import ReferenceResolver
    from resolv.services.understanding import UnderstandingService
    from resolv.services.understanding_context import UnderstandingContext
    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.test_understanding_context import Counter

    context = UnderstandingContext(
        ContextSettings(
            _env_file=None,
            model_context_window_tokens=256000,
            tokenizer_profile="o200k_base",
            provider_input_overhead_tokens=5000,
        ),
        Counter(),
        model_name="fake",
        api_address_fingerprint="f" * 64,
        resolution_prompt=RESOLUTION_PROMPT_V1,
        intent_prompt=INTENT_PROMPT_V1,
    )
    fake = FakeJsonModel(
        [
            AIMessage(content=make_resolution().model_dump_json()),
            AIMessage(
                content=make_classification(["查订单"], ["order"]).model_dump_json()
            ),
        ]
    )
    model = JsonStageModel(fake, context.counter)
    return UnderstandingService(
        factory,
        checkpoints or MemoryCheckpoints(),
        context,
        ReferenceResolver(model, context),
        IntentClassifier(model, context),
    )


@pytest.fixture
def understanding_service(understanding_factory, monkeypatch):
    seed_completed(understanding_factory)
    # Real business entrypoints must never be reached by this independent service.
    import resolv.services.tool_execution as business

    monkeypatch.setattr(
        business.WriteExecutionService,
        "execute",
        lambda *_args, **_kwargs: pytest.fail("business tool called"),
        raising=False,
    )
    try:
        return build_test_service(understanding_factory)
    except ModuleNotFoundError:
        return None  # Tests report missing service as normal RED failures.
