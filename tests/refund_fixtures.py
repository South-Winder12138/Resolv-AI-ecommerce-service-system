"""Task-isolated facts and identities; never selected by production runtime."""

from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from resolv.schemas.refund import OrderBinding, RefundObservation
from resolv.services.hybrid_fingerprints import canonical_hash

NOW = datetime(2026, 10, 16, 4, tzinfo=UTC)


def binding(order_id="A001", namespace="refund-fixture-tenant"):
    return OrderBinding.create(
        order_source=namespace,
        order_id=order_id,
        understanding_id="u",
        primary_task_id="t",
        primary_hash="a" * 64,
        source_refs=(
            {
                "kind": "input",
                "source_id": "current",
                "span": {"start": 0, "end": len(order_id)},
                "json_path": [],
                "quote": order_id,
            },
        ),
    )


def observation(
    tool="query_refund_order",
    *,
    bound=None,
    fields=None,
    call_id="call-1",
    availability="available",
):
    bound = bound or binding()
    return RefundObservation.create(
        observation_id=call_id,
        tool_call_id=call_id,
        tool_name=tool,
        binding_hash=bound.binding_hash,
        source=bound.order_source,
        observed_at=NOW,
        provider_version=None,
        arguments_hash=canonical_hash({}),
        availability=availability,
        fields={
            k: {
                "availability": "available",
                "value": v,
                "applicability": "applicable",
                "original_time": None,
            }
            for k, v in (fields or {}).items()
        },
    )


@contextmanager
def refund_case(factory, *, text="订单A001申请整单退款，商品未激活"):
    from types import SimpleNamespace

    from resolv.models.conversation import ConversationRun
    from resolv.schemas.knowledge_branch import FrozenJSON
    from resolv.schemas.refund import AdapterDescriptor
    from resolv.schemas.refund_flow import RefundModelProfile, RefundProfile
    from resolv.services.conversation_routing import load_primary, run_understanding
    from tests.branch_fixtures import make_routed_service
    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.understanding_fixtures import make_classification, make_resolution

    service = make_routed_service(
        factory,
        MemoryCheckpoints(),
        stage_outputs=[
            make_resolution(resolved_text=text),
            make_classification([text], ["refund_return"]),
        ],
    )
    c = service.create()
    run, _ = service.accept(c.conversation_id, "refund-fixture", text, "agent")

    def model(purpose):
        return RefundModelProfile(
            purpose=purpose,
            model_name="offline",
            model_api_mode="chat_completions",
            api_address_fingerprint="e" * 64,
            system_prompt="只读退款判断",
            prompt_version="fixture",
            tool_schema_fingerprint=canonical_hash([]),
        )

    profile = RefundProfile(
        query_model=model("refund_query"),
        decision_model=model("refund_decision"),
        summary_profile=FrozenJSON.from_value({}),
        adapter=AdapterDescriptor(
            namespace="refund-fixture-tenant",
            implementation_version="1",
            authorization_version="1",
        ),
    )
    with service.execution(run.run_id) as execution:
        run = service.get_run(run.run_id)
        view = run_understanding(run, service.understanding)
        u = service.understanding.record(c.conversation_id, run.understanding_id)
        primary = load_primary(run, view, snapshot_hash=u.snapshot.snapshot_hash)
        with factory.begin() as session:
            row = session.get(ConversationRun, run.run_id)
            row.orchestration_version, row.state_schema_version = "routed-v4", 4
            row.refund_config = profile.model_dump(mode="json")
        yield SimpleNamespace(
            factory=factory,
            service=service,
            run=run,
            execution=execution,
            primary=primary,
            profile=profile,
        )


@pytest.fixture
def refund_owned(tmp_path):
    from resolv.db.engine import create_db_engine, create_session_factory
    from resolv.models.conversation import Conversation
    from resolv.models.refund_flow import ConversationRefundRun  # noqa: F401

    engine = create_db_engine(f"sqlite:///{tmp_path / 'refund-conversation.db'}")
    Conversation.metadata.create_all(engine)
    try:
        with refund_case(create_session_factory(engine)) as h:
            yield h
    finally:
        engine.dispose()
