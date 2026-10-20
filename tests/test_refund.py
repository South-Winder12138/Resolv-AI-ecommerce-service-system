import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from resolv import agent_loop
from resolv.models import Order, Refund
from resolv.repositories.order import OrderRepository
from resolv.repositories.refund import RefundRepository
from resolv.services.refund import RefundService
from resolv.tools.executor import execute_tool_call
from resolv.tools.refund import apply_refund, query_refund_status
from resolv.tools.registry import (
    ToolEffect,
    WriteToolApprovalRequired,
    get_registered_tool,
)
from tests.stream_fakes import as_chunk

pytestmark = pytest.mark.usefixtures("tool_database")


def refund_for(factory, order_no="MH001"):
    with factory() as session:
        order = session.scalar(select(Order).where(Order.order_no == order_no))
        assert order is not None
        refund = session.scalar(select(Refund).where(Refund.order_id == order.id))
        if refund is None:
            return None
        return refund.status, refund.reason


def test_refund_table_fk_unique_and_relationship(tool_database):
    engine = tool_database.kw["bind"]
    db = inspect(engine)
    assert "refunds" in db.get_table_names()
    assert db.get_foreign_keys("refunds")[0]["referred_table"] == "orders"
    assert any(
        "order_id" in item["column_names"]
        for item in db.get_unique_constraints("refunds")
    )

    with tool_database.begin() as session:
        order = session.scalar(select(Order).where(Order.order_no == "MH001"))
        order.refund = Refund(status="pending", reason="商品损坏")
    with tool_database() as session:
        order = session.scalar(select(Order).where(Order.order_no == "MH001"))
        assert order.refund is not None
        assert order.refund.order is order
        assert order.refund.created_at is not None
        assert order.refund.updated_at is not None

    with pytest.raises(IntegrityError), tool_database.begin() as session:
        order = session.scalar(select(Order).where(Order.order_no == "MH001"))
        session.add(Refund(order_id=order.id, status="pending", reason="重复申请"))


def test_refund_order_foreign_key_is_enforced(tool_database):
    with pytest.raises(IntegrityError), tool_database.begin() as session:
        session.add(Refund(order_id=999, status="pending", reason="不存在的订单"))


def test_refund_repository_returns_orm_or_none_and_only_adds(tool_database):
    with tool_database.begin() as session:
        order = OrderRepository(session).get_by_order_no("MH001")
        repository = RefundRepository(session)
        assert repository.get_by_order_id(order.id) is None
        refund = Refund(order_id=order.id, status="pending", reason="商品损坏")
        repository.add(refund)
        session.flush()
        assert repository.get_by_order_id(order.id) is refund
    assert refund_for(tool_database) == ("pending", "商品损坏")


def test_refund_read_distinguishes_missing_order_and_missing_refund(tool_database):
    with tool_database() as session:
        service = RefundService(OrderRepository(session), RefundRepository(session))
        assert service.query_refund_status("MH999") == {
            "found": False,
            "order_id": "MH999",
            "reason": "order_not_found",
        }
        assert service.query_refund_status("MH001") == {
            "found": False,
            "order_id": "MH001",
            "reason": "refund_not_found",
        }


@pytest.mark.parametrize("status", ["pending", "approved", "rejected", "completed"])
def test_refund_read_returns_existing_status(tool_database, status):
    with tool_database.begin() as session:
        order = OrderRepository(session).get_by_order_no("MH001")
        RefundRepository(session).add(
            Refund(order_id=order.id, status=status, reason="商品损坏")
        )
        session.flush()
        # This fixture represents a migrated historical application. Runtime
        # reads must never silently backfill an inconsistent legacy-only row.
        from resolv.services.refund_registration import RefundRegistrationService
        RefundRegistrationService(session).backfill_legacy()
    assert query_refund_status.invoke({"order_id": "MH001"}) == {
        "found": True,
        "order_id": "MH001",
        "status": status,
        "reason": "商品损坏",
    }


def test_apply_refund_commits_pending_and_is_idempotent(tool_database):
    first = apply_refund.invoke({"order_id": "MH001", "reason": "商品损坏"})
    second = apply_refund.invoke({"order_id": "MH001", "reason": "不想要了"})

    assert first == {
        "success": True,
        "order_id": "MH001",
        "status": "pending",
        "changed": True,
    }
    assert second == {
        "success": True,
        "order_id": "MH001",
        "status": "pending",
        "changed": False,
    }
    assert refund_for(tool_database) == ("pending", "商品损坏")
    with tool_database() as session:
        assert len(session.scalars(select(Refund)).all()) == 1


@pytest.mark.parametrize("status", ["processing", "cancelled", "on_hold"])
def test_apply_refund_rejects_non_refundable_order(tool_database, status):
    with tool_database.begin() as session:
        order = OrderRepository(session).get_by_order_no("MH002")
        order.status = status
    assert apply_refund.invoke({"order_id": "MH002", "reason": "商品损坏"}) == {
        "success": False,
        "order_id": "MH002",
        "reason": "refund_not_available",
    }
    assert refund_for(tool_database, "MH002") is None


def test_apply_refund_accepts_delivered_order(tool_database):
    with tool_database.begin() as session:
        order = OrderRepository(session).get_by_order_no("MH002")
        order.status = "delivered"
    result = apply_refund.invoke({"order_id": "MH002", "reason": "商品损坏"})
    assert result["changed"] is True
    assert refund_for(tool_database, "MH002") == ("pending", "商品损坏")


def test_apply_refund_missing_order_is_business_result(tool_database):
    assert apply_refund.invoke({"order_id": "MH999", "reason": "商品损坏"}) == {
        "success": False,
        "order_id": "MH999",
        "reason": "order_not_found",
    }


def test_refund_insert_rolls_back_after_failure(tool_database, monkeypatch):
    original = RefundRepository.add

    def fail_after_insert(self, refund):
        original(self, refund)
        self.session.flush()
        raise RuntimeError("failure after INSERT")

    monkeypatch.setattr(RefundRepository, "add", fail_after_insert)
    with pytest.raises(RuntimeError, match="failure after INSERT"):
        apply_refund.invoke({"order_id": "MH001", "reason": "商品损坏"})
    assert refund_for(tool_database) is None


def test_refund_tool_contract_and_read_write_boundary(tool_database):
    assert get_registered_tool("query_refund_status").effect is ToolEffect.READ
    assert get_registered_tool("apply_refund").effect is ToolEffect.WRITE
    assert set(query_refund_status.args) == {"order_id"}
    assert set(apply_refund.args) == {"order_id", "reason"}
    read = execute_tool_call(
        {"name": "query_refund_status", "args": {"order_id": "MH001"}, "id": "read-1"}
    )
    assert read.output["reason"] == "refund_not_found"
    with pytest.raises(WriteToolApprovalRequired):
        execute_tool_call(
            {
                "name": "apply_refund",
                "args": {"order_id": "MH001", "reason": "商品损坏"},
                "id": "write-1",
            }
        )
    assert refund_for(tool_database) is None


def test_agent_refund_approval_reuses_frozen_action(
    monkeypatch, tool_database, client, api_runtime
):
    class FakeModel:
        def __init__(self):
            self.seen = []

        def bind_tools(self, tools):
            assert apply_refund in tools
            assert query_refund_status in tools
            return self

        def bind(self, **kwargs):
            return self

        def stream(self, messages):
            self.seen.append(list(messages))
            if len(self.seen) == 1:
                yield as_chunk(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "apply_refund",
                                "args": {"order_id": "MH001", "reason": "商品损坏"},
                                "id": "refund-call-1",
                            }
                        ],
                    )
                )
            else:
                yield as_chunk(AIMessage(content="退款申请已提交。"))

    model = FakeModel()
    monkeypatch.setattr(agent_loop, "get_chat_model", lambda: model)
    api_runtime.model = model
    session_id = client.post("/conversations").json()["conversation_id"]
    first = client.post(
        "/agent/chat",
        json={
            "session_id": session_id,
            "request_id": "refund-request",
            "question": "申请退款，订单 MH001，商品损坏",
        },
    ).json()
    assert first["status"] == "approval_required"
    assert first["action"] == {
        "tool": "apply_refund",
        "arguments": {"order_id": "MH001", "reason": "商品损坏"},
    }
    assert "商品损坏" in first["message"]
    assert refund_for(tool_database) is None

    tampered = client.post(
        f"/agent/approvals/{first['approval_id']}",
        json={
            "session_id": session_id,
            "approved": True,
            "arguments": {"order_id": "MH001", "reason": "不想要了"},
        },
    )
    assert tampered.status_code == 422
    assert refund_for(tool_database) is None

    approved = client.post(
        f"/agent/approvals/{first['approval_id']}",
        json={"session_id": session_id, "approved": True},
    )
    assert approved.json()["status"] == "completed"
    assert approved.json()["answer"] == "退款申请已提交。"
    assert refund_for(tool_database) == ("pending", "商品损坏")
    observation = model.seen[1][-1]
    assert isinstance(observation, ToolMessage)
    assert observation.tool_call_id == "refund-call-1"
    assert json.loads(observation.content)["changed"] is True


def test_agent_refund_rejection_does_not_create_refund(monkeypatch, tool_database):
    class FakeModel:
        def bind_tools(self, _tools):
            return self

        def stream(self, messages):
            if not any(isinstance(message, ToolMessage) for message in messages):
                yield as_chunk(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "apply_refund",
                                "args": {"order_id": "MH001", "reason": "商品损坏"},
                                "id": "refund-call-2",
                            }
                        ],
                    )
                )
            else:
                assert json.loads(messages[-1].content)["reason"] == "user_rejected"
                yield as_chunk(AIMessage(content="好的，不申请退款。"))

    monkeypatch.setattr(agent_loop, "get_chat_model", FakeModel)
    paused = agent_loop.run_agent("申请退款")
    assert refund_for(tool_database) is None
    assert agent_loop.resume_agent(
        paused.state, approved=False
    ) == agent_loop.AgentCompleted("好的，不申请退款。")
    assert refund_for(tool_database) is None
