import importlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import event, select

from resolv.models import Order
from resolv.repositories.conversation import ConversationRepository
from resolv.schemas.conversation import ConversationError
from resolv.services.conversation import ConversationService
from tests.conversation_fixtures import MemoryCheckpoints, conversation_factory

__all__ = ["conversation_factory"]


@pytest.fixture
def write_harness(conversation_factory, tool_database):
    module = importlib.import_module("resolv.services.tool_execution")
    service = ConversationService(conversation_factory, MemoryCheckpoints())
    c = service.create()
    run, _ = service.accept(c.conversation_id, "req", "取消 MH002", "agent")
    with service.execution(run.run_id) as execution:
        with conversation_factory.begin() as session:
            action = ConversationRepository(session).stage_action(
                run.run_id,
                execution.epoch,
                1,
                "tc",
                "cancel_order",
                "write",
                {"order_id": "MH002"},
            )
        yield (
            module.WriteExecutionService(conversation_factory, tool_database),
            service,
            action,
            execution,
            tool_database,
        )


def approve(harness):
    write, service, action, execution, business = harness
    run = service.get_run(action.run_id)
    service.decide(action.approval_id, run.conversation_id, True)
    return write, action, execution, business


def test_retry_returns_original_changed_true(write_harness):
    write, action, execution, _ = approve(write_harness)
    first = write.execute(action.action_id, execution)
    second = write.execute(action.action_id, execution)
    assert first.output == {
        "success": True,
        "order_id": "MH002",
        "status": "cancelled",
        "changed": True,
    }
    assert second.output == first.output


def test_receipt_insert_failure_rolls_back_business(write_harness, monkeypatch):
    from resolv.models.tool_execution import ToolExecutionReceipt
    from resolv.repositories.tool_execution import ToolExecutionRepository

    write, action, execution, business = approve(write_harness)

    def fail(*args):
        raise RuntimeError("receipt insert failure")

    monkeypatch.setattr(ToolExecutionRepository, "add", fail)
    with pytest.raises(ConversationError, match="tool_execution_failed"):
        write.execute(action.action_id, execution)
    with business() as session:
        assert (
            session.scalar(select(Order).where(Order.order_no == "MH002")).status
            == "processing"
        )
        assert session.get(ToolExecutionReceipt, action.action_id) is None


def test_business_denial_is_receipted(write_harness):
    from resolv.models.tool_execution import ToolExecutionReceipt

    write, service, original, execution, business = write_harness
    service.decide(
        original.approval_id, service.get_run(original.run_id).conversation_id, False
    )
    with service.session_factory.begin() as session:
        denied = ConversationRepository(session).stage_action(
            original.run_id,
            execution.epoch,
            2,
            "tc2",
            "cancel_order",
            "write",
            {"order_id": "MH001"},
        )
    service.decide(
        denied.approval_id, service.get_run(denied.run_id).conversation_id, True
    )
    result = write.execute(denied.action_id, execution)
    assert result.output["success"] is False
    assert result.output["reason"] == "order_not_cancellable"
    with business() as session:
        assert (
            session.get(ToolExecutionReceipt, denied.action_id).result == result.output
        )


def test_fingerprint_mismatch_rejects(write_harness):
    from resolv.models.tool_execution import ToolExecutionReceipt

    write, action, execution, business = approve(write_harness)
    write.execute(action.action_id, execution)
    with business.begin() as session:
        session.get(ToolExecutionReceipt, action.action_id).arguments_fingerprint = (
            "0" * 64
        )
    with pytest.raises(ConversationError, match="action_conflict"):
        write.execute(action.action_id, execution)


def test_concurrent_same_action_one_commit(write_harness):
    from resolv.models.tool_execution import ToolExecutionReceipt

    write, action, execution, business = approve(write_harness)
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                lambda _: write.execute(action.action_id, execution).output, range(2)
            )
        )
    assert all(result["changed"] is True for result in results)
    with business() as session:
        assert len(session.scalars(select(ToolExecutionReceipt)).all()) == 1


def test_unapproved_action_never_opens_write_transaction(write_harness):
    write, _, action, execution, business = write_harness

    def forbid(conn, cursor, statement, params, context, many):
        pytest.fail("unapproved action accessed business database")

    event.listen(business.kw["bind"], "before_cursor_execute", forbid)
    with pytest.raises(ConversationError, match="approval_required"):
        write.execute(action.action_id, execution)


def test_incremental_upgrade_supports_existing_pre_refund_database(tmp_path):
    from scripts.upgrade_business_db import upgrade
    from sqlalchemy import inspect, text

    from resolv.db.engine import create_db_engine
    from resolv.models import Order, Shipment

    engine = create_db_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    Order.__table__.create(engine)
    Shipment.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO orders (order_no, status, created_at) VALUES ('LEGACY', 'shipped', CURRENT_TIMESTAMP)"))
        original = connection.execute(text("SELECT * FROM orders")).all()
    upgrade(engine)
    upgrade(engine)
    assert set(inspect(engine).get_table_names()) == {"orders", "shipments", "refunds", "tool_execution_receipts"}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT * FROM orders")).all() == original
        assert connection.scalar(text("SELECT COUNT(*) FROM refunds")) == 0
    engine.dispose()


def test_incremental_upgrade_preserves_seeded_rows(tool_database):
    from resolv.models.tool_execution import ToolExecutionReceipt

    module = importlib.import_module("scripts.upgrade_business_db")
    engine = tool_database.kw["bind"]
    ToolExecutionReceipt.__table__.drop(engine)
    module.upgrade(engine)
    module.upgrade(engine)
    with tool_database() as session:
        assert [
            (row.order_no, row.status)
            for row in session.scalars(select(Order).order_by(Order.order_no))
        ] == [("MH001", "shipped"), ("MH002", "processing")]
