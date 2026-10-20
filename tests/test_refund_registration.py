"""The SQLite transaction, rather than a button, decides application uniqueness."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import func, select, text

from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db
from resolv.models import Order, Refund
from resolv.schemas.knowledge_branch import FrozenJSON
from resolv.schemas.refund import RefundFlowError, RefundMoney
from resolv.schemas.refund_submission import RefundRegistrationCommand
from tests.refund_fixtures import NOW, binding


def command(command_id="command-1", *, order_id="A001", namespace="refund-fixture-tenant", reason="不想要了"):
    return RefundRegistrationCommand(
        command_id=command_id, offer_id="offer-" + command_id, request_hash="a" * 64,
        binding=binding(order_id, namespace), subject_ref="verified-subject",
        money=RefundMoney(currency="CNY", minor_units=19900, unit_precision=2), reason=reason,
        decision_hash="b" * 64, evidence_hash="c" * 64, revalidation_hash="d" * 64,
        snapshots=FrozenJSON.from_value({"fixture": "explicit isolated test"}), created_at=NOW,
    )


@pytest.fixture
def business(tmp_path):
    engine = create_db_engine("sqlite:///" + str(tmp_path / "registration.db"))
    init_db(engine)
    yield create_session_factory(engine)
    engine.dispose()


def register(factory, value):
    from resolv.services.refund_registration import RefundRegistrationService

    with factory() as session:
        session.execute(text("BEGIN IMMEDIATE"))
        result = RefundRegistrationService(session).register(value)
        session.commit()
    return result


def test_external_registration_without_local_order(business):
    from resolv.models.refund_application import (
        RefundApplication,
        RefundSubmissionReceipt,
    )

    first = register(business, command())
    assert first.status == "pending" and first.created
    assert register(business, command()) == first
    assert first.money.minor_units_decimal == "19900"
    with business() as session:
        for model, expected in ((RefundApplication, 1), (RefundSubmissionReceipt, 1), (Order, 0), (Refund, 0)):
            assert session.scalar(select(func.count()).select_from(model)) == expected


def test_command_conflict_and_namespace_are_not_merged(business):
    register(business, command())
    with pytest.raises(RefundFlowError, match="refund_request_conflict"):
        register(business, command(reason="替换原因"))
    a = register(business, command("c2", namespace="second-tenant"))
    b = register(business, command("c3", reason="new reason"))
    assert a.application_id != b.application_id
    assert not b.created and b.reason == "不想要了"


def test_new_legacy_race_preserves_original_application(business):
    from resolv.models.refund_application import RefundApplication
    from resolv.repositories.order import OrderRepository
    from resolv.repositories.refund import RefundRepository
    from resolv.services.refund import RefundService

    with business.begin() as session:
        session.add(Order(order_no="MH001", status="delivered"))
    barrier = Barrier(2)

    def run(new):
        barrier.wait(timeout=5)
        if new:
            return register(business, command(namespace="local-business-v1", order_id="MH001"))
        with business() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            result = RefundService(OrderRepository(session), RefundRepository(session)).apply_refund("MH001", "旧工具原因")
            session.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, (True, False)))
    with business() as session:
        app = session.scalar(select(RefundApplication))
        old = session.scalar(select(Refund))
        assert app.reason == old.reason and app.legacy_refund_id == old.id
        assert session.scalar(select(func.count()).select_from(RefundApplication)) == 1
        assert session.scalar(select(func.count()).select_from(Refund)) == 1


def test_receipt_and_compatibility_atomic_rollback(business, monkeypatch):
    from resolv.models.refund_application import (
        RefundApplication,
        RefundSubmissionReceipt,
    )
    from resolv.repositories.refund_application import RefundApplicationRepository

    with business.begin() as session:
        session.add(Order(order_no="MH001", status="shipped"))

    def fail(*args, **kwargs):
        raise RuntimeError("injected receipt failure")

    monkeypatch.setattr(RefundApplicationRepository, "add_receipt", fail)
    with pytest.raises(RuntimeError, match="receipt failure"):
        register(business, command(namespace="local-business-v1", order_id="MH001"))
    with business() as session:
        for model in (RefundApplication, RefundSubmissionReceipt, Refund):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_legacy_and_canonical_mismatch_is_rejected(business):
    from resolv.repositories.order import OrderRepository
    from resolv.repositories.refund import RefundRepository
    from resolv.services.refund import RefundService

    with business.begin() as session:
        order = Order(order_no="MH001", status="shipped")
        order.refund = Refund(status="rejected", reason="old record")
        session.add(order)
    with business() as session, pytest.raises(RefundFlowError, match="refund_store_inconsistent"):
        RefundService(OrderRepository(session), RefundRepository(session)).query_refund_status("MH001")
