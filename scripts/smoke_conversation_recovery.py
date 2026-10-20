"""Explicit fault recovery smoke: real MySQL, script-owned SQLite, fake inference."""

import argparse
import json
import os
import re
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessageChunk
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.dml import Update

from resolv.core.config import ContextSettings, ConversationDatabaseSettings
from resolv.db.checkpoints import CheckpointError, _open_connection, checkpoint_session
from resolv.db.conversation import conversation_engine
from resolv.db.conversation_base import utc_now
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.models import Order
from resolv.models.conversation import ConversationRun
from resolv.models.tool_execution import ToolExecutionReceipt
from resolv.repositories.conversation import ConversationRepository
from resolv.repositories.tool_execution import ToolExecutionRepository
from resolv.schemas.conversation import ConversationError
from resolv.services.context import ContextBuilder
from resolv.services.conversation import ConversationService
from resolv.services.tool_execution import WriteExecutionService


class FakeCounter:
    """Smoke accounting only; production tokenizer is verified separately."""

    def count_text(self, text):
        return len(text)

    def count(self, messages, tool_schemas):
        return sum(len(str(m.content)) for m in messages) + len(
            json.dumps(tool_schemas)
        )


class FakeModel:
    def __init__(self, order_id):
        self.order_id, self.calls = order_id, 0

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    def stream(self, messages):
        self.calls += 1
        if any(m.type == "tool" for m in messages):
            yield AIMessageChunk(content="fake final answer")
        else:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "id": "smoke-refund",
                        "name": "apply_refund",
                        "args": {
                            "order_id": self.order_id,
                            "reason": "isolated recovery smoke",
                        },
                    }
                ],
            )


@contextmanager
def owned_business_db(path, database):
    marker = path.with_suffix(path.suffix + ".resolv-smoke.json")
    if path.exists():
        if (
            not marker.is_file()
            or json.loads(marker.read_text()).get("database") != database
        ):
            raise ValueError("existing business DB is not owned by this smoke")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine(f"sqlite:///{path}")
    factory = create_session_factory(engine)
    try:
        if not marker.exists():
            init_db(engine)
            with factory.begin() as session:
                seed_data(session)
            marker.write_text(
                json.dumps(
                    {"database": database, "owner": "resolv-conversation-smoke"}
                )
            )
        yield factory
    finally:
        engine.dispose()


def make_service(settings, factory, business, order_id, checkpoint_fault=None):
    failed = False

    @contextmanager
    def checkpoints(*, conversation_id):
        nonlocal failed
        with checkpoint_session(settings, conversation_id=conversation_id) as session:
            if checkpoint_fault:
                original = session.saver.put

                def put(config, checkpoint, metadata, new_versions):
                    nonlocal failed
                    values = checkpoint["channel_values"]
                    if not failed and checkpoint_fault(values):
                        failed = True
                        raise RuntimeError("injected checkpoint fault")
                    return original(config, checkpoint, metadata, new_versions)

                session.saver.put = put
            yield session

    model = FakeModel(order_id)
    service = ConversationService(
        factory,
        checkpoints,
        model=model,
        context_builder=ContextBuilder(
            ContextSettings(
                _env_file=None,
                model_context_window_tokens=16000,
                tokenizer_profile="cl100k_base",
            ),
            FakeCounter(),
        ),
        write_service=WriteExecutionService(factory, business),
    )
    return service, model


def verify_exit_fencing(settings, factory, business):
    """Force takeover between stale metadata and exit writes on real MySQL."""
    class AnswerModel(FakeModel):
        def stream(self, messages):
            self.calls += 1
            yield AIMessageChunk(content="fake completed answer")

    service, _ = make_service(settings, factory, business, "unused")
    service.model = AnswerModel("unused")
    conversation = service.create()
    run, _ = service.accept(conversation.conversation_id, "old-error", "fake answer", "chat")
    with service.execution(run.run_id) as old:
        pass
    injected = False

    class ErrorSession(Session):
        def get(self, entity, ident, **kwargs):
            nonlocal injected
            row = super().get(entity, ident, **kwargs)
            if entity is ConversationRun and not injected:
                injected = True
                service.session_factory = factory
                assert list(service.resume(conversation.conversation_id, run.run_id))[-1].type == "done"
            return row

    service.session_factory = sessionmaker(bind=factory.kw["bind"], class_=ErrorSession)
    try:
        service._record_error(run.run_id, old, ConversationError("unsupported_approval_batch"))
    finally:
        service.session_factory = factory
    assert injected and service.get_run(run.run_id).status == "completed"
    assert service.detail(conversation.conversation_id, None, 50).current_run is None
    print(json.dumps({"scenario":"late_error_after_real_takeover", "result":"PASS", "run_id":run.run_id, "action_id":None, "receipt_count":0, "status":"completed"}), flush=True)

    conversation = service.create()
    run, _ = service.accept(conversation.conversation_id, "lost-cleanup", "fake cleanup", "chat")
    newer_context = newer = None
    injected = False

    class CleanupSession(Session):
        def execute(self, statement, *args, **kwargs):
            nonlocal injected, newer_context, newer
            if isinstance(statement, Update) and statement.table.name == "conversation_runs" and not injected:
                injected = True
                service.session_factory = factory
                newer_context = service.execution(run.run_id)
                newer = newer_context.__enter__()
            return super().execute(statement, *args, **kwargs)

    try:
        with service.execution(run.run_id) as old:
            with old.checkpoint_session.serialized(), old.checkpoint_session._connection.cursor() as cursor:
                cursor.execute("SELECT CONNECTION_ID()")
                identity = cursor.fetchone()[0]
            killer = _open_connection(settings)
            try:
                with killer.cursor() as cursor:
                    cursor.execute("KILL CONNECTION %s", (identity,))
            finally:
                killer.close()
            try:
                old.checkpoint_session.assert_owned()
                raise AssertionError("killed lock still owned")
            except CheckpointError:
                old._failed.set()
            service.session_factory = sessionmaker(bind=factory.kw["bind"], class_=CleanupSession)
        saved = service.get_run(run.run_id)
        assert injected and saved.execution_epoch == newer.epoch
        assert saved.execution_owner_id == newer.owner_id and saved.status == "running"
        newer.assert_owned()
        print(json.dumps({"scenario":"lost_lock_cleanup_after_real_takeover", "result":"PASS", "run_id":run.run_id, "action_id":None, "receipt_count":0, "status":"running", "epoch":saved.execution_epoch}), flush=True)
    finally:
        service.session_factory = factory
        if newer_context is not None:
            newer_context.__exit__(None, None, None)


def run_smoke(settings, business_db):
    database = make_url(settings.conversation_database_url.get_secret_value()).database
    with (
        owned_business_db(business_db, database) as business,
        conversation_engine(settings) as engine,
    ):
        factory = create_session_factory(engine)
        verify_exit_fencing(settings, factory, business)
        # Use separate raw MySQL connections, not a fake/two-thread lock.
        service, model = make_service(settings, factory, business, "unused")
        conversation = service.create()
        run, _ = service.accept(conversation.conversation_id, "live-lock", "fake lock check", "agent")
        with service.execution(run.run_id) as old:
            with factory.begin() as session:
                session.get(ConversationRun, run.run_id).lease_expires_at = utc_now() - timedelta(seconds=1)
            assert service.detail(conversation.conversation_id, None, 50).current_run.status == "running"
            try:
                with service.execution(run.run_id):
                    raise AssertionError("expired lease bypassed a live named lock")
            except ConversationError as error:
                assert error.code == "conversation_busy"
        with service.execution(run.run_id):
            try:
                with factory.begin() as session:
                    ConversationRepository(session).save_knowledge(run.run_id, old.epoch, [])
                raise AssertionError("old epoch published after takeover")
            except ConversationError as error:
                assert error.code == "conversation_busy"
        assert model.calls == 0
        print(json.dumps({"scenario":"live_lock_expired_lease_old_epoch", "result":"PASS", "run_id":run.run_id, "action_id":None, "receipt_count":0, "status":"interrupted", "model_calls":0}), flush=True)
        scenarios = [
            "checkpoint_prepare",
            "checkpoint_knowledge",
            "checkpoint_action",
            "sqlite_before_commit",
            "mysql_after_sqlite_commit",
            "checkpoint_answer",
            "lost_done",
        ]
        for scenario in scenarios:
            order_id = "SMOKE" + uuid4().hex[:20]
            with business.begin() as session:
                session.add(Order(order_no=order_id, status="shipped"))
            phases = {
                "checkpoint_prepare": "retrieve_knowledge",
                "checkpoint_knowledge": "call_model",
                "checkpoint_action": "await_approval",
                "checkpoint_answer": "end",
            }
            fault = (
                (lambda values, target=phases[scenario]: values.get("phase") == target)
                if scenario in phases
                else None
            )
            service, model = make_service(settings, factory, business, order_id, fault)
            conversation = service.create()
            run, _ = service.accept(
                conversation.conversation_id, "smoke-request", "fake refund", "agent"
            )
            events = list(service.run(run.run_id))
            if scenario in {
                "checkpoint_prepare",
                "checkpoint_knowledge",
                "checkpoint_action",
            }:
                assert events[-1].type == "error" and not any(
                    e.type in {"done", "approval_required"} for e in events
                )
                events = list(service.resume(conversation.conversation_id, run.run_id))
            assert events[-1].type == "done"
            detail = service.detail(conversation.conversation_id, None, 50)
            assert detail.current_run.status == "waiting_approval"
            action = detail.actions[0]
            service.decide(action.approval_id, conversation.conversation_id, True)
            target = None
            if scenario == "sqlite_before_commit":
                target = patch.object(
                    ToolExecutionRepository,
                    "add",
                    side_effect=RuntimeError("before SQLite commit"),
                )
            elif scenario == "mysql_after_sqlite_commit":
                target = patch.object(
                    ConversationRepository,
                    "save_action_result",
                    side_effect=RuntimeError("after SQLite commit"),
                )
            if target:
                with target:
                    events = list(
                        service.resume(conversation.conversation_id, run.run_id)
                    )
            else:
                events = list(service.resume(conversation.conversation_id, run.run_id))
            if scenario in {
                "sqlite_before_commit",
                "mysql_after_sqlite_commit",
                "checkpoint_answer",
            }:
                assert events[-1].type == "error" and not any(
                    e.type == "done" for e in events
                )
                if scenario == "sqlite_before_commit":
                    with business() as session:
                        assert (
                            session.get(ToolExecutionReceipt, action.action_id) is None
                        )
                if scenario == "mysql_after_sqlite_commit":
                    with business() as session:
                        assert (
                            session.get(ToolExecutionReceipt, action.action_id).result[
                                "changed"
                            ]
                            is True
                        )
                events = list(service.resume(conversation.conversation_id, run.run_id))
            assert events[-1].type == "done"
            if scenario == "lost_done":
                before = model.calls
                events = list(service.resume(conversation.conversation_id, run.run_id))
                assert model.calls == before
            with business() as session:
                receipts = session.scalars(
                    select(ToolExecutionReceipt).where(
                        ToolExecutionReceipt.run_id == run.run_id
                    )
                ).all()
                assert len(receipts) == 1 and receipts[0].result["changed"] is True
                original = receipts[0].result
            with factory() as session:
                saved = ConversationRepository(session).action(action.action_id)
                assert saved.result == original
            assert (
                service.get_run(run.run_id).status == "completed"
                and service.detail(conversation.conversation_id, None, 50).current_run
                is None
            )
            print(
                json.dumps(
                    {
                        "scenario": scenario,
                        "result": "PASS",
                        "run_id": run.run_id,
                        "action_id": action.action_id,
                        "receipt": original,
                        "status": "completed",
                        "model_calls": model.calls,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    print("Conversation recovery smoke PASSED; no formal business rows touched.")


def process_smoke(settings, business_db, phase, manifest, decision, crash):
    """Persist safe IDs so each invocation can run in a fresh OS process."""
    database = make_url(settings.conversation_database_url.get_secret_value()).database
    with (
        owned_business_db(business_db, database) as business,
        conversation_engine(settings) as engine,
    ):
        factory = create_session_factory(engine)
        if phase == "pause":
            if manifest.exists():
                raise ValueError("refuse to replace an existing run manifest")
            order_id = "SMOKE" + uuid4().hex[:20]
            with business.begin() as session:
                session.add(Order(order_no=order_id, status="shipped"))
            service, _ = make_service(settings, factory, business, order_id)
            conversation = service.create()
            run, _ = service.accept(conversation.conversation_id, "process-smoke", "fake refund", "agent")
            events = list(service.run(run.run_id))
            detail = service.detail(conversation.conversation_id, None, 50)
            assert events[-1].type == "done" and detail.current_run.status == "waiting_approval"
            action = detail.actions[0]
            record = {
                "database": database, "business_db": str(business_db),
                "conversation_id": conversation.conversation_id,
                "run_id": run.run_id, "action_id": action.action_id,
                "approval_id": action.approval_id, "order_id": order_id,
                "decision": decision,
            }
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(record))
            print(json.dumps({"phase": phase, "result": "PASS", "run_id": run.run_id, "action_id": action.action_id, "status": "waiting_approval", "receipt_count": 0}), flush=True)
            return
        record = json.loads(manifest.read_text())
        if record["database"] != database or record["business_db"] != str(business_db):
            raise ValueError("manifest resource mismatch")
        service, model = make_service(settings, factory, business, record["order_id"])
        approved = record["decision"] == "approve"
        if phase == "resume":
            service.decide(record["approval_id"], record["conversation_id"], approved)
        if crash:
            if phase != "resume" or not approved:
                raise ValueError("crash requires an approved resume")

            def after_commit(*args, **kwargs):
                print(json.dumps({"phase": "sqlite_committed_process_exit", "run_id": record["run_id"], "action_id": record["action_id"], "exit_code": 23}), flush=True)
                os._exit(23)

            with patch.object(ConversationRepository, "save_action_result", after_commit):
                list(service.resume(record["conversation_id"], record["run_id"]))
            raise AssertionError("expected process exit did not occur")
        events = list(service.resume(record["conversation_id"], record["run_id"]))
        assert events[-1].type == "done"
        with business() as session:
            receipts = session.scalars(select(ToolExecutionReceipt).where(ToolExecutionReceipt.run_id == record["run_id"])).all()
            assert len(receipts) == int(approved)
            original = receipts[0].result if approved else None
            if approved:
                assert original["changed"] is True
        with factory() as session:
            action = ConversationRepository(session).action(record["action_id"])
            assert action.result == original if approved else action.result["reason"] == "user_rejected"
            saved = action.result
        detail = service.detail(record["conversation_id"], None, 50)
        assert detail.current_run is None and service.get_run(record["run_id"]).status == "completed"
        assert [m.role for m in detail.messages] == ["user", "assistant"]
        print(json.dumps({"phase": phase, "result": "PASS", "run_id": record["run_id"], "action_id": record["action_id"], "receipt": saved, "receipt_count": len(receipts), "status": "completed", "model_calls": model.calls, "message_count": len(detail.messages)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-env", required=True)
    parser.add_argument("--business-db", required=True, type=Path)
    parser.add_argument("--phase", choices=["faults", "pause", "resume", "verify"], default="faults")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--decision", choices=["approve", "reject"], default="approve")
    parser.add_argument("--crash-after-commit", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = ConversationDatabaseSettings(
            _env_file=None,
            conversation_database_url=os.environ.get(args.database_env, ""),
        )
        database = make_url(
            settings.conversation_database_url.get_secret_value()
        ).database
        path = args.business_db.resolve()
        formal_directory = Path(__file__).resolve().parents[1] / "data"
        if not re.fullmatch(
            r"resolv_conversations_smoke_[A-Za-z0-9_]+", database or ""
        ) or path.is_relative_to(formal_directory):
            raise ValueError("isolated smoke resources required")
        marker = path.with_suffix(path.suffix + ".resolv-smoke.json")
        if path.exists() and not marker.is_file():
            raise ValueError("unowned existing SQLite")
        if args.phase == "faults":
            if args.manifest or args.crash_after_commit:
                raise ValueError("process flags require a process phase")
            run_smoke(settings, path)
        elif args.manifest is None:
            raise ValueError("process phase requires a manifest")
        else:
            process_smoke(settings, path, args.phase, args.manifest, args.decision, args.crash_after_commit)
    except Exception as exc:  # noqa: BLE001 — safe opt-in boundary; no credentials or exception messages
        raise SystemExit(
            f"Conversation recovery smoke FAILED ({type(exc).__name__}); no later acceptance claim is permitted."
        ) from None


if __name__ == "__main__":
    main()
