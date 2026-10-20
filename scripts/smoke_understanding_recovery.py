"""Opt-in isolated MySQL Gate. Fake inference, no business resources or actions."""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from importlib.metadata import version
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from sqlalchemy import func, select
from sqlalchemy.engine import make_url

from resolv.api.understandings import router
from resolv.core.config import ContextSettings, ConversationDatabaseSettings
from resolv.core.understanding_model import JsonStageModel
from resolv.core.understanding_prompts import INTENT_PROMPT_V1, RESOLUTION_PROMPT_V1
from resolv.db.checkpoints import (
    _open_connection,
    checkpoint_session,
    setup_checkpoint_tables,
)
from resolv.db.conversation import conversation_engine
from resolv.db.engine import create_session_factory
from resolv.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
)
from resolv.models.understanding import ConversationUnderstanding as U
from resolv.repositories.understanding import UnderstandingRepository
from resolv.schemas.understanding import UnderstandingError, UnderstandingInput
from resolv.services.intent_classification import IntentClassifier
from resolv.services.reference_resolution import ReferenceResolver
from resolv.services.understanding import UnderstandingService
from resolv.services.understanding_context import UnderstandingContext

OWNER = "resolv-understanding-recovery-v1"
ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATABASE = re.compile(r"resolv_conversations_smoke_[A-Za-z0-9_]+\Z")


def evidence(scenario, **values):
    print(
        json.dumps(
            {"scenario": scenario, "result": "PASS", **values}, ensure_ascii=False
        ),
        flush=True,
    )


class SmokeCounter:
    def count_text(self, text):
        return len(text) // 4

    def count(self, messages, schemas):
        return sum(self.count_text(m.content) for m in messages)


class FakeStageClient:
    """Only emits predeclared smoke contracts, never used as an accuracy measure."""

    def __init__(self):
        self.phase = 0

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        payload = json.loads(messages[1].content.split("\n", 1)[1])
        text = payload["original_text"]
        pending = payload["pending_requests"]
        if "resolution" not in payload:
            result = {
                "resolved_text": text,
                "bindings": [],
                "unresolved_references": [],
                "continuation_selection": None,
            }
            if text == "它":
                result["unresolved_references"] = [
                    {
                        "span": {"start": 0, "end": 1},
                        "candidate_refs": [],
                        "question": "哪笔订单？",
                    }
                ]
            if text == "继续退款":
                result["continuation_selection"] = pending[0]["task_id"]
        else:
            parts = text.split("，")
            requests = []
            for part in parts:
                intent = "refund_return" if "退款" in part else "order"
                refs = [
                    {
                        "kind": "input",
                        "source_id": "current",
                        "span": {
                            "start": text.index(part),
                            "end": text.index(part) + len(part),
                        },
                        "json_path": [],
                        "quote": part,
                    }
                ]
                if part == "继续退款":
                    refs.append(
                        {
                            "kind": "task",
                            "source_id": pending[0]["task_id"],
                            "span": None,
                            "json_path": ["text"],
                            "quote": pending[0]["text"],
                        }
                    )
                requests.append(
                    {
                        "text": part,
                        "intent": intent,
                        "evidence_refs": refs,
                        "binding_indexes": [],
                        "explicit_priority": None,
                        "priority_evidence_refs": [],
                        "ambiguities": [],
                    }
                )
            result = {"requests": requests}
        return AIMessage(content=json.dumps(result, ensure_ascii=False))


def make_service(settings, factory, checkpoint_factory=None):
    counter = SmokeCounter()
    context = UnderstandingContext(
        ContextSettings(
            _env_file=None,
            model_context_window_tokens=256000,
            tokenizer_profile="o200k_base",
            provider_input_overhead_tokens=5000,
        ),
        counter,
        model_name="fake-smoke",
        api_address_fingerprint="f" * 64,
        resolution_prompt=RESOLUTION_PROMPT_V1,
        intent_prompt=INTENT_PROMPT_V1,
    )
    model = JsonStageModel(FakeStageClient(), counter)
    return UnderstandingService(
        factory,
        checkpoint_factory or partial(checkpoint_session, settings),
        context,
        ReferenceResolver(model, context),
        IntentClassifier(model, context),
    )


def seed(factory):
    id = "understanding-gate-" + uuid4().hex
    with factory.begin() as session:
        c = Conversation(
            conversation_id=id, thread_id=id, title=OWNER, next_message_seq=3
        )
        session.add(c)
        session.flush()
        run, q, a = uuid4().hex, uuid4().hex, uuid4().hex
        session.add(
            ConversationRun(
                run_id=run,
                conversation_id=id,
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
                    conversation_id=id,
                    run_id=run,
                    seq=1,
                    role="user",
                    text="订单 MH001",
                ),
                ConversationMessage(
                    message_id=a,
                    conversation_id=id,
                    run_id=run,
                    seq=2,
                    role="assistant",
                    text="历史记录中的订单 MH001。",
                ),
            ]
        )
    return id


def migrate(engine):
    with engine.begin() as connection:
        config = Config(str(ROOT / "alembic.conversation.ini"))
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


def accept(service, conversation_id, text="查订单", **links):
    return service.accept(
        conversation_id, UnderstandingInput(request_id=uuid4().hex, text=text, **links)
    )[0]


def counts(service):
    return dict(service.resolver.model.calls_by_stage)


def expect_error(function, code=None):
    try:
        function()
    except UnderstandingError as error:
        if code is not None:
            assert error.code == code, (error.code, code)
        return error
    raise AssertionError("injected failure was unexpectedly accepted")


def run_gate(settings, *, setup):
    versions = {
        name: version(name)
        for name in ("langgraph", "langgraph-checkpoint", "langgraph-checkpoint-mysql")
    }
    assert versions == {
        "langgraph": "1.2.12",
        "langgraph-checkpoint": "4.2.0",
        "langgraph-checkpoint-mysql": "3.0.0",
    }
    with conversation_engine(settings) as engine:
        if setup:
            migrate(engine)
            setup_checkpoint_tables(settings)
        factory = create_session_factory(engine)
        c = seed(factory)
        with factory() as session:
            before = session.scalar(
                select(func.count()).select_from(ConversationMessage)
            )
        if setup:
            migrate(engine)
            setup_checkpoint_tables(settings)
        with factory() as session:
            assert (
                session.scalar(select(func.count()).select_from(ConversationMessage))
                == before
            )
        evidence("additive_repeated_setup", conversation_id=c, versions=versions)

        # No accepted half-snapshot on callback failure.
        request = UnderstandingInput(request_id="freeze-failure", text="查订单")
        try:
            with factory.begin() as session:
                UnderstandingRepository(session).create(
                    c, request, lambda _: (_ for _ in ()).throw(ValueError("injected"))
                )
        except ValueError:
            pass
        with factory() as session:
            assert (
                session.scalar(
                    select(U).where(
                        U.conversation_id == c, U.request_id == request.request_id
                    )
                )
                is None
            )
        evidence("atomic_acceptance_rollback")

        service = make_service(settings, factory)
        row = accept(service, c, "查订单，退款")
        original_publish = UnderstandingRepository.publish

        def rollback_publish(repo, *args):
            original_publish(repo, *args)
            raise RuntimeError("injected publish transaction rollback")

        with patch.object(UnderstandingRepository, "publish", rollback_publish):
            expect_error(lambda: service.run(c, row.understanding_id))
        assert service.record(c, row.understanding_id).final_json is None
        assert service.resume(c, row.understanding_id).status == "ready"
        assert counts(service) == {"resolution": 1, "intents": 1}
        evidence(
            "publish_transaction_rollback_reuses_stages",
            understanding_id=row.understanding_id,
            calls=counts(service),
        )

        archive_c = seed(factory)
        service = make_service(settings, factory)
        row = accept(service, archive_c, "查订单，退款")

        def archive_after_publish(point):
            if point == "after_publish":
                with factory.begin() as session:
                    session.get(Conversation, archive_c).status = "archived"

        service.fault_hook = archive_after_publish
        expect_error(
            lambda: service.run(archive_c, row.understanding_id),
            "conversation_archived",
        )
        assert service.get(archive_c, row.understanding_id).status == "failed"
        assert service.list_tasks(archive_c) == []
        evidence(
            "archive_between_publish_and_end_hides_pending",
            understanding_id=row.understanding_id,
        )

        # Real saver writes fail independently after each successful stage and at END.
        for phase in ("resolve_references", "classify_intents", "end"):
            fired = [False]

            @contextmanager
            def checkpoints(*, conversation_id, phase=phase, fired=fired):
                with checkpoint_session(
                    settings, conversation_id=conversation_id
                ) as checkpoint:
                    original = checkpoint.saver.put

                    def put(config, value, metadata, versions):
                        if (
                            value["channel_values"].get("phase") == phase
                            and not fired[0]
                        ):
                            fired[0] = True
                            raise RuntimeError("injected checkpoint failure")
                        return original(config, value, metadata, versions)

                    checkpoint.saver.put = put
                    yield checkpoint

            service = make_service(settings, factory, checkpoints)
            row = accept(service, c, "查订单，退款")
            expect_error(
                lambda service=service, row=row: service.run(c, row.understanding_id)
            )
            assert not service.get(c, row.understanding_id).route_ready
            view = service.resume(c, row.understanding_id)
            assert view.status == "ready" and counts(service) == {
                "resolution": 1,
                "intents": 1,
            }
            evidence(
                "checkpoint_failure_" + phase,
                understanding_id=row.understanding_id,
                calls=counts(service),
            )

        for point in ("after_publish", "after_graph_end"):
            service = make_service(settings, factory)
            fired = [False]

            def fail(actual, point=point, fired=fired):
                if actual == point and not fired[0]:
                    fired[0] = True
                    raise RuntimeError("injected finalization failure")

            service.fault_hook = fail
            row = accept(service, c, "查订单，退款")
            expect_error(
                lambda service=service, row=row: service.run(c, row.understanding_id)
            )
            with factory() as session:
                saved = session.get(U, row.understanding_id).final_json
                ids = [r["task_id"] for r in saved["pending_requests"]]
            assert service.get(c, row.understanding_id).status == "finalizing"
            assert not any(
                t.understanding_id == row.understanding_id
                for t in service.list_tasks(c)
            )
            resumed = service.resume(c, row.understanding_id)
            assert [r.task_id for r in resumed.pending_requests] == ids
            assert (
                service.run(c, row.understanding_id).model_dump()
                == resumed.model_dump()
            )
            evidence(
                point + "_and_lost_json",
                understanding_id=row.understanding_id,
                task_ids=ids,
                calls=counts(service),
            )

        # Original connection is killed; no reconnect can maintain execution rights.
        active, killed = [], [False]

        @contextmanager
        def killable(*, conversation_id):
            with checkpoint_session(
                settings, conversation_id=conversation_id
            ) as checkpoint:
                active[:] = [checkpoint]
                yield checkpoint

        service = make_service(settings, factory, killable)

        def kill(point):
            if point == "after_resolution_saved" and not killed[0]:
                killed[0] = True
                conn = _open_connection(settings)
                try:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "KILL CONNECTION %s", (active[0]._connection.thread_id(),)
                        )
                finally:
                    conn.close()
                active[0].assert_owned()

        service.fault_hook = kill
        row = accept(service, c)
        expect_error(lambda: service.run(c, row.understanding_id), "checkpoint_failed")
        assert service.resume(c, row.understanding_id).status == "ready"
        assert counts(service) == {"resolution": 1, "intents": 1}
        evidence(
            "killed_original_connection_no_reconnect",
            understanding_id=row.understanding_id,
            calls=counts(service),
        )

        # Two independent connections/executors race for one pending request.
        service = make_service(settings, factory)
        cc = seed(factory)
        parent = accept(service, cc, "查订单，退款")
        service.run(cc, parent.understanding_id)
        task = service.list_tasks(cc)[0]
        children = [
            accept(service, cc, "继续退款", continue_task_id=task.task_id)
            for _ in range(2)
        ]
        barrier = Barrier(2)

        def compete(row):
            contender = make_service(settings, factory)

            def sync(point):
                if point == "after_intents_saved":
                    barrier.wait(timeout=30)

            contender.fault_hook = sync
            try:
                return contender.run(cc, row.understanding_id)
            except UnderstandingError as error:
                return error

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(compete, children))
        winners = [r for r in results if not isinstance(r, UnderstandingError)]
        losers = [r for r in results if isinstance(r, UnderstandingError)]
        assert len(winners) == len(losers) == 1 and losers[0].code == "task_conflict"
        assert losers[0].related_id == winners[0].understanding_id
        evidence(
            "two_connections_one_pending_selection",
            task_id=task.task_id,
            winner=winners[0].understanding_id,
        )

        # Parallel clarification accepts use conversation row lock, single effective child.
        ambiguous = accept(service, cc, "它")
        service.run(cc, ambiguous.understanding_id)
        barrier = Barrier(2)

        def clarify(_):
            contender = make_service(settings, factory)
            barrier.wait(timeout=30)
            try:
                return accept(
                    contender,
                    cc,
                    "订单 MH001",
                    clarifies_understanding_id=ambiguous.understanding_id,
                )
            except UnderstandingError as error:
                return error

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(clarify, range(2)))
        winners = [r for r in results if not isinstance(r, UnderstandingError)]
        losers = [r for r in results if isinstance(r, UnderstandingError)]
        assert (
            len(winners) == len(losers) == 1
            and losers[0].related_id == winners[0].understanding_id
        )
        evidence(
            "two_connections_single_clarification_child",
            child_id=winners[0].understanding_id,
        )

        # Epoch fences survive new executor claim, including stale error/cleanup.
        row = accept(service, cc)
        with checkpoint_session(
            settings, conversation_id="understanding:" + row.understanding_id
        ):
            with factory.begin() as session:
                repo = UnderstandingRepository(session)
                first = repo.claim(row.understanding_id, "old-owner")
                second = repo.claim(row.understanding_id, "new-owner")
            for operation in ("release", "record_error"):

                def stale(operation=operation):
                    with factory.begin() as session:
                        method = getattr(UnderstandingRepository(session), operation)
                        args = (
                            row.understanding_id,
                            first.execution_epoch,
                            "old-owner",
                        )
                        return (
                            method(*args, UnderstandingError("stale", True))
                            if operation == "record_error"
                            else method(*args)
                        )

                expect_error(stale, "understanding_busy")
            with factory.begin() as session:
                UnderstandingRepository(session).release(
                    row.understanding_id, second.execution_epoch, "new-owner"
                )
        evidence("stale_epoch_error_and_cleanup_rejected")

        # One same-ID lock cannot be obtained by a second connection.
        with checkpoint_session(
            settings, conversation_id="understanding:" + row.understanding_id
        ):
            expect_error(
                lambda: service.resume(cc, row.understanding_id), "understanding_busy"
            )
        evidence("concurrent_same_id_resume_busy")

        # Chat sentinel checkpoint shares the database but never the thread or lock.
        with checkpoint_session(settings, conversation_id=c) as checkpoint:
            graph = StateGraph(dict)
            graph.add_node("sentinel", lambda _: {"marker": "chat-sentinel"})
            graph.add_edge(START, "sentinel")
            graph.add_edge("sentinel", END)
            chat = graph.compile(checkpointer=checkpoint.saver)
            chat.invoke({}, {"configurable": {"thread_id": c}}, durability="sync")
            before_checkpoint = checkpoint.saver.get_tuple(
                {"configurable": {"thread_id": c}}
            ).checkpoint
            row = accept(service, c)
            service.run(c, row.understanding_id)
            assert (
                checkpoint.saver.get_tuple(
                    {"configurable": {"thread_id": c}}
                ).checkpoint
                == before_checkpoint
            )
        with factory() as session:
            assert (
                session.query(ConversationMessage).filter_by(conversation_id=c).count()
                == 2
            )
            assert session.get(Conversation, c).next_message_seq == 3
            assert session.get(Conversation, c).current_run_id is None
        evidence(
            "chat_checkpoint_messages_and_execution_unchanged", business_tool_calls=0
        )

        app = FastAPI()
        app.state.understanding_runtime = service
        app.include_router(router)
        with TestClient(app) as client:
            body = {"request_id": uuid4().hex, "text": "查订单"}
            view = client.post(f"/conversations/{c}/understandings", json=body)
            assert view.status_code == 200
            id = view.json()["understanding_id"]
            assert (
                client.get(f"/conversations/{c}/understandings/{id}").json()
                == view.json()
            )
            assert (
                client.post(f"/conversations/{c}/understandings", json=body).json()
                == view.json()
            )
            assert (
                client.post(f"/conversations/{c}/understandings/{id}/resume").json()
                == view.json()
            )
        evidence("real_mysql_http_idempotency_and_lost_response", understanding_id=id)


def read_manifest(path, database):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        raise SystemExit("Owned recovery manifest required") from None
    if (
        value.get("owner") != OWNER
        or value.get("database") != database
        or not value.get("conversation_id")
        or not value.get("understanding_id")
    ):
        raise SystemExit("Foreign recovery manifest rejected")
    return value


def run_phase(settings, args):
    database = make_url(settings.conversation_database_url.get_secret_value()).database
    path = args.manifest
    with conversation_engine(settings) as engine:
        factory = create_session_factory(engine)
        service = make_service(settings, factory)
        if args.phase == "pause":
            if path.exists():
                raise SystemExit(
                    "Pause requires a new manifest; existing files preserved"
                )
            c = seed(factory)
            row = accept(service, c, "查订单，退款")
            manifest = {
                "owner": OWNER,
                "database": database,
                "conversation_id": c,
                "understanding_id": row.understanding_id,
                "crash_at": args.crash_at,
                "task_ids": [],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                json.dump(manifest, stream)

            def crash(point):
                expected = (
                    "after_resolution_saved"
                    if args.crash_at == "after_resolution"
                    else "after_publish"
                )
                if point == expected:
                    if args.crash_at == "after_publish":
                        with factory() as session:
                            result = session.get(U, row.understanding_id).final_json
                            manifest["task_ids"] = [
                                r["task_id"] for r in result["pending_requests"]
                            ]
                        path.write_text(json.dumps(manifest))
                    print(
                        json.dumps(
                            {
                                "injected_exit": 23,
                                "crash_at": args.crash_at,
                                "understanding_id": row.understanding_id,
                            }
                        ),
                        flush=True,
                    )
                    os._exit(23)

            service.fault_hook = crash
            service.run(c, row.understanding_id)
            raise AssertionError("pause failed to crash")
        manifest = read_manifest(path, database)
        c, id = manifest["conversation_id"], manifest["understanding_id"]
        with factory() as session:
            if session.get(Conversation, c).title != OWNER:
                raise SystemExit("Foreign conversation rejected")
        if args.phase == "resume":
            view = service.resume(c, id)
            assert view.status == "ready"
            expected = {} if manifest["crash_at"] == "after_publish" else {"intents": 1}
            assert counts(service) == expected
            if manifest["task_ids"]:
                assert [r.task_id for r in view.pending_requests] == manifest[
                    "task_ids"
                ]
            manifest["task_ids"] = [r.task_id for r in view.pending_requests]
            manifest["resume_calls"] = counts(service)
            path.write_text(json.dumps(manifest))
            evidence(
                "independent_process_resume",
                understanding_id=id,
                calls=counts(service),
                task_ids=manifest["task_ids"],
            )
        else:
            view = service.get(c, id)
            assert (
                view.status == "ready"
                and [r.task_id for r in view.pending_requests] == manifest["task_ids"]
            )
            record = service.record(c, id)
            calls = {
                stage: len([a for a in record.invocations if a.stage == stage])
                for stage in ("resolution", "intents")
            }
            assert calls == {"resolution": 1, "intents": 1}
            with checkpoint_session(
                settings, conversation_id="understanding:" + id
            ) as checkpoint:
                raw = checkpoint.saver.get_tuple(
                    {"configurable": {"thread_id": "understanding:" + id}}
                )
                assert raw.checkpoint["channel_values"]["phase"] == "end"
            evidence(
                "independent_process_verify",
                understanding_id=id,
                domain_audit_calls=calls,
                business_tool_calls=0,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-env", required=True)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--phase", choices=("pause", "resume", "verify"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--crash-at", choices=("after_resolution", "after_publish"))
    args = parser.parse_args(argv)
    if (
        args.setup
        and args.phase
        or args.phase
        and not args.manifest
        or args.phase == "pause"
        and not args.crash_at
        or args.phase != "pause"
        and args.crash_at
    ):
        parser.error("Invalid setup/phase/manifest/crash-at combination")
    try:
        settings = ConversationDatabaseSettings(
            _env_file=None,
            conversation_database_url=os.environ.get(args.database_env, ""),
        )
        database = make_url(
            settings.conversation_database_url.get_secret_value()
        ).database
        if not SMOKE_DATABASE.fullmatch(database or ""):
            raise ValueError
    except ValueError:
        parser.error(
            "Explicit isolated resolv_conversations_smoke_* MySQL URL required"
        )
    if args.phase and args.phase != "pause":
        read_manifest(args.manifest, database)
    if args.phase:
        run_phase(settings, args)
    else:
        run_gate(settings, setup=args.setup)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 — credential-safe smoke failure; details are audited in domain records
        raise SystemExit(
            "Understanding MySQL Gate FAILED; inspect safe scenario evidence and domain status."
        ) from None
