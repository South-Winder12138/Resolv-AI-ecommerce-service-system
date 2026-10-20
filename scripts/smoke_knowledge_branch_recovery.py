"""K3: real owned MySQL and checkpoint crash windows; deterministic fake inference."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import os
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
from uuid import uuid4

from langchain_core.messages import AIMessage
from scripts.knowledge_gate_support import (
    OWNERSHIP,
    configured_urls,
    formal_snapshot,
    owned_resources,
    save_json,
    source_fingerprints,
)
from scripts.smoke_three_branch_recovery import lock_probe
from sqlalchemy import func, select

from resolv.db.checkpoints import checkpoint_session
from resolv.db.engine import create_session_factory
from resolv.models.conversation import ConversationAction, ConversationMessage
from resolv.models.knowledge_branch import ConversationKnowledgeReviewItem
from resolv.repositories.knowledge_branch import KnowledgeBranchRepository
from resolv.schemas.hybrid_knowledge import HybridVersionSnapshot
from resolv.schemas.knowledge_branch import KnowledgeModelAudit
from resolv.schemas.knowledge_release import ReleaseSelection
from resolv.services.context import TokenCounter
from resolv.services.hybrid_fingerprints import (
    canonical_hash,
    manifest_hash,
    profile_hash,
)
from resolv.services.knowledge_gate_validation import RECOVERY_WINDOWS as WINDOWS
from resolv.services.knowledge_gate_validation import validate_recovery_gate
from resolv.services.knowledge_release import (
    ROOT,
    WORKSPACE,
    new_output,
    selector_payload,
)

QUESTION = "退货运费由谁承担？"


def append_call(path, purpose):
    with Path(path).open("a") as f:
        f.write(json.dumps({"purpose": purpose}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def make_service(factory, settings, state):
    # Reuse the audited deterministic fixtures; these are explicitly NOT semantic evidence.
    from tests.branch_fixtures import make_routed_service
    from tests.hybrid_fixtures import manifest, profile
    from tests.knowledge_branch_fixtures import branch_profile, hybrid_result
    from tests.understanding_fixtures import make_classification, make_resolution

    selected = ReleaseSelection(
        status="configured",
        release_id="recovery-fixture",
        index_version="fixture-v3",
        collection=profile().collection_name,
        profile=profile(),
        profile_hash=profile_hash(profile()),
        manifest_hash=manifest_hash(manifest()),
    )
    prof = branch_profile().model_copy(
        update={"selection": selected, "index_profile": selected.profile}
    )
    result = (
        hybrid_result((-10.0,))
        if state["window"].startswith("fallback")
        else hybrid_result((2.0, 1.0))
    )

    class Client:
        counter_factory = staticmethod(TokenCounter)

        def call(self, purpose, messages, profile, hook):
            append_call(state["calls"], purpose)
            a = KnowledgeModelAudit(
                invocation_id=uuid4().hex,
                purpose=purpose,
                profile_fingerprint=profile.fingerprint,
                input_hash=canonical_hash(
                    [m.model_dump(mode="json") for m in messages]
                ),
                input_estimate=1000,
                started_at=datetime.now(UTC),
                outcome="started",
            )
            hook(a)
            output = (
                {"standard": QUESTION, "expanded": ["邮费"]}
                if purpose == "knowledge_query"
                else {
                    "useful": not state.get("self_check_false", False),
                    "supporting_window_ids": []
                    if state.get("self_check_false", False)
                    else [result.windows[0].window_id],
                    "missing_information": [],
                    "reason": "Deterministic recovery fixture",
                }
            )
            hook(
                a.model_copy(
                    update={
                        "outcome": "succeeded",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "finish_reason": "stop",
                        "finished_at": datetime.now(UTC),
                    }
                )
            )
            return output

        def invoke_json(self, purpose, messages, output_type, *, profile, audit_hook):
            return output_type.model_validate_json(
                json.dumps(self.call(purpose, messages, profile, audit_hook))
            )

        def invoke_answer(self, messages, *, profile, audit_hook):
            self.call("knowledge_answer", messages, profile, audit_hook)
            return AIMessage(
                content="质量问题由商家承担[1]。",
                response_metadata={"finish_reason": "stop"},
            )

    class Retrievers:
        def count_query_tokens(self, text):
            return len(text)

        def validate_selection(self, selection):
            return HybridVersionSnapshot(
                version_id=selection.index_version,
                profile=selection.profile,
                manifest=manifest(),
                state="ready",
            )

        def retrieve(self, *a, **kw):
            append_call(state["calls"], "retrieval")
            return result

    service = make_routed_service(
        factory,
        partial(checkpoint_session, settings),
        stage_outputs=[
            make_resolution(resolved_text=QUESTION),
            make_classification([QUESTION], ["product_consultation"]),
        ],
    )
    service.orchestration_version = "routed-v3"
    service.knowledge_profile = prof
    client = Client()
    service.knowledge_clients = {
        p: client for p in ("knowledge_query", "knowledge_check", "knowledge_answer")
    }
    service.knowledge_retrievers = Retrievers()
    service.release_selector_path = Path(state["selector"])
    if not service.release_selector_path.exists():
        save_json(service.release_selector_path, selector_payload(selected))
    return service


def state_snapshot(service, state):
    with service.session_factory() as session:
        record = KnowledgeBranchRepository(session, service.owner_id).get(
            state["run_id"]
        )
        counts = {
            "assistant": session.scalar(
                select(func.count())
                .select_from(ConversationMessage)
                .where(
                    ConversationMessage.run_id == state["run_id"],
                    ConversationMessage.role == "assistant",
                )
            ),
            "review": session.scalar(
                select(func.count())
                .select_from(ConversationKnowledgeReviewItem)
                .where(ConversationKnowledgeReviewItem.run_id == state["run_id"])
            ),
            "tools": session.scalar(
                select(func.count())
                .select_from(ConversationAction)
                .where(ConversationAction.run_id == state["run_id"])
            ),
        }
    hashes = {s.stage: s.output_hash for s in record.stages} if record else {}
    run = service.get_run(state["run_id"])
    detail = service.detail(run.conversation_id)
    return {
        "hashes": hashes,
        "counts": counts,
        "status": run.status,
        "public_assistants": sum(m.role == "assistant" for m in detail.messages),
        "call_counts": dict(
            Counter(
                json.loads(line)["purpose"]
                for line in Path(state["calls"]).read_text().splitlines()
            )
        )
        if Path(state["calls"]).exists()
        else {},
    }


def child(path, mode):
    from resolv.core.config import ConversationDatabaseSettings

    state = json.loads(Path(path).read_text())
    if not Path(path).resolve().is_relative_to((ROOT / WORKSPACE).resolve()):
        raise ValueError("Foreign child state path")
    with owned_resources(OWNERSHIP) as (owner, _, engine):
        if state["owner"] != owner["owner"] or state["window"] not in WINDOWS:
            raise ValueError("Foreign crash state")
        _, url = configured_urls()
        settings = ConversationDatabaseSettings(
            _env_file=None,
            conversation_database_url=url.set(
                database=owner["conversation_database"]
            ).render_as_string(hide_password=False),
        )
        factory = create_session_factory(engine)
        service = make_service(factory, settings, state)

        def crash():
            receipt = state_snapshot(service, state)
            receipt.update(reached=True, window=state["window"])
            save_json(Path(state["reached"]), receipt)
            os._exit(86)

        if mode == "crash":
            target = (
                state["window"].replace("fallback_", "knowledge_")
                if state["window"].startswith("fallback")
                else state["window"]
            )
            service.fault_hook = lambda point: crash() if point == target else None
            if target in {"before_end_checkpoint", "after_end_checkpoint"}:
                original_factory = service.checkpoint_factory

                @contextmanager
                def checkpoints(**kwargs):
                    with original_factory(**kwargs) as session:
                        original = session.saver.put

                        def put(config, checkpoint, metadata, versions):
                            end = (
                                checkpoint.get("channel_values", {}).get("phase")
                                == "end"
                                and config["configurable"]["thread_id"]
                                == state["conversation_id"]
                            )
                            if end and target == "before_end_checkpoint":
                                crash()
                            result = original(config, checkpoint, metadata, versions)
                            if end and target == "after_end_checkpoint":
                                crash()
                            return result

                        session.saver.put = put
                        yield session

                service.checkpoint_factory = checkpoints
            if target == "after_completed_commit":
                original = service.mark_completed

                def completed(*args):
                    result = original(*args)
                    crash()
                    return result

                service.mark_completed = completed
        events = list(service.resume(state["conversation_id"], state["run_id"]))
        result = state_snapshot(service, state)
        result["events"] = [e.type for e in events]
        save_json(Path(state["resumed"]), result)
        if result["status"] != "completed":
            return 1
    return 0


def run_window(out, owner, engine, settings, window):
    started = time.monotonic()
    path = out / (window + ".json")
    state = {
        "owner": owner["owner"],
        "window": window,
        "calls": str(out / (window + "-calls.jsonl")),
        "selector": str(out / "fake-selector.json"),
        "reached": str(out / (window + "-reached.json")),
        "resumed": str(out / (window + "-resumed.json")),
    }
    service = make_service(create_session_factory(engine), settings, state)
    c = service.create()
    run, _ = service.accept(c.conversation_id, window, QUESTION, "agent")
    state.update(run_id=run.run_id, conversation_id=c.conversation_id)
    save_json(path, state)
    args = [
        sys.executable,
        str(ROOT / "scripts/smoke_knowledge_branch_recovery.py"),
        "--child-state",
        str(path),
    ]
    crash = subprocess.run(
        [*args, "--child-mode", "crash"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (out / (window + "-crash.log")).write_text(crash.stdout + "\n" + crash.stderr)
    reached = Path(state["reached"])
    if crash.returncode != 86 or not reached.exists():
        raise ValueError("Expected crash window not reached: " + window)
    before = json.loads(reached.read_text())
    if before["status"] != "completed" and before["public_assistants"] != 0:
        raise ValueError("Unfinished result leaked")
    resume = subprocess.run(
        [*args, "--child-mode", "resume"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (out / (window + "-resume.log")).write_text(resume.stdout + "\n" + resume.stderr)
    if resume.returncode != 0:
        raise ValueError("Recovery child failed: " + window)
    after = json.loads(Path(state["resumed"]).read_text())
    stage_purposes = {
        "query": "knowledge_query",
        "retrieval": "retrieval",
        "check": "knowledge_check",
        "answer": "knowledge_answer",
    }
    reused = all(
        after["call_counts"].get(p, 0) == before["call_counts"].get(p, 0)
        for s, p in stage_purposes.items()
        if s in before["hashes"]
    )
    return {
        "elapsed_seconds": time.monotonic() - started,
        "window": window,
        "passed": True,
        "reached": True,
        "child_exit": crash.returncode,
        "status": after["status"],
        "assistant_count": after["counts"]["assistant"],
        "review_count": after["counts"]["review"],
        "stage_hashes_stable": all(
            after["hashes"].get(k) == v for k, v in before["hashes"].items()
        ),
        "saved_calls_reused": reused,
        "business_tool_calls": after["counts"]["tools"],
        "before": before,
        "after": after,
    }


def atomic_probe(engine, settings, out):
    state = {
        "window": "fallback_before_finalize_commit",
        "calls": str(out / "atomic-calls.jsonl"),
        "selector": str(out / "fake-selector.json"),
    }
    service = make_service(create_session_factory(engine), settings, state)
    c = service.create()
    run, _ = service.accept(c.conversation_id, "atomic", QUESTION, "agent")
    state.update(run_id=run.run_id, conversation_id=c.conversation_id)

    def fail(point):
        if point == "knowledge_before_finalize_commit":
            raise RuntimeError("injected atomic finalize failure")

    service.fault_hook = fail
    events = list(service.run(run.run_id))
    if events[-1].type != "error":
        raise ValueError("Atomic failure injection did not fire")
    snap = state_snapshot(service, state)
    return {
        "passed": snap["counts"]["assistant"] == snap["counts"]["review"] == 0,
        "rolled_back_reply": snap["counts"]["assistant"] == 0,
        "rolled_back_review": snap["counts"]["review"] == 0,
    }


def knowledge_epoch_probe(engine, settings, out):
    from resolv.models.conversation import ConversationRun
    from resolv.schemas.conversation import ConversationError
    from resolv.services.conversation_routing import run_understanding

    factory = create_session_factory(engine)
    state = {
        "window": "knowledge_after_answer_saved",
        "calls": str(out / "epoch-calls.jsonl"),
        "selector": str(out / "fake-selector.json"),
    }
    service = make_service(factory, settings, state)
    c = service.create()
    run, _ = service.accept(c.conversation_id, "epoch-control", QUESTION, "agent")
    with service.execution(run.run_id) as execution:
        run_understanding(service.get_run(run.run_id), service.understanding)
        service.knowledge_branch().prepare_query(run.run_id, execution)
        with factory.begin() as session:
            KnowledgeBranchRepository(session, service.owner_id).record_error(
                run.run_id, execution.epoch, "gate_valid_epoch", True
            )
        with factory() as session:
            before = (
                KnowledgeBranchRepository(session, service.owner_id)
                .get(run.run_id)
                .error_code
            )
        with factory.begin() as session:
            row = session.get(ConversationRun, run.run_id)
            row.execution_epoch += 1
            row.execution_owner_id = "gate-successor"
        rejected = False
        try:
            with factory.begin() as session:
                KnowledgeBranchRepository(session, service.owner_id).record_error(
                    run.run_id, execution.epoch, "gate_stale_epoch", False
                )
        except ConversationError as error:
            if error.code != "conversation_busy":
                raise
            rejected = True
        with factory() as session:
            after = (
                KnowledgeBranchRepository(session, service.owner_id)
                .get(run.run_id)
                .error_code
            )
    return {
        "passed": before == "gate_valid_epoch" and rejected and after == before,
        "valid_write_committed": before == "gate_valid_epoch",
        "stale_write_rejected": rejected,
        "unchanged_after_stale": after == before,
    }


def run(report_dir):
    started = time.monotonic()
    from resolv.core.config import ConversationDatabaseSettings

    out = new_output(report_dir)
    out.mkdir(parents=True)
    report = {
        "gate": "K3",
        "passed": False,
        "real_services": True,
        "fake_inference": True,
        "accuracy_evidence": False,
        "business_tool_calls": 0,
        "source_fingerprints": source_fingerprints(
            ("scripts/smoke_three_branch_recovery.py",)
        ),
        "cases": [],
    }
    try:
        before = formal_snapshot()
        with owned_resources(OWNERSHIP) as (owner, _, engine):
            _, url = configured_urls()
            settings = ConversationDatabaseSettings(
                _env_file=None,
                conversation_database_url=url.set(
                    database=owner["conversation_database"]
                ).render_as_string(hide_password=False),
            )
            for window in WINDOWS:
                report["cases"].append(run_window(out, owner, engine, settings, window))
                save_json(out / "partial.json", report)
            state = {
                "window": "knowledge_after_answer_saved",
                "calls": str(out / "lock-calls.jsonl"),
                "selector": str(out / "fake-selector.json"),
            }
            locks = lock_probe(
                make_service(create_session_factory(engine), settings, state)
            )
            epoch_proof = knowledge_epoch_probe(engine, settings, out)
            report["locks_and_fencing"] = dict(
                locks,
                busy_rejected=locks["concurrent_accepts"] == 1,
                stale_epoch_rejected=epoch_proof["passed"],
                knowledge_epoch_control=epoch_proof,
            )
            report["atomicity"] = atomic_probe(engine, settings, out)
        report["preservation"] = {"before": before, "after": formal_snapshot(before)}
        report["elapsed_seconds"] = time.monotonic() - started
        report["passed"] = True
        validate_recovery_gate(report)
        save_json(out / "report.json", report)
        return report
    except Exception as exc:
        report.update(
            passed=False,
            error_type=type(exc).__name__,
            error_code=getattr(exc, "code", "knowledge_recovery_gate_failed"),
        )
        save_json(out / "failure.json", report)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path)
    p.add_argument("--child-state", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--child-mode", choices=["crash", "resume"], help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child_state:
        if a.report_dir or not a.child_mode:
            p.error("invalid child invocation")
        return child(a.child_state, a.child_mode)
    if not a.report_dir or a.child_mode:
        p.error("--report-dir required")
    try:
        report = run(a.report_dir)
    except Exception as exc:  # noqa: BLE001 -- record safe gate failure without credentials
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error_code": getattr(
                        exc, "code", "knowledge_recovery_gate_failed"
                    ),
                }
            )
        )
        return 1
    print(json.dumps({"passed": True, "cases": len(report["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
