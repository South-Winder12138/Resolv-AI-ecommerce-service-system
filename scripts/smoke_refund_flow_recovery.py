"""R3: explicitly owned real storage and process death; deterministic inference only."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import subprocess
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from uuid import uuid4

from scripts.refund_gate_support import (
    ROOT,
    WORKSPACE,
    committed_sources,
    configured_urls,
    new_report_dir,
    owned_refund_resources,
    save_json,
    snapshot_formal_resources,
    validate_manifest,
)
from scripts.smoke_refund_flow import SOURCES as FLOW_SOURCES
from sqlalchemy import func, select, update

from resolv.core.config import ConversationDatabaseSettings
from resolv.db.checkpoints import checkpoint_session
from resolv.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
)
from resolv.models.refund_application import RefundApplication, RefundSubmissionReceipt
from resolv.models.refund_flow import (
    ConversationRefundSubmission,
)
from resolv.repositories.conversation import ConversationRepository
from resolv.repositories.refund_flow import RefundFlowRepository
from resolv.schemas.refund import RefundAccessContext, RefundFlowError
from resolv.schemas.refund_submission import RefundSubmitInput
from resolv.services.hybrid_fingerprints import canonical_hash

WINDOWS = tuple(
    f"refund_{side}_{stage}_saved"
    for stage in (
        "binding",
        "query",
        "retrieval",
        "model_request",
        "observation",
        "decision",
    )
    for side in ("before", "after")
) + tuple(
    f"{side}_{stage}"
    for stage in (
        "finalize_commit",
        "end_checkpoint",
        "completed_commit",
        "submission_creation",
        "revalidation_save",
        "business_commit",
        "receipt_publication",
        "command_terminal_commit",
    )
    for side in ("before", "after")
)
PROBES = (
    "positive_epoch",
    "negative_epoch",
    "named_lock_exclusion",
    "sqlite_atomic_rollback",
    "new_new_race",
    "new_legacy_race",
    "expired_disabled_receipt",
    "missing_receipt",
    "tampered_receipt",
    "later_chat_preserved",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_recovery_report(report):
    windows = report.get("windows", [])
    require(
        [w.get("window") for w in windows] == list(WINDOWS),
        "All 28 distinct ordered windows required",
    )
    for w in windows:
        require(
            w.get("reached") is True
            and w.get("exit_code") == 86
            and w.get("resume_exit_code") == 0,
            "Real reached death and successful resume required",
        )
        for key in (
            "saved_hashes_preserved",
            "saved_calls_reused",
            "receipt_replayed",
            "original_run_preserved",
        ):
            require(w.get(key) is True, "Missing recovery proof: " + key)
        for key in (
            "assistant_count",
            "offer_count",
            "application_count",
            "receipt_count",
        ):
            require(
                type(w.get(key)) is int and w[key] == 1,
                "Exactly one durable result required: " + key,
            )
    require(
        all(report.get("probes", {}).get(p) is True for p in PROBES),
        "Real positive/negative storage probes required",
    )
    require(
        report.get("real_model_calls") == 0 and report.get("formal_preserved") is True,
        "Models or formal resources changed",
    )


def append_call(path, purpose):
    with Path(path).open("a") as f:
        f.write(json.dumps({"purpose": purpose}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def make_service(
    resources, state, scratch, *, checkpoint_factory=None, local_locks=None
):
    """Explicit fixture runtime; never installed in production composition."""
    from tests.branch_fixtures import make_routed_service
    from tests.knowledge_branch_fixtures import branch_profile
    from tests.refund_fixtures import NOW
    from tests.refund_flow_fixtures import flow_case
    from tests.understanding_fixtures import make_classification, make_resolution

    from resolv.services.knowledge_release import selector_payload
    from resolv.services.refund_flow import RefundFlowService
    from resolv.services.refund_submission import RefundSubmissionService

    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=False)
    cleanup = ExitStack()
    h = cleanup.enter_context(flow_case(scratch))
    if checkpoint_factory is None:
        _, url = configured_urls()
        settings = ConversationDatabaseSettings(
            _env_file=None,
            conversation_database_url=url.set(
                database=resources.identity["conversation_database"]
            ).render_as_string(hide_password=False),
        )
        checkpoint_factory = partial(checkpoint_session, settings)
    text = f"订单{state['order_id']}申请整单退款，因为商品未激活"
    c = make_routed_service(
        resources.conversation_factory,
        checkpoint_factory,
        stage_outputs=[
            make_resolution(resolved_text=text),
            make_classification([text], ["refund_return"]),
        ],
        business_factory=resources.business_factory,
    )
    c._fixture_cleanup = cleanup
    c.orchestration_version = "routed-v4"
    c.lease_seconds, c.heartbeat_seconds = 4, 1
    c.knowledge_profile = branch_profile()
    profile = h.profile.model_copy(
        update={
            "adapter": h.profile.adapter.model_copy(
                update={
                    "namespace": "refund-recovery-"
                    + state.get("scope", "offline")
                    + "-"
                    + state["order_id"]
                }
            )
        }
    )
    h.facts.descriptor = profile.adapter
    access = RefundAccessContext(
        subject_ref="fixture-subject",
        namespace=profile.adapter.namespace,
        authorization_version=profile.adapter.authorization_version,
        trusted=True,
    )
    identity = SimpleNamespace(resolve=lambda _: access)
    c.refund_profile = profile
    c.release_selector_path = Path(state["selector"])
    save_json(c.release_selector_path, selector_payload(profile.selection))
    c.refund_clock = lambda: NOW
    clock = NOW + timedelta(seconds=121 if state.get("resume_clock") else 0)
    submissions = RefundSubmissionService(
        resources.conversation_factory,
        resources.business_factory,
        owner_id=c.owner_id,
        identity=identity,
        facts=h.facts,
        selection_reader=lambda: profile.selection,
        profile=profile,
        clock=lambda: clock,
        execution_lock_factory=local_locks,
    )
    flow = RefundFlowService(
        c,
        client=h.client,
        retrievers=h.flow.retrievers,
        identity=identity,
        facts=h.facts,
        registration=submissions,
    )
    c.refund_runtime = SimpleNamespace(
        flow=flow, profile=profile, submissions=submissions
    )
    # The fixture transport records every invocation across process boundaries.
    for obj, name, purpose in (
        (h.client, "prepare", "query"),
        (h.client, "decide", "decision"),
        (h.flow.retrievers, "retrieve", "retrieval"),
    ):
        original = getattr(obj, name)

        def logged(*a, _original=original, _purpose=purpose, **kw):
            append_call(state["calls"], _purpose)
            return _original(*a, **kw)

        setattr(obj, name, logged)
    return c


def snapshot(service, state):
    run = service.get_run(state["run_id"])
    with service.session_factory() as s:
        r = RefundFlowRepository(s, service.owner_id).get(run.run_id)
        offer = RefundFlowRepository(s, service.owner_id).offer_for_run(run.run_id)
        commands = list(
            s.scalars(
                select(ConversationRefundSubmission).where(
                    ConversationRefundSubmission.conversation_id == run.conversation_id
                )
            )
        )
        assistant_count = s.scalar(
            select(func.count())
            .select_from(ConversationMessage)
            .where(
                ConversationMessage.run_id == run.run_id,
                ConversationMessage.role == "assistant",
            )
        )
        current = s.get(Conversation, run.conversation_id).current_run_id
    with service.refund_runtime.submissions.business_factory() as s:
        app = s.scalar(
            select(RefundApplication).where(
                RefundApplication.order_source
                == service.refund_profile.adapter.namespace,
                RefundApplication.order_id == state["order_id"],
            )
        )
        receipt_rows = (
            list(
                s.scalars(
                    select(RefundSubmissionReceipt).where(
                        RefundSubmissionReceipt.application_id == app.application_id
                    )
                )
            )
            if app
            else []
        )
    calls = (
        Counter(
            json.loads(line)["purpose"]
            for line in Path(state["calls"]).read_text().splitlines()
        )
        if Path(state["calls"]).exists()
        else Counter()
    )
    return {
        "status": run.status,
        "epoch": run.execution_epoch,
        "run_id": run.run_id,
        "conversation_id": run.conversation_id,
        "current_run_id": current,
        "assistant_count": assistant_count,
        "offer_count": int(offer is not None),
        "offer_id": offer.offer_id if offer else None,
        "command_id": commands[0].command_id if commands else None,
        "application_count": int(app is not None),
        "receipt_count": len(receipt_rows),
        "receipt_hash": canonical_hash(receipt_rows[0].result)
        if receipt_rows
        else None,
        "hashes": {stage: output for stage, _, output in r.stage_hashes} if r else {},
        "requests": [
            canonical_hash(x.model_dump(mode="json")) for x in r.model_requests
        ]
        if r
        else [],
        "observations": [
            canonical_hash(x.model_dump(mode="json")) for x in r.observations
        ]
        if r
        else [],
        "calls": dict(calls),
        "public_assistants": sum(
            m.role == "assistant" for m in service.detail(run.conversation_id).messages
        ),
    }


def install_crash(service, state):
    target = state["window"]

    def crash():
        save_json(
            Path(state["reached"]),
            snapshot(service, state) | {"reached": True, "window": target},
        )
        os._exit(86)

    service.fault_hook = lambda actual: (
        crash()
        if actual == target
        or (
            target in {"before_finalize_commit", "after_finalize_commit"}
            and actual == "refund_" + target
        )
        else None
    )
    sub = service.refund_runtime.submissions
    aliases = {
        "before_revalidation_save": "refund_after_revalidation",
        "after_revalidation_save": "refund_after_revalidation_save",
        "before_business_commit": "refund_before_business_commit",
        "after_business_commit": "refund_after_business_commit",
        "before_receipt_publication": "refund_before_submission_publish",
        "after_command_terminal_commit": "refund_after_submission_publish",
        "before_command_terminal_commit": "refund_before_submission_publish_commit",
    }
    sub.fault_hook = lambda actual: crash() if aliases.get(target) == actual else None
    if target == "after_receipt_publication":
        original_repo = sub._repo

        def repo(session):
            r = original_repo(session)
            original = r.publish

            def publish(*a, **kw):
                result = original(*a, **kw)
                crash()
                return result

            r.publish = publish
            return r

        sub._repo = repo
    if target in {"before_submission_creation", "after_submission_creation"}:
        original = sub.accept_submission

        def accept(*a, **kw):
            if target.startswith("before"):
                crash()
            result = original(*a, **kw)
            if target.startswith("after"):
                crash()
            return result

        sub.accept_submission = accept
    if target in {"before_completed_commit", "after_completed_commit"}:
        original = service.mark_completed

        def completed(*a, **kw):
            if target.startswith("before"):
                crash()
            result = original(*a, **kw)
            if target.startswith("after"):
                crash()
            return result

        service.mark_completed = completed
    if target in {"before_end_checkpoint", "after_end_checkpoint"}:
        original_factory = service.checkpoint_factory

        @contextmanager
        def checkpoints(**kw):
            with original_factory(**kw) as ck:
                original = ck.saver.put

                def put(config, checkpoint, metadata, versions):
                    end = (
                        checkpoint.get("channel_values", {}).get("phase") == "end"
                        and config["configurable"]["thread_id"]
                        == state["conversation_id"]
                    )
                    if end and target.startswith("before"):
                        crash()
                    result = original(config, checkpoint, metadata, versions)
                    if end and target.startswith("after"):
                        crash()
                    return result

                ck.saver.put = put
                yield ck

        service.checkpoint_factory = checkpoints


def submission_request_id(state):
    return "submit-" + state["scope"] + "-" + state["window"]


def child(path, mode):
    path = Path(path).resolve()
    require(path.is_relative_to((ROOT / WORKSPACE).resolve()), "Foreign child state")
    state = json.loads(path.read_text())
    identity = validate_manifest(Path(state["ownership"]))
    require(
        state["owner"] == identity["owner"] and state["window"] in WINDOWS,
        "Foreign crash target",
    )
    # These children may only connect to the explicitly configured MySQL port.
    _, mysql = configured_urls()

    def audit(event, args):
        if event == "socket.connect":
            address = args[1]
            if isinstance(address, tuple) and (
                address[0] not in {"127.0.0.1", "::1", "localhost"}
                or address[1] != (mysql.port or 3306)
            ):
                raise RuntimeError("Unapproved network transport in deterministic R3")

    sys.addaudithook(audit)
    state["resume_clock"] = mode == "resume"
    with owned_refund_resources(Path(state["ownership"])) as resources:
        service = make_service(
            resources, state, path.parent / (state["window"] + "-" + mode + "-deps")
        )
        try:
            if mode == "crash":
                install_crash(service, state)
            saved = service.get_run(state["run_id"])
            if saved.status != "completed":
                events = list(service.resume(saved.conversation_id, saved.run_id))
                require(
                    service.get_run(saved.run_id).status == "completed",
                    "Chat did not complete: " + str([(e.type, e.data) for e in events]),
                )
            offer = service.get_run(saved.run_id).result_json["refund"]["offer_id"]
            view = service.refund_runtime.submissions.submit(
                saved.conversation_id,
                offer,
                RefundSubmitInput(
                    request_id=submission_request_id(state), confirm=True
                ),
                {},
            )
            require(
                view.status == "completed" and view.receipt.status == "pending",
                "Submission did not complete",
            )
            replay = service.refund_runtime.submissions.get_submission(
                saved.conversation_id, view.submission_id, {}
            )
            result = snapshot(service, state) | {
                "receipt_replayed": replay.receipt == view.receipt,
                "receipt": view.receipt.model_dump(mode="json"),
            }
            save_json(Path(state["resumed"]), result)
        finally:
            service._fixture_cleanup.close()
    return 0


def run_window(resources, target, window, index, ownership):
    out = target / window
    out.mkdir()
    state = {
        "owner": resources.identity["owner"],
        "ownership": str(Path(ownership).resolve()),
        "window": window,
        "order_id": f"R3{index:04d}",
        "scope": canonical_hash(str(target.resolve()))[:12],
        "calls": str(out / "calls.jsonl"),
        "selector": str(out / "selector.json"),
        "reached": str(out / "reached.json"),
        "resumed": str(out / "resumed.json"),
    }
    service = make_service(resources, state, out / "parent-deps")
    try:
        c = service.create()
        run, _ = service.accept(
            c.conversation_id,
            window,
            f"订单{state['order_id']}申请整单退款，因为商品未激活",
            "agent",
        )
        state.update(run_id=run.run_id, conversation_id=c.conversation_id)
    finally:
        service._fixture_cleanup.close()
    path = out / "state.json"
    save_json(path, state)
    args = [sys.executable, str(Path(__file__).resolve()), "--child-state", str(path)]
    dead = subprocess.run(
        [*args, "--child-mode", "crash"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    (out / "crash.log").write_text(dead.stdout + dead.stderr)
    require(
        dead.returncode == 86 and Path(state["reached"]).exists(),
        "Crash window not reached: " + window,
    )
    before = json.loads(Path(state["reached"]).read_text())
    # Only this dead child's owned run lease is expired; machine clock is unchanged.
    with resources.conversation_factory.begin() as s:
        s.execute(
            update(ConversationRun)
            .where(
                ConversationRun.run_id == run.run_id,
                ConversationRun.execution_epoch == before["epoch"],
            )
            .values(lease_expires_at=None)
        )
    resumed = subprocess.run(
        [*args, "--child-mode", "resume"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    (out / "resume.log").write_text(resumed.stdout + resumed.stderr)
    require(
        resumed.returncode == 0 and Path(state["resumed"]).exists(),
        "Recovery failed: " + window,
    )
    after = json.loads(Path(state["resumed"]).read_text())
    require(
        all(after["hashes"].get(k) == v for k, v in before["hashes"].items())
        and after["requests"][: len(before["requests"])] == before["requests"]
        and after["observations"][: len(before["observations"])]
        == before["observations"],
        "Saved immutable values changed",
    )
    reused = all(
        after["calls"].get(p) == before["calls"].get(p)
        for stage, p in (
            ("query", "query"),
            ("retrieval", "retrieval"),
            ("decision", "decision"),
        )
        if stage in before["hashes"]
    )
    return {
        "window": window,
        "reached": True,
        "exit_code": dead.returncode,
        "resume_exit_code": resumed.returncode,
        "saved_hashes_preserved": True,
        "saved_calls_reused": reused,
        "receipt_replayed": after["receipt_replayed"]
        and (
            before["receipt_hash"] is None
            or before["receipt_hash"] == after["receipt_hash"]
        ),
        "original_run_preserved": after["run_id"] == run.run_id
        and after["current_run_id"] is None,
        **{
            k: after[k]
            for k in (
                "assistant_count",
                "offer_count",
                "application_count",
                "receipt_count",
            )
        },
        "before": before,
        "after": after,
        "lease_control": "expire only reached dead owned run; submission uses task clock +121s",
    }


def run_recovery_matrix(*, ownership, report_dir):
    target = new_report_dir(ownership, report_dir)
    report = {
        "gate": "R3",
        "passed": False,
        "windows": [],
        "probes": {},
        "real_model_calls": 0,
        "formal_preserved": False,
    }
    try:
        before = snapshot_formal_resources()
    except Exception as exc:  # noqa: BLE001 — preserve failure, not provider text
        report.update(
            error_type=type(exc).__name__,
            error_code="recovery_snapshot_failed",
            failure_stage="formal_snapshot_before",
        )
        save_json(target / "report.json", report)
        return report
    save_json(target / "preservation-before.json", before)
    try:
        with owned_refund_resources(ownership) as resources:
            for i, w in enumerate(WINDOWS, 1):
                report["windows"].append(run_window(resources, target, w, i, ownership))
                save_json(target / "progress.json", report)
            report["probes"] = storage_probes(resources, target)
    except Exception as exc:  # noqa: BLE001 — no credentials in public reports
        report.update(error_type=type(exc).__name__, error_code="recovery_gate_failed")
    try:
        after = snapshot_formal_resources(layout=before)
    except Exception as exc:  # noqa: BLE001 — preservation is unknown, never passed
        report.update(
            error_type=type(exc).__name__,
            error_code="recovery_snapshot_failed",
            failure_stage="formal_snapshot_after",
        )
        save_json(target / "report.json", report)
        return report
    report["formal_preserved"] = before == after
    save_json(target / "preservation-after.json", after)
    try:
        validate_recovery_report(report)
        report["passed"] = True
    except ValueError:
        pass
    save_json(target / "report.json", report)
    return report


def storage_probes(resources, target, *, checkpoint_factory=None, local_locks=None):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from tests.refund_fixtures import NOW
    from tests.test_refund_registration import command, register

    from resolv.models import Order, Refund
    from resolv.repositories.order import OrderRepository
    from resolv.repositories.refund import RefundRepository
    from resolv.schemas.conversation import ConversationError
    from resolv.schemas.knowledge_release import ReleaseSelection
    from resolv.services.refund import RefundService
    from resolv.services.refund_registration import (
        RefundRegistrationService,
        begin_refund_write,
    )
    from resolv.services.refund_submission import mysql_submission_lock

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    state = {
        "order_id": "R3PROBE",
        "scope": canonical_hash(str(target.resolve()))[:12],
        "window": "probes",
        "calls": str(target / "probe-calls.jsonl"),
        "selector": str(target / "probe-selector.json"),
    }
    service = make_service(
        resources,
        state,
        target / "probe-deps",
        checkpoint_factory=checkpoint_factory,
        local_locks=local_locks,
    )
    scope = state["scope"]
    measured = {}

    def expect_error(function, exception):
        try:
            function()
        except exception:
            return True
        return False

    try:
        c = service.create()
        run, _ = service.accept(
            c.conversation_id,
            "epoch",
            "订单R3PROBE申请整单退款，因为商品未激活",
            "agent",
        )
        with service.execution(run.run_id) as execution:
            with resources.conversation_factory.begin() as session:
                row = ConversationRepository(session, service.owner_id).owned(
                    run.run_id, execution.epoch, execution.owner_id
                )
                row.error_code = "owned-r3-probe"
            with resources.conversation_factory() as session:
                measured["positive_epoch"] = (
                    session.get(ConversationRun, run.run_id).error_code
                    == "owned-r3-probe"
                )

            def stale():
                with resources.conversation_factory.begin() as session:
                    ConversationRepository(session, service.owner_id).owned(
                        run.run_id, execution.epoch - 1, execution.owner_id
                    ).error_code = "stale-write"

            measured["negative_epoch"] = expect_error(stale, ConversationError)
            with resources.conversation_factory.begin() as session:
                row = ConversationRepository(session, service.owner_id).owned(
                    run.run_id, execution.epoch, execution.owner_id
                )
                measured["negative_epoch"] &= row.error_code == "owned-r3-probe"
                row.error_code = None
        require(
            list(service.resume(c.conversation_id, run.run_id))[-1].type == "done",
            "Probe chat failed",
        )
        offer_id = service.get_run(run.run_id).result_json["refund"]["offer_id"]
        sub = service.refund_runtime.submissions
        request = RefundSubmitInput(request_id="r3-probe-submit-" + scope, confirm=True)
        accepted = sub.accept_submission(c.conversation_id, offer_id, request, {})
        from resolv.schemas.refund_submission import RefundRevalidationAudit

        with sub.submission_execution(accepted.submission_id):
            pass  # Release epoch 1, then prove the prior valid epoch is fenced.
        with sub.submission_execution(accepted.submission_id) as execution:
            audit = RefundRevalidationAudit(
                attempt_id="epoch-probe", outcome="started", at=NOW
            )
            with resources.conversation_factory.begin() as session:
                sub._repo(session).audit(execution.lease, audit)
            with resources.conversation_factory() as session:
                measured["positive_epoch"] &= any(
                    a.attempt_id == "epoch-probe"
                    for a in sub._repo(session)
                    .get(accepted.submission_id)
                    .revalidation_audits
                )

            def stale_submission():
                with resources.conversation_factory.begin() as session:
                    sub._repo(session).audit(
                        execution.lease.model_copy(
                            update={"epoch": execution.lease.epoch - 1}
                        ),
                        audit,
                    )

            measured["negative_epoch"] &= expect_error(
                stale_submission, RefundFlowError
            )
        lock_factory = local_locks or (
            lambda cid: mysql_submission_lock(resources.conversation_engine, cid)
        )
        with (
            lock_factory(accepted.submission_id),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            measured["named_lock_exclusion"] = pool.submit(
                expect_error,
                lambda: sub.execute(accepted.submission_id, {}),
                RefundFlowError,
            ).result()
            # execute coalesces busy commands to a read-only running view.
            if not measured["named_lock_exclusion"]:

                def contend():
                    with lock_factory(accepted.submission_id):
                        return False

                measured["named_lock_exclusion"] = pool.submit(
                    expect_error, contend, RefundFlowError
                ).result()
        with resources.business_factory() as session:
            begin_refund_write(session)
            rollback = command(
                "r3-rollback-" + scope,
                order_id="R3ROLLBACK",
                namespace="refund-r3-rollback",
            )
            RefundRegistrationService(session).register(rollback)
            session.rollback()
        with resources.business_factory() as session:
            measured["sqlite_atomic_rollback"] = (
                session.scalar(
                    select(RefundApplication).where(
                        RefundApplication.command_id == "r3-rollback-" + scope
                    )
                )
                is None
                and session.get(RefundSubmissionReceipt, "r3-rollback-" + scope) is None
            )
        barrier = Barrier(2)

        def race_new(i):
            barrier.wait(timeout=5)
            return register(
                resources.business_factory,
                command(
                    "r3-new-" + str(i) + "-" + scope,
                    order_id="R3RACE",
                    namespace="refund-r3-race-" + scope,
                ),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(race_new, (1, 2)))
        measured["new_new_race"] = (
            results[0].application_id == results[1].application_id
            and sum(r.created for r in results) == 1
        )
        with resources.business_factory.begin() as session:
            session.add(Order(order_no="R3LOCAL-" + scope, status="delivered"))
        barrier = Barrier(2)

        def race_legacy(new):
            barrier.wait(timeout=5)
            if new:
                return register(
                    resources.business_factory,
                    command(
                        "r3-legacy-race-" + scope,
                        order_id="R3LOCAL-" + scope,
                        namespace="local-business-v1",
                    ),
                )
            with resources.business_factory() as session:
                result = RefundService(
                    OrderRepository(session), RefundRepository(session)
                ).apply_refund("R3LOCAL-" + scope, "r3 original legacy reason")
                session.commit()
                return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(race_legacy, (True, False)))
        with resources.business_factory() as session:
            app = session.scalar(
                select(RefundApplication).where(
                    RefundApplication.order_source == "local-business-v1",
                    RefundApplication.order_id == "R3LOCAL-" + scope,
                )
            )
            old = session.get(Refund, app.legacy_refund_id)
            measured["new_legacy_race"] = old is not None and old.reason == app.reason
        view = sub.execute(accepted.submission_id, {})
        require(
            view.status == "completed" and view.receipt.status == "pending",
            "Probe submit failed",
        )
        sub.clock = lambda: NOW + timedelta(hours=1)
        sub.selection_reader = lambda: ReleaseSelection(status="disabled")
        replay = sub.submit(c.conversation_id, offer_id, request, {})
        measured["expired_disabled_receipt"] = replay.receipt == view.receipt
        with resources.business_factory.begin() as session:
            row = session.get(RefundSubmissionReceipt, view.submission_id)
            saved = {col.name: getattr(row, col.name) for col in row.__table__.columns}
            session.delete(row)
        try:
            measured["missing_receipt"] = expect_error(
                lambda: sub.get_submission(c.conversation_id, view.submission_id, {}),
                RefundFlowError,
            )
        finally:
            with resources.business_factory.begin() as session:
                session.add(RefundSubmissionReceipt(**saved))
        with resources.business_factory.begin() as session:
            row = session.get(RefundSubmissionReceipt, view.submission_id)
            row.result = {**row.result, "reason": "tampered-r3"}
        try:
            measured["tampered_receipt"] = expect_error(
                lambda: sub.get_submission(c.conversation_id, view.submission_id, {}),
                RefundFlowError,
            )
        finally:
            with resources.business_factory.begin() as session:
                session.get(RefundSubmissionReceipt, view.submission_id).result = saved[
                    "result"
                ]
        later, _ = service.accept(c.conversation_id, "later-chat", "你好", "chat")
        with resources.conversation_factory() as session:
            before = session.get(Conversation, c.conversation_id).current_run_id
        sub.get_submission(c.conversation_id, view.submission_id, {})
        with resources.conversation_factory() as session:
            measured["later_chat_preserved"] = (
                session.get(Conversation, c.conversation_id).current_run_id
                == before
                == later.run_id
            )
        save_json(target / "storage-probes.json", measured)
        return measured
    finally:
        service._fixture_cleanup.close()


def frontend_source_required(name):
    return name.startswith("frontend/") and not name.endswith(".tsbuildinfo")


def frontend_dist(report_path):
    import re
    from hashlib import sha256

    report_path = Path(report_path).resolve()
    require(
        report_path
        in {
            (ROOT / WORKSPACE / "snapshot-frontend-02/report.json").resolve(),
            (ROOT / WORKSPACE / "snapshot-review-fixed-01/report.json").resolve(),
        },
        "Only explicit Task14 or review-fixed exact export is accepted",
    )
    report = json.loads(report_path.read_text())
    require(
        report.get("passed") is True
        and re.fullmatch(r"[0-9a-f]{40}", report.get("candidate_commit", ""))
        and re.fullmatch(r"[0-9a-f]{40}", report.get("tree", ""))
        and report["frontend"].get("passed") is True
        and report["frontend"].get("checks")
        == dict.fromkeys(("test", "typecheck", "build", "lint"), 0)
        and report["frontend"]["revision"]
        == report["python"]["revision"]
        == report["candidate_commit"]
        and report["frontend"]["tree"] == report["python"]["tree"] == report["tree"],
        "Mixed or unverified frontend export",
    )
    for args in (
        [report["candidate_commit"], report["tree"], "--", "."],
        [report["candidate_commit"], "HEAD", "--", "frontend"],
    ):
        require(
            subprocess.run(
                ["git", "diff", "--quiet", *args], cwd=ROOT, check=False
            ).returncode
            == 0,
            "Export tree or current committed frontend source differs",
        )
    source = report_path.parent / "frontend/frontend"
    for name, digest in report["frontend"]["source_fingerprints"].items():
        if not frontend_source_required(name):
            continue
        path = source / name.removeprefix("frontend/")
        require(
            path.is_file() and sha256(path.read_bytes()).hexdigest() == digest,
            "Frontend source differs from verified export",
        )
    dist = source / "dist"
    require((dist / "index.html").is_file(), "Verified dist missing")
    return dist, {
        "commit": report["candidate_commit"],
        "tree": report["tree"],
        "report_hash": canonical_hash(report),
        "dist_hashes": {
            str(p.relative_to(dist)): sha256(p.read_bytes()).hexdigest()
            for p in dist.rglob("*")
            if p.is_file()
        },
    }


def make_http_app(
    resources, target, dist, *, checkpoint_factory=None, local_locks=None
):
    from contextlib import asynccontextmanager
    from threading import Event

    from fastapi import FastAPI, Request
    from fastapi.staticfiles import StaticFiles
    from langchain_core.messages import AIMessage
    from tests.branch_fixtures import OfflineUsageModel
    from tests.refund_fixtures import NOW
    from tests.test_conversation_graph import Model
    from tests.understanding_fixtures import make_classification, make_resolution

    from resolv.api import agent, chat, conversations, refunds, understandings

    state = {
        "order_id": "R3UI001",
        "scope": canonical_hash(str(target.resolve()))[:12],
        "window": "http",
        "calls": str(target / "http-calls.jsonl"),
        "selector": str(target / "http-selector.json"),
    }
    service = make_service(
        resources,
        state,
        target / "http-deps",
        checkpoint_factory=checkpoint_factory,
        local_locks=local_locks,
    )
    service.owner_id = "refund-ui-" + resources.identity["owner"][:16]
    service.understanding.owner_id = service.owner_id
    # UI accepts the user's current text into the deterministic upstream contract.
    stage = service.understanding.resolver.model.client

    class Stages:
        def invoke(self, messages, **kw):
            raise RuntimeError("Use per-accept fixed stage sequence")

    original_accept = service.accept

    def accept(cid, rid, question, mode, **kw):
        stage.replies = [
            AIMessage(
                content=make_resolution(resolved_text=question).model_dump_json()
            ),
            AIMessage(
                content=make_classification(
                    [question], ["refund_return"]
                ).model_dump_json()
            ),
        ]
        service.model = OfflineUsageModel(
            Model([AIMessage(content="这是隔离恢复验证中的普通聊天回复。")] * 20)
        )
        return original_accept(cid, rid, question, mode, **kw)

    service.accept = accept
    clock = SimpleNamespace(now=NOW)
    service.refund_clock = lambda: clock.now
    sub = service.refund_runtime.submissions
    sub.owner_id = service.owner_id
    sub.clock = lambda: clock.now
    blocked = Event()
    blocked.set()
    publish_failure = Event()
    original_fault = sub._fault

    def fault(point):
        if point == "refund_before_revalidation":
            require(blocked.wait(90), "Owned UI block timed out")
        if point == "refund_before_submission_publish" and publish_failure.is_set():
            publish_failure.clear()
            raise OSError("Task-isolated publication fault")
        return original_fault(point)

    sub._fault = fault

    @asynccontextmanager
    async def lifespan(app):
        app.state.refund_workers = refunds.RefundWorkers()
        try:
            yield
        finally:
            blocked.set()
            app.state.refund_workers.close()
            service._fixture_cleanup.close()

    app = FastAPI(lifespan=lifespan)
    app.state.conversation_runtime = service
    app.state.understanding_runtime = service.understanding
    app.state.refund_wait_seconds = 0.05
    app.state.refund_identity_context_provider = lambda request: {}

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    for router in (
        agent.router,
        chat.router,
        conversations.router,
        refunds.router,
        understandings.router,
    ):
        app.include_router(router, prefix="/api")
    requests = []

    @app.middleware("http")
    async def record(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            requests.append(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                }
            )
            save_json(target / "http-requests.json", requests)
        return response

    nonce = uuid4().hex

    @app.post("/__refund_fixture/" + nonce + "/{action}")
    def control(action: str):
        if action == "block":
            blocked.clear()
        elif action == "release":
            blocked.set()
        elif action == "expire":
            clock.now += timedelta(minutes=31)
        elif action == "fail_publish_once":
            publish_failure.set()
        else:
            raise ValueError("Unknown task control")
        return {"task_fixture": True, "action": action}

    @app.get("/__refund_fixture/" + nonce + "/counts")
    def counts():
        with resources.business_factory() as session:
            apps = list(
                session.scalars(
                    select(RefundApplication).where(
                        RefundApplication.order_source
                        == service.refund_profile.adapter.namespace
                    )
                )
            )
        return {"applications": len(apps), "ids": [a.application_id for a in apps]}

    app.mount("/", StaticFiles(directory=dist, html=True), name="verified-frontend")
    return app, nonce


def serve_http(ownership, target, snapshot_report):
    import socket

    import uvicorn

    dist, receipt = frontend_dist(snapshot_report)
    target = new_report_dir(ownership, target)
    before = snapshot_formal_resources()
    save_json(target / "preservation-before.json", before)
    with owned_refund_resources(ownership) as resources:
        app, nonce = make_http_app(resources, target, dist)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port = listener.getsockname()[1]
            save_json(
                target / "server.json",
                {
                    "url": f"http://127.0.0.1:{port}",
                    "control_base": f"http://127.0.0.1:{port}/__refund_fixture/{nonce}",
                    "pid": os.getpid(),
                    "fixture": True,
                    "frontend": receipt,
                },
            )
            server = uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
            )
            try:
                server.run(sockets=[listener])
            finally:
                after = snapshot_formal_resources(layout=before)
                save_json(target / "preservation-after.json", after)
                save_json(
                    target / "server-finished.json",
                    {"formal_preserved": before == after, "frontend": receipt},
                )


def recovery_destination(correction, commit, hashes):
    import re

    public = ROOT / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    if correction is None:
        destination = public / "R3.json"
        lineage = None
    else:
        prior = Path(correction).resolve()
        match = re.fullmatch(r"R3(?:-corrected(?:-(\d+))?)?\.json", prior.name)
        require(
            prior.parent == public.resolve() and match,
            "Only retained R3 evidence can authorize correction",
        )
        original = json.loads(prior.read_text())
        require(
            original.get("gate") == "R3"
            and original.get("candidate_commit") != commit
            and original.get("source_fingerprints")
            and original["source_fingerprints"] != hashes,
            "Committed changed sources required",
        )
        if original.get("passed"):
            validate_recovery_report(original)
        number = 1 if prior.name == "R3.json" else int(match[1] or "1") + 1
        destination = public / (
            "R3-corrected.json" if number == 1 else f"R3-corrected-{number}.json"
        )
        lineage = canonical_hash(original)
    require(not destination.exists(), "Refusing to overwrite actual R3 evidence")
    return destination, lineage


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("recovery", "http"))
    p.add_argument("--ownership", type=Path)
    p.add_argument("--report-dir", type=Path)
    p.add_argument("--frontend-snapshot-report", type=Path)
    p.add_argument("--correction-of", type=Path)
    p.add_argument("--child-state", type=Path)
    p.add_argument("--child-mode", choices=("crash", "resume"))
    a = p.parse_args(argv)
    if a.child_state:
        return child(a.child_state, a.child_mode)
    require(
        a.mode and a.ownership and a.report_dir,
        "Explicit owned gate arguments required",
    )
    commit, hashes = committed_sources(
        tuple(
            dict.fromkeys(
                (
                    *FLOW_SOURCES,
                    "scripts/smoke_refund_flow_recovery.py",
                    *(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("*.py")),
                )
            )
        )
    )
    if a.mode == "http":
        serve_http(a.ownership, a.report_dir, a.frontend_snapshot_report)
        return 0
    public, lineage = recovery_destination(a.correction_of, commit, hashes)
    report = run_recovery_matrix(ownership=a.ownership, report_dir=a.report_dir)
    report.update(
        candidate_commit=commit, source_fingerprints=hashes, correction_of=lineage
    )
    require(not public.exists(), "Immutable R3 evidence already exists")
    save_json(public, report)
    print(json.dumps({"passed": report["passed"], "windows": len(report["windows"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
