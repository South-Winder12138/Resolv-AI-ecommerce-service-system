"""Opt-in MySQL and actual process-death recovery with deterministic inference."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import os
import subprocess
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx2
from fastapi.testclient import TestClient
from scripts.logistics_gate_support import (
    SOURCE_ROOT,
    logistics_child,
    network_events,
    new_logistics_report_dir,
    owned_logistics_resources,
    owned_profile,
    save_json,
    serve_owned_app,
    snapshot_logistics_protection,
    validate_logistics_artifact,
    validate_logistics_ownership,
    validate_service_port,
)
from scripts.smoke_logistics_mcp import counts, require
from sqlalchemy import event, func, select
from sqlalchemy.exc import SQLAlchemyError

from resolv.models.conversation import (
    ConversationAction,
    ConversationMessage,
    ConversationRun,
)
from resolv.models.logistics import ConversationLogisticsAttempt as Attempt
from resolv.repositories.conversation import ConversationRepository, fingerprint
from resolv.repositories.logistics import LogisticsAttemptRepository
from resolv.schemas.conversation import ConversationError
from resolv.services.logistics_client import MCPLogisticsClient
from resolv.services.logistics_demo import create_logistics_demo_app
from resolv.services.logistics_execution import project_logistics_observation

WINDOWS = {"C1": "after_attempt_prepared", "C2": "after_remote_result", "C3": "after_result_commit"}


def validate_logistics_recovery_report(report):
    require(report.get("format") == "logistics-recovery-v1", "Unknown recovery report")
    require([w.get("case") for w in report.get("windows", [])] == list(WINDOWS), "All three ordered crash windows required")
    require(report.get("storage_dialect") == "mysql", "Actual MySQL evidence required")
    require(report.get("real_model_calls") == 0 and report.get("formal_preserved") is True, "Models/formal protection failed")
    require(report.get("protection_before") and report["protection_before"] == report.get("protection_after"), "Protection snapshots differ")
    for row, before, after in zip(report["windows"], (0, 1, 1), (1, 2, 1), strict=True):
        require(row.get("hook") == WINDOWS[row["case"]] and row.get("reached") is True and row.get("exit_code") == 86 and row.get("resume_exit_code") == 0, "Actual hook/death/resume evidence required")
        require(row.get("death_evidence", {}).get("hook") == row["hook"] and row["death_evidence"].get("attempt_id"), "Missing durable death evidence")
        require((row.get("remote_calls_before"), row.get("remote_calls"), row.get("api_requests"), row.get("network_event_count")) == (before, after, after, after), "Crash window HTTP counts differ")
        require(row.get("attempt_states") == (["completed"] if row["case"] == "C3" else ["unknown", "completed"]), "Unknown attempt history lost")
        require(row.get("refresh_extra_calls") == 0 and row.get("assistant_count") == 1, "Refresh redispatched or answer duplicated")
        require(all(row.get(k) is True for k in ("original_attempt_preserved", "result_hash_verified", "prepared_durable")), "Durable proof missing")
    probes = report.get("probes", {})
    require(set(probes) == {"atomic_rollback", "commit_ack_unknown", "stale_epoch", "concurrent_resume"}, "All storage probes required")
    require(all(p.get("passed") is True for p in probes.values()), "Storage probe failed")
    require(probes["atomic_rollback"].get("action_pending") is True and probes["atomic_rollback"].get("completed_attempts") == 0, "Atomic rollback not proven")
    require(probes["commit_ack_unknown"].get("fault_reached") is True and probes["commit_ack_unknown"].get("remote_calls") == probes["commit_ack_unknown"].get("completed_attempts") == 1, "Commit acknowledgement not proven")
    require(probes["stale_epoch"].get("rejected") is True and probes["stale_epoch"].get("action_pending") is True, "Epoch fencing not proven")
    require(probes["concurrent_resume"].get("requests") == 2 and probes["concurrent_resume"].get("remote_calls") == probes["concurrent_resume"].get("completed_attempts") == 1, "Concurrent resume not proven")


def _window(ownership, directory, case):
    obj = validate_logistics_ownership(ownership)
    base = f"http://127.0.0.1:{obj['ports']['resolv']}"
    state_file = directory / (case + "-state.json")
    death_file = directory / (case + "-death.json")
    before_counts = counts(network_events(ownership))
    with logistics_child(ownership, directory, role="resolv", worker_module="scripts.smoke_logistics_recovery", extra=("--crash-hook", WINDOWS[case], "--death-file", str(death_file))) as child:
        with httpx2.Client(trust_env=False, timeout=30) as http:
            conversation = http.post(base + "/conversations").json()["conversation_id"]
            request_id = "crash-" + uuid4().hex
            payload = {"session_id": conversation, "request_id": request_id, "question": "查询订单 MOCK001 的物流"}
            try:
                http.post(base + "/agent/chat", json=payload)
            except httpx2.HTTPError:
                pass  # Actual exit and durable hook are independently checked below.
        require(child.wait(timeout=10) == 86, "Crash child did not exit86")
    require(death_file.is_file(), "Crash hook was not reached")
    death = json.loads(death_file.read_text())
    with owned_logistics_resources(ownership) as owned, owned.conversation_factory() as session:
        run = session.scalar(select(ConversationRun).where(ConversationRun.conversation_id == conversation, ConversationRun.request_id == request_id))
        require(run is not None, "Crash run missing")
        attempt = session.get(Attempt, death["attempt_id"])
        require(attempt is not None and attempt.run_id == run.run_id, "Prepared audit not durable")
        original = {"attempt_id": attempt.attempt_id, "started_at": attempt.started_at.isoformat(), "arguments_hash": attempt.arguments_hash, "config_hash": attempt.config_hash}
        state = {"conversation_id": conversation, "run_id": run.run_id, "payload": payload}
    save_json(state_file, state)
    after_crash_counts = counts(network_events(ownership))
    with logistics_child(ownership, directory, role="resolv"):
        with httpx2.Client(trust_env=False, timeout=30) as http:
            refresh = http.get(base + "/conversations/" + conversation)
            require(refresh.status_code == 200, "Crash refresh failed")
            http.post(base + "/agent/chat", json=payload)
        require(counts(network_events(ownership)) == after_crash_counts, "Refresh/request replay dispatched remote read")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT / "src") + os.pathsep + str(SOURCE_ROOT)
        commands = []
        for index in range(2 if case == "C1" else 1):
            output = directory / f"{case}-resume-{index}.json"
            commands.append(subprocess.Popen([sys.executable, "-m", "scripts.smoke_logistics_recovery", "--ownership", str(ownership), "--resume-client", str(state_file), "--client-result", str(output)], cwd=SOURCE_ROOT, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        exits = [p.wait(timeout=40) for p in commands]
        require(exits == [0] * len(exits), "Resume client failed")
        client_results = [json.loads((directory / f"{case}-resume-{index}.json").read_text()) for index in range(len(commands))]
        require(any(r["completed"] for r in client_results), "No resume completed")
    final_counts = counts(network_events(ownership))
    with owned_logistics_resources(ownership) as owned, owned.conversation_factory() as session:
        attempts = session.scalars(select(Attempt).where(Attempt.run_id == state["run_id"]).order_by(Attempt.attempt_no)).all()
        require(attempts and attempts[-1].state == "completed", "No completed attempt after resume")
        completed = attempts[-1]
        action = session.get(ConversationAction, completed.action_id)
        from resolv.schemas.logistics import ShipmentEnvelope

        envelope = ShipmentEnvelope.model_validate_json(json.dumps(completed.envelope))
        hash_verified = fingerprint(completed.envelope) == completed.result_hash and action.result == project_logistics_observation(envelope, attempt_id=completed.attempt_id)
        assistant_count = session.scalar(select(func.count()).select_from(ConversationMessage).where(ConversationMessage.run_id == state["run_id"], ConversationMessage.role == "assistant"))
        preserved = original == {"attempt_id": attempts[0].attempt_id, "started_at": attempts[0].started_at.isoformat(), "arguments_hash": attempts[0].arguments_hash, "config_hash": attempts[0].config_hash}
        row = {"case": case, "hook": WINDOWS[case], "reached": death["hook"] == WINDOWS[case], "exit_code": 86, "resume_exit_code": max(exits),
            "remote_calls_before": after_crash_counts[0] - before_counts[0], "remote_calls": final_counts[0] - before_counts[0],
            "api_requests": final_counts[1] - before_counts[1], "network_event_count": final_counts[0] - before_counts[0],
            "refresh_extra_calls": 0, "assistant_count": assistant_count, "attempt_states": [a.state for a in attempts],
            "original_attempt_preserved": preserved, "result_hash_verified": hash_verified, "prepared_durable": True,
            "death_evidence": death, "concurrent_requests": len(commands), "client_results": client_results}
    return row


def _storage_probes(ownership, concurrent):
    probes = {"concurrent_resume": {"passed": concurrent["concurrent_requests"] == 2 and concurrent["remote_calls"] == 1,
        "requests": concurrent["concurrent_requests"], "remote_calls": concurrent["remote_calls"], "completed_attempts": concurrent["attempt_states"].count("completed")}}
    app = create_logistics_demo_app(ownership)
    with TestClient(app):
        service = app.state.conversation_runtime

        def new_action(execution, run):
            with service.session_factory.begin() as session:
                return ConversationRepository(session, service.owner_id).stage_action(run.run_id, execution.epoch, 1, "probe", "query_logistics", "read", {"tracking_no": "MOCKSF001"})

        def new_run():
            c = service.create()
            return service.accept(c.conversation_id, uuid4().hex, "查询运单 MOCKSF001 的物流", "agent")[0]

        run = new_run()
        with service.execution(run.run_id) as execution:
            action = new_action(execution, run)
            with service.session_factory.begin() as session:
                attempt = LogisticsAttemptRepository(session, service.owner_id).prepare(action.action_id, execution.epoch, config_hash=fingerprint(service.get_run(run.run_id).business_config), tracking_no="MOCKSF001")
            result = MCPLogisticsClient(owned_profile(ownership)).query("MOCKSF001", request_id=attempt.attempt_id)
            try:
                with service.session_factory.begin() as session:
                    LogisticsAttemptRepository(session, service.owner_id).complete(attempt.attempt_id, execution.epoch, result, project_logistics_observation(result, attempt_id=attempt.attempt_id))
                    raise RuntimeError("rollback gate")
            except RuntimeError as error:
                require(str(error) == "rollback gate", "Unexpected rollback error")
            with service.session_factory() as session:
                pending = session.get(ConversationAction, action.action_id).execution_status == "pending"
                count = session.scalar(select(func.count()).select_from(Attempt).where(Attempt.action_id == action.action_id, Attempt.state == "completed"))
                probes["atomic_rollback"] = {"passed": pending and count == 0, "action_pending": pending, "completed_attempts": count}
        run = new_run()
        with service.execution(run.run_id) as execution:
            action = new_action(execution, run)
            reached = []

            def flushed(session, context):
                if any(isinstance(a, Attempt) and a.state == "completed" for a in session.dirty):
                    session.info["logistics_gate_ack"] = True

            def committed(session):
                if session.info.pop("logistics_gate_ack", False):
                    reached.append(True)
                    raise SQLAlchemyError("gate acknowledgement lost")

            initial = counts(network_events(ownership))[0]
            event.listen(service.session_factory.class_, "after_flush", flushed)
            event.listen(service.session_factory.class_, "after_commit", committed)
            try:
                service.logistics_read_service.read(action.action_id, execution)
            finally:
                event.remove(service.session_factory.class_, "after_flush", flushed)
                event.remove(service.session_factory.class_, "after_commit", committed)
            with service.session_factory() as session:
                count = session.scalar(select(func.count()).select_from(Attempt).where(Attempt.action_id == action.action_id, Attempt.state == "completed"))
            probes["commit_ack_unknown"] = {"passed": bool(reached) and count == 1, "fault_reached": bool(reached), "completed_attempts": count, "remote_calls": counts(network_events(ownership))[0] - initial}
        run = new_run()
        with service.execution(run.run_id) as execution:
            action = new_action(execution, run)
            with service.session_factory.begin() as session:
                repo = LogisticsAttemptRepository(session, service.owner_id)
                attempt = repo.prepare(action.action_id, execution.epoch, config_hash=fingerprint(service.get_run(run.run_id).business_config), tracking_no="MOCKSF001")
            with service.session_factory.begin() as session:
                session.get(ConversationRun, run.run_id).execution_epoch += 1
            rejected = False
            try:
                with service.session_factory.begin() as session:
                    LogisticsAttemptRepository(session, service.owner_id).mark_unknown(attempt.attempt_id, execution.epoch)
            except ConversationError as error:
                rejected = error.code == "conversation_busy"
            with service.session_factory() as session:
                pending = session.get(ConversationAction, action.action_id).execution_status == "pending"
            probes["stale_epoch"] = {"passed": rejected and pending, "rejected": rejected, "action_pending": pending}
    return probes


def run_logistics_recovery_matrix(ownership, report_dir):
    before = snapshot_logistics_protection()
    directory = new_logistics_report_dir(ownership, report_dir)
    report = {"format": "logistics-recovery-v1", "windows": [], "probes": {}, "real_model_calls": 0, "formal_preserved": False, "protection_before": fingerprint(before)}
    try:
        identity = validate_logistics_ownership(ownership)
        report.update(source_revision=identity["source_revision"], source_root=identity["source_root"])
        with owned_logistics_resources(ownership) as owned:
            report["storage_dialect"] = owned.conversation_factory.kw["bind"].dialect.name
        with logistics_child(ownership, directory, role="api", worker_module="scripts.smoke_logistics_recovery", extra=("--delay-api",)), logistics_child(ownership, directory, role="mcp"):
            for case in WINDOWS:
                report["windows"].append(_window(ownership, directory, case))
                save_json(directory / "progress.json", report)
            report["probes"] = _storage_probes(ownership, report["windows"][0])
        after = snapshot_logistics_protection(layout=before)
        report.update(protection_after=fingerprint(after), formal_preserved=before == after)
        validate_logistics_recovery_report(report)
        report["status"] = "passed"
        return report
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        save_json(directory / "report.json", report)


def _worker(args):
    role = "api" if args.delay_api else "resolv"
    validate_service_port(args.ownership, role, args.port)
    if args.delay_api:
        from scripts.run_mock_logistics import create_app

        inner = create_app(args.ownership, args.port)

        async def delayed(scope, receive, send):
            if scope["type"] == "http" and scope["path"].startswith("/v1/shipments/"):
                await asyncio.sleep(0.25)
            await inner(scope, receive, send)

        app = delayed
    else:
        validate_logistics_artifact(args.ownership, args.death_file)
        app = create_logistics_demo_app(args.ownership)
        owner = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with owner(app):
                def crash(stage, attempt):
                    if stage == args.crash_hook:
                        save_json(args.death_file, {"hook": stage, "attempt_id": attempt.attempt_id})
                        os._exit(86)

                app.state.conversation_runtime.logistics_read_service.fault_hook = crash
                yield

        app.router.lifespan_context = lifespan
    serve_owned_app(app, args.ownership, role, args.port)


def _resume_client(args):
    obj = validate_logistics_ownership(args.ownership)
    validate_logistics_artifact(args.ownership, args.resume_client)
    validate_logistics_artifact(args.ownership, args.client_result)
    state = json.loads(args.resume_client.read_text())
    url = f"http://127.0.0.1:{obj['ports']['resolv']}/conversations/{state['conversation_id']}/runs/{state['run_id']}/resume"
    with httpx2.Client(trust_env=False, timeout=30) as http:
        result = http.post(url)
    completed = result.status_code == 200 and '"status":"completed"' in result.text.replace(" ", "")
    busy = result.status_code == 409 and "conversation_busy" in result.text
    # SSE can carry an error while HTTP status remains200.
    busy = busy or (result.status_code == 200 and '"code":"conversation_busy"' in result.text.replace(" ", ""))
    save_json(args.client_result, {"completed": completed, "busy": busy, "http_status": result.status_code})
    require(completed or busy, "Unexpected resume response")


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--ownership", required=True, type=Path)
    value.add_argument("--report-dir", type=Path)
    value.add_argument("--port", type=int)
    value.add_argument("--crash-hook", choices=list(WINDOWS.values()))
    value.add_argument("--death-file", type=Path)
    value.add_argument("--delay-api", action="store_true")
    value.add_argument("--resume-client", type=Path)
    value.add_argument("--client-result", type=Path)
    return value


def main():
    args = parser().parse_args()
    if args.resume_client:
        _resume_client(args)
    elif args.crash_hook or args.delay_api:
        _worker(args)
    else:
        require(args.report_dir is not None, "report-dir required")
        run_logistics_recovery_matrix(args.ownership, args.report_dir)


if __name__ == "__main__":
    main()
