"""Single approved L1-L6 execution with fsynced reservations and HTTP permits."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import fcntl
import json
import re
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import datetime
from hashlib import sha256
from threading import RLock
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import httpx2
from scripts.logistics_gate_support import (
    REPOSITORY_ROOT,
    SOURCE_ROOT,
    _read_json,
    logistics_child,
    network_events,
    new_logistics_report_dir,
    owned_logistics_resources,
    save_json,
    serve_owned_app,
    snapshot_logistics_protection,
    validate_logistics_ownership,
    validate_service_port,
)
from scripts.smoke_logistics_mcp import counts, require, validate_wire_report
from scripts.smoke_logistics_recovery import validate_logistics_recovery_report
from sqlalchemy import func, select

from resolv.models.conversation import (
    Conversation,
    ConversationAction,
    ConversationMessage,
    ConversationRun,
)
from resolv.models.logistics import ConversationLogisticsAttempt as Attempt
from resolv.models.understanding import ConversationUnderstanding
from resolv.repositories.conversation import fingerprint
from resolv.services.logistics_demo import create_logistics_demo_app

CASES_PATH = SOURCE_ROOT / "tests/fixtures/logistics_model_cases.json"
PERMIT = ContextVar("logistics_model_http_permit", default=None)


class LogisticsModelBudget:
    ledger_filename = "budget.json"
    call_limit = 48
    case_ids = tuple(f"L{i}" for i in range(1, 7))
    report_format = "logistics-model-v1"
    continuation_flag = None

    def __init__(self, ownership, revision):
        self.ownership = Path(ownership)
        self.identity = validate_logistics_ownership(ownership)
        require(revision == self.identity["source_revision"], "Model candidate differs from ownership")
        self.revision, self.path = revision, self.ownership.parent / self.ledger_filename
        self.cases = [c for c in json.loads(CASES_PATH.read_text())["cases"] if c["id"] in self.case_ids]
        self._mutex = RLock()

    @contextmanager
    def transaction(self):
        with self._mutex, self.path.with_suffix(".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            require(validate_logistics_ownership(self.ownership)["source_revision"] == self.revision, "Model budget source changed")
            value = _read_json(self.path)
            self.verify_references(value)
            yield value
            save_json(self.path, value)

    def read(self):
        with self._mutex:
            return _read_json(self.path)

    def verify_references(self, value):
        pass

    def open(self):
        with self.transaction() as value:
            require(value.get("state") == "unopened" and not value.get("calls"), "Terminal/used model ledger cannot be reused")
            value.update(state="active", source_revision=self.revision, cases_hash=fingerprint(self.cases),
                active_case=None, http_calls=0, cases={c["id"]: {"state": "not_started"} for c in self.cases})

    def begin_case(self, case_id):
        with self.transaction() as value:
            require(value["state"] == "active" and value["active_case"] is None, "Model ledger not ready")
            remaining = [c["id"] for c in self.cases if value["cases"][c["id"]]["state"] == "not_started"]
            require(remaining and remaining[0] == case_id, "Model case order/once rule violated")
            previous = [c["id"] for c in self.cases if c["id"] != case_id and self.cases.index(c) < next(i for i, r in enumerate(self.cases) if r["id"] == case_id)]
            require(all(value["cases"][c]["state"] == "passed" for c in previous), "Previous model case failed")
            value["cases"][case_id] = {"state": "running"}
            value["active_case"] = case_id

    def reserve(self, purpose):
        with self.transaction() as value:
            case_id = value.get("active_case")
            require(value.get("state") == "active" and case_id, "No active approved model case")
            require(purpose in {"resolution", "intents", "business"}, "Forbidden model purpose")
            rows = [r for r in value["calls"] if r["case_id"] == case_id and r["purpose"] == purpose]
            require(len(rows) < (6 if purpose == "business" else 1) and len(value["calls"]) < self.call_limit, "Model budget exhausted")
            permit = {"id": uuid4().hex, "case_id": case_id, "purpose": purpose, "revision": self.revision}
            value["calls"].append({**permit, "state": "reserved", "http_sends": 0})
            return permit

    def consume(self, permit):
        with self.transaction() as value:
            require(permit is not None and permit.get("revision") == self.revision and permit.get("case_id") == value.get("active_case") and value.get("state") == "active", "Unreserved provider HTTP request")
            row = next((r for r in value["calls"] if r["id"] == permit["id"]), None)
            require(row is not None and row["state"] == "reserved" and row["http_sends"] == 0 and value["http_calls"] < self.call_limit, "Additional provider send forbidden")
            row.update(state="http_started", http_sends=1)
            value["http_calls"] += 1

    def http_result(self, permit, *, status=None, failed=False):
        with self.transaction() as value:
            row = next(r for r in value["calls"] if r["id"] == permit["id"])
            row.update(state="transport_failed" if failed else "http_completed", http_status=status)

    def invocation_result(self, permit, *, passed):
        with self.transaction() as value:
            row = next(r for r in value["calls"] if r["id"] == permit["id"])
            row["invocation_outcome"] = "completed" if passed else "failed"

    def finish_case(self, case_id, *, passed):
        with self.transaction() as value:
            require(value.get("state") == "active" and value.get("active_case") == case_id, "Model case not active")
            value["cases"][case_id]["state"] = "passed" if passed else "failed"
            value["active_case"] = None
            if not passed:
                value["state"] = "closed"
                for case in value["cases"].values():
                    if case["state"] == "not_started":
                        case.update(state="skipped", reason="first_unexpected_failure")
            elif all(c["state"] == "passed" for c in value["cases"].values()):
                value["state"] = "completed"


class LogisticsContinuationBudget(LogisticsModelBudget):
    ledger_filename = "budget-C1.json"
    call_limit = 40
    case_ids = ("L2", "L3", "L4", "L5", "L6")
    parent_http_calls = 5
    report_format = "logistics-model-C1-v1"
    continuation_flag = "--continuation-C1"

    def __init__(self, ownership, revision):
        super().__init__(ownership, revision)
        root = self.ownership.parent
        self.parent_paths = (root / "budget.json", root / "model-once/report.json", root / "L1-adjudication.json")
        parent, report, receipt = (_read_json(p) for p in self.parent_paths)
        require(parent.get("state") == "closed" and parent.get("http_calls") == 5 and parent.get("max_model_calls") == 48 and parent.get("retrievals") == 0, "C1 requires original closed five-call ledger")
        require(len(parent.get("calls", [])) == 5 and all(c.get("case_id") == "L1" and c.get("http_sends") == 1 for c in parent["calls"]), "Parent attempts differ")
        require(all(parent.get("cases", {}).get(c, {}).get("state") == "skipped" for c in self.case_ids), "C1 case already attempted")
        require(report.get("budget") == parent and report.get("source_revision") == parent.get("source_revision"), "Parent report differs")
        require(receipt.get("status") == "passed_from_existing_receipt" and receipt.get("new_model_calls") == 0 and receipt.get("model_calls") == 5
            and receipt.get("source_revision") == parent["source_revision"] and receipt.get("original_report_sha256") == sha256(self.parent_paths[1].read_bytes()).hexdigest(), "Invalid L1 adjudication")
        self.previous_evidence = receipt["evidence"]
        require(self.previous_evidence.get("model_calls") == parent["calls"], "L1 calls differ from parent")
        validate_model_case(json.loads(CASES_PATH.read_text())["cases"][0], self.previous_evidence)
        self.references = self._reference_hashes()
        if self.path.exists():
            self.verify_references(_read_json(self.path))

    def _reference_hashes(self):
        return {str(p.relative_to(self.ownership.parent)): sha256(p.read_bytes()).hexdigest() for p in self.parent_paths}

    def verify_references(self, value):
        require(value.get("parent_hashes") == self.references == self._reference_hashes(), "C1 immutable parent reference changed")
        require(value.get("source_revision") == self.revision and value.get("max_model_calls") == 40, "C1 candidate/budget differs")
        require(value.get("http_calls", 0) + 5 <= 48, "Combined model limit exceeded")

    def prepare(self):
        with self._mutex, self.path.with_suffix(".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            require(not self.path.exists(), "C1 ledger already exists; cannot reuse")
            require(self.references == self._reference_hashes(), "C1 parent changed before preparation")
            save_json(self.path, {"state": "unopened", "max_model_calls": 40, "retrievals": 0, "calls": [], "cases": {},
                "http_calls": 0, "source_revision": self.revision, "parent_hashes": self.references})


class LogisticsFinalCasesBudget(LogisticsModelBudget):
    ledger_filename = "budget-C2.json"
    call_limit = 24
    case_ids = ("L4", "L5", "L6")
    parent_http_calls = 14
    report_format = "logistics-model-C2-v1"
    continuation_flag = "--continuation-C2"

    def __init__(self, ownership, revision):
        super().__init__(ownership, revision)
        root = self.ownership.parent
        self.parent_paths = tuple(root / name for name in (
            "budget.json", "model-once/report.json", "L1-adjudication.json", "budget-C1.json", "model-C1/report.json", "L1-L3-final-adjudication.json",
        ))
        parent, original, l1, child, continuation, receipt = (_read_json(p) for p in self.parent_paths)
        for ledger, report, count, maximum in ((parent, original, 5, 48), (child, continuation, 9, 40)):
            require(ledger.get("state") == "closed" and ledger.get("retrievals") == 0 and ledger.get("http_calls") == count
                and len(ledger.get("calls", [])) == count and ledger.get("max_model_calls") == maximum, "C2 requires the original closed 14-call history")
            require(report.get("budget") == ledger and report.get("source_revision") == ledger.get("source_revision"), "C2 parent report differs")
            require(all(ledger.get("cases", {}).get(case, {}).get("state") == "skipped" for case in self.case_ids), "C2 case already attempted")
        self.references = self._reference_hashes()
        require(receipt.get("format") == "logistics-receipt-adjudication-v1" and receipt.get("model_http_requests_added") == 0
            and receipt.get("cumulative_model_http_requests") == 14 and receipt.get("retrievals") == 0
            and receipt.get("preserved_hashes") == {k: v for k, v in self.references.items() if k != "L1-L3-final-adjudication.json"}, "C2 adjudication provenance differs")
        rows = receipt.get("cases", [])
        require([row.get("case") for row in rows] == ["L1", "L2", "L3"] and all(row.get("passed") is True for row in rows), "C2 requires three passing preserved receipts")
        previous = {"L1": l1.get("evidence"), **{row["case"]: row.get("evidence") for row in continuation.get("cases", [])}}
        fixed = {case["id"]: case for case in json.loads(CASES_PATH.read_text())["cases"]}
        for row, count in zip(rows, (5, 5, 4), strict=True):
            evidence = row["evidence"]
            require(evidence == previous.get(row["case"]), "C2 adjudicated evidence changed")
            calls = [call for call in (parent["calls"] + child["calls"]) if call["case_id"] == row["case"]]
            require(len(calls) == count and evidence.get("model_calls") == calls, "C2 previous sends differ")
            validate_model_case(fixed[row["case"]], evidence)
        self._extend_prior_history()
        self.references = self._reference_hashes()
        if self.path.exists():
            self.verify_references(_read_json(self.path))

    def _extend_prior_history(self):
        pass

    def _reference_hashes(self):
        return {str(p.relative_to(self.ownership.parent)): sha256(p.read_bytes()).hexdigest() for p in self.parent_paths}

    def verify_references(self, value):
        require(value.get("parent_hashes") == self.references == self._reference_hashes(), "Continuation immutable parent reference changed")
        require(value.get("source_revision") == self.revision and value.get("max_model_calls") == self.call_limit, "Continuation candidate/budget differs")
        require(value.get("http_calls", 0) <= self.call_limit and value.get("http_calls", 0) + self.parent_http_calls <= 38, "Continuation cumulative limit exceeded")

    def prepare(self):
        with self._mutex, self.path.with_suffix(".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            require(not self.path.exists(), "Continuation ledger already exists; cannot reuse")
            require(self.references == self._reference_hashes(), "Continuation parent changed before preparation")
            save_json(self.path, {"state": "unopened", "max_model_calls": self.call_limit, "retrievals": 0, "calls": [], "cases": {},
                "http_calls": 0, "source_revision": self.revision, "parent_hashes": self.references})


class LogisticsLastTwoCasesBudget(LogisticsFinalCasesBudget):
    ledger_filename = "budget-C3.json"
    call_limit = 16
    case_ids = ("L5", "L6")
    parent_http_calls = 18
    report_format = "logistics-model-C3-v1"
    continuation_flag = "--continuation-C3"

    def _extend_prior_history(self):
        root = self.ownership.parent
        ledger_path, report_path, receipt_path = (root / name for name in ("budget-C2.json", "model-C2/report.json", "L4-adjudication.json"))
        ledger, report, receipt = (_read_json(p) for p in (ledger_path, report_path, receipt_path))
        require(ledger.get("state") == "closed" and ledger.get("http_calls") == 4 and len(ledger.get("calls", [])) == 4
            and ledger.get("max_model_calls") == 24 and ledger.get("retrievals") == 0 and ledger.get("parent_hashes") == self.references, "C3 requires preserved closed L4 history")
        require(report.get("budget") == ledger and report.get("source_revision") == ledger.get("source_revision")
            and report.get("cumulative_model_http_calls") == 18 and [r.get("case") for r in report.get("cases", [])] == ["L4"], "C3 original L4 report differs")
        require(all(ledger.get("cases", {}).get(case, {}).get("state") == "skipped" for case in self.case_ids), "C3 case already attempted")
        require(receipt.get("format") == "logistics-case-adjudication-v1" and receipt.get("case") == "L4"
            and receipt.get("status") == "passed_from_existing_receipt" and receipt.get("new_model_calls") == 0 and receipt.get("model_calls") == 4
            and receipt.get("original_report_sha256") == sha256(report_path.read_bytes()).hexdigest(), "C3 L4 adjudication differs")
        evidence = receipt["evidence"]
        require(evidence == report["cases"][0].get("evidence") and evidence.get("model_calls") == ledger["calls"]
            and all(call.get("case_id") == "L4" for call in ledger["calls"]), "C3 L4 receipt changed")
        case = next(c for c in json.loads(CASES_PATH.read_text())["cases"] if c["id"] == "L4")
        validate_model_case(case, evidence)
        self.parent_paths += (ledger_path, report_path, receipt_path)


def _model_budget_type(c1=False, c2=False, c3=False):
    choices = [cls for enabled, cls in ((c1, LogisticsContinuationBudget), (c2, LogisticsFinalCasesBudget), (c3, LogisticsLastTwoCasesBudget)) if enabled]
    require(len(choices) <= 1, "Continuation modes are mutually exclusive")
    return choices[0] if choices else LogisticsModelBudget


def validate_persisted_l1(ownership, ledger):
    expected = ledger.previous_evidence
    with owned_logistics_resources(ownership) as owned, owned.conversation_factory() as session:
        run = session.get(ConversationRun, expected["run_id"])
        require(run is not None and run.status == "completed", "Persisted L1 run missing/completed state changed")
        conversation = session.get(Conversation, run.conversation_id)
        message = session.get(ConversationMessage, run.answer_message_id)
        require(conversation is not None and conversation.owner_id == ledger.identity["owner_id"] and conversation.status == "active", "Original L1 conversation unavailable")
        require(message is not None and message.text == expected["answer"] and run.model_audits == expected["business_audits"] and (run.result_json or {}).get("branch") == "business", "Original L1 answer/audit changed")
        latest = session.scalar(select(ConversationMessage).where(ConversationMessage.conversation_id == run.conversation_id).order_by(ConversationMessage.seq.desc()).limit(1))
        require(conversation.current_run_id is None and latest is not None and latest.message_id == run.answer_message_id, "Original conversation latest L1 turn changed")
        require(run.business_config.get("logistics_backend") == _read_json(Path(ownership).parent / "profile.json"), "L1 logistics profile changed")
        actions = session.scalars(select(ConversationAction).where(ConversationAction.run_id == run.run_id).order_by(ConversationAction.model_step)).all()
        require([{"tool_name": a.tool_name, "effect": a.effect, "arguments": a.arguments, "result": a.result, "execution_status": a.execution_status} for a in actions] == expected["actions"], "L1 persisted tools changed")
        for proof in expected["attempts"]:
            attempt = session.get(Attempt, proof["attempt_id"])
            require(attempt is not None and attempt.run_id == run.run_id and attempt.state == "completed" and attempt.envelope == proof["envelope"] and fingerprint(attempt.envelope) == attempt.result_hash, "L1 persisted observation changed")
        return run.conversation_id


@contextmanager
def model_permit(value):
    token = PERMIT.set(value)
    try:
        yield
    finally:
        PERMIT.reset(token)


@contextmanager
def provider_http_guard(ledger, base_url):
    target = urlsplit(base_url)

    def provider(request):
        url = urlsplit(str(request.url))
        matches = (url.scheme, url.netloc) == (target.scheme, target.netloc) and url.path.startswith(target.path.rstrip("/") + "/")
        if not matches:
            require(PERMIT.get() is None, "Provider request redirected outside approved endpoint")
            require(url.hostname == "127.0.0.1" and url.port in ledger.identity["ports"].values(), "Unapproved HTTP destination")
            return False
        ledger.consume(PERMIT.get())
        return True

    with ExitStack() as stack:
        for library in (httpx, httpx2):
            # send() can perform multiple redirects/auth exchanges internally.
            # Both pinned clients dispatch each physical attempt through this method.
            original = library.Client._send_single_request
            original_async = library.AsyncClient._send_single_request

            def guarded(client, request, *args, _original=original, **kwargs):
                counted, permit = provider(request), PERMIT.get()
                try:
                    response = _original(client, request, *args, **kwargs)
                except BaseException:
                    if counted:
                        ledger.http_result(permit, failed=True)
                    raise
                if counted:
                    ledger.http_result(permit, status=response.status_code)
                return response

            async def guarded_async(client, request, *args, _original=original_async, **kwargs):
                counted, permit = provider(request), PERMIT.get()
                try:
                    response = await _original(client, request, *args, **kwargs)
                except BaseException:
                    if counted:
                        ledger.http_result(permit, failed=True)
                    raise
                if counted:
                    ledger.http_result(permit, status=response.status_code)
                return response

            stack.enter_context(patch.object(library.Client, "_send_single_request", guarded))
            stack.enter_context(patch.object(library.AsyncClient, "_send_single_request", guarded_async))
        yield


class LogisticsModelGuard:
    def __init__(self, ledger):
        self.ledger = ledger

    def wrap(self, purpose, client):
        if purpose == "understanding":
            return StageGuard(self.ledger, client)
        require(purpose == "business", "Forbidden model purpose")
        return BusinessGuard(self.ledger, client)


class StageGuard:
    def __init__(self, ledger, client):
        self.ledger, self.client = ledger, client

    def invoke(self, stage, *args, **kwargs):
        require(stage in {"resolution", "intents"}, "Forbidden understanding purpose")
        if args:
            payload = json.loads(args[0][-1].content.split("\n", 1)[1])
            active = self.ledger.read()["active_case"]
            expected = next(c["text"] for c in self.ledger.cases if c["id"] == active)
            require(payload["original_text"] == expected, "Frozen model input differs")
        permit, passed = self.ledger.reserve(stage), False
        try:
            with model_permit(permit):
                result = self.client.invoke(stage, *args, **kwargs)
            if stage == "intents":
                require(len(result[0].requests) == 1 and result[0].requests[0].intent == "logistics", "Wrong route; downstream retrieval/models forbidden")
            passed = True
            return result
        finally:
            self.ledger.invocation_result(permit, passed=passed)


class BusinessGuard:
    def __init__(self, ledger, client):
        self.ledger, self.client = ledger, client

    def bind_tools(self, tools):
        return BusinessGuard(self.ledger, self.client.bind_tools(tools))

    def bind(self, **kwargs):
        return BusinessGuard(self.ledger, self.client.bind(**kwargs))

    def stream(self, messages):
        permit, passed = self.ledger.reserve("business"), False
        try:
            with model_permit(permit):
                yield from self.client.stream(messages)
            passed = True
        finally:
            self.ledger.invocation_result(permit, passed=passed)


def run_logistics_model_cases(ownership, revision, report_dir, *, execute_approved_budget=False, continuation_c1=False, continuation_c2=False, continuation_c3=False):
    require(execute_approved_budget, "--execute-approved-budget required")
    ledger_type = _model_budget_type(continuation_c1, continuation_c2, continuation_c3)
    continuation = ledger_type is not LogisticsModelBudget
    identity = validate_logistics_ownership(ownership)
    require(revision == identity["source_revision"], "Candidate revision differs")
    validate_export_origin()
    gates = validate_model_prerequisites(ownership, revision)
    ledger = ledger_type(ownership, revision)
    conversation = validate_persisted_l1(ownership, ledger) if continuation_c1 else None
    if continuation:
        require(not ledger.path.exists(), "Continuation ledger already exists; cannot rerun")
    before = snapshot_logistics_protection()
    directory = new_logistics_report_dir(ownership, report_dir)
    report = {"format": ledger.report_format, "source_revision": revision, "source_root": str(SOURCE_ROOT),
        "gates": gates, "cases": [], "protection_before": fingerprint(before), "retrievals": 0}
    if continuation:
        ledger.prepare()
        report.update(parent_hashes=ledger.references, original_conversation_id=conversation)
    ledger.open()
    try:
        with logistics_child(ownership, directory, role="api"), logistics_child(ownership, directory, role="mcp"), logistics_child(
            ownership, directory, role="resolv", worker_module="scripts.smoke_logistics_model", extra=("--real-worker", "--execute-approved-budget", *((ledger.continuation_flag,) if continuation else ())),
        ):
            execute = LiveModelCases(ownership, ledger, directory, conversation=conversation)
            report["cases"] = execute_fixed_cases(ledger, execute, progress=lambda rows: save_json(directory / "progress.json", {"cases": rows}))
        after = snapshot_logistics_protection(layout=before)
        report.update(protection_after=fingerprint(after), formal_preserved=before == after)
        require(before == after, "Formal protection snapshot differs")
        report["status"] = "passed" if len(report["cases"]) == len(ledger.cases) and all(c["passed"] for c in report["cases"]) else "failed"
        return report
    except BaseException:
        report["status"] = "failed"
        with ledger.transaction() as value:
            if value["state"] == "active":
                value["state"] = "closed"
                for case in value["cases"].values():
                    if case["state"] in {"running", "not_started"}:
                        case.update(state="skipped", reason="controller_interrupted")
        raise
    finally:
        report["budget"] = ledger.read()
        report["cumulative_model_http_calls"] = report["budget"]["http_calls"] + getattr(ledger, "parent_http_calls", 0)
        save_json(directory / "report.json", report)


def validate_export_origin():
    require((SOURCE_ROOT / ".logistics-source.json").is_file(), "Live validation requires an immutable export")
    for name, module in tuple(sys.modules.items()):
        if name == "resolv" or name.startswith(("resolv.", "scripts.")):
            path = getattr(module, "__file__", None)
            if path:
                require(Path(path).resolve().is_relative_to(SOURCE_ROOT), "Project import escaped candidate export")


def validate_model_prerequisites(ownership, revision):
    offline_path = SOURCE_ROOT.parents[1] / "report.json"
    offline = json.loads(offline_path.read_bytes())
    require(offline.get("passed") is True and offline.get("candidate_commit") == revision and offline.get("exported_source") == str(SOURCE_ROOT), "Exact offline candidate gate missing")
    proofs = {"offline": str(offline_path)}
    for kind, validator in (("logistics-wire-v1", validate_wire_report), ("logistics-recovery-v1", validate_logistics_recovery_report)):
        for path in sorted(Path(ownership).parent.glob("*/report.json")):
            report = json.loads(path.read_bytes())
            if report.get("format") == kind and report.get("status") == "passed" and report.get("source_revision") == revision and report.get("source_root") == str(SOURCE_ROOT):
                validator(report)
                proofs[kind] = str(path)
                break
        require(kind in proofs, "Same-candidate wire/recovery gate missing")
    return proofs


def execute_fixed_cases(ledger, execute, *, progress=None):
    rows = []
    for case in ledger.cases:
        ledger.begin_case(case["id"])
        evidence = None
        try:
            evidence = execute(case)
            validate_model_case(case, evidence)
        except Exception as error:  # noqa: BLE001 -- retain safe failure class, never provider exception text
            rows.append({"case": case["id"], "passed": False, "failure_type": type(error).__name__, "evidence": evidence})
            if evidence is not None:
                rows[-1]["failed_check"] = str(error)  # Only our fixed acceptance assertions run after collection.
            ledger.finish_case(case["id"], passed=False)
            if progress:
                progress(rows)
            break
        else:
            rows.append({"case": case["id"], "passed": True, "evidence": evidence})
            ledger.finish_case(case["id"], passed=True)
            if progress:
                progress(rows)
    return rows


ANSWER_TIME = re.compile(
    r"\d{4}[-年/]\d{1,2}[-月/]\d{1,2}日?(?:[ T]*\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:UTC|Z|[+-]\d{2}:\d{2}))?)?"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
)
UNKNOWN_FACT = r"暂无|尚无|未知|未提供|没有|无.*记录|为空|未返回|无法|不详|无详细"
NEGATION = r"(?:不|未|没有|并非|不是|尚未|还没).{0,4}"


def _matches_saved_time(text, value):
    if value is None:
        return False
    saved = datetime.fromisoformat(value)
    normalized = text.replace("年", "-").replace("月", "-").replace("日", " ").replace("/", "-").strip()
    normalized = re.sub(r"\s*UTC$", "+00:00", normalized)
    try:
        if re.search(r"(?:Z|[+-]\d{2}:\d{2})$", normalized):
            parsed = datetime.fromisoformat(normalized)
            saved = saved.astimezone(parsed.tzinfo)
            if not re.search(r"\d{1,2}:\d{2}:\d{2}", normalized):
                return parsed.replace(second=0, microsecond=0) == saved.replace(second=0, microsecond=0)
            return parsed.replace(microsecond=0) == saved.replace(microsecond=0)
        formats = ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%H:%M", "%H:%M:%S")
        for fmt in formats:
            try:
                parsed = datetime.strptime(normalized, fmt)  # noqa: DTZ007 -- compare displayed parts in the saved time's zone
            except ValueError:
                continue
            return parsed.strftime(fmt) == saved.strftime(fmt)
        return False
    except ValueError:
        return False


def validate_answer_facts(case, evidence, answer):
    """Conservative fixed-case checks; retained prose still needs human review.

    These predicates catch concrete contradictory/unsupported claims. They are
    not a general Chinese semantic evaluator or a replacement for receipt review.
    """
    case_id = case["id"]
    require(not re.search(r"预计|将于|明天|后天|今天送达|今日送达", answer), "Unsupported delivery promise")
    attempts = evidence["attempts"]
    envelope = attempts[0]["envelope"] if attempts else None
    shipment = envelope.get("shipment") if envelope else None
    if shipment:
        known = [shipment.get("shipped_at"), shipment.get("signed_at"), envelope.get("observed_at")]
        known.extend(e["occurred_at"] for e in shipment.get("events", []))
        for match in ANSWER_TIME.finditer(answer):
            candidates = known
            if case_id == "L3":
                prefix = re.split(r"[，,。；;\n]", answer[:match.start()])[-1]
                # A known timestamp cannot justify attributing it to signed_at.
                labels = list(re.finditer(r"发货|寄出|查询|观察|签收|送达", prefix))
                label = labels[-1].group() if labels else None
                candidates = [shipment.get("shipped_at")] if label in {"发货", "寄出"} else [envelope.get("observed_at")] if label in {"查询", "观察"} else []
            require(any(_matches_saved_time(match.group(), value) for value in candidates), "Unsupported or misattributed time")
    if case_id in {"L1", "L2"}:
        require(any(word in answer for word in ("运输", "在途", "转运")), "In-transit semantics missing")
        require(not re.search(NEGATION + r"(?:运输|在途|转运|上海)", answer), "Negated observed transit/location")
        require(not re.search(r"已(?:经)?(?:签收|送达)", answer), "Contradictory delivered status")
        if case_id == "L2":
            require("上海" in answer, "Current location missing")
    elif case_id == "L3":
        require(shipment["signed_at"] is None and re.search(r"已(?:经)?(?:签收|送达)|delivered", answer), "Delivered status missing")
        require(not re.search(NEGATION + r"(?:签收|送达)(?!时间|日期|时刻)", answer), "Negated delivered status")
        require(not re.search(r"今天|昨天|今日|昨日|前天|上周|昨晚|今早|刚刚", answer), "Unsupported relative time")
        for clause in re.split(r"[，,。；;\n]", answer):
            if re.search(r"地点|位置|位于|经过|途经|到达|抵达|轨迹|节点", clause):
                require(re.search(UNKNOWN_FACT, clause), "Unknown location/trajectory invented")
    elif case_id in {"L4", "L5"}:
        for status in re.finditer(r"运输中|在途|转运|已(?:经)?(?:发货|签收|送达)", answer):
            prefix = re.split(r"[，,。；;？！?!\n]", answer[:status.start()])[-1]
            # A question or conditional recommendation does not assert a fact.
            non_assertion = re.search(r"(?:是否|有没有|有无)\s*$|(?:如果|假如|假设|若)[^，,。；;？！?!\n]*$", prefix)
            require(non_assertion, "Unobserved shipment status")
        if case_id == "L4":
            require(re.search(r"运单|物流|包裹", answer) and re.search(r"未(?:查询|查|检索|找)到|不存在|没有(?:查|找)到|找不到|查不到|暂无.{0,8}(?:运单|物流信息|物流记录)", answer), "Not-found semantics missing")
        else:
            require(evidence["actions"][0]["result"]["tracking_no"] is None and re.search(r"(?:暂无|尚无|尚未|没有|未生成|无可).{0,12}(?:运单|快递单|物流单)|(?:运单|快递单|物流单).{0,12}(?:暂无|尚无|尚未|没有|未生成|为空|未知)", answer), "No-tracking semantics missing")


def validate_model_case(case, evidence):
    require(evidence.get("error_code") == case.get("error_code"), "Unexpected model/tool error")
    require(evidence.get("mcp_calls") == evidence.get("api_requests") == case["mcp_calls"], "Business wire counts differ")
    require(evidence.get("understanding_status") == "ready" and evidence.get("intent") == "logistics", "Understanding/route failed")
    require(evidence.get("refresh_extra_calls") == 0 and evidence.get("replay_equal") is True and evidence.get("sse_equal") is True, "HTTP/SSE/readback/replay differs")
    require(evidence.get("versions") == ["routed-v4", 4, 3, 2] and evidence.get("input_budget") == 60000, "Runtime/config versions differ")
    calls = evidence.get("model_calls", [])
    require(calls and all(c.get("http_sends") == 1 and c.get("http_status") == 200 and c.get("state") == "http_completed" and c.get("invocation_outcome") == "completed" for c in calls), "Model transport/assembly failed")
    require([c["purpose"] for c in calls[:2]] == ["resolution", "intents"] and all(c["purpose"] == "business" for c in calls[2:]) and 3 <= len(calls) <= 8, "Model purposes/budget differ")
    audits = evidence.get("business_audits", [])
    expected_phases = (["order"] if case["id"] in {"L1", "L2", "L5", "L6"} else []) + (["logistics"] if case["mcp_calls"] else []) + ([] if case.get("error_code") else ["answer"])
    require([a.get("tool_policy", {}).get("phase") for a in audits] == expected_phases and len(audits) == len(calls) - 2, "Wrong phase/extra business rounds")
    for audit in audits:
        policy = audit["tool_policy"]
        require(audit.get("error_code") is None and audit.get("actual_input") is not None and audit.get("actual_output") is not None, "Missing successful usage audit")
        expected = [] if policy["phase"] == "answer" else ["query_" + ("order" if policy["phase"] == "order" else "logistics")]
        require(audit.get("returned_tool_names") == expected and list(policy["allowed_tool_names"]) == expected, "Actual native tools differ")
        # The business parser accepts complete native calls with stop or tool_calls.
        # Tool names/arguments, policy and durable observations prove execution.
        require(audit.get("finish_reason") in ({"stop", "tool_calls"} if expected else {"stop"}), "Model finish reason differs")
    actions = evidence.get("actions", [])
    expected_tools = ["query_order"] if case["id"] in {"L1", "L2", "L5", "L6"} else []
    if case["mcp_calls"]:
        expected_tools.append("query_logistics")
    require([a["tool_name"] for a in actions] == expected_tools and all(a["effect"] == "read" for a in actions), "Tool order/authority differs")
    for action in actions:
        if action["tool_name"] == "query_order":
            expected_order = {"L1": "MOCK001", "L2": "MOCK001", "L5": "MOCK002", "L6": "MOCK003"}[case["id"]]
            require(action["arguments"] == {"order_id": expected_order}, "Order binding differs")
        else:
            require(action["arguments"] == {"tracking_no": case["tracking_no"]}, "Tracking binding differs")
    attempts = evidence.get("attempts", [])
    require(len(attempts) == case["mcp_calls"], "Attempt count differs")
    if case.get("error_code"):
        require(evidence.get("run_status") == "interrupted" and evidence.get("assistant_count") == 0 and evidence.get("http_status") == 503, "L6 must interrupt without an answer")
        require(attempts[0]["state"] == "failed" and attempts[0]["error_code"] == "logistics_timeout", "L6 expected timeout missing")
        return
    require(evidence.get("run_status") == "completed" and evidence.get("assistant_count") == 1 and evidence.get("http_status") == 200, "Completed answer missing")
    answer = evidence.get("answer", "")
    require(answer.startswith("【模拟物流】") and evidence.get("reply_kind") == "answer", "Simulated answer source/kind missing")
    if attempts:
        attempt = attempts[0]
        require(attempt["state"] == "completed" and attempt["hash_verified"] is True, "Verified durable observation missing")
        envelope = attempt["envelope"]
        require(envelope["tracking_no"] == case["tracking_no"] and envelope["outcome"] == case.get("outcome", "found"), "Observation identity/outcome differs")
        if "status" in case:
            require(envelope["shipment"]["status"] == case["status"], "Shipment status differs")
    validate_answer_facts(case, evidence, answer)


def model_replay_proof(evidence, proof):
    run_id = evidence["run_id"]
    detail, replay, sse = (proof[key] for key in ("detail", "replay", "sse"))
    saved = [m for m in detail["body"].get("messages", []) if m.get("run_id") == run_id and m.get("role") == "assistant"]
    frames = [json.loads(line[6:]) for line in sse["text"].splitlines() if line.startswith("data: ")]
    if evidence["error_code"]:
        # Existing API contract: first failure is503; duplicate requests return
        # a200 interrupted snapshot and SSE run_snapshot/done, without retrying.
        initial = proof["initial"]
        replay_equal = (initial["status"] == 503 and initial["body"].get("detail", {}).get("code") == evidence["error_code"]
            and initial["body"]["detail"].get("run_id") == run_id and detail["status"] == 200 and not saved
            and replay["status"] == 200 and replay["body"] == {"status": evidence["run_status"], "run_id": run_id})
        sse_equal = (sse["status"] == 200 and all(f.get("run_id") == run_id for f in frames)
            and not any(f.get("type") in {"message", "error"} for f in frames)
            and any(f.get("type") == "run_snapshot" and f.get("status") == evidence["run_status"] for f in frames)
            and any(f.get("type") == "done" and f.get("status") == evidence["run_status"] and f.get("snapshot") is True for f in frames))
    else:
        replay_equal = (detail["status"] == 200 and replay == proof["initial"] and len(saved) == 1
            and saved[0]["text"] == evidence["answer"] == proof["initial"]["body"].get("answer"))
        sse_equal = sse["status"] == 200 and any(f.get("type") == "message" and f.get("run_id") == run_id and f.get("text") == evidence["answer"] and f.get("schema_version") == 3 for f in frames)
    return {"replay_equal": bool(replay_equal), "sse_equal": bool(sse_equal)}


class LiveModelCases:
    def __init__(self, ownership, ledger, directory, *, conversation=None):
        self.ownership, self.ledger, self.directory = ownership, ledger, directory
        self.identity = validate_logistics_ownership(ownership)
        self.conversation = conversation

    def __call__(self, case):
        require(case["id"] != "L2" or self.conversation is not None, "L2 requires verified L1 conversation")
        base = f"http://127.0.0.1:{self.identity['ports']['resolv']}"
        before_events = {e["event_id"] for e in network_events(self.ownership)}
        with httpx2.Client(trust_env=False, timeout=600) as http:
            if case["id"] != "L2":
                response = http.post(base + "/conversations")
                require(response.status_code == 200, "Conversation create failed")
                self.conversation = response.json()["conversation_id"]
            payload = {"session_id": self.conversation, "request_id": "model-" + case["id"] + "-" + uuid4().hex, "question": case["text"]}
            response = http.post(base + "/agent/chat", json=payload)
            body = response.json()
            save_json(self.directory / (case["id"] + "-http.json"), {"status": response.status_code, "body": body})
            with owned_logistics_resources(self.ownership) as owned, owned.conversation_factory() as session:
                run = session.scalar(select(ConversationRun).where(ConversationRun.conversation_id == self.conversation, ConversationRun.request_id == payload["request_id"]))
                require(run is not None, "Persisted run missing; inspect private HTTP evidence")
                understanding = session.get(ConversationUnderstanding, run.understanding_id)
                actions = session.scalars(select(ConversationAction).where(ConversationAction.run_id == run.run_id).order_by(ConversationAction.model_step)).all()
                attempts = session.scalars(select(Attempt).where(Attempt.run_id == run.run_id).order_by(Attempt.attempt_no)).all()
                assistant_count = session.scalar(select(func.count()).select_from(ConversationMessage).where(ConversationMessage.run_id == run.run_id, ConversationMessage.role == "assistant"))
                reply = run.result_json or {}
                answer = session.get(ConversationMessage, run.answer_message_id)
                evidence = {"run_id": run.run_id, "run_status": run.status, "error_code": run.error_code,
                    "http_status": response.status_code, "assistant_count": assistant_count, "answer": answer.text if answer else None, "reply_kind": reply.get("reply_kind"),
                    "versions": [run.orchestration_version, run.state_schema_version, reply.get("schema_version", 3), run.business_config["config_version"]],
                    "input_budget": run.business_config["app_max_input_tokens"], "understanding_status": understanding.status if understanding else None,
                    "intent": (understanding.classification or {}).get("output", {}).get("requests", [{}])[0].get("intent") if understanding else None,
                    "resolution": understanding.resolution if understanding else None, "understanding_audits": understanding.invocations if understanding else [],
                    "business_audits": run.model_audits,
                    "actions": [{"tool_name": a.tool_name, "effect": a.effect, "arguments": a.arguments, "result": a.result, "execution_status": a.execution_status} for a in actions],
                    "attempts": [{"attempt_id": a.attempt_id, "action_id": a.action_id, "state": a.state, "error_code": a.error_code, "envelope": a.envelope,
                        "hash_verified": a.envelope is not None and fingerprint(a.envelope) == a.result_hash and a.projection == next((x.result for x in actions if x.action_id == a.action_id), None)} for a in attempts]}
            events = [e for e in network_events(self.ownership) if e["event_id"] not in before_events]
            evidence["mcp_calls"], evidence["api_requests"] = counts(events)
            evidence["network_events"] = events
            calls_before = self.ledger.read()["http_calls"]
            wire_before = counts(network_events(self.ownership))
            detail = http.get(base + "/conversations/" + self.conversation)
            replay = http.post(base + "/agent/chat", json=payload)
            sse = http.post(base + "/agent/stream", json=payload)
            proof = {"initial": {"status": response.status_code, "body": body},
                "detail": {"status": detail.status_code, "body": detail.json()},
                "replay": {"status": replay.status_code, "body": replay.json()}, "sse": {"status": sse.status_code, "text": sse.text}}
            save_json(self.directory / (case["id"] + "-replay.json"), proof)
            evidence.update(model_replay_proof(evidence, proof))
            evidence["refresh_extra_calls"] = (self.ledger.read()["http_calls"] - calls_before) + sum(b - a for a, b in zip(wire_before, counts(network_events(self.ownership)), strict=True))
            evidence["model_calls"] = [c for c in self.ledger.read()["calls"] if c["case_id"] == case["id"]]
            save_json(self.directory / (case["id"] + "-evidence.json"), evidence)
            return evidence


def _preserved_run_facts(owned, run_id):
    with owned.conversation_factory() as session:
        run = session.get(ConversationRun, run_id)
        require(run is not None, "Preserved run missing")
        conversation = session.get(Conversation, run.conversation_id)
        require(conversation.owner_id == owned.identity["owner_id"], "Preserved run owner differs")
        question = session.get(ConversationMessage, run.question_message_id)
        return {"run_id": run_id, "session_id": run.conversation_id, "request_id": run.request_id, "question": question.text,
            "status": run.status, "error_code": run.error_code, "model_audits": run.model_audits,
            "answer_message_id": run.answer_message_id, "result_json": run.result_json,
            "actions": [{"id": a.action_id, "arguments": a.arguments, "result": a.result, "status": a.execution_status}
                for a in session.scalars(select(ConversationAction).where(ConversationAction.run_id == run_id).order_by(ConversationAction.model_step)).all()],
            "attempts": [{"id": a.attempt_id, "state": a.state, "error": a.error_code, "envelope": a.envelope, "hash": a.result_hash}
                for a in session.scalars(select(Attempt).where(Attempt.run_id == run_id).order_by(Attempt.attempt_no)).all()]}


def run_preserved_error_replay(ownership, report_dir, *, case_id="L6"):
    require(case_id == "L6", "Only L6 preserved-error replay is supported")
    identity = validate_logistics_ownership(ownership)
    validate_export_origin()
    gates = validate_model_prerequisites(ownership, identity["source_revision"])
    root = Path(ownership).parent
    evidence = _read_json(root / "model-C3/L6-evidence.json")
    original = _read_json(root / "model-C3/L6-http.json")
    preserved = _read_json(root / "C3-preserved-hashes.json")
    for name in ("budget-C3.json", "model-C3/report.json", "model-C3/L6-evidence.json", "model-C3/L6-http.json"):
        preserved[name] = sha256((root / name).read_bytes()).hexdigest()
    ledgers = [_read_json(root / name) for name in ("budget.json", "budget-C1.json", "budget-C2.json", "budget-C3.json")]
    require(all(b["state"] in {"closed", "completed"} for b in ledgers) and sum(b["http_calls"] for b in ledgers) == 26, "Replay requires terminal unchanged 26-call history")
    case = next(c for c in json.loads(CASES_PATH.read_text())["cases"] if c["id"] == case_id)
    with owned_logistics_resources(ownership) as owned:
        before_run = _preserved_run_facts(owned, evidence["run_id"])
    require(before_run["status"] == "interrupted" and before_run["error_code"] == "logistics_timeout"
        and before_run["question"] == case["text"], "Wrong preserved error run")
    before = snapshot_logistics_protection()
    before_wire = counts(network_events(ownership))
    directory = new_logistics_report_dir(ownership, report_dir)
    with logistics_child(ownership, directory, role="resolv"):
        # This standard development process has fake clients only. No API/MCP
        # process is started, and any inference/remote read changes the proof.
        base = f"http://127.0.0.1:{identity['ports']['resolv']}"
        with httpx2.Client(trust_env=False, timeout=30) as http:
            detail = http.get(base + "/conversations/" + before_run["session_id"])
            payload = {k: before_run[k] for k in ("session_id", "request_id", "question")}
            replay = http.post(base + "/agent/chat", json=payload)
            sse = http.post(base + "/agent/stream", json=payload)
    proof = {"initial": original, "detail": {"status": detail.status_code, "body": detail.json()},
        "replay": {"status": replay.status_code, "body": replay.json()}, "sse": {"status": sse.status_code, "text": sse.text}}
    save_json(directory / "L6-replay.json", proof)
    with owned_logistics_resources(ownership) as owned:
        after_run = _preserved_run_facts(owned, evidence["run_id"])
    require(before_run == after_run, "Read replay changed preserved run facts")
    require(before_wire == counts(network_events(ownership)), "Read replay added API/MCP calls")
    require(all(sha256((root / name).read_bytes()).hexdigest() == value for name, value in preserved.items()), "Read replay changed original evidence")
    after = snapshot_logistics_protection(layout=before)
    require(before == after, "Read replay changed protected resources")
    corrected = {**evidence, **model_replay_proof(evidence, proof), "refresh_extra_calls": 0}
    validate_model_case(case, corrected)
    report = {"format": "logistics-preserved-error-replay-v1", "status": "passed", "source_revision": identity["source_revision"],
        "source_root": str(SOURCE_ROOT), "gates": gates, "model_http_requests_added": 0, "remote_calls_added": 0,
        "preserved_hashes": preserved, "preserved_run_unchanged": True, "formal_preserved": True,
        "original_replay_raw_was_not_saved": True, "proof": str(directory / "L6-replay.json"), "evidence": corrected}
    save_json(directory / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--real-worker", action="store_true")
    parser.add_argument("--execute-approved-budget", action="store_true")
    parser.add_argument("--replay-preserved-error", action="store_true", help="Zero-model read replay of the already persisted L6 error")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--continuation-C1", dest="continuation_c1", action="store_true", help="Explicit approved C1: only L2-L6; preserve parent ledger")
    continuation.add_argument("--continuation-C2", dest="continuation_c2", action="store_true", help="Explicit approved C2: only L4-L6; preserve all earlier ledgers")
    continuation.add_argument("--continuation-C3", dest="continuation_c3", action="store_true", help="Approved harness exception: only unattempted L5-L6")
    args = parser.parse_args()
    if args.replay_preserved_error:
        require(args.report_dir is not None and not args.real_worker and not any((args.continuation_c1, args.continuation_c2, args.continuation_c3)), "Read replay mode cannot execute model cases")
        run_preserved_error_replay(args.ownership, args.report_dir)
        return
    require(args.execute_approved_budget, "--execute-approved-budget required")
    if args.real_worker:
        from resolv.core.config import Settings

        validate_export_origin()
        identity = validate_logistics_ownership(args.ownership)
        validate_service_port(args.ownership, "resolv", args.port)
        ledger_type = _model_budget_type(args.continuation_c1, args.continuation_c2, args.continuation_c3)
        ledger = ledger_type(args.ownership, identity["source_revision"])
        require(ledger.read()["state"] == "active", "Model ledger is not active")
        app = create_logistics_demo_app(args.ownership, real_models=True, model_guard=LogisticsModelGuard(ledger).wrap)
        with provider_http_guard(ledger, Settings(_env_file=REPOSITORY_ROOT / ".env").openai_base_url):
            serve_owned_app(app, args.ownership, "resolv", args.port)
    else:
        require(args.revision and args.report_dir, "revision and report-dir required")
        report = run_logistics_model_cases(args.ownership, args.revision, args.report_dir, execute_approved_budget=True,
            continuation_c1=args.continuation_c1, continuation_c2=args.continuation_c2, continuation_c3=args.continuation_c3)
        raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
