"""A true flag cannot substitute for retrieval, MPS, usage or provenance evidence."""

import copy

import pytest
from scripts.knowledge_gate_support import validate_ownership
from scripts.smoke_knowledge_branch import validate_knowledge_gate


def positive_report():
    return {
        "gate": "K2",
        "passed": True,
        "real_services": True,
        "knowledge_scope": "isolated",
        "probes": {
            name: {"passed": True}
            for name in [
                "one_lane_empty",
                "bm25_fault",
                "sql_fault",
                "reranker_fault",
                "corrupt_authority",
                "low_score",
                "selfcheck_insufficient",
            ]
        },
        "business_tool_calls": 0,
        "source_fingerprints": {"src/example.py": "a" * 64},
        "release": {
            "release_manifest_hash": "a" * 64,
            "profile_hash": "b" * 64,
            "manifest_hash": "c" * 64,
        },
        "local_inference": {
            "device": "mps",
            "dtype": "torch.float32",
            "batch_size": 4,
            "forward_calls": 2,
            "dense_device": "cpu",
            "dense_dtype": "torch.float32",
            "model_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        },
        "preservation": {"before": {"old": "same"}, "after": {"old": "same"}},
        "cases": [
            {
                "case_id": "normal",
                "passed": True,
                "expected": "knowledge",
                "status": "completed",
                "business_tool_calls": 0,
                "counts": {
                    "dense_count": 5,
                    "bm25_count": 5,
                    "center_count": 5,
                    "scored_count": 5,
                    "selected_count": 2,
                },
                "evidence": {
                    "evidence_hash": "e" * 64,
                    "input_tokens": 800,
                    "budget": 3000,
                    "windows": ["w1"],
                    "citations": {"w1": ["a", "b"]},
                    "chunks": ["a", "b"],
                    "used_citation_numbers": [1],
                },
                "signals": {
                    "passed": True,
                    "total": 0.5,
                    "contributions": [0.3, 0.1, 0.05, 0.05],
                },
                "selfcheck": {"useful": True, "supporting_window_ids": ["w1"]},
                "audits": [
                    {
                        "purpose": p,
                        "outcome": "succeeded",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "input_estimate": 200,
                    }
                    for p in [
                        "resolution",
                        "intent",
                        "knowledge_query",
                        "knowledge_check",
                        "knowledge_answer",
                    ]
                ],
                "query_tokens": 20,
                "pair_tokens": [100, 200],
                "same_evidence": True,
                "transport_tools": 0,
            }
        ],
    }


def test_complete_fixture_report_passes_structural_validation_only():
    validate_knowledge_gate(positive_report())


@pytest.mark.parametrize(
    "broken",
    [
        "empty",
        "scored",
        "mps",
        "dtype",
        "tools",
        "usage",
        "closure",
        "budget",
        "evidence",
        "preservation",
        "selfcheck",
        "signals",
    ],
)
def test_rejects_incomplete_positive_evidence(broken):
    r = copy.deepcopy(positive_report())
    c = r["cases"][0]
    if broken == "empty":
        c["counts"]["dense_count"] = c["counts"]["bm25_count"] = 0
    elif broken == "scored":
        c["counts"]["scored_count"] = 0
    elif broken == "mps":
        r["local_inference"]["forward_calls"] = 0
    elif broken == "dtype":
        r["local_inference"]["dtype"] = "torch.float16"
    elif broken == "tools":
        r["business_tool_calls"] = 1
    elif broken == "usage":
        c["audits"][0]["input_tokens"] = None
    elif broken == "closure":
        c["evidence"]["chunks"].pop()
    elif broken == "budget":
        c["pair_tokens"] = [3073]
    elif broken == "evidence":
        c["same_evidence"] = False
    elif broken == "preservation":
        r["preservation"]["after"] = {}
    elif broken == "selfcheck":
        c["selfcheck"]["supporting_window_ids"] = ["alien"]
    elif broken == "signals":
        c["signals"]["total"] = 0.8
    with pytest.raises(ValueError):
        validate_knowledge_gate(r)


def test_foreign_ownership_manifest_rejected_before_resource_use(tmp_path):
    path = tmp_path / "ownership.json"
    path.write_text('{"owner":"foreign"}')
    with pytest.raises(ValueError):
        validate_ownership(path)


def test_activation_requires_both_isolated_and_formal_k2_proofs():
    from resolv.services.knowledge_release import validate_gate_report

    with pytest.raises(ValueError):
        validate_gate_report("K2", positive_report(), None)


def test_case_report_uses_persisted_actual_evidence_and_refresh(
    knowledge_factory, tmp_path, monkeypatch
):
    from scripts import smoke_knowledge_branch as gate
    from scripts import smoke_knowledge_branch_recovery as recovery

    from tests.conversation_fixtures import MemoryCheckpoints

    memory = MemoryCheckpoints()
    monkeypatch.setattr(
        recovery, "checkpoint_session", lambda _settings, **kwargs: memory(**kwargs)
    )
    monkeypatch.setattr(gate, "begin_provider_case", lambda *a: None)
    state = {
        "window": "knowledge_after_answer_saved",
        "calls": str(tmp_path / "calls.jsonl"),
        "selector": str(tmp_path / "selector.json"),
    }
    service = recovery.make_service(knowledge_factory, None, state)
    report = gate.run_case(
        service,
        {},
        {
            "case_id": "offline",
            "question": recovery.QUESTION,
            "expected": "knowledge",
            "facts": [],
        },
        "isolated",
        tmp_path,
    )
    assert report["passed"] and report["same_evidence"]
    assert report["evidence"]["used_citation_numbers"] == [1]
    assert report["business_tool_calls"] == 0


from tests.knowledge_branch_fixtures import knowledge_factory

__all__ = ["knowledge_factory"]


def test_existing_foreign_database_is_detected_before_claim_or_creation(
    tmp_path, monkeypatch
):
    from scripts import knowledge_gate_support as support
    from sqlalchemy.engine import make_url

    monkeypatch.setattr(support, "ROOT", tmp_path)
    marker = tmp_path / support.WORKSPACE / "ownership.json"
    monkeypatch.setattr(support, "OWNERSHIP", marker)
    k = make_url("mysql+pymysql://fixture@localhost/resolv_knowledge")
    c = make_url("mysql+pymysql://fixture@localhost/resolv_conversations")
    monkeypatch.setattr(support, "configured_urls", lambda: (k, c))
    monkeypatch.setattr(
        support,
        "database_exists",
        lambda url: url.database == support.CONVERSATION_DB,
        raising=False,
    )
    monkeypatch.setattr(
        support,
        "_owned_database",
        lambda *a, **k: pytest.fail(
            "created resources before complete collision preflight"
        ),
    )
    with pytest.raises(ValueError, match="existing"):
        support.initialize_resources(
            support.CONVERSATION_DB, tmp_path / support.WORKSPACE / "business.db"
        )
    assert not marker.exists()


def test_isolated_k2_requires_empty_fault_and_two_gate_control_probes():
    report = positive_report()
    report.pop("probes", None)
    with pytest.raises(ValueError):
        validate_knowledge_gate(report)


def test_fault_probe_guard_rejects_a_formal_collection_before_accessing_resources():
    from types import SimpleNamespace

    from scripts.smoke_knowledge_branch import run_isolated_probes

    with pytest.raises(ValueError, match="owned"):
        run_isolated_probes(
            SimpleNamespace(knowledge_retrievers=object()),
            {"collection": "owned"},
            SimpleNamespace(collection="resolv_knowledge_20261015_v1"),
            None,
        )


@pytest.mark.parametrize(
    "name", ["smoke_knowledge_branch.py", "smoke_knowledge_branch_recovery.py"]
)
def test_direct_cli_entry_help_works_without_pytest_pythonpath(name):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / name), "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--report-dir" in result.stdout


def test_resource_inventory_connects_without_selecting_the_nonexistent_database(
    monkeypatch,
):
    from scripts import knowledge_gate_support as support
    from sqlalchemy.engine import make_url

    class Result:
        def scalar(self):
            return None

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, *a, **kw):
            return Result()

    class Engine:
        def connect(self):
            return Connection()

        def dispose(self):
            pass

    def engine(url):
        assert url.database is None, "inventory selected the not-yet-created DB"
        return Engine()

    monkeypatch.setattr(support, "_engine", engine)
    assert (
        support.database_exists(
            make_url("mysql+pymysql://fixture@localhost/not_created_yet")
        )
        is False
    )


def test_admin_bootstrap_sql_is_limited_to_new_owned_databases_and_existing_account():
    from scripts.knowledge_gate_support import (
        CONVERSATION_DB,
        OWNER_KIND,
        owned_bootstrap_sql,
    )

    owner = "a" * 32
    obj = {
        "kind": OWNER_KIND,
        "owner": owner,
        "knowledge_database": "mhknow_gate_" + owner[:16],
        "conversation_database": CONVERSATION_DB,
        "collection": "mhknow_gate_" + owner[:16],
    }
    sql = owned_bootstrap_sql(obj, ["resolv@%"])
    assert "GRANT ALL PRIVILEGES ON *.*" not in sql
    assert (
        "ON `resolv_knowledge`.*" not in sql
        and "ON `resolv_conversations`.*" not in sql
    )
    assert sql.count("CREATE DATABASE") == 2 and sql.count("CREATE USER") == 0
    with pytest.raises(ValueError):
        owned_bootstrap_sql(
            {**obj, "knowledge_database": "resolv_knowledge"}, ["resolv@%"]
        )
    with pytest.raises(ValueError):
        owned_bootstrap_sql(obj, ["bad'account@%"])


def test_scoped_database_grants_escape_mysql_wildcards():
    from scripts.knowledge_gate_support import (
        CONVERSATION_DB,
        OWNER_KIND,
        owned_bootstrap_sql,
    )

    owner = "b" * 32
    obj = {
        "kind": OWNER_KIND,
        "owner": owner,
        "knowledge_database": "mhknow_gate_" + owner[:16],
        "conversation_database": CONVERSATION_DB,
        "collection": "mhknow_gate_" + owner[:16],
    }
    for line in owned_bootstrap_sql(obj, ["resolv@%"]).splitlines():
        if line.startswith("GRANT "):
            identifier = line.split("`")[1]
            assert "_" not in identifier.replace("\\_", ""), (
                "MySQL database privilege wildcard was not escaped"
            )


def test_admin_preflight_rejects_server_identity_mismatch(monkeypatch):
    from types import SimpleNamespace

    from scripts import knowledge_gate_support as support
    from sqlalchemy.engine import make_url

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, *a, **kw):
            return SimpleNamespace(scalar_one=lambda: "resolv@%")

    monkeypatch.setattr(
        support,
        "_engine",
        lambda url: SimpleNamespace(connect=lambda: Connection(), dispose=lambda: None),
    )
    monkeypatch.setattr(
        support, "database_identity", lambda *a: {"server_fingerprint": "0" * 64}
    )
    statements = []

    def admin(sql):
        statements.append(sql)
        return ["0"] if "partial_revokes" in sql else ["foreign-server"]

    monkeypatch.setattr(support, "_admin_mysql", admin)
    owner = "c" * 32
    obj = {
        "kind": support.OWNER_KIND,
        "owner": owner,
        "knowledge_database": "mhknow_gate_" + owner[:16],
        "conversation_database": support.CONVERSATION_DB,
        "collection": "mhknow_gate_" + owner[:16],
    }
    with pytest.raises(ValueError, match="identity"):
        support.admin_preflight(
            obj, [make_url("mysql+pymysql://fixture@localhost/resolv_knowledge")]
        )
    assert all(sql.startswith("SELECT ") for sql in statements)


@pytest.mark.parametrize("scope", ["isolated", "formal-readonly"])
def test_gate_resumes_only_the_existing_journaled_case_without_repeat_understanding(
    knowledge_factory, tmp_path, monkeypatch, scope
):
    from scripts import smoke_knowledge_branch as gate
    from scripts import smoke_knowledge_branch_recovery as recovery

    from tests.conversation_fixtures import MemoryCheckpoints

    memory = MemoryCheckpoints()
    monkeypatch.setattr(
        recovery, "checkpoint_session", lambda _settings, **kwargs: memory(**kwargs)
    )
    state = {
        "window": "knowledge_after_answer_saved",
        "calls": str(tmp_path / "calls.jsonl"),
        "selector": str(tmp_path / "selector.json"),
    }
    service = recovery.make_service(knowledge_factory, None, state)
    c = service.create()
    run, _ = service.accept(c.conversation_id, "offline", recovery.QUESTION, "agent")

    def crash(point):
        if point == "after_understanding_terminal":
            raise RuntimeError("checkpoint boundary")

    service.fault_hook = crash
    assert list(service.run(run.run_id))[-1].type == "error"
    service.fault_hook = None
    before = len(service.understanding.resolver.model.client.calls)
    monkeypatch.setattr(
        gate,
        "begin_provider_case",
        lambda *a: pytest.fail("re-sending a journaled provider case"),
    )
    case = {
        "case_id": "offline",
        "question": recovery.QUESTION,
        "expected": "knowledge",
        "facts": [],
    }
    report = gate.run_case(
        service,
        {"provider_cases": [scope + ":offline"]},
        case,
        scope,
        tmp_path,
        resume_run=run.run_id,
    )
    assert report["passed"] and report["run_id"] == run.run_id
    assert len(service.understanding.resolver.model.client.calls) == before == 2
    with pytest.raises(ValueError):
        gate.run_case(
            service,
            {"provider_cases": []},
            {**case, "question": "different"},
            "isolated",
            tmp_path,
            resume_run=run.run_id,
        )


def test_gate_counts_recovered_invalid_reference_attempts_and_still_requires_one_answer():
    r = positive_report()
    c = r["cases"][0]
    valid = next(a for a in c["audits"] if a["purpose"] == "knowledge_check")
    valid["invocation_id"] = "valid-check"
    c["audits"].append({**valid, "invocation_id": "invalid-check"})
    c["recovered_invalid_check_invocations"] = ["invalid-check"]
    c["selfcheck_invocation_id"] = "valid-check"
    validate_knowledge_gate(r)
    with pytest.raises(ValueError):
        validate_knowledge_gate(
            {**r, "cases": [{**c, "recovered_invalid_check_invocations": []}]}
        )
    with pytest.raises(ValueError):
        validate_knowledge_gate(
            {
                **r,
                "cases": [
                    {**c, "recovered_invalid_check_invocations": ["foreign-check"]}
                ],
            }
        )


def test_gate_retains_structured_bad_reference_output_before_rejecting_it(
    knowledge_factory, tmp_path
):
    from scripts import smoke_knowledge_branch as gate

    from resolv.schemas.conversation import ConversationError
    from tests.test_knowledge_routing_graph import harness

    h = harness(
        knowledge_factory,
        tmp_path,
        check_reply={
            "useful": True,
            "supporting_window_ids": ["0" * 64],
            "missing_information": [],
            "reason": "bad reference fixture",
        },
    )
    case = {
        "case_id": h.run.request_id,
        "question": h.run.question,
        "expected": "knowledge",
        "facts": [],
    }
    with pytest.raises(ConversationError, match="self_check_error"):
        gate.run_case(
            h.conversation,
            {"provider_cases": ["isolated:" + h.run.request_id]},
            case,
            "isolated",
            tmp_path,
            resume_run=h.run.run_id,
        )
    import json

    outputs = [
        json.loads(line)
        for line in (tmp_path / (case["case_id"] + "-public-model-outputs.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert next(o for o in outputs if o["purpose"] == "knowledge_check")["output"][
        "supporting_window_ids"
    ] == ["0" * 64]
    assert h.counts["knowledge_answer"] == 0


def test_formal_resume_inventory_requires_task_report_and_every_journaled_case(
    tmp_path, monkeypatch
):
    import json

    from scripts import smoke_knowledge_branch as gate

    monkeypatch.setattr(gate, "ROOT", tmp_path)
    root = tmp_path / gate.WORKSPACE
    prior = root / "gate-k2-formal-01"
    prior.mkdir(parents=True)
    (prior / "failure.json").write_text(
        json.dumps({"knowledge_scope": "formal-readonly"})
    )
    (prior / "a-failure.json").write_text(
        json.dumps({"case_id": "a", "run_id": "run-a"})
    )
    owner = {"provider_cases": ["formal-readonly:a"]}
    assert gate.formal_resume_inventory(prior, owner) == {"a": "run-a"}
    with pytest.raises(ValueError):
        gate.formal_resume_inventory(prior.parent, owner)
    with pytest.raises(ValueError):
        gate.formal_resume_inventory(
            prior, {"provider_cases": ["formal-readonly:missing"]}
        )


def test_explicit_failed_upstream_receipt_requires_successful_replacement():
    r = positive_report()
    c = r["cases"][0]
    c["audits"].append(
        {
            "purpose": "intents",
            "invocation_id": "timed-out",
            "outcome": "failed",
            "input_tokens": None,
            "output_tokens": None,
        }
    )
    with pytest.raises(ValueError):
        validate_knowledge_gate(r)
    c["recovered_failed_upstream_invocations"] = ["timed-out"]
    c["recovery_failure_receipts"] = [
        {"failed_upstream_invocation": "timed-out", "sha256": "a" * 64}
    ]
    validate_knowledge_gate(r)
    c["audits"] = [a for a in c["audits"] if a["purpose"] != "intent"]
    with pytest.raises(ValueError):
        validate_knowledge_gate(r)


def formal_report():
    """Structural-only complete fixture, not real provider semantic evidence."""
    from resolv.schemas.knowledge_gate_cases import FORMAL_CASES

    report = positive_report()
    report["knowledge_scope"] = "formal-readonly"
    base = report["cases"][0]
    report["cases"] = []
    for rule in FORMAL_CASES:
        case = copy.deepcopy(base)
        case.update(
            case_id=rule["case_id"],
            question=rule["question"],
            expected=rule["expected"],
            answer=" ".join(a[0] for a in rule["facts"]),
            knowledge_record_present=rule["expected"] != "refund",
        )
        meta = {
            "branch": "knowledge",
            "reply_kind": "answer",
            "knowledge": {
                "decision": "sufficient",
                "citations": [
                    {"source": "knowledge/releases/fixture/after-sales-manual.md"}
                ],
            },
            "suggested_actions": [],
        }
        if rule["expected"] == "fallback":
            meta.update(
                reply_kind="knowledge_fallback",
                knowledge={"citations": [], "chunks": []},
                suggested_actions=[{"kind": "handoff"}],
            )
        elif rule["expected"] == "refund":
            meta.update(branch="refund_flow", reply_kind="unavailable", knowledge=None)
        case["metadata"] = meta
        report["cases"].append(case)
    return report
