"""Gate validation rejects incomplete evidence and preserves existing output paths."""

import pytest
from scripts import smoke_knowledge_branch_model as smoke


def test_existing_directory_is_rejected_before_client_creation(tmp_path, monkeypatch):
    target = tmp_path / "already"
    target.mkdir()
    old = target / "report.json"
    old.write_bytes(b"keep original")
    monkeypatch.setattr(smoke, "WORKSPACE", tmp_path)
    monkeypatch.setattr(smoke, "run_calls", lambda: pytest.fail("created real clients"))
    assert smoke.main(["--report-dir", str(target)]) != 0
    assert old.read_bytes() == b"keep original"


@pytest.mark.parametrize("report", [{}, {"passed": True}, {"business_tool_calls": 1}])
def test_gate_rejects_fabricated_or_missing_usage_evidence(report):
    with pytest.raises(ValueError):
        smoke.validate_model_gate(report)


def test_failure_receipt_keeps_attempted_model_invocations(tmp_path, monkeypatch):
    import json
    from datetime import UTC, datetime

    from resolv.schemas.knowledge_branch import (
        KnowledgeBranchError,
        KnowledgeModelAudit,
    )

    monkeypatch.setattr(smoke, "WORKSPACE", tmp_path)
    audit = KnowledgeModelAudit(
        invocation_id="attempt",
        purpose="knowledge_query",
        profile_fingerprint="a" * 64,
        input_hash="b" * 64,
        input_estimate=100,
        started_at=datetime.now(UTC),
        outcome="started",
    )

    def fail(*, audit_hook):
        audit_hook(audit)
        raise KnowledgeBranchError("knowledge_model_unavailable", True)

    monkeypatch.setattr(smoke, "run_calls", fail)
    target = tmp_path / "failed"
    assert smoke.main(["--report-dir", str(target)]) == 1
    report = json.loads((target / "failure.json").read_text())
    assert report["audits"][0]["invocation_id"] == "attempt"
    assert report["business_tool_calls"] == 0
