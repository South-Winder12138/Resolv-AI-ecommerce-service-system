import hashlib

import pytest

from tests.branch_fixtures import branch_factory
from tests.test_routing_graph import harness
from tests.test_smoke_three_branch_recovery import api

__all__ = ["branch_factory"]


@pytest.mark.parametrize(
    "missing",
    [
        "atomic_accept",
        "locks_and_fencing",
        "after_understanding_terminal",
        "approval_process_recovery",
        "after_sqlite_commit",
        "after_final_result",
        "after_sixth_model_step",
        "legacy_and_migrations",
    ],
)
def test_every_required_real_scenario_is_mandatory(missing):
    module = api()
    report = {
        "fake_inference": True,
        "scenarios": {name: {"passed": True} for name in module.REQUIRED_SCENARIOS},
    }
    del report["scenarios"][missing]
    with pytest.raises(ValueError):
        module.assert_gate_report(report)


def test_verify_is_readonly_and_cannot_claim_model_accuracy(
    branch_factory, tool_database
):
    module = api()
    service, run, _ = harness(branch_factory, tool_database, "投诉", "complaint")
    list(service.run(run.run_id))
    record = service.understanding.record(run.conversation_id, run.understanding_id)
    path = tool_database.kw["bind"].url.database
    from pathlib import Path

    before = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = module.verify_state(
        service,
        {
            "run_id": run.run_id,
            "conversation_id": run.conversation_id,
            "understanding_id": run.understanding_id,
            "snapshot_hash": record.snapshot.snapshot_hash,
            "window": "after_final_result",
            "order_id": "MH002",
        },
        business_db=Path(path),
    )
    assert (
        result["passed"] is True
        and result["fake_inference"] is True
        and result["accuracy_evidence"] is False
    )
    assert result["readonly_business"] is True
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == before


def test_real_gate_context_can_rehydrate_its_frozen_count_config(branch_factory):
    from resolv.core.config import ContextSettings

    service = api().make_service(None, branch_factory, None)
    settings = service.context_builder.settings
    assert ContextSettings(_env_file=None, **settings.model_dump()) == settings


def test_recovery_gate_uses_the_current_required_tool_policy(branch_factory):
    service = api().make_service(None, branch_factory, None)
    assert (
        service.business_run_config.decision_policy_version == "require-observations-v1"
    )
