from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.test_logistics_contracts import api
from tests.test_logistics_gate_support import prepared

__all__ = ["prepared"]


def recovery_report():
    windows = []
    for case, stage, before, after, states in [
        ("C1", "after_attempt_prepared", 0, 1, ["unknown", "completed"]),
        ("C2", "after_remote_result", 1, 2, ["unknown", "completed"]),
        ("C3", "after_result_commit", 1, 1, ["completed"]),
    ]:
        windows.append({"case": case, "hook": stage, "reached": True, "exit_code": 86, "resume_exit_code": 0,
            "remote_calls_before": before, "remote_calls": after, "api_requests": after,
            "refresh_extra_calls": 0, "assistant_count": 1,
            "attempt_states": states, "original_attempt_preserved": True,
            "result_hash_verified": True, "prepared_durable": True,
            "death_evidence": {"hook": stage, "attempt_id": "a" * 32},
            "network_event_count": after,
        })
    return {"format": "logistics-recovery-v1", "windows": windows, "real_model_calls": 0,
        "formal_preserved": True, "protection_before": "b" * 64, "protection_after": "b" * 64,
        "storage_dialect": "mysql", "probes": {
            "atomic_rollback": {"passed": True, "action_pending": True, "completed_attempts": 0},
            "commit_ack_unknown": {"passed": True, "fault_reached": True, "remote_calls": 1, "completed_attempts": 1},
            "stale_epoch": {"passed": True, "rejected": True, "action_pending": True},
            "concurrent_resume": {"passed": True, "requests": 2, "remote_calls": 1, "completed_attempts": 1},
        }}


def test_recovery_needs_reached_exit_and_durable_proof():
    module = api("scripts.smoke_logistics_recovery")
    module.validate_logistics_recovery_report(recovery_report())
    for field, value in [("reached", False), ("exit_code", 1), ("remote_calls", 0), ("prepared_durable", False), ("attempt_states", ["completed"]), ("network_event_count", 0)]:
        report = deepcopy(recovery_report())
        report["windows"][0][field] = value
        with pytest.raises(ValueError):
            module.validate_logistics_recovery_report(report)


@pytest.mark.parametrize("mutation", ["missing_window", "missing_probe", "fake_sql", "unreached_commit_ack", "formal_failure", "real_model"])
def test_recovery_report_rejects_incomplete_or_unproved_storage(mutation):
    report = deepcopy(recovery_report())
    if mutation == "missing_window":
        report["windows"].pop()
    elif mutation == "missing_probe":
        report["probes"].pop("stale_epoch")
    elif mutation == "fake_sql":
        report["storage_dialect"] = "sqlite"
    elif mutation == "unreached_commit_ack":
        report["probes"]["commit_ack_unknown"]["fault_reached"] = False
    elif mutation == "formal_failure":
        report["formal_preserved"] = False
    else:
        report["real_model_calls"] = 1
    with pytest.raises(ValueError):
        api("scripts.smoke_logistics_recovery").validate_logistics_recovery_report(report)


def test_fault_worker_cannot_write_outside_task_resources(prepared, monkeypatch):
    module = api("scripts.smoke_logistics_recovery")
    h = prepared
    args = SimpleNamespace(ownership=h.manifest, port=18103, delay_api=False, crash_hook="after_attempt_prepared", death_file=h.workspace.parent / "foreign-death.json")
    monkeypatch.setattr(module, "create_logistics_demo_app", lambda *args: pytest.fail("foreign fault path reached app construction"))
    with pytest.raises(ValueError, match="artifact"):
        module._worker(args)
    assert not args.death_file.exists()
