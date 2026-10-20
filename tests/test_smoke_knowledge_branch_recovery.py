"""A child error or missing reached receipt cannot pass a recovery gate."""

import pytest
from scripts.smoke_knowledge_branch_recovery import WINDOWS, validate_recovery_gate


def report():
    return {
        "gate": "K3",
        "passed": True,
        "real_services": True,
        "fake_inference": True,
        "accuracy_evidence": False,
        "business_tool_calls": 0,
        "source_fingerprints": {"source": "a" * 64},
        "preservation": {"before": {"old": "same"}, "after": {"old": "same"}},
        "cases": [
            {
                "window": w,
                "passed": True,
                "reached": True,
                "child_exit": 86,
                "status": "completed",
                "assistant_count": 1,
                "review_count": 1 if "fallback" in w else 0,
                "stage_hashes_stable": True,
                "saved_calls_reused": True,
                "business_tool_calls": 0,
            }
            for w in WINDOWS
        ],
        "locks_and_fencing": {
            "passed": True,
            "busy_rejected": True,
            "stale_epoch_rejected": True,
            "knowledge_epoch_control": {
                "passed": True,
                "valid_write_committed": True,
                "stale_write_rejected": True,
                "unchanged_after_stale": True,
            },
        },
        "atomicity": {
            "passed": True,
            "rolled_back_reply": True,
            "rolled_back_review": True,
        },
    }


def test_complete_synthetic_recovery_receipt_validates_only_behavior():
    validate_recovery_gate(report())


@pytest.mark.parametrize(
    "kind",
    [
        "exit",
        "reached",
        "missing",
        "duplicate",
        "tools",
        "atomic",
        "hashes",
        "epoch-control",
    ],
)
def test_incomplete_recovery_proof_is_not_success(kind):
    r = report()
    if kind == "exit":
        r["cases"][0]["child_exit"] = 1
    elif kind == "reached":
        r["cases"][0]["reached"] = False
    elif kind == "missing":
        r["cases"].pop()
    elif kind == "duplicate":
        r["cases"][0]["assistant_count"] = 2
    elif kind == "tools":
        r["business_tool_calls"] = 1
    elif kind == "atomic":
        r["atomicity"]["rolled_back_review"] = False
    elif kind == "epoch-control":
        r["locks_and_fencing"]["knowledge_epoch_control"]["valid_write_committed"] = (
            False
        )
    elif kind == "hashes":
        r["cases"][0]["stage_hashes_stable"] = False
    with pytest.raises(ValueError):
        validate_recovery_gate(r)


def test_recovery_harness_exercises_real_offline_graph_and_atomic_fallback(
    knowledge_factory, tmp_path, monkeypatch
):
    from scripts import smoke_knowledge_branch_recovery as gate

    from tests.conversation_fixtures import MemoryCheckpoints

    memory = MemoryCheckpoints()
    monkeypatch.setattr(
        gate, "checkpoint_session", lambda _settings, **kwargs: memory(**kwargs)
    )
    state = {
        "window": "knowledge_after_answer_saved",
        "calls": str(tmp_path / "calls.jsonl"),
        "selector": str(tmp_path / "selector.json"),
    }
    service = gate.make_service(knowledge_factory, None, state)
    c = service.create()
    run, _ = service.accept(c.conversation_id, "offline", gate.QUESTION, "agent")
    events = list(service.run(run.run_id))
    assert events[-1].type == "done", [(e.type, e.data) for e in events]
    assert service.get_run(run.run_id).result_json["knowledge"][
        "used_citation_numbers"
    ] == [1]
    assert gate.atomic_probe(knowledge_factory.kw["bind"], None, tmp_path) == {
        "passed": True,
        "rolled_back_reply": True,
        "rolled_back_review": True,
    }


from tests.knowledge_branch_fixtures import knowledge_factory

__all__ = ["knowledge_factory"]


def test_epoch_probe_first_commits_valid_write_then_rejects_same_write_from_stale_owner(
    knowledge_factory, tmp_path, monkeypatch
):
    from scripts import smoke_knowledge_branch_recovery as gate

    from tests.conversation_fixtures import MemoryCheckpoints

    memory = MemoryCheckpoints()
    monkeypatch.setattr(
        gate, "checkpoint_session", lambda _settings, **kwargs: memory(**kwargs)
    )
    proof = gate.knowledge_epoch_probe(knowledge_factory.kw["bind"], None, tmp_path)
    assert proof == {
        "passed": True,
        "valid_write_committed": True,
        "stale_write_rejected": True,
        "unchanged_after_stale": True,
    }
