import importlib
from pathlib import Path

import pytest


def api():
    try:
        return importlib.import_module("scripts.smoke_three_branch_accuracy")
    except ModuleNotFoundError:
        pytest.fail("real routed acceptance runner missing")


def test_formal_business_path_cannot_reach_acceptance_connection(tmp_path):
    with pytest.raises(ValueError):
        api().validate_accuracy_isolation(
            "resolv_conversations", Path("data/resolv.db")
        )


def test_gold_hash_is_checked_before_model_invocation(tmp_path):
    fixture = tmp_path / "gold.json"
    fixture.write_text("changed gold")
    with pytest.raises(ValueError):
        api().assert_gold_hash(fixture, "0" * 64)


def test_known_usage_does_not_hide_a_failed_model_invocation():
    from resolv.services.branch_evaluation import BLOCKERS

    counters = dict.fromkeys(BLOCKERS, 0)
    api().count_model_audit(
        {
            "actual_input": 100,
            "actual_output": 10,
            "estimate": 200,
            "error_code": "model_response_invalid",
        },
        counters,
    )
    assert counters["technical_errors"] == 1
    assert counters["usage_unknown"] == 0


def test_rejected_tool_requests_still_count_as_entity_errors():
    from resolv.services.branch_evaluation import BLOCKERS

    counters = dict.fromkeys(BLOCKERS, 0)
    case = {
        "orders": [],
        "expect": {"allowed_tools": ["query_order"], "allowed_orders": ["A"]},
    }
    api().count_requested_tool(
        {"name": "query_order", "args": {"order_id": "B"}}, case, counters
    )
    assert counters["wrong_entity_bindings"] == 1


def test_stage_observer_keeps_only_validated_public_objects_and_restores_models():
    from types import SimpleNamespace

    from resolv.schemas.understanding import ResolutionResult

    result = ResolutionResult(
        resolved_text="原话",
        bindings=[],
        unresolved_references=[],
        continuation_selection=None,
    )
    original = SimpleNamespace(
        invoke=lambda *args, **kwargs: (result, SimpleNamespace(invocation_id="i"))
    )
    service = SimpleNamespace(
        understanding=SimpleNamespace(
            resolver=SimpleNamespace(model=original),
            classifier=SimpleNamespace(model=original),
        )
    )
    with api().observe_stage_outputs(service) as outputs:
        actual = service.understanding.classifier.model.invoke("intents")
        assert actual[0] is result
        assert outputs == {"i": result.model_dump(mode="json")}
    assert service.understanding.resolver.model is original
    assert service.understanding.classifier.model is original


def test_documented_file_cli_can_load_without_project_pythonpath():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/smoke_three_branch_accuracy.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_isolated_lifespan_closes_sync_clients_outside_event_loop(
    monkeypatch, tmp_path
):
    import asyncio
    from contextlib import contextmanager
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    module = api()

    @contextmanager
    def environment(*args):
        yield

    @contextmanager
    def runtime():
        try:
            yield SimpleNamespace(understanding=object())
        finally:
            asyncio.run(asyncio.sleep(0))

    monkeypatch.setattr(module, "isolated_environment", environment)
    monkeypatch.setattr(module, "routed_conversation_runtime", runtime)
    with TestClient(
        module.create_isolated_app("isolated", tmp_path / "accuracy.db")
    ) as client:
        assert client.get("/health").json() == {"status": "ok"}
