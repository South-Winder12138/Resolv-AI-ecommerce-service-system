import importlib

import pytest


def module():
    try:
        return importlib.import_module("scripts.smoke_understanding_accuracy")
    except ModuleNotFoundError:
        pytest.fail("accuracy CLI missing")


def test_audited_model_keeps_only_parsed_schema_output():
    from langchain_core.messages import AIMessage, HumanMessage

    from resolv.schemas.understanding import ResolutionResult
    from tests.understanding_fixtures import FakeJsonModel, make_resolution

    mod = module()
    assert hasattr(mod, "AuditedStageModel")
    model = mod.AuditedStageModel(
        FakeJsonModel([AIMessage(content=make_resolution().model_dump_json())])
    )
    output, audit = model.invoke(
        "resolution",
        [HumanMessage(content="查订单")],
        ResolutionResult,
        6000,
        2000,
        lambda _: None,
    )
    assert model.parsed_outputs[audit.invocation_id] == output.model_dump(mode="json")


def test_prediction_counts_explicit_identifiers_without_a_pronoun_binding():
    from types import SimpleNamespace

    from tests.understanding_fixtures import make_pending_task

    task = make_pending_task("a", "refund_return", text="订单 MH002 申请退款")
    view = {
        "primary_request": task.model_dump(mode="json"),
        "pending_requests": [],
        "branch": "refund_flow",
        "resolution": {},
        "status": "ready",
    }
    assert module().prediction(SimpleNamespace(case_id="explicit"), view, None, {}, 0)[
        "primary"
    ]["entities"] == ["MH002"]


def test_prediction_does_not_misreport_quoted_completion_as_business_success():
    from types import SimpleNamespace

    from tests.understanding_fixtures import make_pending_task

    text = "商家说‘已完成退款’，请核实退款进度"
    task = make_pending_task("a", "refund_return", text=text)
    view = {
        "primary_request": task.model_dump(mode="json"),
        "pending_requests": [],
        "branch": "refund_flow",
        "resolution": {"resolved_text": text},
        "stage_results": {"intents": {"output": {"requests": [{"text": text}]}}},
        "status": "ready",
    }
    assert (
        module().prediction(
            SimpleNamespace(case_id="quoted", text=text), view, None, {}, 0
        )["violations"]
        == []
    )


def test_accuracy_diagnostics_distinguish_empty_sdk_stream_without_error_text():
    from langchain_core.messages import HumanMessage

    from resolv.schemas.understanding import ResolutionResult, UnderstandingError
    from tests.understanding_fixtures import FakeJsonModel

    model = module().AuditedStageModel(
        FakeJsonModel([ValueError("No generations found in stream.")])
    )
    with pytest.raises(UnderstandingError, match="model_output_invalid"):
        model.invoke(
            "resolution",
            [HumanMessage(content="hi")],
            ResolutionResult,
            6000,
            2000,
            lambda _: None,
        )
    assert model.sdk_diagnostics == [
        {"exception_type": "ValueError", "empty_stream": True}
    ]


def test_imported_source_preserves_times_and_summary_coverage(understanding_factory):
    from datetime import datetime

    from sqlalchemy import select

    from resolv.models.conversation import (
        ConversationAction,
        ConversationMessage,
        ConversationSummaryRecord,
    )

    mod = module()
    stamp = "2026-09-01T08:00:00+00:00"
    history = [
        {
            "question": "订单 MH001",
            "answer": "历史观察",
            "user_observed_at": stamp,
            "assistant_observed_at": stamp,
            "actions": [{"result": {"order_id": "MH001"}, "observed_at": stamp}],
        },
        {"question": "后来 MH002", "answer": "较新问题", "actions": []},
    ]
    c = mod.seed_history(
        understanding_factory,
        history,
        {"confirmed_facts": ["订单 MH001"]},
        summary_meta={
            "covered_turns": 1,
            "version": 3,
            "model_name": "source-model",
            "created_at": stamp,
        },
    )
    with understanding_factory() as session:
        message = session.scalar(
            select(ConversationMessage).where(
                ConversationMessage.conversation_id == c, ConversationMessage.seq == 1
            )
        )
        assert message.created_at == datetime.fromisoformat(stamp)
        action = session.scalar(select(ConversationAction))
        assert action.result == {
            "order_id": "MH001"
        } and action.completed_at == datetime.fromisoformat(stamp)
        summary = session.scalar(
            select(ConversationSummaryRecord).where(
                ConversationSummaryRecord.conversation_id == c
            )
        )
        assert (
            summary.covered_until_seq == 2
            and summary.version == 3
            and summary.model_name == "source-model"
        )


def test_accuracy_rejects_formal_database_before_model(monkeypatch, tmp_path):
    mod = module()
    monkeypatch.setenv(
        "BAD_URL", "mysql+pymysql://u:p@localhost/resolv_conversations?charset=utf8mb4"
    )
    monkeypatch.setattr(
        mod, "run_accuracy", lambda *_: pytest.fail("unsafe resource access")
    )
    with pytest.raises(SystemExit):
        mod.main(
            ["--database-env", "BAD_URL", "--report", str(tmp_path / "report.json")]
        )


def test_existing_source_requires_explicit_pair(monkeypatch, tmp_path):
    mod = module()
    monkeypatch.setenv(
        "TEST_URL",
        "mysql+pymysql://u:p@localhost/resolv_conversations_smoke_test?charset=utf8mb4",
    )
    with pytest.raises(SystemExit):
        mod.main(
            [
                "--database-env",
                "TEST_URL",
                "--report",
                str(tmp_path / "report.json"),
                "--existing-conversation-id",
                "x",
            ]
        )
