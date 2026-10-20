import importlib

import pytest
from pydantic import ValidationError

from tests.branch_fixtures import make_view


def api():
    try:
        return (
            importlib.import_module("resolv.schemas.branch_execution"),
            importlib.import_module("resolv.services.branch_replies"),
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"branch contract missing: {exc.name}")


@pytest.mark.parametrize(
    "intent,answer,kind,actions",
    [
        (
            "complaint",
            "很抱歉给您带来不好的体验。您可以通过下方入口整理问题说明，查看转人工或投诉工单草稿。",
            "answer",
            ["transfer_human", "create_ticket"],
        ),
        (
            "human",
            "您可以通过下方转人工入口整理问题说明。本期仅提供草稿确认，尚未实际转接。",
            "answer",
            ["transfer_human"],
        ),
        (
            "chitchat",
            "您好！我可以协助您咨询商品、订单、物流或售后问题，请告诉我您需要什么帮助。",
            "answer",
            [],
        ),
        (
            "other",
            "请把问题描述得更具体一些，例如您想查询什么，或者遇到了什么情况。",
            "answer",
            [],
        ),
        ("product_consultation", "该分支尚未接入", "unavailable", []),
        ("refund_return", "该分支尚未接入", "unavailable", []),
        ("after_sales", "该分支尚未接入", "unavailable", []),
    ],
)
def test_fixed_routes_have_only_preview_actions(
    intent, answer, kind, actions, monkeypatch
):
    _, replies = api()
    from resolv.core import llm
    from resolv.tools import executor

    def forbidden(*args, **kwargs):
        pytest.fail("fixed reply reached business dependency")

    monkeypatch.setattr(llm, "get_chat_model", forbidden)
    monkeypatch.setattr(executor, "execute_tool_call", forbidden)
    reply = replies.render_reply("run-1", make_view("原始主诉求", intent), mode="agent")
    assert reply.answer == answer
    assert reply.metadata.reply_kind == kind
    assert [a.type for a in reply.metadata.suggested_actions] == actions
    assert all(a.preview_only is True for a in reply.metadata.suggested_actions)
    assert reply.metadata.understanding_id == "u-1"
    assert reply.metadata.primary_task_id == "task-1"


def test_draft_uses_primary_only_and_ids_are_stable():
    _, replies = api()
    view = make_view("投诉 B 商家说‘已退款’，若未处理请说明", "complaint")
    view.original_text = "查 A，另" + view.original_text
    first = replies.render_reply("r1", view, mode="agent")
    again = replies.render_reply("r1", view, mode="agent")
    other = replies.render_reply("r2", view, mode="agent")
    assert first == again
    assert {a.suggestion_id for a in first.metadata.suggested_actions}.isdisjoint(
        a.suggestion_id for a in other.metadata.suggested_actions
    )
    for action in first.metadata.suggested_actions:
        assert action.draft.content == "投诉 B 商家说‘已退款’，若未处理请说明"
        assert "查 A" not in action.draft.content


def test_draft_retains_full_verified_clause_when_model_task_text_is_shortened():
    _, replies = api()
    text = "我要投诉：商家说“已退款”，但我未收到，请给我整理投诉内容"
    view = make_view(text, "complaint")
    view.primary_request.text = "我要投诉：商家说"
    view.original_text = "先查订单 MH001，" + text
    reply = replies.render_reply("r", view, mode="agent")
    assert all(a.draft.content == text for a in reply.metadata.suggested_actions)
    assert all(
        "先查订单" not in a.draft.content for a in reply.metadata.suggested_actions
    )


def test_continued_draft_uses_verified_task_clause_instead_of_continue_command():
    from resolv.schemas.understanding import EvidenceRef

    _, replies = api()
    view = make_view("继续", "complaint")
    view.primary_request.text = "投诉"
    view.primary_request.evidence_refs.append(
        EvidenceRef(
            kind="task",
            source_id="pending-complaint",
            span=None,
            json_path=["text"],
            quote="投诉：包装破损且商家拒绝说明",
        )
    )
    reply = replies.render_reply("r", view, mode="agent")
    assert all(
        a.draft.content == "投诉：包装破损且商家拒绝说明"
        for a in reply.metadata.suggested_actions
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"preview_only": False},
        {"preview_only": 1},
        {"preview_only": "true"},
        {"approval_id": "a"},
        {"label": 12},
        {"draft": {"title": "t", "content": "c", "submit": True}},
    ],
)
def test_preview_contract_rejects_execution_and_coercion(patch):
    schema, _ = api()
    with pytest.raises(ValidationError):
        schema.SuggestedAction.model_validate(
            {
                "suggestion_id": "s",
                "type": "transfer_human",
                "label": "转人工",
                "draft": {"title": "标题", "content": "内容"},
                **patch,
            }
        )


@pytest.mark.parametrize("intent", ["order", "logistics"])
def test_chat_mode_cannot_execute_and_agent_requires_business(intent):
    _, replies = api()
    view = make_view("查询订单 MH001", intent)
    assert replies.render_reply("r", view, mode="agent") is None
    reply = replies.render_reply("r", view, mode="chat")
    assert reply.answer == "该入口不支持业务执行，请使用客服聊天入口"
    assert reply.metadata.suggested_actions == []
    assert reply.metadata.reply_kind == "unavailable"


def test_clarification_preserves_original_question_without_actions():
    _, replies = api()
    view = make_view("取消那单", "order", ready=False)
    reply = replies.render_reply("r", view, mode="agent")
    assert reply.answer == "请问您指的是哪笔订单？"
    assert reply.metadata.reply_kind == "clarification"
    assert reply.metadata.clarifies_understanding_id == "u-1"
    assert reply.metadata.suggested_actions == []


@pytest.mark.parametrize(
    "patch",
    [
        {"continue_task_id": ""},
        {"continue_task_id": 1},
        {"continue_task_id": "x" * 65},
        {"continue_task_id": "t", "clarifies_understanding_id": "u"},
        {"branch": "business"},
    ],
)
def test_routing_links_follow_understanding_constraints(patch):
    schema, _ = api()
    with pytest.raises(ValidationError):
        schema.RoutingInput.model_validate(patch)
