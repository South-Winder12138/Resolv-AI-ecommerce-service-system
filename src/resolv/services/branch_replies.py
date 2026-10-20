"""Deterministic replies and local preview drafts; no inference or execution."""

from resolv.repositories.conversation import fingerprint
from resolv.schemas.branch_execution import (
    ActionDraft,
    BranchReply,
    ReplyMetadata,
    SuggestedAction,
)
from resolv.schemas.conversation import ConversationError
from resolv.schemas.understanding import UnderstandingView


def draft_content(primary):
    # Action evidence was verified before selection. Keep the user's complete
    # clause, including quoted claims/conditions, even if the model shortened
    # its task label. A continuation command is not the old task's content.
    for kind in ("task", "understanding", "input"):
        refs = [
            ref
            for ref in primary.evidence_refs
            if ref.kind == kind and (kind != "task" or ref.json_path == ["text"])
        ]
        if not refs:
            continue
        quotes = []
        for ref in sorted(refs, key=lambda r: r.span.start if r.span else 0):
            if any(ref.quote in quote for quote in quotes):
                continue
            quotes = [quote for quote in quotes if quote not in ref.quote]
            quotes.append(ref.quote)
        return "\n".join(quotes)
    return primary.text


def render_reply(
    run_id: str, view: UnderstandingView, *, mode: str
) -> BranchReply | None:
    if mode not in {"agent", "chat"}:
        raise ConversationError("invalid_request")
    if view.status not in {"ready", "needs_clarification"}:
        raise ConversationError("understanding_not_ready", True)
    primary = view.primary_request
    metadata = ReplyMetadata(
        reply_kind="answer",
        understanding_id=view.understanding_id,
        primary_task_id=primary.task_id if primary else None,
        branch=view.branch,
    )
    if not view.route_ready:
        if not view.clarification_question:
            raise ConversationError("understanding_state_conflict")
        metadata.reply_kind = "clarification"
        metadata.clarifies_understanding_id = view.understanding_id
        return BranchReply(answer=view.clarification_question, metadata=metadata)
    if primary is None:
        raise ConversationError("understanding_state_conflict")
    intent = primary.intent
    actions = []
    if view.branch == "escalate":
        answer = "很抱歉给您带来不好的体验。您可以通过下方入口整理问题说明，查看转人工或投诉工单草稿。"
        actions = ["transfer_human", "create_ticket"]
    elif view.branch == "fallback_script":
        answer = (
            "您好！我可以协助您咨询商品、订单、物流或售后问题，请告诉我您需要什么帮助。"
            if intent == "chitchat"
            else "请把问题描述得更具体一些，例如您想查询什么，或者遇到了什么情况。"
        )
    elif view.branch in {"knowledge", "refund_flow"}:
        answer = "该分支尚未接入"
        metadata.reply_kind = "unavailable"
    elif view.branch == "business" and intent == "human":
        answer = (
            "您可以通过下方转人工入口整理问题说明。本期仅提供草稿确认，尚未实际转接。"
        )
        actions = ["transfer_human"]
    elif view.branch == "business" and intent in {"order", "logistics"}:
        if mode == "agent":
            return None
        answer = "该入口不支持业务执行，请使用客服聊天入口"
        metadata.reply_kind = "unavailable"
    else:
        raise ConversationError("understanding_state_conflict")
    for action_type in actions:
        label = "转人工" if action_type == "transfer_human" else "投诉工单"
        metadata.suggested_actions.append(
            SuggestedAction(
                suggestion_id=fingerprint(run_id + action_type),
                type=action_type,
                label=label,
                draft=ActionDraft(
                    title=label + "内容确认", content=draft_content(primary)
                ),
            )
        )
    return BranchReply(answer=answer, metadata=metadata)
