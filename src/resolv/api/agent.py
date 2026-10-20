"""Agent HTTP paths all share the durable conversation Service."""

from typing import Annotated, Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from resolv.api.chat import ChatRequest
from resolv.api.conversations import call, http_error, runtime
from resolv.api.sse import conversation_sse
from resolv.schemas.branch_execution import SuggestedAction
from resolv.schemas.conversation import ConversationError
from resolv.schemas.knowledge_branch import KnowledgePublicResult, canonical_json
from resolv.schemas.refund_submission import RefundPublicResult
from resolv.schemas.understanding import Branch

router = APIRouter(prefix="/agent")
AgentChatRequest = ChatRequest


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=64)
    approved: StrictBool


class ActionSummary(BaseModel):
    tool: str
    arguments: dict


class CompletedResponse(BaseModel):
    status: Literal["completed"] = "completed"
    run_id: str
    answer: str
    schema_version: Literal[1] = 1
    reply_kind: (
        Literal["answer", "clarification", "unavailable", "step_limit"] | None
    ) = None
    understanding_id: str | None = None
    primary_task_id: str | None = None
    branch: Branch | None = None
    clarifies_understanding_id: str | None = None
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)


class ApprovalRequiredResponse(BaseModel):
    status: Literal["approval_required"] = "approval_required"
    run_id: str
    approval_id: str
    action_id: str
    message: str
    action: ActionSummary


class CompletedResponseV2(CompletedResponse):
    schema_version: Literal[2] = 2
    reply_kind: (
        Literal[
            "answer", "clarification", "unavailable", "step_limit", "knowledge_fallback"
        ]
        | None
    ) = None
    knowledge: KnowledgePublicResult | None = None


class CompletedResponseV3(CompletedResponseV2):
    schema_version: Literal[3] = 3
    refund: RefundPublicResult | None = None


class RunStateResponse(BaseModel):
    status: Literal[
        "running", "waiting_approval", "interrupted", "finalizing", "failed"
    ]
    run_id: str


CompletedHttpResponse = Annotated[
    CompletedResponse | CompletedResponseV2 | CompletedResponseV3,
    Field(discriminator="schema_version"),
]
AgentHttpResponse = Annotated[
    CompletedHttpResponse | ApprovalRequiredResponse | RunStateResponse,
    Field(discriminator="status"),
]


def response_for_events(service, run_id, events):
    seen = list(events)
    error = next((event for event in reversed(seen) if event.type == "error"), None)
    if error:
        raise http_error(
            ConversationError(
                error.data["code"], error.data.get("retryable", False), run_id
            )
        )
    approval = next(
        (event for event in reversed(seen) if event.type == "approval_required"), None
    )
    if approval:
        data = approval.data
        return ApprovalRequiredResponse(
            run_id=run_id,
            approval_id=data["approval_id"],
            action_id=data["action_id"],
            message=data.get("message", data.get("summary", "请确认操作")),
            action=ActionSummary(tool=data["tool"], arguments=data["arguments"]),
        )
    message = next((event for event in reversed(seen) if event.type == "message"), None)
    if message:
        metadata = service.reply_metadata(run_id)
        target = {1: CompletedResponse, 2: CompletedResponseV2, 3: CompletedResponseV3}[
            metadata["schema_version"]
        ]
        return target.model_validate_json(
            canonical_json(
                {"run_id": run_id, "answer": message.data["text"], **metadata}
            )
        )
    run = service.get_run(run_id)
    return RunStateResponse(status=run.status, run_id=run_id)


@router.post("/chat", response_model=AgentHttpResponse)
def agent_chat(body: AgentChatRequest, request: Request):
    service = runtime(request)
    run, created = call(
        service.accept,
        body.session_id,
        body.request_id,
        body.question,
        "agent",
        routing_input=body.routing(),
    )
    return response_for_events(
        service,
        run.run_id,
        service.run(run.run_id) if created else service.snapshot_events(run.run_id),
    )


@router.post("/stream")
def agent_stream(body: AgentChatRequest, request: Request):
    service = runtime(request)
    run, created = call(
        service.accept,
        body.session_id,
        body.request_id,
        body.question,
        "agent",
        routing_input=body.routing(),
    )
    return StreamingResponse(
        conversation_sse(
            service.run(run.run_id) if created else service.snapshot_events(run.run_id)
        ),
        media_type="text/event-stream",
    )


@router.post("/approvals/{approval_id}", response_model=AgentHttpResponse)
def decide_approval(approval_id: str, body: ApprovalDecision, request: Request):
    service = runtime(request)
    action = call(
        service.decide, approval_id, body.session_id, body.approved, require_pause=True
    )
    return response_for_events(
        service, action.run_id, service.resume(body.session_id, action.run_id)
    )
