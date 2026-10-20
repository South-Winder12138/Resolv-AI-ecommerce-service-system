from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import Field

from resolv.api.conversations import call, runtime
from resolv.api.sse import conversation_sse
from resolv.schemas.branch_execution import RoutingInput

router = APIRouter(prefix="/chat")


class ChatRequest(RoutingInput):
    session_id: str = Field(min_length=1, max_length=64)
    request_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1)

    def routing(self):
        return RoutingInput(
            clarifies_understanding_id=self.clarifies_understanding_id,
            continue_task_id=self.continue_task_id,
            reassesses_refund_offer_id=self.reassesses_refund_offer_id,
        )


@router.post("/stream")
def stream_chat_endpoint(body: ChatRequest, request: Request):
    service = runtime(request)
    run, created = call(
        service.accept,
        body.session_id,
        body.request_id,
        body.question,
        "chat",
        routing_input=body.routing(),
    )
    events = service.run(run.run_id) if created else service.snapshot_events(run.run_id)
    return StreamingResponse(conversation_sse(events), media_type="text/event-stream")


@router.delete("/sessions/{session_id}")
def archive_session(session_id: str, request: Request):
    call(runtime(request).archive, session_id)
    return {"status": "archived"}
