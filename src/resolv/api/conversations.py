"""Thin persistent conversation HTTP boundary; synchronous work runs in worker threads."""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from resolv.api.sse import conversation_sse
from resolv.schemas.conversation import ConversationError

router = APIRouter(prefix="/conversations")


def runtime(request):
    value = getattr(request.app.state, "conversation_runtime", None)
    if value is None:
        raise HTTPException(
            503,
            detail={
                "code": "conversation_store_failed",
                "retryable": True,
                "run_id": None,
            },
        )
    return value


def http_error(error):
    status = (
        404
        if error.code in {"conversation_not_found", "approval_not_found", "refund_not_found"}
        else 422
        if error.code in {"invalid_query", "invalid_request"}
        else 503
        if error.retryable and error.code not in {"conversation_busy"}
        else 409
    )
    return HTTPException(
        status,
        detail={
            "code": error.code,
            "run_id": error.run_id,
            "retryable": error.retryable,
        },
    )


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ConversationError as error:
        raise http_error(error) from None
    except SQLAlchemyError:
        raise http_error(ConversationError("conversation_store_failed", True)) from None
    except Exception:  # noqa: BLE001 — do not expose model/driver errors from request acceptance
        raise http_error(ConversationError("conversation_store_failed", True)) from None


@router.post("")
def create(request: Request):
    return call(runtime(request).create)


@router.get("")
def list_conversations(
    request: Request, cursor: str | None = None, limit: int = Query(50, ge=1, le=100)
):
    records = call(runtime(request).list, cursor, limit)
    return {
        "conversations": records,
        "next_cursor": records[-1].conversation_id if len(records) == limit else None,
    }


@router.get("/{conversation_id}")
def detail(
    conversation_id: str,
    request: Request,
    before_seq: int | None = Query(None, gt=0),
    limit: int = Query(50, ge=1, le=100),
):
    return call(runtime(request).detail, conversation_id, before_seq, limit)


@router.post("/{conversation_id}/runs/{run_id}/resume")
def resume(conversation_id: str, run_id: str, request: Request):
    service = runtime(request)
    detail = call(service.detail, conversation_id, None, 1)
    run = call(service.get_run, run_id)
    if run.conversation_id != conversation_id or detail.conversation.status != "active":
        raise http_error(ConversationError("run_state_conflict", run_id=run_id))
    return StreamingResponse(
        conversation_sse(service.resume(conversation_id, run_id)),
        media_type="text/event-stream",
    )
