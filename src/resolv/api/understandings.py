"""Four independent synchronous JSON endpoints; no chat or branch execution."""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from resolv.schemas.understanding import (
    UnderstandingError,
    UnderstandingInput,
    UnderstandingView,
)


class UnderstandingRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handle(request):
            id = request.path_params.get("understanding_id")
            try:
                return await original(request)
            except RequestValidationError:
                raise http_error(
                    UnderstandingError("invalid_request", understanding_id=id)
                ) from None
            except HTTPException as error:
                if (
                    isinstance(error.detail, dict)
                    and error.detail.get("understanding_id") is None
                    and id
                ):
                    error.detail = error.detail | {"understanding_id": id}
                raise

        return handle


router = APIRouter(prefix="/conversations", route_class=UnderstandingRoute)


def runtime(request):
    service = getattr(request.app.state, "understanding_runtime", None)
    if service is None:
        raise http_error(UnderstandingError("understanding_unavailable", True))
    return service


def http_error(error):
    status = (
        404
        if error.code
        in {"conversation_not_found", "understanding_not_found", "task_not_found"}
        else 422
        if error.code == "invalid_request"
        else 409
        if error.code
        in {
            "request_conflict",
            "understanding_busy",
            "conversation_archived",
            "understanding_incompatible",
            "task_conflict",
            "clarification_conflict",
            "continuation_conflict",
        }
        else 503
    )
    return HTTPException(
        status,
        detail={
            "code": error.code,
            "retryable": error.retryable,
            "understanding_id": error.understanding_id,
            "related_id": error.related_id,
        },
    )


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except UnderstandingError as error:
        raise http_error(error) from None
    except SQLAlchemyError:
        raise http_error(
            UnderstandingError("understanding_store_failed", True)
        ) from None
    except ValidationError:
        raise http_error(UnderstandingError("understanding_incompatible")) from None


@router.post("/{conversation_id}/understandings", response_model=UnderstandingView)
def create(conversation_id: str, input: UnderstandingInput, request: Request):
    service = runtime(request)
    row, created = call(service.accept, conversation_id, input)
    return call(
        service.run if created else service.get, conversation_id, row.understanding_id
    )


@router.get(
    "/{conversation_id}/understandings/{understanding_id}",
    response_model=UnderstandingView,
)
def get(conversation_id: str, understanding_id: str, request: Request):
    return call(runtime(request).get, conversation_id, understanding_id)


@router.post(
    "/{conversation_id}/understandings/{understanding_id}/resume",
    response_model=UnderstandingView,
)
def resume(conversation_id: str, understanding_id: str, request: Request):
    return call(runtime(request).resume, conversation_id, understanding_id)


@router.get("/{conversation_id}/understanding-tasks")
def tasks(
    conversation_id: str,
    request: Request,
    status: str = Query("pending", pattern="^(pending|selected)$"),
    cursor: str | None = None,
    limit: int = Query(50, ge=1, le=100),
):
    values = call(runtime(request).list_tasks, conversation_id, status, cursor, limit)
    return {
        "requests": values,
        "next_cursor": values[-1].task_id if len(values) == limit else None,
    }
