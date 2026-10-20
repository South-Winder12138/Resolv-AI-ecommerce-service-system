"""Owned public projections and bounded waits for standalone submission workers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import RLock

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from resolv.api.conversations import runtime
from resolv.schemas.refund import RefundContract, RefundFlowError
from resolv.schemas.refund_submission import (
    RefundOfferView,
    RefundSubmissionView,
    RefundSubmitInput,
)

router = APIRouter(prefix="/conversations")


class RefundResumeInput(RefundContract):
    pass


class RefundWorkers:
    """Process-local coalescing; durable command locks remain authoritative."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="refund-submit"
        )
        self.lock, self.jobs = RLock(), {}

    def start(self, command_id, function):
        with self.lock:
            old = self.jobs.get(command_id)
            if old is not None and not old.done():
                return old
            future = self.executor.submit(function)
            self.jobs[command_id] = future

            def cleanup(done):
                with self.lock:
                    if self.jobs.get(command_id) is done:
                        del self.jobs[command_id]

            future.add_done_callback(cleanup)
            return future

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def http_error(error, command_id=None):
    status = (
        404
        if error.code == "refund_not_found"
        else 422
        if error.code == "refund_request_invalid"
        else 503
        if (
            error.retryable
            or error.code
            in {
                "refund_identity_unavailable",
                "refund_unavailable",
                "refund_store_inconsistent",
                "refund_runtime_unavailable",
            }
        )
        else 409
    )
    return HTTPException(
        status,
        detail={
            "code": error.code,
            "retryable": error.retryable,
            "run_id": error.run_id,
            "submission_id": error.submission_id or command_id,
        },
    )


def call(function, *args, command_id=None, **kwargs):
    try:
        return function(*args, **kwargs)
    except HTTPException:
        raise
    except RefundFlowError as error:
        raise http_error(error, command_id) from None
    except Exception:  # noqa: BLE001 — provider/driver details are never public
        raise http_error(
            RefundFlowError("refund_store_failed", True), command_id
        ) from None


def environment(request):
    owner = getattr(runtime(request), "refund_runtime", None)
    if owner is None:
        raise RefundFlowError("refund_runtime_unavailable", True)
    provider = getattr(request.app.state, "refund_identity_context_provider", None)
    # Only an installed server provider can supply identity context. No HTTP
    # body, customer header or query parameter is promoted here.
    return owner.submissions, provider(request) if provider else None


def response(view, *, pending=False):
    if view.receipt is not None:
        return JSONResponse(view.model_dump(mode="json"))
    if view.error_code is not None:
        raise http_error(
            RefundFlowError(
                view.error_code, view.retryable, submission_id=view.submission_id
            )
        )
    return JSONResponse(
        view.model_dump(mode="json"),
        status_code=202
        if pending or view.status in {"created", "running", "interrupted"}
        else 200,
    )


async def execute(request, service, context, accepted):
    if accepted.status == "completed":
        return response(accepted)
    workers = getattr(request.app.state, "refund_workers", None)
    cid = accepted.submission_id
    if workers is None:
        raise http_error(RefundFlowError("refund_runtime_unavailable", True), cid)
    future = call(
        workers.start, cid, partial(service.execute, cid, context), command_id=cid
    )
    wrapped = asyncio.wrap_future(future)
    # The request can leave after its bounded wait. Retrieve a later exception
    # without logging provider/driver details or cancelling the durable job.
    wrapped.add_done_callback(
        lambda done: None if done.cancelled() else done.exception()
    )
    try:
        view = await asyncio.wait_for(
            asyncio.shield(wrapped),
            timeout=getattr(request.app.state, "refund_wait_seconds", 5),
        )
    except TimeoutError:
        if wrapped.done():
            view = call(wrapped.result, command_id=cid)
        else:
            # Timeout never cancels or reallocates the accepted command.
            return response(
                accepted.model_copy(
                    update={"status": "running", "error_code": None, "retryable": False}
                ),
                pending=True,
            )
    except Exception as error:  # noqa: BLE001 — preserve command association without private exceptions
        raise http_error(
            error
            if isinstance(error, RefundFlowError)
            else RefundFlowError("refund_store_failed", True),
            cid,
        ) from None
    return response(view)


@router.get(
    "/{conversation_id}/refund-offers/{offer_id}", response_model=RefundOfferView
)
def get_offer(conversation_id: str, offer_id: str, request: Request):
    service, context = call(environment, request)
    return call(service.get_offer, conversation_id, offer_id, context)


@router.post(
    "/{conversation_id}/refund-offers/{offer_id}/submit",
    response_model=RefundSubmissionView,
)
async def submit(
    conversation_id: str, offer_id: str, body: RefundSubmitInput, request: Request
):
    # FastAPI has already checked types. Check raw JSON too, before duplicate
    # object keys can be lost by a generic JSON decoder.
    try:
        body = RefundSubmitInput.model_validate_json(await request.body())
    except (ValueError, TypeError):
        raise http_error(RefundFlowError("refund_request_invalid")) from None
    service, context = await run_in_threadpool(call, environment, request)
    accepted = await run_in_threadpool(
        call, service.accept_submission, conversation_id, offer_id, body, context
    )
    return await execute(request, service, context, accepted)


@router.get(
    "/{conversation_id}/refund-submissions/{submission_id}",
    response_model=RefundSubmissionView,
)
def get_submission(conversation_id: str, submission_id: str, request: Request):
    service, context = call(environment, request)
    return call(
        service.get_submission,
        conversation_id,
        submission_id,
        context,
        command_id=submission_id,
    )


@router.post(
    "/{conversation_id}/refund-submissions/{submission_id}/resume",
    response_model=RefundSubmissionView,
)
async def resume(
    conversation_id: str,
    submission_id: str,
    request: Request,
    body: RefundResumeInput | None = None,
):
    service, context = await run_in_threadpool(call, environment, request)
    accepted = await run_in_threadpool(
        call,
        service.get_submission,
        conversation_id,
        submission_id,
        context,
        command_id=submission_id,
    )
    return await execute(request, service, context, accepted)
