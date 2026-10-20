import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from resolv.api.agent import router as agent_router
from resolv.api.chat import router as chat_router
from resolv.api.conversations import router as conversations_router
from resolv.api.refunds import RefundWorkers
from resolv.api.refunds import router as refunds_router
from resolv.api.understandings import router as understandings_router
from resolv.services.routed_runtime import routed_conversation_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.knowledge_chat_runtime = None
    app.state.conversation_runtime = None
    app.state.understanding_runtime = None
    owner = routed_conversation_runtime()
    entered = False
    workers = None
    try:
        service = await run_in_threadpool(owner.__enter__)
        entered = True
        app.state.conversation_runtime = service
        app.state.understanding_runtime = service.understanding
        workers = RefundWorkers()
        app.state.refund_workers = workers
        yield
    finally:
        info = sys.exc_info()
        app.state.conversation_runtime = None
        app.state.understanding_runtime = None
        app.state.knowledge_chat_runtime = None
        app.state.refund_workers = None
        try:
            if workers is not None:
                await run_in_threadpool(workers.close)
        finally:
            if entered:
                await run_in_threadpool(owner.__exit__, *info)


app = FastAPI(title="Resolv", lifespan=lifespan)
app.state.conversation_runtime = None
app.state.knowledge_chat_runtime = None
app.state.understanding_runtime = None
app.state.refund_workers = None
app.include_router(understandings_router)
app.include_router(conversations_router)
app.include_router(refunds_router)
app.include_router(chat_router)
app.include_router(agent_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
