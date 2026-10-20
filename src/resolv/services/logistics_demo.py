"""Explicitly assembled local demo. Never enters the formal app lifespan."""

import json
import re
from contextlib import ExitStack, asynccontextmanager

from fastapi import FastAPI
from langchain_core.messages import AIMessage, AIMessageChunk
from scripts.logistics_gate_support import (
    REPOSITORY_ROOT,
    owned_logistics_resources,
    owned_profile,
    validate_logistics_ownership,
)

from resolv.core.business_model import get_business_model, make_business_config
from resolv.core.config import (
    ContextSettings,
    Settings,
    SummaryModelSettings,
    UnderstandingSettings,
)
from resolv.core.context_budget import apply_context_budget
from resolv.core.knowledge_branch_config import runtime_knowledge_branch_profile
from resolv.core.refund_config import default_refund_profile
from resolv.core.understanding_model import (
    JsonStageModel,
    get_understanding_model,
    understanding_model_profile,
)
from resolv.core.understanding_prompts import (
    INTENT_PROMPT_V1,
    INTENT_VERSION,
    RESOLUTION_PROMPT_V1,
    RESOLUTION_VERSION,
)
from resolv.repositories.conversation import fingerprint
from resolv.schemas.branch_execution import BusinessRunConfigV2
from resolv.schemas.refund import AdapterDescriptor
from resolv.services.context import ContextBuilder, TokenCounter
from resolv.services.conversation import ConversationService
from resolv.services.intent_classification import IntentClassifier
from resolv.services.logistics_client import MCPLogisticsClient
from resolv.services.logistics_execution import LogisticsReadService
from resolv.services.reference_resolution import ReferenceResolver
from resolv.services.routed_runtime import business_context_settings, close_model
from resolv.services.tool_execution import WriteExecutionService
from resolv.services.understanding import UnderstandingService
from resolv.services.understanding_context import UnderstandingContext


class DemoUnderstandingClient:
    """Deterministic isolated wire driver, not a model or customer inference."""

    def bind(self, **kwargs):
        self.parameters = kwargs
        return self

    def invoke(self, messages):
        payload = json.loads(messages[-1].content.split("\n", 1)[1])
        text = payload["original_text"]
        if "resolution" not in payload:
            result = {"resolved_text": text, "bindings": [], "unresolved_references": [], "continuation_selection": None}
        else:
            result = {"requests": [{
                "text": text, "intent": "logistics", "evidence_refs": [payload["current_input_ref"]],
                "binding_indexes": [], "explicit_priority": None, "priority_evidence_refs": [], "ambiguities": [],
            }]}
        return AIMessage(content=json.dumps(result, ensure_ascii=False), usage_metadata={"input_tokens": 100, "output_tokens": 100, "total_tokens": 200})


class DemoBusinessModel:
    def __init__(self):
        self.names = []

    def bind_tools(self, tools):
        self.names = [tool.name for tool in tools]
        return self

    def bind(self, **kwargs):
        self.parameters = kwargs
        if kwargs.get("tool_choice") == "none":
            self.names = []
        return self

    def stream(self, messages):
        calls, text = [], "模拟物流查询结果已保存，以下为已验证的观察。"
        if self.names:
            name = self.names[0]
            human = next(m.content for m in reversed(messages) if m.type == "human")
            match = re.search(r"MOCK(?:SF)?[0-9]+", human)
            if not match:
                raise ValueError("Demo input requires explicit MOCK object")
            value = match.group()
            if name == "query_logistics" and not value.startswith("MOCKSF"):
                prior = next(json.loads(m.content) for m in reversed(messages) if m.type == "tool" and m.name == "query_order")
                value = prior["tracking_no"]
            key = "order_id" if name == "query_order" else "tracking_no"
            calls = [{"name": name, "id": "demo-" + name, "index": 0, "args": json.dumps({key: value})}]
            text = ""
        else:
            results = [json.loads(m.content) for m in messages if m.type == "tool"]
            if results:
                text += json.dumps(results[-1], ensure_ascii=False)
        yield AIMessageChunk(content=text, tool_call_chunks=calls,
            response_metadata={"finish_reason": "tool_calls" if calls else "stop"},
            usage_metadata={"input_tokens": 100, "output_tokens": 100, "total_tokens": 200})


def create_logistics_demo_app(ownership_path, *, real_models=False, model_guard=None):
    if real_models and model_guard is None:
        raise ValueError("Real models require an injected budget guard")
    identity = validate_logistics_ownership(ownership_path)
    backend = owned_profile(ownership_path)

    @asynccontextmanager
    async def lifespan(app):
        with ExitStack() as cleanup:
            owned = cleanup.enter_context(owned_logistics_resources(ownership_path))
            settings = Settings(_env_file=REPOSITORY_ROOT / ".env") if real_models else Settings(
                _env_file=None, openai_api_key="offline-demo", openai_base_url="https://offline.invalid/v1", openai_model="logistics-demo-fake",
            )
            config = BusinessRunConfigV2.model_validate({
                **make_business_config(settings).model_dump(mode="json"),
                "config_version": 2, "prompt_version": "business-logistics-mcp-v1",
                "logistics_backend": backend,
            })
            budgets = apply_context_budget(ContextSettings(_env_file=REPOSITORY_ROOT / ".env") if real_models else ContextSettings(_env_file=None, model_context_window_tokens=256000, tokenizer_profile="o200k_base"))
            counter = TokenCounter(budgets.tokenizer_profile)
            options = UnderstandingSettings(_env_file=None)
            context = UnderstandingContext(
                budgets, counter, model_name=settings.openai_model,
                api_address_fingerprint=fingerprint(settings.openai_base_url),
                resolution_prompt=RESOLUTION_PROMPT_V1, resolution_version=RESOLUTION_VERSION,
                intent_prompt=INTENT_PROMPT_V1, intent_version=INTENT_VERSION,
                overhead_override=options.provider_input_overhead_tokens,
                **understanding_model_profile(settings, options.reasoning_effort),
            )
            if real_models:
                uclient = get_understanding_model(settings, options.timeout_seconds)
                cleanup.callback(close_model, uclient)
                bclient = get_business_model(settings, config)
                cleanup.callback(close_model, bclient)
                umodel = model_guard("understanding", JsonStageModel(uclient, counter))
                bclient = model_guard("business", bclient)
            else:
                umodel, bclient = JsonStageModel(DemoUnderstandingClient(), counter), DemoBusinessModel()
            understanding = UnderstandingService(
                owned.conversation_factory, owned.checkpoint_factory, context,
                ReferenceResolver(umodel, context), IntentClassifier(umodel, context), owner_id=identity["owner_id"],
            )
            summary_settings = SummaryModelSettings(
                _env_file=None, api_key="offline", base_url="https://offline.invalid/v1", model_name="disabled-summary",
                model_context_window_tokens=256000,
            )
            knowledge = runtime_knowledge_branch_profile(settings, understanding_config=context.config, business_config=config, summary_settings=summary_settings)
            refund = default_refund_profile(
                understanding_config=context.config, business_config=config, summary_profile={},
                adapter_descriptor=AdapterDescriptor(namespace="unconnected", implementation_version="1", authorization_version="1"),
            )
            service = ConversationService(
                owned.conversation_factory, owned.checkpoint_factory, owner_id=identity["owner_id"],
                understanding=understanding, orchestration_version="routed-v4",
                business_run_config=config, knowledge_profile=knowledge, refund_profile=refund,
                release_selector_path=ownership_path.parent / "disabled-selector.json",
                model=bclient, context_builder=ContextBuilder(business_context_settings(config, budgets), counter),
                write_service=WriteExecutionService(owned.conversation_factory, owned.business_factory),
                logistics_read_service=LogisticsReadService(owned.conversation_factory, MCPLogisticsClient(backend), backend),
                run_config={"prompt_version": config.prompt_version, "state_schema_version": 4},
            )
            app.state.conversation_runtime = service
            app.state.understanding_runtime = understanding
            app.state.knowledge_chat_runtime = None
            app.state.refund_workers = None
            try:
                yield
            finally:
                app.state.conversation_runtime = app.state.understanding_runtime = None

    app = FastAPI(title="Resolv isolated logistics demo", lifespan=lifespan)
    from resolv.api.agent import router as agent_router
    from resolv.api.chat import router as chat_router
    from resolv.api.conversations import router as conversations_router
    from resolv.api.understandings import router as understandings_router

    for router in (agent_router, chat_router, conversations_router, understandings_router):
        app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_mode": "real" if real_models else "fake", "owner_id": identity["owner_id"], "resource_id": identity["resource_id"], "source_revision": identity["source_revision"]}

    return app
