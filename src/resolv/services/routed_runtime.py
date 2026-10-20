"""Lifespan resources for routed chat, including a lazy knowledge resource pool."""

import asyncio
from contextlib import ExitStack, contextmanager
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url

from resolv.core.business_model import get_business_model, make_business_config
from resolv.core.config import (
    ContextSettings,
    ConversationExecutionSettings,
    Settings,
    SummaryModelSettings,
)
from resolv.core.context_budget import apply_context_budget, apply_summary_budget
from resolv.core.knowledge_branch_config import runtime_knowledge_branch_profile
from resolv.core.knowledge_branch_model import KnowledgeModelClient
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.schemas.conversation import ConversationError
from resolv.schemas.understanding import UnderstandingError
from resolv.services.business_summary import (
    AuditedSummaryService,
    get_summary_client,
    summary_profile,
)
from resolv.services.context import ContextBuilder, TokenCounter
from resolv.services.conversation import ConversationService
from resolv.services.knowledge_branch_runtime import default_knowledge_retrievers
from resolv.services.tool_execution import WriteExecutionService
from resolv.services.understanding_runtime import understanding_runtime


def close_model(client):
    try:
        try:
            client.root_client.close()
        finally:
            asyncio.run(client.root_async_client.close())
    except Exception:  # noqa: BLE001 — credentials can occur in client teardown errors
        raise ConversationError("model_close_failed", True) from None


def business_context_settings(config, base=None):
    data = base.model_dump() if base is not None else {}
    data.update(
        {
            key: getattr(config, key)
            for key in (
                "tokenizer_profile",
                "model_context_window_tokens",
                "app_max_input_tokens",
                "provider_input_overhead_tokens",
                "reserved_output_tokens",
                "safety_margin_tokens",
                "recent_history_target_tokens",
            )
        }
    )
    # The business branch uses this builder without knowledge. The knowledge
    # branch creates its own builder with the frozen evidence budget.
    data["knowledge_context_max_tokens"] = 1
    return ContextSettings(_env_file=None, **data)


def validate_conversation_runtime_store(factory, *, require_logistics=False):
    engine = factory.kw["bind"]
    required = {
        "conversations",
        "conversation_runs",
        "conversation_messages",
        "conversation_actions",
        "conversation_summaries",
        "conversation_understandings",
        "conversation_understanding_requests",
        "conversation_knowledge_runs", "conversation_knowledge_review_items",
        "conversation_refund_runs", "conversation_refund_offers", "conversation_refund_submissions", "conversation_refund_request_aliases",
        "checkpoints",
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoint_migrations",
        "conversation_alembic_version",
    }
    if not required <= set(inspect(engine).get_table_names()):
        raise ConversationError("conversation_store_failed", True)
    with factory() as session:
        revision = session.scalar(text("SELECT version_num FROM conversation_alembic_version"))
    if revision not in {"20261016_0005", "20261020_0006"}:
        raise ConversationError("conversation_store_failed", True)
    if (require_logistics or revision == "20261020_0006") and "conversation_logistics_attempts" not in inspect(engine).get_table_names():
        raise ConversationError("conversation_store_failed", True)
    if require_logistics and revision != "20261020_0006":
        raise ConversationError("conversation_store_failed", True)


@contextmanager
def routed_conversation_runtime():
    try:
        with ExitStack() as cleanup:
            understanding = cleanup.enter_context(understanding_runtime(required=True))
            if understanding is None:
                raise ConversationError("understanding_unavailable", True)
            factory = understanding.session_factory
            validate_conversation_runtime_store(factory)
            settings, execution_settings = Settings(), ConversationExecutionSettings()
            config = make_business_config(settings)
            summary_settings = apply_summary_budget(SummaryModelSettings())
            config.summary_profile = summary_profile(summary_settings)
            budgets = business_context_settings(config, apply_context_budget(ContextSettings()))
            counter = TokenCounter(config.tokenizer_profile)
            business_url = make_url(settings.database_url)
            if (
                business_url.drivername != "sqlite"
                or not business_url.database
                or not Path(business_url.database).is_file()
            ):
                raise ConversationError("conversation_store_failed", True)
            business_engine = create_db_engine(settings.database_url)
            cleanup.callback(business_engine.dispose)
            if not {"orders", "refunds", "shipments", "tool_execution_receipts", "refund_applications", "refund_submission_receipts"} <= set(
                inspect(business_engine).get_table_names()
            ):
                raise ConversationError("conversation_store_failed", True)
            model = get_business_model(settings, config)
            cleanup.callback(close_model, model)
            summary_model = get_summary_client(summary_settings)
            cleanup.callback(close_model, summary_model)
            knowledge_retrievers = default_knowledge_retrievers()
            cleanup.callback(knowledge_retrievers.close)
            knowledge_client = KnowledgeModelClient(settings)
            cleanup.callback(knowledge_client.close)
            knowledge_profile = runtime_knowledge_branch_profile(settings,
                understanding_config=understanding.context.config, business_config=config, summary_settings=summary_settings)
            service = ConversationService(
                factory,
                understanding.checkpoint_factory,
                owner_id=execution_settings.owner_id,
                lease_seconds=execution_settings.lease_seconds,
                heartbeat_seconds=execution_settings.heartbeat_seconds,
                understanding=understanding,
                orchestration_version="routed-v4",
                knowledge_profile=knowledge_profile,
                knowledge_clients={purpose: knowledge_client for purpose in ("knowledge_query", "knowledge_check", "knowledge_answer")},
                knowledge_retrievers=knowledge_retrievers,
                business_run_config=config,
                model=model,
                context_builder=ContextBuilder(budgets, counter),
                summary_service=AuditedSummaryService(
                    summary_model, counter, summary_settings
                ),
                max_iterations=config.max_agent_steps,
                write_service=WriteExecutionService(
                    factory, create_session_factory(business_engine)
                ),
                run_config={
                    "prompt_version": config.prompt_version,
                    "state_schema_version": 4,
                },
            )
            service.knowledge_retrievers = knowledge_retrievers
            from resolv.services.refund_runtime import create_refund_runtime
            service.refund_runtime = create_refund_runtime(service, settings=settings)
            service.refund_profile = service.refund_runtime.profile
            cleanup.callback(service.refund_runtime.close)
            yield service
    except ConversationError:
        raise
    except UnderstandingError as error:
        raise ConversationError(error.code, error.retryable) from None
    except Exception:  # noqa: BLE001 — initialization never exposes credentials
        raise ConversationError("conversation_store_failed", True) from None
