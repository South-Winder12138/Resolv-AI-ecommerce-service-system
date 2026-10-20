"""Optional resources, isolated from chat, knowledge and business stores."""

import asyncio
from contextlib import contextmanager
from functools import partial

from pydantic import ValidationError
from sqlalchemy import inspect

from resolv.core.config import (
    ContextSettings,
    ConversationDatabaseSettings,
    Settings,
    UnderstandingSettings,
)
from resolv.core.context_budget import apply_context_budget
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
from resolv.db.checkpoints import checkpoint_session
from resolv.db.conversation import conversation_engine
from resolv.db.engine import create_session_factory
from resolv.repositories.conversation import fingerprint
from resolv.schemas.understanding import UnderstandingError
from resolv.services.context import TokenCounter
from resolv.services.intent_classification import IntentClassifier
from resolv.services.reference_resolution import ReferenceResolver
from resolv.services.understanding import UnderstandingService
from resolv.services.understanding_context import UnderstandingContext


@contextmanager
def resources(options):
    database = ConversationDatabaseSettings()
    with conversation_engine(database) as engine:
        if not {
            "conversation_understandings",
            "conversation_understanding_requests",
        }.issubset(inspect(engine).get_table_names()):
            raise UnderstandingError("understanding_unavailable", True)
        settings, budgets = Settings(), apply_context_budget(ContextSettings())
        counter = TokenCounter(budgets.tokenizer_profile)
        context = UnderstandingContext(
            budgets,
            counter,
            model_name=settings.openai_model,
            api_address_fingerprint=fingerprint(settings.openai_base_url),
            resolution_prompt=RESOLUTION_PROMPT_V1,
            resolution_version=RESOLUTION_VERSION,
            intent_prompt=INTENT_PROMPT_V1,
            intent_version=INTENT_VERSION,
            overhead_override=options.provider_input_overhead_tokens,
            **understanding_model_profile(settings, options.reasoning_effort),
        )
        client = get_understanding_model(settings, options.timeout_seconds)
        try:
            model = JsonStageModel(client, counter)
            yield UnderstandingService(
                create_session_factory(engine),
                partial(checkpoint_session, database),
                context,
                ReferenceResolver(model, context),
                IntentClassifier(model, context),
            )
        finally:
            try:
                client.root_client.close()
            finally:
                asyncio.run(client.root_async_client.close())


@contextmanager
def understanding_runtime(*, required=False):
    try:
        options = UnderstandingSettings()
    except ValidationError:
        if required:
            raise UnderstandingError("understanding_unavailable", True) from None
        yield None
        return
    if not options.enabled and not required:
        yield None
        return
    owner = resources(options)
    try:
        service = owner.__enter__()
    except Exception:  # noqa: BLE001 — independent startup availability, never log credentials
        if required:
            raise UnderstandingError("understanding_unavailable", True) from None
        yield None
        return
    try:
        yield service
    finally:
        try:
            owner.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 — resource cleanup must not leak credentials
            raise UnderstandingError("understanding_unavailable", True) from None
