"""Independent summary configuration and usage, outside the business step counter."""

from time import monotonic
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI

from resolv.repositories.conversation import ConversationRepository, fingerprint
from resolv.schemas.branch_execution import BusinessAudit
from resolv.schemas.conversation import ConversationError
from resolv.services.summarization import SummaryService


def summary_profile(settings):
    return settings.model_dump(mode="json", exclude={"api_key", "base_url"}) | {
        "api_address_fingerprint": fingerprint(settings.base_url),
        "model_api_mode": "chat_completions"
        if settings.model_name.startswith("qwen3.8-")
        else "responses",
    }


def get_summary_client(settings):
    responses = summary_profile(settings)["model_api_mode"] == "responses"
    params = {
        "model": settings.model_name,
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "use_responses_api": responses,
        "streaming": True,
        "max_retries": 0,
        "max_completion_tokens": settings.reserved_output_tokens,
        "timeout": settings.timeout_seconds,
    }
    if responses:
        params.update(
            output_version="responses/v1",
            use_previous_response_id=False,
            truncation="disabled",
        )
    else:
        params.update(
            stream_usage=True, top_p=0.1, extra_body={"enable_thinking": False}
        )
    return ChatOpenAI(**params)


class AuditedSummaryModel:
    def __init__(self, client, counter, settings, *, audit_hook, attempt=1):
        self.client, self.counter, self.settings, self.audit_hook, self.attempt = (
            client,
            counter,
            settings,
            audit_hook,
            attempt,
        )

    def invoke(self, messages):
        started = monotonic()
        settings = self.settings
        audit = BusinessAudit(
            invocation_id=uuid4().hex,
            purpose="summary",
            model_step=1,
            attempt=self.attempt,
            model_name=settings.model_name,
            profile_fingerprint=fingerprint(summary_profile(settings)),
            estimate=0,
        )
        try:
            audit.estimate = (
                self.counter.count(messages, [])
                + settings.provider_input_overhead_tokens
            )
            if audit.estimate > settings.input_limit:
                raise ConversationError("summary_input_budget_exceeded", True)
            full = None
            for chunk in self.client.bind(
                max_completion_tokens=settings.reserved_output_tokens,
                timeout=settings.timeout_seconds,
            ).stream(messages):
                if not isinstance(chunk, AIMessageChunk):
                    raise ConversationError("summary_response_invalid", True)
                full = chunk if full is None else full + chunk
            if full is None or full.tool_calls or full.invalid_tool_calls:
                raise ConversationError("summary_response_invalid", True)
            usage = full.usage_metadata or {}
            if any(
                type(usage.get(key)) is not int or usage[key] < 0
                for key in ("input_tokens", "output_tokens")
            ):
                raise ConversationError("summary_usage_unknown", True)
            audit.actual_input, audit.actual_output = (
                usage["input_tokens"],
                usage["output_tokens"],
            )
            finish = full.response_metadata.get(
                "finish_reason"
            ) or full.response_metadata.get("status")
            audit.finish_reason = (
                finish
                if finish in {"stop", "completed", "length", "incomplete"}
                else None
            )
            if finish not in {"stop", "completed"} or not full.text.strip():
                raise ConversationError("summary_response_incomplete", True)
            if audit.actual_input > audit.estimate:
                raise ConversationError("summary_input_underestimated", True)
            if audit.actual_output > settings.reserved_output_tokens:
                raise ConversationError("summary_output_budget_exceeded", True)
            answer = AIMessage(
                content=full.text,
                usage_metadata={
                    "input_tokens": audit.actual_input,
                    "output_tokens": audit.actual_output,
                    "total_tokens": audit.actual_input + audit.actual_output,
                },
                response_metadata={"finish_reason": audit.finish_reason},
            )
        except Exception as caught:  # noqa: BLE001 — independent summary provider boundary
            error = (
                caught
                if isinstance(caught, ConversationError)
                else ConversationError("summary_model_unavailable", True)
            )
            audit.error_code = error.code
            audit.elapsed_seconds = monotonic() - started
            self.audit_hook(audit)
            raise error from None
        audit.elapsed_seconds = monotonic() - started
        self.audit_hook(audit)
        return answer


class AuditedSummaryService:
    def __init__(self, client, counter, settings):
        self.client, self.counter, self.settings = client, counter, settings

    def for_run(self, run, execution):
        from resolv.core.config import SummaryModelSettings

        saved = (run.business_config or {}).get("summary_profile")
        if not saved:
            raise ConversationError("run_state_incompatible", run_id=run.run_id)
        if any(
            saved.get(key) != summary_profile(self.settings).get(key)
            for key in ("model_name", "api_address_fingerprint", "model_api_mode")
        ):
            raise ConversationError("run_state_incompatible", run_id=run.run_id)
        values = {
            key: value
            for key, value in saved.items()
            if key in SummaryModelSettings.model_fields
        }
        settings = SummaryModelSettings(
            _env_file=None,
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            **values,
        )

        def audit_hook(audit):
            execution.assert_owned()
            with execution.service.session_factory.begin() as session:
                ConversationRepository(
                    session, execution.service.owner_id
                ).record_model_audit(run.run_id, execution.epoch, audit)

        return SummaryService(
            AuditedSummaryModel(
                self.client,
                self.counter,
                settings,
                audit_hook=audit_hook,
                attempt=run.attempt,
            ),
            self.counter,
            settings,
        )
