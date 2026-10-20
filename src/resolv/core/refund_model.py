"""Audited refund-only transport: complete native reads or strict public JSON."""

import asyncio
from contextlib import ExitStack
from time import monotonic
from uuid import uuid4

from langchain_core.messages import (
    AIMessageChunk,
    SystemMessage,
    messages_to_dict,
)
from langchain_openai import ChatOpenAI

from resolv.core.refund_diagnostics import failure_diagnostic, transport_diagnostic
from resolv.core.refund_native import (
    RETURN_DECISION,
    decision_tools,
    native_decision_schema,
    uses_native_return,
)
from resolv.core.refund_prompts import (
    COMPLETE_DECISION_PROMPT_VERSION,
    CONTEXT_IDENTITY_PROMPT_VERSION,
    STRUCTURED_DECISION_PROMPT_VERSION,
)
from resolv.core.refund_response import (
    ResponseDecodeError,
    decode_refund_response,
    strict_json,
)
from resolv.core.understanding_model import compact_schema
from resolv.db.conversation_base import utc_now
from resolv.schemas.knowledge_branch import canonical_json
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_flow import (
    RefundDecision,
    RefundDecisionOutput,
    RefundModelAudit,
    RefundModelProfile,
    RefundModelResponse,
    RefundQueryOutput,
)
from resolv.services.context import TokenCounter
from resolv.services.hybrid_fingerprints import canonical_hash

__all__ = [
    "RefundModelClient",
    "refund_input_estimate",
    "refund_request_parameters",
    "strict_json",
]


def refund_request_parameters(
    profile, *, prepare=False, require_tool=False, final_round=False, required_tool=None
):
    if final_round and (prepare or require_tool):
        raise RefundFlowError("refund_model_profile_mismatch")
    params = {
        "max_completion_tokens": profile.reserved_output_tokens,
        "timeout": profile.timeout_seconds,
    }
    if profile.top_p is not None:
        params["top_p"] = profile.top_p
    if profile.reasoning_effort is not None:
        params["reasoning_effort"] = profile.reasoning_effort
    if profile.enable_thinking is not None:
        params["extra_body"] = {"enable_thinking": profile.enable_thinking}
        if profile.thinking_budget is not None:
            params["extra_body"]["thinking_budget"] = profile.thinking_budget
    if prepare:
        schema = compact_schema(RefundQueryOutput.model_json_schema())
        if profile.output_schema is None or profile.output_schema.value != schema:
            raise RefundFlowError("refund_model_profile_mismatch")
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "refund_query", "strict": True, "schema": schema},
        }
    elif uses_native_return(profile):
        if (
            profile.output_schema is None
            or profile.output_schema.value != native_decision_schema()
        ):
            raise RefundFlowError("refund_model_profile_mismatch")
        if require_tool:
            if required_tool not in {"query_refund_order", "query_refund_status"}:
                raise RefundFlowError("refund_model_profile_mismatch")
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": required_tool},
            }
        elif final_round:
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": RETURN_DECISION},
            }
    elif final_round:
        params["tool_choice"] = "none"
    elif require_tool:
        params["tool_choice"] = "required"
    if (
        not prepare
        and not require_tool
        and profile.prompt_version
        in {
            STRUCTURED_DECISION_PROMPT_VERSION,
            COMPLETE_DECISION_PROMPT_VERSION,
            CONTEXT_IDENTITY_PROMPT_VERSION,
        }
    ):
        schema = (
            RefundDecisionOutput
            if profile.prompt_version
            in {COMPLETE_DECISION_PROMPT_VERSION, CONTEXT_IDENTITY_PROMPT_VERSION}
            else RefundDecision
        ).model_json_schema()
        if profile.output_schema is None or profile.output_schema.value != schema:
            raise RefundFlowError("refund_model_profile_mismatch")
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "refund_decision",
                "strict": True,
                "schema": compact_schema(schema),
            },
        }
    return params


def refund_input_estimate(
    messages,
    profile,
    counter,
    *,
    prepare=False,
    require_tool=False,
    final_round=False,
    required_tool=None,
):
    schemas = (
        ()
        if prepare
        else decision_tools(profile, require_tool=require_tool, final_round=final_round)
    )
    count = counter.count(messages, schemas)
    parameters = counter.count_text(
        canonical_json(
            refund_request_parameters(
                profile,
                prepare=prepare,
                require_tool=require_tool,
                final_round=final_round,
                required_tool=required_tool,
            )
        )
    )
    if any(type(v) is not int or v < 0 for v in (count, parameters)):
        raise RefundFlowError("refund_model_token_count_invalid")
    return count + parameters + profile.provider_input_overhead_tokens


class RefundModelClient:
    def __init__(
        self,
        settings,
        *,
        client_factory=ChatOpenAI,
        counter_factory=TokenCounter,
        diagnostic_hook=None,
    ):
        self.settings, self.client_factory, self.counter_factory = (
            settings,
            client_factory,
            counter_factory,
        )
        self.diagnostic_hook = diagnostic_hook
        self._clients = {}
        self._closed = False

    def _emit_diagnostic(self, diagnostic):
        try:
            self.diagnostic_hook(diagnostic)
        except Exception:  # noqa: BLE001 -- an explicitly enabled evidence sink must not fail silently
            raise RefundFlowError(
                "refund_model_diagnostic_store_failed", True
            ) from None

    def _client(self, profile):
        if (
            self.settings.openai_model != profile.model_name
            or canonical_hash(self.settings.openai_base_url)
            != profile.api_address_fingerprint
        ):
            raise RefundFlowError("refund_model_profile_mismatch")
        if profile.fingerprint not in self._clients:
            key = self.settings.openai_api_key
            params = {
                "model": profile.model_name,
                "api_key": key.get_secret_value()
                if hasattr(key, "get_secret_value")
                else key,
                "base_url": self.settings.openai_base_url,
                "max_retries": 0,
                "timeout": profile.timeout_seconds,
                "use_responses_api": profile.model_api_mode == "responses",
                "streaming": profile.purpose == "refund_decision"
                or profile.model_api_mode == "responses",
            }
            if profile.model_api_mode == "responses":
                params.update(
                    output_version="responses/v1",
                    use_previous_response_id=False,
                    truncation="disabled",
                )
            else:
                params["stream_usage"] = True
            self._clients[profile.fingerprint] = self.client_factory(**params)
        return self._clients[profile.fingerprint]

    def close(self):
        if self._closed:
            return
        self._closed = True

        def close_one(client):
            try:
                try:
                    client.root_client.close()
                finally:
                    asyncio.run(client.root_async_client.close())
            except Exception as exc:
                raise RefundFlowError("refund_model_close_failed", True) from exc

        with ExitStack() as stack:
            for client in self._clients.values():
                stack.callback(close_one, client)
            self._clients.clear()

    def prepare(self, messages, *, profile, audit_hook):
        output, _ = self._invoke(
            messages,
            profile=profile,
            model_step=0,
            attempt=1,
            require_tool=False,
            audit_hook=audit_hook,
            prepare=True,
        )
        return output

    def decide(
        self,
        messages,
        *,
        profile,
        model_step,
        attempt,
        require_tool,
        audit_hook,
        final_round=False,
        required_tool=None,
    ):
        output, audit = self._invoke(
            messages,
            profile=profile,
            model_step=model_step,
            attempt=attempt,
            require_tool=require_tool,
            final_round=final_round,
            required_tool=required_tool,
            audit_hook=audit_hook,
            prepare=False,
        )
        calls, decision = output
        value = {
            "tool_calls": [c.model_dump(mode="json") for c in calls],
            "decision": decision.model_dump(mode="json") if decision else None,
        }
        return RefundModelResponse(
            model_step=model_step,
            tool_calls=calls,
            decision=decision,
            input_hash=audit.input_hash,
            output_hash=canonical_hash(value),
            audit=audit,
        )

    def _invoke(
        self,
        messages,
        *,
        profile,
        model_step,
        attempt,
        require_tool,
        audit_hook,
        prepare,
        final_round=False,
        required_tool=None,
    ):
        if self._closed:
            raise RefundFlowError("refund_model_client_closed")
        profile = RefundModelProfile.model_validate_json(profile.model_dump_json())
        purpose = "refund_query" if prepare else "refund_decision"
        expected_tools = canonical_hash([] if prepare else decision_tools(profile))
        if (
            profile.purpose != purpose
            or profile.tool_schema_fingerprint != expected_tools
            or not messages
            or not isinstance(messages[0], SystemMessage)
            or messages[0].content != profile.system_prompt
        ):
            raise RefundFlowError("refund_model_profile_mismatch")
        params = refund_request_parameters(
            profile,
            prepare=prepare,
            require_tool=require_tool,
            final_round=final_round,
            required_tool=required_tool,
        )
        counter = self.counter_factory(profile.tokenizer_profile)
        estimate = refund_input_estimate(
            messages,
            profile,
            counter,
            prepare=prepare,
            require_tool=require_tool,
            final_round=final_round,
            required_tool=required_tool,
        )
        if estimate > profile.input_limit:
            raise RefundFlowError("refund_model_input_budget_exceeded")
        audit = RefundModelAudit(
            invocation_id=uuid4().hex,
            purpose=purpose,
            profile_fingerprint=profile.fingerprint,
            input_hash=canonical_hash(messages_to_dict(messages)),
            model_step=model_step,
            attempt=attempt,
            input_estimate=estimate,
            started_at=utc_now(),
            outcome="started",
        )
        start = monotonic()
        audit_hook(audit)
        try:
            model = self._client(profile)
            if not prepare:
                model = model.bind_tools(
                    list(
                        decision_tools(
                            profile, require_tool=require_tool, final_round=final_round
                        )
                    )
                )
            model = model.bind(**params)
            if prepare and profile.model_api_mode == "chat_completions":
                reply = model.invoke(messages)
            else:
                reply = None
                for chunk in model.stream(messages):
                    if not isinstance(chunk, AIMessageChunk):
                        raise RefundFlowError("refund_model_output_invalid", True)
                    reply = chunk if reply is None else reply + chunk
        except Exception as exc:  # noqa: BLE001 -- retain no private provider text
            error = (
                exc
                if isinstance(exc, RefundFlowError)
                else RefundFlowError("refund_model_unavailable", True)
            )
            audit_hook(
                audit.model_copy(
                    update={
                        "outcome": "failed"
                        if isinstance(exc, RefundFlowError)
                        else "unknown",
                        "error_code": error.code,
                        "finished_at": utc_now(),
                        "elapsed_seconds": monotonic() - start,
                    }
                )
            )
            if self.diagnostic_hook is not None:
                self._emit_diagnostic(transport_diagnostic(exc, error.code, audit))
            raise error from None
        try:
            output, usage, finish = decode_refund_response(
                reply,
                profile,
                prepare=prepare,
                require_tool=require_tool,
                final_round=final_round,
                counter=counter,
                estimate=estimate,
                required_tool=required_tool,
            )
        except ResponseDecodeError as exc:
            error = RefundFlowError(exc.code, exc.retryable)
            usage, finish = exc.usage, exc.finish

            def known(key):
                return (
                    usage[key]
                    if type(usage.get(key)) is int and usage[key] >= 0
                    else None
                )

            audit_hook(
                audit.model_copy(
                    update={
                        "outcome": "failed",
                        "error_code": error.code,
                        "input_tokens": known("input_tokens"),
                        "output_tokens": known("output_tokens"),
                        "finished_at": utc_now(),
                        "elapsed_seconds": monotonic() - start,
                        "finish_reason": finish
                        if finish
                        in {
                            "stop",
                            "completed",
                            "tool_calls",
                            "length",
                            "content_filter",
                        }
                        else None,
                    }
                )
            )
            if self.diagnostic_hook is not None:
                self._emit_diagnostic(
                    failure_diagnostic(
                        exc,
                        reply,
                        audit,
                        prepare=prepare,
                        require_tool=require_tool,
                        final_round=final_round,
                        required_tool=required_tool,
                    )
                )
            raise error from None
        audit = audit.model_copy(
            update={
                "outcome": "succeeded",
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "finished_at": utc_now(),
                "elapsed_seconds": monotonic() - start,
                "finish_reason": finish,
            }
        )
        audit_hook(audit)
        return output, audit
