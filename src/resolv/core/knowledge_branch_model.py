"""Audited, tool-free transport. Only complete validated public output escapes."""

import asyncio
import json
from contextlib import ExitStack
from time import monotonic
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage
from langchain_openai import ChatOpenAI

from resolv.core.knowledge_branch_prompts import PROMPTS, KnowledgeQueryOutput
from resolv.core.understanding_model import (
    compact_schema,
    public_text,
    understanding_model_profile,
    unique_object,
)
from resolv.db.conversation_base import utc_now
from resolv.schemas.knowledge_branch import (
    FrozenJSON,
    KnowledgeBranchError,
    KnowledgeModelAudit,
    KnowledgeModelProfile,
    SufficiencyResult,
    canonical_json,
)
from resolv.services.context import TokenCounter
from resolv.services.hybrid_fingerprints import canonical_hash


def make_knowledge_model_profile(settings, *, purpose, reference_profile):
    if purpose not in PROMPTS:
        raise KnowledgeBranchError("knowledge_model_profile_mismatch")
    ref = (
        reference_profile.model_dump()
        if hasattr(reference_profile, "model_dump")
        else dict(reference_profile)
    )
    if purpose == "knowledge_answer":
        rule = {
            "model_api_mode": ref.get("model_api_mode", "chat_completions"),
            "enable_thinking": ref.get("enable_thinking", False),
            "thinking_budget": None,
            "top_p": ref.get("top_p", 0.1),
            "reasoning_effort": ref.get("reasoning_effort"),
        }
    else:
        rule = understanding_model_profile(
            settings, ref.get("reasoning_effort") or "low"
        )
    schema = {
        "knowledge_query": KnowledgeQueryOutput,
        "knowledge_check": SufficiencyResult,
    }.get(purpose)
    return KnowledgeModelProfile(
        purpose=purpose,
        model_name=settings.openai_model,
        api_address_fingerprint=canonical_hash(settings.openai_base_url),
        model_api_mode=rule["model_api_mode"],
        tokenizer_profile=ref["tokenizer_profile"],
        model_context_window_tokens=ref["model_context_window_tokens"],
        app_max_input_tokens=ref["app_max_input_tokens"],
        provider_input_overhead_tokens=ref["provider_input_overhead_tokens"],
        reserved_output_tokens=ref["reserved_output_tokens"],
        safety_margin_tokens=ref["safety_margin_tokens"],
        recent_history_target_tokens=ref.get("recent_history_target_tokens", 2000),
        timeout_seconds=float(ref.get("timeout_seconds", 60)),
        max_retries=0,
        enable_thinking=rule.get("enable_thinking"),
        thinking_budget=rule.get("thinking_budget"),
        top_p=rule.get("top_p"),
        reasoning_effort=rule.get("reasoning_effort"),
        system_prompt=PROMPTS[purpose],
        prompt_version=f"{purpose}-v2" if purpose == "knowledge_check" else f"{purpose}-v1",
        output_format=(rule.get("output_format", "json_schema") if schema else "text"),
        output_schema=FrozenJSON.from_value(compact_schema(schema.model_json_schema()))
        if schema
        else None,
        streaming=purpose == "knowledge_answer"
        or rule["model_api_mode"] == "responses",
    )


def _invalid_constant(_):
    raise ValueError("Nonfinite JSON")


def knowledge_request_parameters(profile, output_type=None):
    params = {
        "max_completion_tokens": profile.reserved_output_tokens,
        "timeout": profile.timeout_seconds,
    }
    if output_type is not None:
        schema = compact_schema(output_type.model_json_schema())
        if profile.output_schema is None or profile.output_schema.value != schema:
            raise KnowledgeBranchError("knowledge_model_profile_mismatch")
        if profile.output_format == "json_schema":
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": profile.purpose,
                    "strict": True,
                    "schema": schema,
                },
            }
        elif profile.output_format == "json_object":
            params["response_format"] = {"type": "json_object"}
        else:
            raise KnowledgeBranchError("knowledge_model_profile_mismatch")
    if profile.top_p is not None:
        params["top_p"] = profile.top_p
    if profile.reasoning_effort is not None:
        params["reasoning_effort"] = profile.reasoning_effort
    if profile.enable_thinking is not None:
        params["extra_body"] = {"enable_thinking": profile.enable_thinking}
        if profile.thinking_budget is not None:
            params["extra_body"]["thinking_budget"] = profile.thinking_budget
    return params


class KnowledgeModelClient:
    def __init__(
        self, settings, *, client_factory=ChatOpenAI, counter_factory=TokenCounter
    ):
        self.settings, self.client_factory, self.counter_factory = (
            settings,
            client_factory,
            counter_factory,
        )
        self._clients = {}
        self._closed = False

    def close(self):
        if self._closed:
            return
        self._closed = True

        def close_client(client):
            try:
                try:
                    client.root_client.close()
                finally:
                    asyncio.run(client.root_async_client.close())
            except Exception as exc:
                raise KnowledgeBranchError(
                    "knowledge_model_close_failed", True
                ) from exc

        with ExitStack() as cleanup:
            for client in self._clients.values():
                cleanup.callback(close_client, client)
            self._clients.clear()

    def _client(self, profile):
        if (
            self.settings.openai_model != profile.model_name
            or canonical_hash(self.settings.openai_base_url)
            != profile.api_address_fingerprint
        ):
            raise KnowledgeBranchError("knowledge_model_profile_mismatch")
        if profile.fingerprint not in self._clients:
            key = self.settings.openai_api_key
            parameters = {
                "model": profile.model_name,
                "api_key": key.get_secret_value()
                if hasattr(key, "get_secret_value")
                else key,
                "base_url": self.settings.openai_base_url,
                "max_retries": 0,
                "use_responses_api": profile.model_api_mode == "responses",
                "streaming": profile.streaming,
                "timeout": profile.timeout_seconds,
            }
            if profile.model_api_mode == "responses":
                parameters.update(
                    output_version="responses/v1",
                    use_previous_response_id=False,
                    truncation="disabled",
                )
            else:
                parameters["stream_usage"] = True
            self._clients[profile.fingerprint] = self.client_factory(**parameters)
        return self._clients[profile.fingerprint]

    def invoke_json(self, purpose, messages, output_type, *, profile, audit_hook):
        if purpose not in {"knowledge_query", "knowledge_check"}:
            raise KnowledgeBranchError("knowledge_model_profile_mismatch")
        return self._invoke(purpose, messages, output_type, profile, audit_hook)

    def invoke_answer(self, messages, *, profile, audit_hook):
        return self._invoke("knowledge_answer", messages, None, profile, audit_hook)

    def _invoke(self, purpose, messages, output_type, profile, audit_hook):
        if self._closed:
            raise KnowledgeBranchError("knowledge_model_client_closed")
        profile = KnowledgeModelProfile.model_validate_json(profile.model_dump_json())
        if (
            profile.purpose != purpose
            or not messages
            or not isinstance(messages[0], SystemMessage)
            or messages[0].content != profile.system_prompt
        ):
            raise KnowledgeBranchError("knowledge_model_profile_mismatch")
        params = knowledge_request_parameters(profile, output_type)
        counter = self.counter_factory(profile.tokenizer_profile)
        estimate = (
            counter.count(messages, ())
            + counter.count_text(canonical_json(params))
            + profile.provider_input_overhead_tokens
        )
        if type(estimate) is not int or estimate < 0 or estimate > profile.input_limit:
            raise KnowledgeBranchError("knowledge_model_input_budget_exceeded", True)
        started = monotonic()
        audit = KnowledgeModelAudit(
            invocation_id=uuid4().hex,
            purpose=purpose,
            profile_fingerprint=profile.fingerprint,
            input_hash=canonical_hash([m.model_dump(mode="json") for m in messages]),
            input_estimate=estimate,
            started_at=utc_now(),
            outcome="started",
        )
        # Hooks deliberately sit outside the provider exception boundary.
        audit_hook(audit)
        try:
            model = self._client(profile).bind(**params)
            if output_type is not None:
                reply = model.invoke(messages)
            else:
                reply = None
                for chunk in model.stream(messages):
                    if not isinstance(chunk, AIMessageChunk):
                        raise KnowledgeBranchError(
                            "knowledge_model_output_invalid", True
                        )
                    reply = chunk if reply is None else reply + chunk
        except Exception as exc:  # noqa: BLE001 -- provider details never enter public errors
            error = (
                exc
                if isinstance(exc, KnowledgeBranchError)
                else KnowledgeBranchError("knowledge_model_unavailable", True)
            )
            outcome = "failed" if isinstance(exc, KnowledgeBranchError) else "unknown"
            audit_hook(
                audit.model_copy(
                    update={
                        "outcome": outcome,
                        "error_code": error.code,
                        "finished_at": utc_now(),
                        "elapsed_seconds": monotonic() - started,
                    }
                )
            )
            raise error from None
        usage = (reply.usage_metadata or {}) if isinstance(reply, AIMessage) else {}
        finish = None
        try:
            if (
                not isinstance(reply, AIMessage)
                or reply.tool_calls
                or reply.invalid_tool_calls
                or reply.additional_kwargs.get("tool_calls")
            ):
                raise KnowledgeBranchError("knowledge_model_output_invalid", True)
            metadata = reply.response_metadata
            finish = metadata.get("finish_reason")
            if profile.model_api_mode == "responses":
                finish = (
                    "completed" if metadata.get("status") == "completed" else finish
                )
            if finish not in {"stop", "completed"} or metadata.get("status") in {
                "failed",
                "incomplete",
            }:
                raise KnowledgeBranchError("knowledge_model_output_invalid", True)
            if any(
                type(usage.get(k)) is not int or usage[k] < 0
                for k in ("input_tokens", "output_tokens", "total_tokens")
            ):
                raise KnowledgeBranchError("knowledge_model_usage_unknown", True)
            if usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]:
                raise KnowledgeBranchError("knowledge_model_usage_unknown", True)
            if usage["input_tokens"] > estimate:
                raise KnowledgeBranchError("knowledge_model_input_underestimated", True)
            text = public_text(reply)
            if not text.strip():
                raise KnowledgeBranchError("knowledge_model_output_invalid", True)
            if (
                usage["output_tokens"] > profile.reserved_output_tokens
                or counter.count_text(text) > profile.reserved_output_tokens
            ):
                raise KnowledgeBranchError(
                    "knowledge_model_output_budget_exceeded", True
                )
            if output_type is not None:
                data = json.loads(
                    text,
                    object_pairs_hook=unique_object,
                    parse_constant=_invalid_constant,
                )
                output = output_type.model_validate_json(canonical_json(data))
            else:
                output = AIMessage(
                    content=text,
                    response_metadata={"finish_reason": finish},
                    usage_metadata=usage,
                )
        except Exception as exc:  # noqa: BLE001 -- malformed provider output is a technical failure
            error = (
                exc
                if isinstance(exc, KnowledgeBranchError)
                else KnowledgeBranchError("knowledge_model_output_invalid", True)
            )
            audit_hook(
                audit.model_copy(
                    update={
                        "outcome": "failed",
                        "error_code": error.code,
                        "input_tokens": usage.get("input_tokens")
                        if type(usage.get("input_tokens")) is int
                        and usage["input_tokens"] >= 0
                        else None,
                        "output_tokens": usage.get("output_tokens")
                        if type(usage.get("output_tokens")) is int
                        and usage["output_tokens"] >= 0
                        else None,
                        "finished_at": utc_now(),
                        "elapsed_seconds": monotonic() - started,
                        "finish_reason": finish
                        if finish in {"stop", "completed", "length", "content_filter"}
                        else None,
                    }
                )
            )
            raise error from None
        audit_hook(
            audit.model_copy(
                update={
                    "outcome": "succeeded",
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "finish_reason": finish,
                    "finished_at": utc_now(),
                    "elapsed_seconds": monotonic() - started,
                }
            )
        )
        return output
