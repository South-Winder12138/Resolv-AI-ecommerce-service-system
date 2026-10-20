"""Independent provider client and safe per-invocation audit, without tools."""

import json
from collections import Counter
from urllib.parse import urlsplit
from uuid import uuid4

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from resolv.db.conversation_base import utc_now
from resolv.schemas.understanding import StageAudit, UnderstandingError


def understanding_model_profile(settings, reasoning_effort="low"):
    url = urlsplit(settings.openai_base_url)
    if (
        settings.openai_model.startswith("qwen3.8-")
        and (url.hostname or "").endswith(".aliyuncs.com")
        and url.path.rstrip("/").endswith("/compatible-mode/v1")
    ):
        return {
            "model_api_mode": "chat_completions",
            "reasoning_effort": None,
            "enable_thinking": True,
            "thinking_budget": 512,
            "top_p": 0.1,
            "output_format": "json_schema",
            "schema_constraint_version": "source-catalog-v6",
        }
    return {
        "model_api_mode": "responses",
        "reasoning_effort": reasoning_effort,
        "enable_thinking": None,
        "thinking_budget": None,
        "top_p": None,
        "output_format": "json_object",
        "schema_constraint_version": None,
    }


def get_understanding_model(settings, timeout_seconds):
    responses = understanding_model_profile(settings)["model_api_mode"] == "responses"
    parameters = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
        "base_url": settings.openai_base_url,
        "use_responses_api": responses,
        "streaming": responses,
        "max_retries": 0,
        "timeout": timeout_seconds,
    }
    if responses:
        parameters.update(
            output_version="responses/v1",
            use_previous_response_id=False,
            truncation="disabled",
        )
    return ChatOpenAI(**parameters)


def compact_schema(value):
    if isinstance(value, dict):
        return {k: compact_schema(v) for k, v in value.items() if k != "title"}
    if isinstance(value, list):
        return [compact_schema(v) for v in value]
    return value


def stage_response_format(stage, schema, output_format):
    if output_format == "json_object":
        return {"type": "json_object"}
    if output_format != "json_schema" or schema is None:
        raise UnderstandingError("understanding_incompatible")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "understanding_" + stage,
            "schema": compact_schema(schema),
            "strict": True,
        },
    }


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def public_text(reply):
    if reply.tool_calls or reply.invalid_tool_calls:
        raise UnderstandingError("model_output_invalid", True)
    if reply.response_metadata.get("finish_reason") in {
        "length",
        "content_filter",
    } or reply.response_metadata.get("status") in {"incomplete", "failed"}:
        raise UnderstandingError("model_output_invalid", True)
    if isinstance(reply.content, str):
        return reply.content
    text = []
    for block in reply.content:
        if not isinstance(block, dict):
            raise UnderstandingError("model_output_invalid", True)
        if block.get("type") == "reasoning":
            continue  # Never store reasoning blocks or summary text.
        if block.get("type") not in {"text", "output_text"} or not isinstance(
            block.get("text"), str
        ):
            raise UnderstandingError("model_output_invalid", True)
        text.append(block["text"])
    return "".join(text)


class JsonStageModel:
    def __init__(self, client, counter=None):
        self.client, self.counter = client, counter
        self.calls_by_stage = Counter()

    def invoke(
        self,
        stage,
        messages,
        output_type,
        input_estimate,
        reserved_output_tokens,
        audit_hook,
        *,
        reasoning_effort=None,
        enable_thinking=None,
        thinking_budget=None,
        top_p=None,
        output_format="json_object",
        output_schema=None,
    ):
        self.calls_by_stage[stage] += 1
        audit = StageAudit(
            stage=stage,
            invocation_id=uuid4().hex,
            attempt=self.calls_by_stage[stage],
            input_estimate=input_estimate,
            input_tokens=None,
            output_tokens=None,
            started_at=utc_now(),
            finished_at=None,
            outcome="started",
            verified_refs=[],
        )
        audit_hook(audit)
        try:
            parameters = {
                "max_completion_tokens": reserved_output_tokens,
                "response_format": stage_response_format(
                    stage, output_schema, output_format
                ),
            }
            if reasoning_effort is not None:
                parameters["reasoning_effort"] = reasoning_effort
            if top_p is not None:
                parameters["top_p"] = top_p
            if enable_thinking is not None:
                parameters["extra_body"] = {"enable_thinking": enable_thinking}
                if thinking_budget is not None:
                    parameters["extra_body"]["thinking_budget"] = thinking_budget
            reply = self.client.bind(**parameters).invoke(messages)
            usage = reply.usage_metadata or {}
            audit.input_tokens = usage.get("input_tokens")
            audit.output_tokens = usage.get("output_tokens")
            if audit.input_tokens is not None and audit.input_tokens > input_estimate:
                raise UnderstandingError("model_input_underestimated", True)
            text = public_text(reply)
            if (
                audit.output_tokens is not None
                and audit.output_tokens > reserved_output_tokens
            ) or (
                self.counter and self.counter.count_text(text) > reserved_output_tokens
            ):
                raise UnderstandingError("model_output_budget_exceeded", True)
            data = json.loads(
                text,
                object_pairs_hook=unique_object,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    ValueError("invalid constant")
                ),
            )
            output = output_type.model_validate(data)
        except UnderstandingError as error:
            failure = error
        except (ValidationError, ValueError, TypeError, AttributeError):
            failure = UnderstandingError("model_output_invalid", True)
        except Exception:  # noqa: BLE001 — redact SDK/transport errors at credential boundary
            failure = UnderstandingError("model_unavailable", True)
        else:
            audit = audit.model_copy(
                update={"finished_at": utc_now(), "outcome": "succeeded"}
            )
            audit_hook(audit)
            return output, audit
        audit = audit.model_copy(update={"finished_at": utc_now(), "outcome": "failed"})
        audit_hook(audit)
        raise failure from None
