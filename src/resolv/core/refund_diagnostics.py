"""Explicit public-response diagnostics. Never persist SDK credentials or reasoning."""

import json
import re

from langchain_core.messages import AIMessageChunk
from pydantic import ValidationError

from resolv.core.refund_response import ResponseDecodeError, decode_refund_response
from resolv.services.context import TokenCounter
from resolv.services.hybrid_fingerprints import canonical_hash


def public_snapshot(reply):
    content = getattr(reply, "content", "")
    if isinstance(content, str):
        content = re.sub(
            r"(?is)<(think|thinking|analysis|reasoning)>.*?(?:</\1>|$)",
            "[reasoning omitted]",
            content,
        )
    if isinstance(content, list):
        content = [
            block
            if isinstance(block, dict)
            and block.get("type")
            in {"text", "output_text", "function_call", "tool_call"}
            else {"type": "unsupported"}
            for block in content
            if not isinstance(block, dict)
            or block.get("type") not in {"reasoning", "thinking", "analysis"}
        ]
    return {
        "content": content,
        "tool_calls": getattr(reply, "tool_calls", []),
        "invalid_tool_calls": getattr(reply, "invalid_tool_calls", []),
        "tool_call_chunks": getattr(reply, "tool_call_chunks", []),
        "additional_kwargs": {"tool_calls": reply.additional_kwargs["tool_calls"]}
        if getattr(reply, "additional_kwargs", {}).get("tool_calls")
        else {},
        "response_metadata": {
            k: v
            for k, v in getattr(reply, "response_metadata", {}).items()
            if k in {"finish_reason", "status"}
        },
        "usage_metadata": getattr(reply, "usage_metadata", None),
    }


def failure_diagnostic(
    error, reply, audit, *, prepare, require_tool, final_round, required_tool=None
):
    response = public_snapshot(reply)
    cause = error.cause
    issues = (
        [
            {"loc": list(e["loc"]), "type": e["type"]}
            for e in cause.errors(
                include_input=False, include_context=False, include_url=False
            )
        ]
        if isinstance(cause, ValidationError)
        else []
    )
    if isinstance(cause, ValidationError):
        rules = {
            "Value error, Status must not assert eligibility": "status_verdict_conflict",
            "Value error, Eligibility verdict requires sourced policy and facts": "sourced_support_missing",
            "Value error, Unresolved conditions forbid eligibility verdict": "unresolved_eligibility_conditions",
            "Value error, Only eligible decisions can offer an application": "offer_requires_eligible",
        }
        for issue, detail in zip(
            issues,
            cause.errors(include_input=False, include_context=False, include_url=False),
            strict=True,
        ):
            if detail["msg"] in rules:
                issue["rule"] = rules[detail["msg"]]
    return {
        "schema_version": 1,
        "invocation_id": audit.invocation_id,
        "profile_fingerprint": audit.profile_fingerprint,
        "input_hash": audit.input_hash,
        "input_estimate": audit.input_estimate,
        "model_step": audit.model_step,
        "stage": error.stage,
        "exception_type": type(cause).__name__,
        "error_code": error.code,
        "issues": issues,
        "json_error": {
            "position": cause.pos,
            "line": cause.lineno,
            "column": cause.colno,
        }
        if isinstance(cause, json.JSONDecodeError)
        else None,
        "prepare": prepare,
        "require_tool": require_tool,
        "final_round": final_round,
        "response": response,
        "response_hash": canonical_hash(response),
        "retention_scope": "public answer and native tool payload only; reasoning/headers/credentials omitted",
        **({"required_tool": required_tool} if required_tool is not None else {}),
    }


def replay_failed_response(diagnostic, profile):
    if diagnostic.get("replayable") is False:
        raise ValueError("No complete response exists for request failure replay")
    if (
        diagnostic.get("schema_version") != 1
        or diagnostic.get("profile_fingerprint") != profile.fingerprint
        or canonical_hash(diagnostic["response"]) != diagnostic.get("response_hash")
    ):
        raise ValueError("Changed diagnostic or frozen model profile")
    reply = AIMessageChunk.model_construct(**diagnostic["response"])
    try:
        decode_refund_response(
            reply,
            profile,
            prepare=diagnostic["prepare"],
            require_tool=diagnostic["require_tool"],
            final_round=diagnostic["final_round"],
            counter=TokenCounter(profile.tokenizer_profile),
            estimate=diagnostic["input_estimate"],
            required_tool=diagnostic.get("required_tool"),
        )
    except ResponseDecodeError as exc:
        return {
            "stage": exc.stage,
            "exception_type": type(exc.cause).__name__,
            "error_code": exc.code,
            "response_hash_matches": True,
            "provider_calls": 0,
        }
    raise ValueError("Recorded failed response did not reproduce a decode failure")


def transport_diagnostic(cause, error_code, audit):
    body = getattr(cause, "body", {})
    body = body.get("error", body) if isinstance(body, dict) else {}
    safe = {
        k: body[k]
        for k in ("code", "type", "param")
        if isinstance(body, dict)
        and isinstance(body.get(k), str)
        and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,128}", body[k])
    }
    status = getattr(cause, "status_code", None)
    return {
        "schema_version": 1,
        "invocation_id": audit.invocation_id,
        "profile_fingerprint": audit.profile_fingerprint,
        "input_hash": audit.input_hash,
        "input_estimate": audit.input_estimate,
        "model_step": audit.model_step,
        "stage": "transport",
        "exception_type": type(cause).__name__,
        "error_code": error_code,
        "http_status": status if type(status) is int else None,
        "provider_error": safe,
        "issues": [],
        "json_error": None,
        "response": None,
        "response_hash": canonical_hash(None),
        "replayable": False,
    }
