"""Shared SSE frame encoding for chat and Agent HTTP streams."""

import json


def encode_sse(event: str, data: dict[str, object]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def conversation_sse(events):
    from resolv.schemas.conversation import ConversationError

    try:
        for event in events:
            payload = {"type": event.type, "run_id": event.run_id, **event.data}
            if event.type == "message" and payload.get("schema_version") in {2, 3}:
                from resolv.schemas.branch_execution import parse_reply_metadata

                metadata = parse_reply_metadata(
                    {
                        k: v
                        for k, v in payload.items()
                        if k not in {"type", "run_id", "text"}
                    }
                )
                payload = {
                    "type": event.type,
                    "run_id": event.run_id,
                    "text": payload["text"],
                    **metadata.model_dump(mode="json"),
                }
            if event.type == "knowledge_sources":
                payload["hits"] = payload.pop("sources", [])
            if event.type == "approval_required":
                payload["message"] = payload.pop(
                    "summary", payload.get("message", "请确认操作")
                )
            yield encode_sse(event.type, payload)
    except ConversationError as error:
        yield encode_sse(
            "error",
            {
                "type": "error",
                "run_id": error.run_id,
                "code": error.code,
                "retryable": error.retryable,
            },
        )
    except Exception:  # noqa: BLE001 — streaming errors cannot expose private driver/model data
        yield encode_sse(
            "error",
            {
                "type": "error",
                "run_id": None,
                "code": "conversation_store_failed",
                "retryable": True,
            },
        )
