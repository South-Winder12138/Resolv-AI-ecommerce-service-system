"""Canonical JSON text inside a versioned JSON envelope, with legacy read support.

MySQL's native numeric JSON representation can change a binary double by one ULP.
Immutable evidence hashes need the original value, so the server stores its canonical
text as a JSON string. ORM callers continue to receive the original dict/list.
"""

import json

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator

_TAG = "__resolv_exact_json__"


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class ExactJSON(TypeDecorator):
    impl = JSON
    cache_ok = True

    def __init__(self, none_as_null=False, *, only_reply_v2=False):
        super().__init__(none_as_null=none_as_null)
        self.none_as_null = none_as_null
        self.only_reply_v2 = only_reply_v2
        self.should_evaluate_none = self.impl.should_evaluate_none

    def process_bind_param(self, value, dialect):
        if value is None or value is JSON.NULL:
            return value
        if self.only_reply_v2 and (
            not isinstance(value, dict) or value.get("schema_version") not in (2, 3)
        ):
            return value
        return {_TAG: 1, "payload": _canonical(value)}

    def process_result_value(self, value, dialect):
        if not isinstance(value, dict) or _TAG not in value:
            return value
        if (
            set(value) != {_TAG, "payload"}
            or type(value[_TAG]) is not int
            or value[_TAG] != 1
            or not isinstance(value["payload"], str)
        ):
            raise ValueError("Invalid exact JSON envelope")
        decoded = json.loads(value["payload"])
        if _canonical(decoded) != value["payload"]:
            raise ValueError("Noncanonical exact JSON payload")
        return decoded
