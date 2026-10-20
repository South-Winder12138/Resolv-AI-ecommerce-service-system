"""Human-approved expectation amendment; original case bytes and policies stay fixed."""

import json
from hashlib import sha256

from scripts import smoke_refund_flow as legacy

CASES = "tests/fixtures/refund_flow/semantic_cases-v2.json"
CASES_SHA256 = "4c45697b8b65e68bb63096f0a763bb527073c3286ffea021805be04f01135205"
EXPLICIT_POLICY = "tests/fixtures/refund_flow/explicit-time-policy.md"
EXPLICIT_SHA256 = "37600d15c05f7f99f78fcb7447a7ca5976bf496d4271a7155085a317558c1273"


def load_cases():
    raw = (legacy.ROOT / CASES).read_bytes()
    legacy.require(
        sha256(raw).hexdigest() == CASES_SHA256, "Amended expectation bytes changed"
    )
    value = legacy.validate_cases(json.loads(raw))
    legacy.require(
        value.get("expectation_revision") == 2, "Expectation revision required"
    )
    original = legacy.load_cases()
    for before, after in zip(original["cases"], value["cases"], strict=True):
        if before["case_id"] in {"F1", "F7"}:
            legacy.require(
                {k: v for k, v in before.items() if k != "expected"}
                == {k: v for k, v in after.items() if k != "expected"},
                "Original facts/policy/identity must be retained",
            )
            expected = after["expected"]
            legacy.require(
                expected["verdict"] == "undetermined"
                and expected["next_actions"] == ["human_draft"]
                and expected["offer_count"] == 0,
                "Ambiguous policy cannot offer application",
            )
        else:
            legacy.require(before == after, "Unapproved case amendment")
    return value


def explicit_policy():
    raw = (legacy.ROOT / EXPLICIT_POLICY).read_bytes()
    legacy.require(
        sha256(raw).hexdigest() == EXPLICIT_SHA256, "Isolated positive policy changed"
    )
    return raw.decode()
