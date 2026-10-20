"""Structural gate evidence checks; these do not manufacture accuracy evidence."""

import math

from resolv.schemas.knowledge_gate_cases import FORMAL_CASES

RECOVERY_WINDOWS = tuple(
    f"knowledge_{side}_{stage}_saved"
    for stage in ("query", "retrieval", "check", "answer_input", "answer")
    for side in ("before", "after")
) + (
    "knowledge_before_finalize_commit",
    "knowledge_after_finalize_commit",
    "before_end_checkpoint",
    "after_end_checkpoint",
    "after_completed_commit",
    "fallback_before_finalize_commit",
    "fallback_after_finalize_commit",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_formal_case(result, case):
    meta, answer = result["metadata"], result["answer"]
    expected = case["expected"]
    if expected == "knowledge":
        if (
            meta["branch"] != "knowledge"
            or meta["reply_kind"] != "answer"
            or meta["knowledge"]["decision"] != "sufficient"
        ):
            raise ValueError("Expected sufficient knowledge: " + case["case_id"])
        for alternatives in case["facts"]:
            if not any(text in answer for text in alternatives):
                raise ValueError("Missing expected policy fact: " + case["case_id"])
        for filename in case.get("required_sources", []):
            if not any(
                c["source"].endswith("/" + filename)
                for c in meta["knowledge"]["citations"]
            ):
                raise ValueError("Missing required policy source: " + filename)
    elif expected == "fallback":
        if (
            meta["reply_kind"] != "knowledge_fallback"
            or meta["knowledge"]["citations"]
            or meta["knowledge"]["chunks"]
            or not meta["suggested_actions"]
        ):
            raise ValueError("Expected safe insufficient result")
    elif expected == "refund":
        if (
            meta["branch"] != "refund_flow"
            or meta["reply_kind"] != "unavailable"
            or meta["knowledge"] is not None
            or result.get("knowledge_record_present")
        ):
            raise ValueError("Specific order refund changed branch")


def validate_knowledge_gate(report):
    _require(
        report.get("gate") == "K2"
        and report.get("passed") is True
        and report.get("real_services") is True
        and report.get("business_tool_calls") == 0
        and bool(report.get("source_fingerprints")),
        "Incomplete K2 header",
    )
    if report.get("knowledge_scope") == "combined":
        scopes = report.get("reports", [])
        _require(
            {r.get("knowledge_scope") for r in scopes}
            == {"isolated", "formal-readonly"}
            and len(scopes) == 2,
            "Both isolated and formal evidence required",
        )
        for scope in scopes:
            validate_knowledge_gate(scope)
        _require(
            report.get("release")
            == next(
                r["release"]
                for r in scopes
                if r["knowledge_scope"] == "formal-readonly"
            ),
            "Formal binding differs",
        )
        return
    _require(
        report.get("knowledge_scope") in {"isolated", "formal-readonly"},
        "Invalid K2 scope",
    )
    if report.get("knowledge_scope") == "isolated":
        probes = report.get("probes", {})
        expected = {
            "one_lane_empty",
            "bm25_fault",
            "sql_fault",
            "reranker_fault",
            "corrupt_authority",
            "low_score",
            "selfcheck_insufficient",
        }
        _require(
            set(probes) == expected
            and all(p.get("passed") is True for p in probes.values()),
            "Missing isolated boundary probes",
        )
    release = report.get("release", {})
    _require(
        all(
            release.get(k)
            for k in ("release_manifest_hash", "profile_hash", "manifest_hash")
        ),
        "Missing release identity",
    )
    local = report.get("local_inference", {})
    _require(
        local.get("device") == "mps"
        and local.get("dtype") == "torch.float32"
        and local.get("batch_size") == 4
        and local.get("forward_calls", 0) > 0
        and local.get("dense_device") == "cpu"
        and local.get("dense_dtype") == "torch.float32"
        and local.get("model_revision") == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "Actual pinned MPS FP32 forward required",
    )
    preservation = report.get("preservation", {})
    _require(
        bool(preservation.get("before"))
        and preservation["before"] == preservation.get("after"),
        "Preservation mismatch",
    )
    cases = report.get("cases", [])
    _require(
        bool(cases) and len({c.get("case_id") for c in cases}) == len(cases),
        "Missing or duplicate cases",
    )
    if report["knowledge_scope"] == "formal-readonly":
        expected = {c["case_id"]: c for c in FORMAL_CASES}
        _require(
            {c.get("case_id") for c in cases} == set(expected),
            "Missing approved formal cases",
        )
        for case in cases:
            rule = expected[case["case_id"]]
            _require(
                case.get("question") == rule["question"]
                and case.get("expected") == rule["expected"],
                "Formal case contract changed",
            )
            validate_formal_case(case, rule)
    positive = 0
    for case in cases:
        _require(
            case.get("passed") is True
            and case.get("status") == "completed"
            and case.get("business_tool_calls") == 0
            and case.get("transport_tools") == 0,
            "Case did not complete safely",
        )
        audits = case.get("audits", [])
        _require(bool(audits), "Missing real model calls")
        invalid_list = case.get("recovered_invalid_check_invocations", [])
        invalid = set(invalid_list)
        checks = [a for a in audits if a.get("purpose") == "knowledge_check"]
        _require(
            len(invalid) == len(invalid_list)
            and len(invalid) <= 2
            and invalid.issubset({a.get("invocation_id") for a in checks}),
            "Invalid technical recovery receipts",
        )
        if invalid:
            _require(
                case.get("selfcheck_invocation_id")
                in {a.get("invocation_id") for a in checks} - invalid,
                "Final selfcheck audit is not validated",
            )
        validated_audits = [a for a in audits if a.get("invocation_id") not in invalid]
        upstream = set(case.get("recovered_failed_upstream_invocations", []))
        receipts = case.get("recovery_failure_receipts", [])
        _require(
            len(upstream) <= 2
            and upstream
            == {
                r.get("failed_upstream_invocation")
                for r in receipts
                if r.get("failed_upstream_invocation")
            },
            "Missing upstream recovery receipts",
        )
        for invocation in upstream:
            failures = [a for a in audits if a.get("invocation_id") == invocation]
            _require(
                len(failures) == 1
                and failures[0].get("purpose") in {"resolution", "intents"}
                and failures[0].get("outcome") in {"failed", "unknown"}
                and any(
                    a.get("purpose", "").rstrip("s")
                    == failures[0]["purpose"].rstrip("s")
                    and a.get("outcome") == "succeeded"
                    for a in audits
                ),
                "Unrecovered upstream call",
            )
        for audit in audits:
            if audit.get("invocation_id") in upstream:
                continue
            if audit.get("invocation_id") in invalid and audit.get("outcome") in {
                "failed",
                "unknown",
            }:
                continue

            _require(
                audit.get("outcome") == "succeeded"
                and type(audit.get("input_tokens")) is int
                and type(audit.get("output_tokens")) is int
                and 0 < audit["input_tokens"] <= audit.get("input_estimate", -1)
                and audit["output_tokens"] > 0,
                "Missing usage/budget evidence",
            )
        if case.get("expected") != "knowledge":
            continue
        positive += 1
        counts = case.get("counts", {})
        _require(
            counts.get("dense_count", 0) + counts.get("bm25_count", 0) > 0
            and all(
                counts.get(k, 0) > 0
                for k in ("center_count", "scored_count", "selected_count")
            ),
            "Empty positive retrieval",
        )
        _require(
            0 < case.get("query_tokens", 0) <= 512
            and bool(case.get("pair_tokens"))
            and max(case["pair_tokens"]) <= 3072,
            "Token boundary violated",
        )
        evidence = case.get("evidence", {})
        windows = evidence.get("windows", [])
        citations = evidence.get("citations", {})
        chunks = evidence.get("chunks", [])
        _require(
            bool(windows)
            and len(set(windows)) == len(windows)
            and set(windows) == set(citations)
            and set(chunks) == {id for ids in citations.values() for id in ids}
            and len(chunks) == len(set(chunks)),
            "Reference closure violated",
        )
        used = evidence.get("used_citation_numbers", [])
        _require(
            evidence.get("evidence_hash")
            and 0
            < evidence.get("input_tokens", 0)
            <= evidence.get("budget", -1)
            <= 3000
            and bool(used)
            and all(type(n) is int and 1 <= n <= len(windows) for n in used)
            and case.get("same_evidence") is True,
            "Actual evidence/budget differs",
        )
        signals = case.get("signals", {})
        _require(
            signals.get("passed") is True
            and len(signals.get("contributions", [])) == 4
            and math.isclose(
                sum(signals["contributions"]), signals.get("total", -1), abs_tol=1e-12
            )
            and signals["total"] >= 0.26,
            "Invalid local gate signals",
        )
        check = case.get("selfcheck", {})
        _require(
            check.get("useful") is True
            and bool(check.get("supporting_window_ids"))
            and set(check["supporting_window_ids"]).issubset(windows),
            "Invalid sufficiency evidence",
        )
        _require(
            sum(a["purpose"] == "knowledge_answer" for a in validated_audits) == 1
            and sum(a["purpose"] == "knowledge_query" for a in validated_audits) == 1
            and sum(a["purpose"] == "knowledge_check" for a in validated_audits) == 1,
            "Repeated or missing knowledge model call",
        )
    _require(positive > 0, "No positive real retrieval case")


def validate_recovery_gate(report):
    _require(
        report.get("gate") == "K3"
        and report.get("passed") is True
        and report.get("real_services") is True
        and report.get("fake_inference") is True
        and report.get("accuracy_evidence") is False
        and report.get("business_tool_calls") == 0
        and bool(report.get("source_fingerprints")),
        "Invalid recovery header",
    )
    p = report.get("preservation", {})
    _require(
        bool(p.get("before")) and p["before"] == p.get("after"),
        "Recovery changed formal records",
    )
    cases = report.get("cases", [])
    _require(
        len(cases) == len(RECOVERY_WINDOWS)
        and {c.get("window") for c in cases} == set(RECOVERY_WINDOWS),
        "Incomplete recovery matrix",
    )
    for c in cases:
        _require(
            c.get("passed") is True
            and c.get("reached") is True
            and c.get("child_exit") == 86
            and c.get("status") == "completed"
            and c.get("assistant_count") == 1
            and c.get("review_count")
            == (1 if c["window"].startswith("fallback") else 0)
            and c.get("stage_hashes_stable") is True
            and c.get("saved_calls_reused") is True
            and c.get("business_tool_calls") == 0,
            "Missing actual crash/recovery receipt",
        )
    lock = report.get("locks_and_fencing", {})
    atom = report.get("atomicity", {})
    epoch = lock.get("knowledge_epoch_control", {})
    _require(
        all(
            lock.get(k) is True
            for k in ("passed", "busy_rejected", "stale_epoch_rejected")
        )
        and all(
            epoch.get(k) is True
            for k in (
                "passed",
                "valid_write_committed",
                "stale_write_rejected",
                "unchanged_after_stale",
            )
        )
        and all(
            atom.get(k) is True
            for k in ("passed", "rolled_back_reply", "rolled_back_review")
        ),
        "Missing fencing/atomicity proof",
    )
