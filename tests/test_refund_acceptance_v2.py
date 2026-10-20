"""Versioned expectations preserve the original policy and all historical cases."""

from pathlib import Path


def test_ambiguous_policy_expectation_changes_only_F1_and_F7():
    from scripts.refund_acceptance_v2 import load_cases
    from scripts.smoke_refund_flow import load_cases as original

    old, new = original(), load_cases()
    assert new["expectation_revision"] == 2
    for before, after in zip(old["cases"], new["cases"], strict=True):
        if before["case_id"] not in {"F1", "F7"}:
            assert before == after
            continue
        assert {k: v for k, v in before.items() if k != "expected"} == {
            k: v for k, v in after.items() if k != "expected"
        }
        assert before["expected"]["verdict"] == "eligible"
        assert after["expected"]["verdict"] == "undetermined"
        assert after["expected"]["next_actions"] == ["human_draft"]
        assert after["expected"]["offer_count"] == 0
        assert after["expected"]["policy_quotes"] == ["自签收之日起 7 天内"]


def test_explicit_positive_policy_is_separate_and_defines_elapsed_seconds():
    from scripts.refund_acceptance_v2 import explicit_policy

    policy = explicit_policy()
    assert "仅用于隔离验证" in policy
    assert "168小时内" in policy
    assert "整单退款" in policy
    assert "168" not in Path("tests/fixtures/refund_flow/returns-policy.md").read_text()
