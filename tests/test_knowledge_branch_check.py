"""Two gates use the same actual bundle; model errors are technical failures."""

import json

import pytest

from resolv.schemas.knowledge import KnowledgeFileConfig
from resolv.schemas.knowledge_branch import KnowledgeBranchError
from resolv.services.knowledge_branch_check import check_knowledge_evidence
from resolv.services.knowledge_confidence import score_knowledge_evidence
from tests.knowledge_branch_fixtures import branch_profile, bundle, primary
from tests.test_knowledge_branch_query import Client, prepare


def check(client, b, *, p=None, standard=None):
    p = p or primary()
    q = prepare(Client({"standard": standard or p.primary.text, "expanded": []}), p)
    profile = branch_profile()
    signals = score_knowledge_evidence(
        b,
        file_configs={"d" * 64: KnowledgeFileConfig(content_type="policy")},
        profile=profile,
    )
    return check_knowledge_evidence(
        p, q, b, signals, client=client, profile=profile, audit_hook=lambda _: None
    )


@pytest.mark.parametrize("logits", [(), (-10.0,)])
def test_empty_or_low_score_skips_selfcheck(logits):
    client = Client(RuntimeError("must not call"))
    decision = check(client, bundle(logits))
    assert client.calls == [] and not decision.evidence_strong
    assert decision.source == "retrieval_low_conf" and decision.self_check is None


def test_sufficient_conditional_answer_uses_original_and_same_evidence_hash():
    b = bundle((2.0, 1.0))
    client = Client(
        {
            "useful": True,
            "supporting_window_ids": [b.windows[0].window_id],
            "missing_information": [],
            "reason": "按质量问题与个人原因给条件化说明",
        }
    )
    d = check(client, b)
    assert d.evidence_strong and d.source is None and len(client.calls) == 1
    sent = json.loads(client.calls[0][1][1].content)
    assert sent["primary_text"] == primary().primary.text
    assert sent["standard"] == primary().primary.text
    assert sent["evidence_hash"] == d.evidence_hash == b.evidence_hash
    assert sent["evidence"] == b.payload


@pytest.mark.parametrize(
    "patch",
    [
        {"supporting_window_ids": []},
        {"supporting_window_ids": ["0" * 64]},
        {"missing_information": ["运费"]},
        {"reason": ""},
    ],
)
def test_invalid_support_is_selfcheck_technical_error(patch):
    b = bundle((2.0,))
    reply = {
        "useful": True,
        "supporting_window_ids": [b.windows[0].window_id],
        "missing_information": [],
        "reason": "明确",
    } | patch
    with pytest.raises(KnowledgeBranchError, match="self_check_error"):
        check(Client(reply), b)


def test_rewrite_audience_change_cannot_be_released():
    b = bundle((2.0,))
    p = primary("普通档用户未满99元基础运费是多少？")
    client = Client(
        {
            "useful": False,
            "supporting_window_ids": [],
            "missing_information": [],
            "reason": "检索问题与原主诉求人群不一致",
        }
    )
    d = check(client, b, p=p, standard="银卡用户未满99元基础运费是多少？")
    assert not d.evidence_strong and d.source == "self_check"
    sent = client.calls[0][1]
    assert "原主诉求" in sent[0].content and "身份" in sent[0].content
    assert "普通档" in sent[1].content and "银卡" in sent[1].content


def test_missing_terms_and_unresolved_conflict_are_insufficient():
    for reason in ("缺少退货运费关键条款", "同条件政策冲突未解决"):
        d = check(
            Client(
                {
                    "useful": False,
                    "supporting_window_ids": [],
                    "missing_information": [reason],
                    "reason": reason,
                }
            ),
            bundle((2.0,)),
        )
        assert d.source == "self_check" and not d.evidence_strong


def test_selfcheck_timeout_is_not_insufficient():
    with pytest.raises(KnowledgeBranchError, match="self_check_error"):
        check(Client(TimeoutError()), bundle((2.0,)))


def test_selfcheck_storage_failure_is_preserved():
    with pytest.raises(KnowledgeBranchError, match="execution_lost"):
        check(Client(KnowledgeBranchError("execution_lost")), bundle((2.0,)))


def test_selfcheck_storage_value_error_is_not_converted_to_model_failure():
    with pytest.raises(ValueError, match="storage hash"):
        check(Client(ValueError("storage hash")), bundle((2.0,)))


def test_selfcheck_explicitly_names_allowed_window_ids_without_changing_evidence():
    b = bundle((2.0, 1.0))
    client = Client(
        {
            "useful": True,
            "supporting_window_ids": [b.windows[0].window_id],
            "missing_information": [],
            "reason": "条款充分",
        }
    )
    check(client, b)
    sent = json.loads(client.calls[0][1][1].content)
    assert sent["allowed_supporting_window_ids"] == [w.window_id for w in b.windows]
    assert sent["evidence"] == b.payload and sent["evidence_hash"] == b.evidence_hash
