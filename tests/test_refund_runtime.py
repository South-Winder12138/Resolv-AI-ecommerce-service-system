import importlib
from types import SimpleNamespace

import pytest

from resolv.schemas.branch_execution import BusinessRunConfig
from resolv.schemas.refund import RefundFlowError
from tests.test_refund_routing_graph import events, harness


def test_runtime_defaults_unconnected_without_provider_or_sqlite_writes(tmp_path):
    module = importlib.import_module("resolv.services.refund_runtime")
    with harness(tmp_path) as h:
        h.service.business_run_config = BusinessRunConfig.model_validate(
            h.service.get_run(h.run.run_id).business_config
        )
        h.service.knowledge_retrievers = h.flow.retrievers
        runtime = module.create_refund_runtime(
            h.service, settings=SimpleNamespace(), business_factory=h.business
        )
        try:
            assert runtime.profile.adapter.connected is False
            with pytest.raises(RefundFlowError, match="refund_identity_unavailable"):
                runtime.submissions.identity.resolve({"customer": "client claim"})
            assert runtime.flow.client._clients == {}
        finally:
            runtime.close()


def test_v4_unconnected_is_public_unavailable_and_does_not_query(tmp_path):
    from resolv.services.refund_adapters import UnconnectedRefundIdentity

    with harness(tmp_path) as h:
        h.flow.identity = UnconnectedRefundIdentity()
        final = events(h)
        assert final[-1].type == "done", [(e.type, e.data) for e in final]
        saved = h.service.get_run(h.run.run_id)
        assert saved.result_json["refund"]["offer_id"] is None
        assert saved.result_json["reply_kind"] == "unavailable"
        assert h.counts["retrieve"] == h.counts["decide"] == 0


def test_promote_reply_v3_preserves_closed_v2_contract():
    from resolv.schemas.branch_execution import ReplyMetadata
    from resolv.services.refund_runtime import promote_reply_v3

    old = ReplyMetadata(
        reply_kind="answer",
        understanding_id="u",
        primary_task_id="t",
        branch="business",
    )
    new = promote_reply_v3(old)
    assert new.schema_version == 3 and new.knowledge is None and new.refund is None
