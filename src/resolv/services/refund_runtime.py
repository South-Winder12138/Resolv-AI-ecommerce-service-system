"""Explicit refund wiring. No schema creation, seed, selector publication or fake facts."""

from dataclasses import dataclass

from sqlalchemy import inspect

from resolv.core.refund_config import default_refund_profile
from resolv.core.refund_model import RefundModelClient
from resolv.schemas.branch_execution import parse_reply_metadata
from resolv.schemas.conversation import ConversationError
from resolv.schemas.knowledge_branch import canonical_json
from resolv.schemas.refund_flow import RefundProfile
from resolv.schemas.refund_submission import ReplyMetadataV3
from resolv.services.knowledge_branch_runtime import read_release_selection
from resolv.services.refund_adapters import (
    UnconnectedRefundFactsAdapter,
    UnconnectedRefundIdentity,
)
from resolv.services.refund_flow import RefundFlowService
from resolv.services.refund_submission import RefundSubmissionService


def promote_reply_v3(value):
    metadata = (
        value if hasattr(value, "schema_version") else parse_reply_metadata(value)
    )
    data = metadata.model_dump(mode="json")
    return ReplyMetadataV3.model_validate_json(
        canonical_json(
            data
            | {
                "schema_version": 3,
                "knowledge": data.get("knowledge"),
                "refund": data.get("refund"),
            }
        )
    )


@dataclass
class RefundRuntime:
    profile: RefundProfile
    flow: RefundFlowService
    submissions: RefundSubmissionService
    client: RefundModelClient

    def close(self):
        self.client.close()


def create_refund_runtime(
    conversation, *, settings, identity=None, facts=None, business_factory=None
):
    business_factory = business_factory or getattr(
        conversation.write_service, "business_session_factory", None
    )
    if business_factory is None or not {
        "refund_applications",
        "refund_submission_receipts",
        "refunds",
        "orders",
    } <= set(inspect(business_factory.kw["bind"]).get_table_names()):
        raise ConversationError("conversation_store_failed", True)
    if not {
        "conversation_refund_runs",
        "conversation_refund_offers",
        "conversation_refund_submissions",
        "conversation_refund_request_aliases",
    } <= set(inspect(conversation.session_factory.kw["bind"]).get_table_names()):
        raise ConversationError("conversation_store_failed", True)
    identity = identity if identity is not None else UnconnectedRefundIdentity()
    facts = facts if facts is not None else UnconnectedRefundFactsAdapter()
    config = conversation.business_run_config
    profile = default_refund_profile(
        understanding_config=conversation.understanding.context.config,
        business_config=config,
        summary_profile=config.summary_profile or {},
        adapter_descriptor=facts.descriptor,
    )
    client = RefundModelClient(settings)
    submissions = RefundSubmissionService(
        conversation.session_factory,
        business_factory,
        owner_id=conversation.owner_id,
        identity=identity,
        facts=facts,
        selection_reader=lambda: read_release_selection(
            conversation.release_selector_path
        ),
        profile=profile,
    )
    flow = RefundFlowService(
        conversation,
        client=client,
        retrievers=conversation.knowledge_retrievers,
        identity=identity,
        facts=facts,
        registration=submissions,
    )
    return RefundRuntime(profile, flow, submissions, client)
