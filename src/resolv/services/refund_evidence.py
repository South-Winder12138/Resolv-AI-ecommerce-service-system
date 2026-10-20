"""Reuse authoritative hybrid retrieval and complete-prefix evidence, without scoring."""

from pydantic import ValidationError

from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from resolv.schemas.knowledge_branch import (
    KnowledgeBranchError,
    KnowledgeRetrievalStage,
)
from resolv.schemas.knowledge_release import ReleaseSelection
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_flow import PreparedRefundQuery
from resolv.services.hybrid_fingerprints import manifest_hash
from resolv.services.knowledge_branch_evidence import select_knowledge_evidence


def retrieve_refund_evidence(query, *, selection, retrievers, profile, counter):
    query = PreparedRefundQuery.model_validate_json(query.model_dump_json())
    selection = ReleaseSelection.model_validate_json(selection.model_dump_json())
    if selection.fingerprint != profile.selection.fingerprint:
        raise RefundFlowError("refund_release_conflict")
    if selection.status == "disabled":
        raise RefundFlowError("refund_unavailable")
    if selection.status == "invalid":
        raise RefundFlowError("refund_release_invalid")
    try:
        version = retrievers.validate_selection(selection)
        if (
            version.version_id != selection.index_version
            or version.state != "ready"
            or manifest_hash(version.manifest) != selection.manifest_hash
            or version.profile != selection.profile
        ):
            raise RefundFlowError("refund_release_conflict")
        result = retrievers.retrieve(
            selection,
            query.standard,
            bm25_text=query.bm25_text,
            timeout_seconds=profile.decision_model.timeout_seconds,
        )
        if (
            result.index_version != selection.index_version
            or result.index_profile_hash != selection.profile_hash
            or result.retrieval_profile != profile.retrieval_profile
        ):
            raise RefundFlowError("refund_release_conflict")
        bundle = select_knowledge_evidence(
            result, counter=counter, max_tokens=profile.evidence_max_tokens
        )
        return KnowledgeRetrievalStage.from_result(
            result,
            bundle=bundle,
            file_configs=tuple(
                (d.document_id, d.file_config) for d in version.manifest.documents
            ),
        )
    except KnowledgeBranchError as exc:
        raise RefundFlowError(
            "refund_" + exc.code.removeprefix("knowledge_"), exc.retryable
        ) from exc
    except HybridRetrievalError as exc:
        raise RefundFlowError("refund_retrieval_failed", True) from exc
    except (OSError, TimeoutError) as exc:
        raise RefundFlowError("refund_retrieval_failed", True) from exc
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        raise RefundFlowError("refund_evidence_invalid") from exc
