"""Add knowledge-purpose audit events around the existing summary implementation."""

from time import monotonic
from uuid import uuid4

from resolv.db.conversation_base import utc_now
from resolv.schemas.knowledge_branch import KnowledgeBranchError, KnowledgeModelAudit
from resolv.services.business_summary import summary_profile
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.summarization import (
    SUMMARY_PROMPT,
    SummaryFailure,
    SummaryService,
)


class _AuditedModel:
    def __init__(self, base, profile, hook):
        self.base, self.profile, self.hook = base, profile, hook
        self.fatal = None

    def record(self, audit):
        try:
            self.hook(audit)
        except Exception as exc:
            self.fatal = exc
            raise

    def invoke(self, messages):
        self.fatal = None
        start = monotonic()
        p = self.profile.summary_profile
        audit = KnowledgeModelAudit(
            invocation_id=uuid4().hex,
            purpose="summary",
            profile_fingerprint=p.fingerprint,
            input_hash=canonical_hash([m.model_dump(mode="json") for m in messages]),
            input_estimate=self.base.counter.count(messages, [])
            + self.base.settings.provider_input_overhead_tokens,
            started_at=utc_now(),
            outcome="started",
        )
        self.record(audit)
        try:
            response = self.base.model.invoke(messages)
        except Exception as exc:
            code = getattr(exc, "code", "")
            if code.startswith("summary_"):
                self.record(
                    audit.model_copy(
                        update={
                            "outcome": "unknown"
                            if code == "summary_model_unavailable"
                            else "failed",
                            "error_code": code,
                            "finished_at": utc_now(),
                            "elapsed_seconds": monotonic() - start,
                        }
                    )
                )
            else:
                self.fatal = exc
            raise
        usage = response.usage_metadata or {}
        self.record(
            audit.model_copy(
                update={
                    "outcome": "succeeded",
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "finish_reason": response.response_metadata.get("finish_reason")
                    or response.response_metadata.get("status"),
                    "finished_at": utc_now(),
                    "elapsed_seconds": monotonic() - start,
                }
            )
        )
        return response


class _Candidates:
    def __init__(self, model, service):
        self.model, self.service = model, service

    def create_candidate(self, previous, older):
        try:
            return self.service.create_candidate(previous, older)
        except SummaryFailure:
            if self.model.fatal is not None:
                raise self.model.fatal
            raise


class KnowledgeSummaryAdapter:
    def __init__(self, base, profile, audit_hook):
        self.base, self.profile, self.audit_hook = base, profile, audit_hook

    def for_run(self, run, execution):
        base = (
            self.base.for_run(run, execution)
            if hasattr(self.base, "for_run")
            else self.base
        )
        if (
            self.profile.summary_config is None
            or self.profile.summary_config.value != summary_profile(base.settings)
        ):
            raise KnowledgeBranchError("knowledge_summary_profile_mismatch")
        p, cfg = self.profile.summary_profile, summary_profile(base.settings)
        if (
            p.model_name != cfg["model_name"]
            or p.api_address_fingerprint != cfg["api_address_fingerprint"]
            or p.model_api_mode != cfg["model_api_mode"]
            or p.system_prompt != SUMMARY_PROMPT
            or p.model_context_window_tokens != cfg["model_context_window_tokens"]
            or p.app_max_input_tokens != cfg["input_max_tokens"]
            or p.reserved_output_tokens != cfg["reserved_output_tokens"]
            or p.provider_input_overhead_tokens != cfg["provider_input_overhead_tokens"]
            or p.safety_margin_tokens != cfg["safety_margin_tokens"]
            or p.timeout_seconds != cfg["timeout_seconds"]
        ):
            raise KnowledgeBranchError("knowledge_summary_profile_mismatch")
        model = _AuditedModel(base, self.profile, self.audit_hook)
        return _Candidates(model, SummaryService(model, base.counter, base.settings))
