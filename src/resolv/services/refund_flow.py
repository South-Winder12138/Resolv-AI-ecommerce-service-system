"""Recoverable serial refund stages; no application writer is reachable here."""

from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
)

from resolv.core.config import ContextSettings
from resolv.core.refund_model import refund_input_estimate
from resolv.db.conversation_base import utc_now
from resolv.repositories.conversation import ConversationRepository
from resolv.repositories.refund_flow import RefundFlowRepository
from resolv.repositories.understanding import UnderstandingRepository
from resolv.schemas.branch_execution import PrimaryExecution
from resolv.schemas.hybrid_knowledge import HybridDiagnostics, HybridRetrievalResult
from resolv.schemas.knowledge_branch import KnowledgeRetrievalStage, canonical_json
from resolv.schemas.refund import OrderBinding, RefundAccessContext, RefundFlowError
from resolv.schemas.refund_flow import (
    PreparedRefundQuery,
    RefundBindingStage,
    RefundProfile,
    RefundReadAudit,
    RefundScopeReply,
)
from resolv.services.context import ContextBudgetExceeded
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.knowledge_branch_evidence import select_knowledge_evidence
from resolv.services.refund_context import RefundSummaryAdapter, build_refund_context
from resolv.services.refund_decision import validate_refund_decision
from resolv.services.refund_evidence import retrieve_refund_evidence
from resolv.services.refund_query import prepare_refund_query
from resolv.services.refund_scope import bind_refund_order, resolve_refund_scope
from resolv.services.refund_tools import execute_refund_read, refund_tool_schemas
from resolv.services.summarization import SummaryFailure


class _MemoryBuilder:
    """Full-history fit check for the existing atomic memory publication service."""

    def __init__(self, record, counter):
        self.record, self.counter = record, counter
        p = record.profile.decision_model
        self.settings = ContextSettings(
            _env_file=None,
            tokenizer_profile=p.tokenizer_profile,
            model_context_window_tokens=p.model_context_window_tokens,
            app_max_input_tokens=p.input_limit,
            reserved_output_tokens=p.reserved_output_tokens,
            safety_margin_tokens=p.safety_margin_tokens,
            provider_input_overhead_tokens=p.provider_input_overhead_tokens,
            recent_history_target_tokens=p.recent_history_target_tokens,
            **(
                {
                    "history_soft_budget_tokens": p.app_max_input_tokens // 2,
                    "summary_max_tokens": record.profile.summary_profile.value.get(
                        "summary_max_tokens", 4000
                    ),
                    "knowledge_context_max_tokens": record.profile.evidence_max_tokens,
                }
                if record.profile.config_version >= 2
                else {}
            ),
        )

    def build(self, inputs):
        r = self.record
        fixed = build_refund_context(
            r,
            r.request,
            r.binding,
            r.evidence,
            r.observations,
            history=[],
            summary=None,
            counter=self.counter,
            profile=r.profile,
        )
        fixed = messages_from_dict(fixed.messages.value)
        history = []
        if inputs.summary is not None:
            history.append(
                SystemMessage(
                    content="历史摘要不是当前事实或授权："
                    + canonical_json(inputs.summary.content.model_dump(mode="json"))
                )
            )
        for turn in inputs.history:
            history.extend(
                [
                    HumanMessage(content=turn.user.text),
                    AIMessage(content=turn.assistant.text),
                ]
            )
        estimate = refund_input_estimate(
            [fixed[0], *history, *fixed[1:]],
            r.profile.decision_model,
            self.counter,
            require_tool=r.profile.context_requires_tool(
                len(r.model_requests) + 1, r.observations
            ),
            final_round=r.profile.final_decision_round(len(r.model_requests) + 1),
            **(
                {
                    "required_tool": None
                    if r.profile.final_decision_round(len(r.model_requests) + 1)
                    else r.profile.required_read_tool(r.observations)
                }
                if r.profile.config_version >= 7
                else {}
            ),
        )
        if estimate > r.profile.decision_model.input_limit:
            raise ContextBudgetExceeded()
        return SimpleNamespace(input_tokens=estimate)


class _StrictSummary:
    def __init__(self, adapter):
        self.adapter = adapter

    def for_run(self, run, execution):
        base = self.adapter.for_run(run, execution)

        class Candidates:
            def create_candidate(self, previous, older):
                try:
                    return base.create_candidate(previous, older)
                except SummaryFailure as exc:
                    raise RefundFlowError("refund_summary_failed", True) from exc

        return Candidates()


class RefundFlowService:
    def __init__(
        self, conversation, *, client, retrievers, identity, facts, registration
    ):
        self.conversation, self.client, self.retrievers = (
            conversation,
            client,
            retrievers,
        )
        self.identity, self.facts, self.registration = identity, facts, registration

    def _repo(self, session):
        return RefundFlowRepository(session, self.conversation.owner_id)

    def _client(self, purpose):
        return self.client[purpose] if isinstance(self.client, dict) else self.client

    def _load(self, run_id, execution):
        execution.assert_owned()
        run = self.conversation.get_run(run_id)
        u = self.conversation.understanding.record(
            run.conversation_id, run.understanding_id
        )
        self.conversation.understanding.compatible(u)
        view = self.conversation.understanding.view(u)
        if (
            run.orchestration_version != "routed-v4"
            or run.mode != "agent"
            or u.original_text != run.question
            or not view.route_ready
            or view.primary_request is None
        ):
            raise RefundFlowError("refund_identity_conflict")
        primary = PrimaryExecution(
            conversation_id=run.conversation_id,
            understanding_id=run.understanding_id,
            snapshot_hash=u.snapshot.snapshot_hash,
            primary=view.primary_request,
        )
        profile = RefundProfile.model_validate_json(canonical_json(run.refund_config))
        if self.facts.descriptor != profile.adapter:
            raise RefundFlowError("refund_adapter_conflict")
        with self.conversation.session_factory.begin() as s:
            record = self._repo(s).create(run_id, execution.epoch, primary, profile)
        return run, u, primary, record

    def _access(self, record):
        value = self.identity.resolve(
            getattr(self.conversation, "refund_identity_context", None)
        )
        value = RefundAccessContext.model_validate_json(value.model_dump_json())
        if (
            value.namespace != record.profile.adapter.namespace
            or value.authorization_version
            != record.profile.adapter.authorization_version
        ):
            raise RefundFlowError("refund_authorization_conflict")
        subjects = {
            o.fields["subject_ref"].value
            for o in record.observations
            if "subject_ref" in o.fields
        }
        if subjects and subjects != {value.subject_ref}:
            raise RefundFlowError("refund_authorization_conflict")
        return value

    def _get(self, run_id):
        with self.conversation.session_factory() as s:
            return self._repo(s).get(run_id)

    def _save_stage(self, run_id, execution, record, stage, payload):
        execution.assert_owned()
        self.conversation.fault(f"refund_before_{stage}_saved")
        digest = (
            record.stage_hashes[-1][2] if record.stage_hashes else record.primary_hash
        )
        with self.conversation.session_factory.begin() as s:
            result = self._repo(s).save_stage(
                run_id, execution.epoch, stage, payload, expected_input_hash=digest
            )
            if result.error_code is not None:
                result = self._repo(s).record_error(
                    run_id, execution.epoch, None, False
                )
        self.conversation.fault(f"refund_after_{stage}_saved")
        return result

    def _phase(self, run_id, execution, stage, operation):
        run, u, p, record = self._load(run_id, execution)
        if any(name == stage for name, _, _ in record.stage_hashes) or (
            record.scope_reply and stage != "binding"
        ):
            return record
        return self._guard(
            run_id,
            execution,
            lambda: self._save_stage(
                run_id, execution, record, stage, operation(run, u, p, record)
            ),
        )

    def _guard(self, run_id, execution, operation):
        try:
            return operation()
        except Exception as exc:
            execution.assert_owned()
            try:
                with self.conversation.session_factory.begin() as s:
                    self._repo(s).record_error(
                        run_id,
                        execution.epoch,
                        getattr(exc, "code", "refund_stage_failed"),
                        getattr(exc, "retryable", True),
                    )
            except Exception as store:
                raise RefundFlowError("refund_store_failed", True, run_id) from store
            raise

    def _audit(self, run_id, execution):
        def save(audit):
            execution.assert_owned()
            with self.conversation.session_factory.begin() as s:
                self._repo(s).record_audit(run_id, execution.epoch, audit)

        return save

    def _unknown_models(self, record, execution, purpose):
        linked = {r.audit.invocation_id for r in record.model_requests}
        if purpose == "refund_query" and record.query:
            return
        latest = {a.invocation_id: a for a in record.audits if a.purpose == purpose}
        for audit in latest.values():
            if audit.invocation_id not in linked and audit.outcome in {
                "started",
                "succeeded",
            }:
                self._audit(record.run_id, execution)(
                    audit.model_copy(
                        update={
                            "outcome": "unknown",
                            "finished_at": utc_now(),
                            "error_code": "refund_response_not_saved",
                        }
                    )
                )

    def bind(self, run_id, execution):
        def operation(run, u, p, record):
            def loader(ref):
                with self.conversation.session_factory() as s:
                    return UnderstandingRepository(
                        s, self.conversation.owner_id
                    ).load_source(run.conversation_id, ref, u.snapshot.cutoff_seq)

            scope = resolve_refund_scope(p, u.snapshot, source_loader=loader)
            if scope.scope_reply:
                return RefundBindingStage(scope_reply=scope.scope_reply)
            binding = bind_refund_order(
                p,
                u.snapshot,
                order_source=record.profile.adapter.namespace,
                source_loader=loader,
            )
            if isinstance(binding, RefundScopeReply):
                return RefundBindingStage(scope_reply=binding)
            if not isinstance(binding, OrderBinding):
                raise RefundFlowError("refund_identity_conflict")
            try:
                self._access(record)
            except RefundFlowError as exc:
                if exc.code != "refund_identity_unavailable":
                    raise
                return RefundBindingStage(
                    binding=binding,
                    scope_reply=RefundScopeReply(
                        kind="unavailable",
                        reason="客户身份与订单系统尚未接入，暂不能自动判断或提交退款。",
                    ),
                )
            return RefundBindingStage(binding=binding, request=scope.request)

        return self._phase(run_id, execution, "binding", operation)

    def prepare_query(self, run_id, execution):
        def operation(run, u, p, record):
            if record.binding is None or record.request is None:
                raise RefundFlowError("refund_stage_predecessor_missing")
            if record.profile.selection.status != "configured":
                return PreparedRefundQuery(
                    standard=p.primary.text,
                    expanded=(),
                    bm25_text=p.primary.text,
                    status="prepared",
                    input_hash=canonical_hash(
                        {
                            "primary": p.model_dump(mode="json"),
                            "binding": record.binding.model_dump(mode="json"),
                            "selection": record.profile.selection.model_dump(
                                mode="json"
                            ),
                        }
                    ),
                )
            self._unknown_models(record, execution, "refund_query")
            return prepare_refund_query(
                record.request,
                p,
                record.binding,
                client=self._client("refund_query"),
                profile=record.profile,
                query_token_count=self.retrievers.count_query_tokens,
                audit_hook=self._audit(run_id, execution),
            )

        return self._phase(run_id, execution, "query", operation)

    def retrieve(self, run_id, execution):
        def operation(run, u, p, record):
            if record.query is None:
                raise RefundFlowError("refund_stage_predecessor_missing")
            counter = self._client("refund_decision").counter_factory(
                record.profile.decision_model.tokenizer_profile
            )
            if record.profile.selection.status != "configured":
                result = HybridRetrievalResult(
                    index_version="refund-no-policy",
                    index_profile_hash=canonical_hash({"policy": "not-configured"}),
                    retrieval_profile=record.profile.retrieval_profile,
                    windows=(),
                    chunks={},
                    diagnostics=HybridDiagnostics(),
                )
                return KnowledgeRetrievalStage.from_result(
                    result,
                    bundle=select_knowledge_evidence(
                        result,
                        counter=counter,
                        max_tokens=record.profile.evidence_max_tokens,
                    ),
                    file_configs=(),
                )
            return retrieve_refund_evidence(
                record.query,
                selection=record.profile.selection,
                retrievers=self.retrievers,
                profile=record.profile,
                counter=counter,
            )

        return self._phase(run_id, execution, "retrieval", operation)

    def _context(self, run, record, execution):
        counter = self._client("refund_decision").counter_factory(
            record.profile.decision_model.tokenizer_profile
        )
        build_refund_context(
            record,
            record.request,
            record.binding,
            record.evidence,
            record.observations,
            history=[],
            summary=None,
            counter=counter,
            profile=record.profile,
        )
        summary = (
            _StrictSummary(
                RefundSummaryAdapter(
                    self.conversation.summary_service,
                    record.profile,
                    self._audit(run.run_id, execution),
                )
            )
            if self.conversation.summary_service
            else None
        )
        pinned = self.conversation.prepare_memory(
            run.run_id,
            execution,
            context_builder=_MemoryBuilder(record, counter),
            working_question=record.primary_json.value["primary"]["text"],
            system_prompt=record.profile.decision_model.system_prompt,
            knowledge=[],
            tool_schemas=refund_tool_schemas(),
            summary_service_override=summary,
        )
        with self.conversation.session_factory() as s:
            repo = ConversationRepository(s, self.conversation.owner_id)
            previous = repo.summary(run.conversation_id, pinned.summary_version)
            history = repo.completed_turns(
                run.conversation_id,
                after_seq=previous.covered_until_seq if previous else 0,
                cutoff_seq=pinned.history_cutoff_seq,
            )
        return build_refund_context(
            record,
            record.request,
            record.binding,
            record.evidence,
            record.observations,
            history=history,
            summary=previous,
            counter=counter,
            profile=record.profile,
        )

    def main_agent(self, run_id, execution):
        run, _, _, record = self._load(run_id, execution)
        if record.scope_reply or record.decision or record.step_limit:
            return record
        if record.retrieval is None:
            raise RefundFlowError("refund_stage_predecessor_missing")

        def operation():
            current = self._get(run_id)
            if current.model_requests:
                last = current.model_requests[-1]
                if last.decision is not None:
                    value = validate_refund_decision(
                        last.decision,
                        request=current.request,
                        binding=current.binding,
                        evidence=current.evidence,
                        observations=current.observations,
                        profile=current.profile,
                    )
                    return self._save_stage(
                        run_id, execution, current, "decision", value
                    )
                completed = {o.tool_call_id for o in current.observations}
                if any(c.id not in completed for c in last.tool_calls):
                    return current
            order = next(
                (
                    o
                    for o in reversed(current.observations)
                    if o.tool_name == "query_refund_order"
                ),
                None,
            )
            status = next(
                (
                    o
                    for o in reversed(current.observations)
                    if o.tool_name == "query_refund_status"
                ),
                None,
            )
            for obs in (order, status):
                if obs and obs.availability in {"not_found", "forbidden"}:
                    raise RefundFlowError("refund_not_found")
                if obs and obs.availability != "available":
                    raise RefundFlowError("refund_unavailable")
            if (
                status
                and current.profile.selection.status != "configured"
                and current.request.operation != "status"
                and status.fields.get("local_application")
                and status.fields["local_application"].value is None
            ):
                raise RefundFlowError(
                    "refund_release_invalid"
                    if current.profile.selection.status == "invalid"
                    else "refund_unavailable"
                )
            if len(current.model_requests) >= current.profile.max_agent_steps:
                with self.conversation.session_factory.begin() as s:
                    return self._repo(s).mark_step_limit(run_id, execution.epoch)
            self._unknown_models(current, execution, "refund_decision")
            context = self._context(run, current, execution)
            model_step = len(current.model_requests) + 1
            final_round = current.profile.final_decision_round(model_step)
            response = self._client("refund_decision").decide(
                messages_from_dict(context.messages.value),
                profile=current.profile.decision_model,
                model_step=model_step,
                attempt=run.attempt,
                require_tool=not final_round and (order is None or status is None),
                audit_hook=self._audit(run_id, execution),
                **(
                    {
                        "required_tool": None
                        if final_round
                        else current.profile.required_read_tool(current.observations)
                    }
                    if current.profile.config_version >= 7
                    else {}
                ),
                **(
                    {"final_round": final_round}
                    if current.profile.config_version >= 3
                    else {}
                ),
            )
            execution.assert_owned()
            if response.input_hash != context.input_hash:
                raise RefundFlowError("refund_model_input_conflict")
            value = None
            try:
                if final_round and response.tool_calls:
                    raise RefundFlowError("refund_model_output_invalid", True)
                if any(
                    c.name not in {s["function"]["name"] for s in refund_tool_schemas()}
                    or c.args.value != {}
                    for c in response.tool_calls
                ):
                    raise RefundFlowError("refund_tool_scope_violation")
                if response.decision is not None:
                    value = validate_refund_decision(
                        response.decision,
                        request=current.request,
                        binding=current.binding,
                        evidence=current.evidence,
                        observations=current.observations,
                        profile=current.profile,
                    )
            except RefundFlowError as exc:
                self._audit(run_id, execution)(
                    response.audit.model_copy(
                        update={
                            "outcome": "failed",
                            "error_code": exc.code,
                            "finished_at": utc_now(),
                        }
                    )
                )
                raise
            self.conversation.fault("refund_before_model_request_saved")
            with self.conversation.session_factory.begin() as s:
                result = self._repo(s).save_model_request(
                    run_id, execution.epoch, response
                )
                if (
                    response.tool_calls
                    and response.model_step == result.profile.max_agent_steps
                ):
                    result = self._repo(s).mark_step_limit(run_id, execution.epoch)
            self.conversation.fault("refund_after_model_request_saved")
            if response.decision is not None:
                result = self._save_stage(run_id, execution, result, "decision", value)
            return result

        return self._guard(run_id, execution, operation)

    def _read_audit(self, run_id, execution, audit):
        execution.assert_owned()
        with self.conversation.session_factory.begin() as s:
            return self._repo(s).record_read_audit(run_id, execution.epoch, audit)

    def read_next(self, run_id, execution):
        _, _, _, record = self._load(run_id, execution)
        if (
            record.scope_reply
            or record.decision
            or record.step_limit
            or not record.model_requests
        ):
            return record

        def operation():
            last = record.model_requests[-1]
            if last.model_step >= record.profile.max_agent_steps:
                with self.conversation.session_factory.begin() as s:
                    return self._repo(s).mark_step_limit(run_id, execution.epoch)
            existing = {o.tool_call_id: o for o in record.observations}
            call = next((c for c in last.tool_calls if c.id not in existing), None)
            if call is None:
                return record
            audits = [
                a
                for a in record.read_audits
                if a.model_step == last.model_step and a.tool_call_id == call.id
            ]
            latest = {a.invocation_id: a for a in audits}
            for audit in latest.values():
                if audit.outcome == "started":
                    self._read_audit(
                        run_id,
                        execution,
                        audit.model_copy(
                            update={
                                "outcome": "unknown",
                                "finished_at": utc_now(),
                                "error_code": "refund_observation_not_saved",
                            }
                        ),
                    )
            audit = RefundReadAudit(
                invocation_id=uuid4().hex,
                model_step=last.model_step,
                tool_call_id=call.id,
                tool_name=call.name,
                binding_hash=record.binding.binding_hash,
                profile_fingerprint=record.profile.adapter.fingerprint,
                input_hash=canonical_hash(
                    {
                        "binding_hash": record.binding.binding_hash,
                        "args": call.args.value,
                    }
                ),
                attempt=max((a.attempt for a in audits), default=0) + 1,
                started_at=utc_now(),
                outcome="started",
            )
            self._read_audit(run_id, execution, audit)
            try:
                access = self._access(record)
                observation = execute_refund_read(
                    call,
                    binding=record.binding,
                    access=access,
                    adapter=self.facts,
                    registration=self.registration,
                    observations=record.observations,
                )
                if observation.error_code:
                    raise RefundFlowError("refund_provider_failed", True)
            except Exception as exc:
                code = (
                    exc.code
                    if isinstance(exc, RefundFlowError)
                    else "refund_provider_failed"
                )
                self._read_audit(
                    run_id,
                    execution,
                    audit.model_copy(
                        update={
                            "outcome": "failed"
                            if isinstance(exc, RefundFlowError) and not exc.retryable
                            else "unknown",
                            "finished_at": utc_now(),
                            "error_code": code,
                        }
                    ),
                )
                if isinstance(exc, RefundFlowError):
                    raise
                raise RefundFlowError("refund_provider_failed", True) from exc
            execution.assert_owned()
            self.conversation.fault("refund_before_observation_saved")
            with self.conversation.session_factory.begin() as s:
                result = self._repo(s).save_observation(
                    run_id, execution.epoch, last.model_step, observation
                )
                result = self._repo(s).record_read_audit(
                    run_id,
                    execution.epoch,
                    audit.model_copy(
                        update={
                            "outcome": "succeeded",
                            "finished_at": utc_now(),
                            "observation_id": observation.observation_id,
                        }
                    ),
                )
                if result.error_code:
                    result = self._repo(s).record_error(
                        run_id, execution.epoch, None, False
                    )
            self.conversation.fault("refund_after_observation_saved")
            return result

        return self._guard(run_id, execution, operation)
