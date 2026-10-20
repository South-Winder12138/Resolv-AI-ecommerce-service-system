"""routed-v4: independent refund stages; original graphs remain version-pinned."""

from sqlalchemy import select

from resolv.knowledge_routing_graph import (
    KnowledgeConversationState,
    KnowledgeRoutedNodes,
)
from resolv.models.conversation import ConversationAction
from resolv.repositories.conversation import ConversationRepository
from resolv.repositories.refund_flow import RefundFlowRepository
from resolv.routing_graph import build_routing_graph
from resolv.schemas.branch_execution import parse_business_config, parse_reply_metadata
from resolv.schemas.conversation import ConversationError
from resolv.schemas.refund_flow import RefundProfile
from resolv.services.conversation_routing import load_primary
from resolv.services.refund_offer import finalize_refund_reply
from resolv.services.refund_reply_projection import build_refund_reply
from resolv.services.refund_runtime import promote_reply_v3


class RefundConversationState(KnowledgeConversationState):
    refund_run_id: str | None
    refund_stage: str | None
    refund_binding_hash: str | None
    refund_evidence_hash: str | None


class RefundRoutedNodes(KnowledgeRoutedNodes):
    @property
    def refund(self):
        return self.service.refund_branch()

    def initial_state(self, run):
        return super().initial_state(run) | {
            "state_schema_version": 4,
            "refund_run_id": None,
            "refund_stage": None,
            "refund_binding_hash": None,
            "refund_evidence_hash": None,
        }

    def is_refund(self, state):
        return state.get("branch") == "refund_flow" and state.get("mode") == "agent"

    def route_by_intent(self, state):
        run = self.service.get_run(state["run_id"])
        u = self.service.understanding.record(run.conversation_id, run.understanding_id)
        view = self.service.understanding.view(u)
        if run.mode == "agent" and view.route_ready and view.branch == "refund_flow":
            primary = load_primary(run, view, snapshot_hash=u.snapshot.snapshot_hash)
            profile = RefundProfile.model_validate_json(
                __import__("json").dumps(run.refund_config)
            )
            return {
                "branch": "refund_flow",
                "primary": primary.model_dump(mode="json"),
                "remaining_iterations": profile.max_agent_steps,
                "phase": "refund_scope_and_mode",
            }
        return super().route_by_intent(state)

    def validate_metadata(self, state, run, view):
        data = state.get("result_metadata")
        if self.is_refund(state) and view.route_ready:
            if state.get("branch") != view.branch:
                raise ConversationError("run_state_conflict", run_id=run.run_id)
            if data is None:
                return
            with self.service.session_factory() as s:
                repo = RefundFlowRepository(s, self.service.owner_id)
                record, offer = repo.get(run.run_id), repo.offer_for_run(run.run_id)
            reply = build_refund_reply(record, offer)
            if (
                data != reply.metadata.model_dump(mode="json")
                or state.get("final_answer") != reply.answer
            ):
                raise ConversationError("run_state_conflict", run_id=run.run_id)
            return
        old = dict(state)
        if data is not None and data.get("schema_version") == 3:
            meta = parse_reply_metadata(data)
            if meta.refund is not None:
                raise ConversationError("run_state_conflict", run_id=run.run_id)
            old["result_metadata"] = meta.model_dump(
                mode="json", exclude={"refund"}
            ) | {"schema_version": 2}
        return super().validate_metadata(old, run, view)

    def refund_scope_and_mode(self, state):
        self.primary(state)
        return {"phase": "refund_bind"}

    def _state(self, record, phase):
        run = self.service.get_run(record.run_id)
        return {
            "refund_run_id": record.run_id,
            "refund_stage": next(reversed(record.stage_hashes), None),
            "refund_binding_hash": record.binding.binding_hash
            if record.binding
            else None,
            "refund_evidence_hash": record.evidence.evidence_hash
            if record.evidence
            else None,
            "model_step": len(record.model_requests),
            "remaining_iterations": record.profile.max_agent_steps
            - len(record.model_requests),
            "summary_version": run.summary_version,
            "history_cutoff_seq": run.history_cutoff_seq,
            "phase": phase,
        }

    def refund_bind(self, state):
        self.primary(state)
        record = self.refund.bind(state["run_id"], self.execution)
        return self._state(
            record, "finalize_answer" if record.scope_reply else "prepare_refund_query"
        )

    def prepare_refund_query(self, state):
        self.primary(state)
        record = self.refund.prepare_query(state["run_id"], self.execution)
        return self._state(record, "retrieve_refund_evidence")

    def retrieve_refund_evidence(self, state):
        self.primary(state)
        record = self.refund.retrieve(state["run_id"], self.execution)
        return self._state(record, "main_agent")

    def main_agent(self, state):
        if not self.is_refund(state):
            return super().main_agent(state)
        self.primary(state)
        record = self.refund.main_agent(state["run_id"], self.execution)
        return self._state(record, "validate_decision")

    def validate_decision(self, state):
        if not self.is_refund(state):
            return super().validate_decision(state)
        self.primary(state)
        _, _, _, record = self.refund._load(state["run_id"], self.execution)
        return self._state(
            record,
            "limit_reply"
            if record.step_limit
            else "finalize_answer"
            if record.decision
            else "agent_tools",
        )

    def agent_tools(self, state):
        if not self.is_refund(state):
            return super().agent_tools(state)
        self.primary(state)
        record = self.refund.read_next(state["run_id"], self.execution)
        ids = {o.tool_call_id for o in record.observations}
        pending = any(c.id not in ids for c in record.model_requests[-1].tool_calls)
        return self._state(record, "agent_tools" if pending else "main_agent")

    def limit_reply(self, state):
        if not self.is_refund(state):
            return super().limit_reply(state)
        self.primary(state)
        return {"phase": "finalize_answer"}

    def finalize_answer(self, state):
        if not self.is_refund(state):
            result = super().finalize_answer(state)
            return result | {
                "result_metadata": promote_reply_v3(
                    result["result_metadata"]
                ).model_dump(mode="json")
            }
        self.primary(state)
        _, _, _, record = self.refund._load(state["run_id"], self.execution)
        reply = finalize_refund_reply(
            self.service, state["run_id"], self.execution, record
        )
        self.service.fault("after_final_result")
        return self._state(record, "end") | {
            "final_answer": reply.answer,
            "result_metadata": reply.metadata.model_dump(mode="json"),
        }

    def restore_terminal_state(self, state, run):
        restored = super().restore_terminal_state(state, run)
        with self.service.session_factory() as s:
            record = RefundFlowRepository(s, self.service.owner_id).get(run.run_id)
        return restored | (self._state(record, "end") if record else {})


def build_refund_routing_graph(nodes, saver):
    return build_routing_graph(nodes, saver, knowledge=True, refund=True)


def reconcile_refund_run(service, run, execution):
    nodes = RefundRoutedNodes(service, execution)
    state = (
        service._graph(execution)
        .get_state({"configurable": {"thread_id": run.conversation_id}})
        .values
    )
    with service.session_factory() as s:
        repo = ConversationRepository(s, service.owner_id)
        if (
            s.scalar(
                select(ConversationAction.action_id).where(
                    ConversationAction.run_id == run.run_id
                )
            )
            is not None
        ):
            raise ConversationError("refund_tool_scope_violation", run_id=run.run_id)
        if state and state.get("run_id") != run.run_id:
            old = repo.run(state.get("run_id"))
            if old.conversation_id != run.conversation_id or old.status not in {
                "completed",
                "failed",
            }:
                raise ConversationError("run_state_conflict", run_id=run.run_id)
            return run
    _, _, _, record = nodes.refund._load(run.run_id, execution)
    current = service.refund_profile
    if current is not None and any(
        getattr(getattr(record.profile, purpose), key)
        != getattr(getattr(current, purpose), key)
        for purpose in ("query_model", "decision_model")
        for key in (
            "model_name",
            "model_api_mode",
            "api_address_fingerprint",
            "tool_schema_fingerprint",
        )
    ):
        raise ConversationError("run_state_incompatible", run_id=run.run_id)
    if state:
        before_route = state.get("phase") in {"understand_input", "route_by_intent"}
        count_limit = (
            parse_business_config(run.business_config).max_agent_steps
            if before_route
            else record.profile.max_agent_steps
        )
        if (
            state.get("state_schema_version") != 4
            or run.state_schema_version != 4
            or state.get("conversation_id") != run.conversation_id
            or state.get("mode") != "agent"
            or state.get("branch") not in {None, "refund_flow"}
            or state.get("tool_calls")
            or state.get("messages")
            or state.get("action_id") is not None
            or state.get("refund_run_id") not in {None, run.run_id}
            or state.get("model_step", 0) > len(record.model_requests)
            or state.get("model_step", 0) + state.get("remaining_iterations", 0)
            != count_limit
        ):
            raise ConversationError("run_state_conflict", run_id=run.run_id)
        if state.get("primary") is not None:
            nodes.primary(state)
        if state.get("refund_binding_hash") not in {
            None,
            record.binding.binding_hash if record.binding else None,
        } or state.get("refund_evidence_hash") not in {
            None,
            record.evidence.evidence_hash if record.evidence else None,
        }:
            raise ConversationError("run_state_conflict", run_id=run.run_id)
    return run
