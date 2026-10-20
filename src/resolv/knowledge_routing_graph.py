"""routed-v3 adds persisted knowledge gates; the v2 business graph stays available."""

from sqlalchemy import select

from resolv.models.conversation import ConversationAction
from resolv.repositories.conversation import ConversationRepository
from resolv.routing_graph import (
    RoutedConversationState,
    RoutedNodes,
    build_routing_graph,
)
from resolv.schemas.branch_execution import ReplyMetadata, parse_reply_metadata
from resolv.schemas.conversation import ConversationError
from resolv.schemas.knowledge_branch import (
    BranchReplyV2,
    KnowledgeBranchError,
    ReplyMetadataV2,
)
from resolv.services.conversation_routing import load_primary
from resolv.services.knowledge_branch import knowledge_answer_reply


class KnowledgeConversationState(RoutedConversationState):
    knowledge_record_id: str | None
    knowledge_stage: str | None
    knowledge_evidence_hash: str | None


def promote_metadata(value):
    metadata = parse_reply_metadata(value)
    if isinstance(metadata, ReplyMetadata):
        data = metadata.model_dump(mode="json") | {
            "schema_version": 2,
            "knowledge": None,
        }
        return parse_reply_metadata(data)
    return metadata


class KnowledgeRoutedNodes(RoutedNodes):
    @property
    def knowledge(self):
        return self.service.knowledge_branch()

    def initial_state(self, run):
        return super().initial_state(run) | {
            "state_schema_version": 3,
            "knowledge_record_id": None,
            "knowledge_stage": None,
            "knowledge_evidence_hash": None,
        }

    def route_by_intent(self, state):
        self.execution.assert_owned()
        run = self.service.get_run(state["run_id"])
        record = self.service.understanding.record(
            run.conversation_id, run.understanding_id
        )
        view = self.service.understanding.view(record)
        if view.route_ready and view.branch == "knowledge":
            p = load_primary(run, view, snapshot_hash=record.snapshot.snapshot_hash)
            return {
                "branch": "knowledge",
                "primary": p.model_dump(mode="json"),
                "phase": "prepare_knowledge_query",
            }
        return super().route_by_intent(state)

    def primary(self, state):
        primary = super().primary(state)
        if primary.primary.branch == "knowledge" and state.get("phase") in {
            "prepare_business",
            "prepare_memory",
            "validate_decision",
            "agent_tools",
            "stage_action",
            "await_approval",
            "execute_write",
            "limit_reply",
        }:
            raise KnowledgeBranchError("knowledge_business_tools_forbidden")
        return primary

    def validate_metadata(self, state, run, view):
        if state.get("branch") != view.branch:
            raise ConversationError("run_state_conflict", run_id=run.run_id)
        data = state.get("result_metadata")
        if view.branch != "knowledge" or not view.route_ready:
            old_state = dict(state)
            if data is not None:
                meta = parse_reply_metadata(data)
                if isinstance(meta, ReplyMetadataV2):
                    if meta.knowledge is not None:
                        raise ConversationError("run_state_conflict", run_id=run.run_id)
                    old_state["result_metadata"] = {
                        k: v
                        for k, v in meta.model_dump(mode="json").items()
                        if k != "knowledge"
                    } | {"schema_version": 1}
            return super().validate_metadata(old_state, run, view)
        if data is None:
            return
        _, _, record = self.knowledge._load(run.run_id, self.execution)
        reply = (
            knowledge_answer_reply(record, record.answer)
            if record.check and record.check.evidence_strong
            else self.knowledge.fallback(run.run_id, self.execution)
        )
        if (
            data != reply.metadata.model_dump(mode="json")
            or state.get("final_answer") != reply.answer
        ):
            raise ConversationError("run_state_conflict", run_id=run.run_id)

    def prepare_knowledge_query(self, state):
        self.primary(state)
        self.knowledge.prepare_query(state["run_id"], self.execution)
        return {
            "knowledge_record_id": state["run_id"],
            "knowledge_stage": "query",
            "phase": "retrieve_knowledge",
        }

    def retrieve_knowledge(self, state):
        self.primary(state)
        record = self.knowledge.retrieve(state["run_id"], self.execution)
        return {
            "knowledge_record_id": state["run_id"],
            "knowledge_stage": "retrieval",
            "knowledge_evidence_hash": record.retrieval.bundle.evidence_hash,
            "phase": "confidence_check",
        }

    def confidence_check(self, state):
        self.primary(state)
        record = self.knowledge.check(state["run_id"], self.execution)
        return {
            "knowledge_stage": "check",
            "knowledge_evidence_hash": record.check.evidence_hash,
            "phase": "main_agent" if record.check.evidence_strong else "fallback_reply",
        }

    def main_agent(self, state):
        primary = self.primary(state)
        if primary.primary.branch != "knowledge":
            return super().main_agent(state)
        answer = self.knowledge.answer(state["run_id"], self.execution)
        run, _, record = self.knowledge._load(state["run_id"], self.execution)
        reply = knowledge_answer_reply(record, answer)
        return {
            "final_answer": reply.answer,
            "result_metadata": reply.metadata.model_dump(mode="json"),
            "knowledge_stage": "answer",
            "summary_version": run.summary_version,
            "history_cutoff_seq": run.history_cutoff_seq,
            "phase": "finalize_answer",
        }

    def fallback_reply(self, state):
        self.primary(state)
        reply = self.knowledge.fallback(state["run_id"], self.execution)
        return {
            "final_answer": reply.answer,
            "result_metadata": reply.metadata.model_dump(mode="json"),
            "phase": "finalize_answer",
        }

    def finalize_answer(self, state):
        self.execution.assert_owned()
        run = self.service.get_run(state["run_id"])
        u = self.service.understanding.record(run.conversation_id, run.understanding_id)
        view = self.service.understanding.view(u)
        self.validate_metadata(state, run, view)
        if view.route_ready and view.branch == "knowledge":
            self.knowledge.finalize(
                run.run_id,
                self.execution,
                BranchReplyV2(
                    answer=state["final_answer"],
                    metadata=ReplyMetadataV2.model_validate_json(
                        __import__("json").dumps(state["result_metadata"])
                    ),
                ),
            )
            metadata = state["result_metadata"]
        else:
            meta = promote_metadata(state["result_metadata"])
            self.service.commit_answer(
                run.run_id, self.execution.epoch, state["final_answer"], metadata=meta
            )
            metadata = meta.model_dump(mode="json")
        self.service.fault("after_final_result")
        return {"phase": "end", "result_metadata": metadata}

    def restore_terminal_state(self, state, run):
        u = self.service.understanding.record(run.conversation_id, run.understanding_id)
        view = self.service.understanding.view(u)
        primary = (
            load_primary(run, view, snapshot_hash=u.snapshot.snapshot_hash)
            if view.route_ready
            else None
        )
        return state | {
            "branch": view.branch,
            "primary": primary.model_dump(mode="json") if primary else None,
            "result_metadata": run.result_json,
            "summary_version": run.summary_version,
            "history_cutoff_seq": run.history_cutoff_seq,
        }


def build_knowledge_routing_graph(nodes, saver):
    return build_routing_graph(nodes, saver, knowledge=True)


def reconcile_knowledge_run(service, run, execution):
    if run.orchestration_version == "routed-v4":
        from resolv.refund_routing_graph import RefundRoutedNodes
        nodes = RefundRoutedNodes(service, execution)
    else:
        nodes = KnowledgeRoutedNodes(service, execution)
    graph = service._graph(execution)
    state = graph.get_state({"configurable": {"thread_id": run.conversation_id}}).values
    with service.session_factory() as session:
        repo = ConversationRepository(session, service.owner_id)
        if (
            session.scalar(
                select(ConversationAction.action_id).where(
                    ConversationAction.run_id == run.run_id
                )
            )
            is not None
        ):
            raise ConversationError(
                "knowledge_business_tools_forbidden", run_id=run.run_id
            )
        if state and state.get("run_id") != run.run_id:
            old = repo.run(state.get("run_id"))
            if old.conversation_id != run.conversation_id or old.status not in {
                "completed",
                "failed",
            }:
                raise ConversationError("run_state_conflict", run_id=run.run_id)
            return run
    _, _, record = nodes.knowledge._load(run.run_id, execution)
    if state:
        if (
            state.get("state_schema_version") != (4 if run.orchestration_version == "routed-v4" else 3)
            or run.state_schema_version != (4 if run.orchestration_version == "routed-v4" else 3)
            or state.get("conversation_id") != run.conversation_id
            or state.get("mode") != run.mode
            or state.get("branch") not in {None, "knowledge"}
            or state.get("model_step") != 0
            or state.get("tool_calls")
            or state.get("messages")
            or state.get("action_id") is not None
            or state.get("knowledge_record_id") not in {None, run.run_id}
        ):
            raise ConversationError("run_state_conflict", run_id=run.run_id)
        if state.get("primary") is not None:
            nodes.primary(state)
        evidence_hash = (
            record.retrieval.bundle.evidence_hash if record.retrieval else None
        )
        if state.get("knowledge_evidence_hash") not in {None, evidence_hash}:
            raise ConversationError("run_state_conflict", run_id=run.run_id)
    return run
