"""Six sequential nodes and synchronous isolated graph END."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from resolv.repositories.understanding import UnderstandingRepository
from resolv.schemas.understanding import RouteSelection, UnderstandingError
from resolv.understanding_rules import select_route


class UnderstandingState(TypedDict):
    understanding_id: str
    conversation_id: str
    state_schema_version: int
    context_hash: str
    phase: str
    resolution: dict | None
    classification: dict | None
    selection: dict | None
    result_status: str | None


class UnderstandingNodes:
    def __init__(self, service, execution):
        self.service, self.execution = service, execution

    def current(self, state):
        self.execution.assert_owned()
        with self.service.session_factory.begin() as session:
            repo = UnderstandingRepository(session, self.service.owner_id)
            row = repo.owned(
                self.execution.understanding_id,
                self.execution.epoch,
                self.execution.execution_owner_id,
            )
            record = repo.get(row.conversation_id, row.understanding_id)
        self.service.compatible(record)
        if type(state.get("state_schema_version")) is not int:
            raise UnderstandingError(
                "understanding_incompatible", understanding_id=record.understanding_id
            )
        if state.get("selection") is not None:
            if record.resolution is None or record.classification is None:
                raise UnderstandingError(
                    "understanding_incompatible",
                    understanding_id=record.understanding_id,
                )
            expected = select_route(
                record.snapshot, record.resolution.output, record.classification.output
            )
            if state["selection"] != expected.model_dump(mode="json") or state.get(
                "result_status"
            ) != ("ready" if expected.route_ready else "needs_clarification"):
                raise UnderstandingError(
                    "understanding_incompatible",
                    understanding_id=record.understanding_id,
                )
        for field in ("resolution", "classification"):
            saved = getattr(record, field)
            if state.get(field) is not None and (
                saved is None or state[field] != saved.model_dump(mode="json")
            ):
                raise UnderstandingError(
                    "understanding_incompatible",
                    understanding_id=record.understanding_id,
                )
        if (
            state["understanding_id"] != record.understanding_id
            or state["conversation_id"] != record.conversation_id
            or state["context_hash"] != record.snapshot.snapshot_hash
            or state["state_schema_version"] != 1
        ):
            raise UnderstandingError(
                "understanding_incompatible", understanding_id=record.understanding_id
            )
        return record

    def load_context(self, state):
        self.current(state)
        return {"phase": "load_context"}

    def stage(self, state, name):
        record = self.current(state)
        result = record.resolution if name == "resolution" else record.classification
        if result is None:

            def loader(ref):
                self.execution.assert_owned()
                with self.service.session_factory.begin() as session:
                    repo = UnderstandingRepository(session, self.service.owner_id)
                    repo.owned(
                        record.understanding_id,
                        self.execution.epoch,
                        self.execution.execution_owner_id,
                    )
                    return repo.load_source(
                        record.conversation_id, ref, record.snapshot.cutoff_seq
                    )

            def audit_hook(audit):
                self.execution.assert_owned()
                with self.service.session_factory.begin() as session:
                    UnderstandingRepository(
                        session, self.service.owner_id
                    ).record_invocation(
                        record.understanding_id,
                        self.execution.epoch,
                        self.execution.execution_owner_id,
                        audit,
                    )

            if name == "resolution":
                result = self.service.resolver.resolve(
                    record.snapshot, loader, audit_hook
                )
                if (
                    record.continue_task_id is not None
                    and result.output.continuation_selection != record.continue_task_id
                ):
                    raise UnderstandingError(
                        "continuation_conflict",
                        understanding_id=record.understanding_id,
                    )
            else:
                if record.resolution is None:
                    raise UnderstandingError("understanding_state_conflict")
                result = self.service.classifier.classify(
                    record.snapshot, record.resolution.output, loader, audit_hook
                )
            self.execution.assert_owned()
            with self.service.session_factory.begin() as session:
                UnderstandingRepository(session, self.service.owner_id).save_stage(
                    record.understanding_id,
                    self.execution.epoch,
                    self.execution.execution_owner_id,
                    name,
                    result.output,
                    result.audit,
                )
            self.service.fault("after_" + name + "_saved")
        return {
            "phase": "resolve_references"
            if name == "resolution"
            else "classify_intents",
            "resolution"
            if name == "resolution"
            else "classification": result.model_dump(mode="json"),
        }

    def resolve_references(self, state):
        return self.stage(state, "resolution")

    def classify_intents(self, state):
        return self.stage(state, "intents")

    def select_route(self, state):
        record = self.current(state)
        selection = select_route(
            record.snapshot, record.resolution.output, record.classification.output
        )
        return {
            "phase": "select_route",
            "selection": selection.model_dump(mode="json"),
            "result_status": "ready"
            if selection.route_ready
            else "needs_clarification",
        }

    def publish_result(self, state):
        record = self.current(state)
        with self.service.session_factory.begin() as session:
            UnderstandingRepository(session, self.service.owner_id).publish(
                record.understanding_id,
                self.execution.epoch,
                self.execution.execution_owner_id,
                RouteSelection.model_validate(state["selection"]),
            )
        self.service.fault("after_publish")
        return {"phase": "publish_result"}

    def end(self, state):
        self.current(state)
        return {"phase": "end"}


def build_understanding_graph(nodes, saver):
    graph = StateGraph(UnderstandingState)
    names = (
        "load_context",
        "resolve_references",
        "classify_intents",
        "select_route",
        "publish_result",
        "end",
    )
    previous = START
    for name in names:
        graph.add_node(name, getattr(nodes, name))
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile(checkpointer=saver)
