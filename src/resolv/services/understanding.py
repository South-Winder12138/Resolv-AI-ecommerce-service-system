"""Independent orchestration. Returns complete only after saved END and terminal commit."""

from contextvars import Context

from sqlalchemy.exc import SQLAlchemyError

from resolv.db.checkpoints import CheckpointError
from resolv.repositories.understanding import TERMINAL, UnderstandingRepository
from resolv.schemas.understanding import UnderstandingError, UnderstandingView
from resolv.services.understanding_context import snapshot_digest
from resolv.services.understanding_execution import understanding_execution
from resolv.understanding_graph import UnderstandingNodes, build_understanding_graph


class UnderstandingService:
    def __init__(
        self,
        session_factory,
        checkpoint_factory,
        context,
        resolver,
        classifier,
        owner_id="local",
        fault_hook=None,
    ):
        self.session_factory, self.checkpoint_factory = (
            session_factory,
            checkpoint_factory,
        )
        self.context, self.resolver, self.classifier = context, resolver, classifier
        self.owner_id, self.fault_hook = owner_id, fault_hook

    def fault(self, point):
        if self.fault_hook:
            self.fault_hook(point)

    def accept(self, conversation_id, input):
        with self.session_factory.begin() as session:
            return UnderstandingRepository(session, self.owner_id).create(
                conversation_id,
                input,
                lambda sources: self.context.freeze(sources, input),
            )

    def record(self, conversation_id, id):
        with self.session_factory() as session:
            return UnderstandingRepository(session, self.owner_id).get(
                conversation_id, id
            )

    def compatible(self, record):
        frozen, current = record.snapshot.config, self.context.config
        if (
            record.snapshot.snapshot_hash != snapshot_digest(record.snapshot)
            or record.original_text != record.snapshot.original_text
            or record.snapshot.owner_id != self.owner_id
            or record.snapshot.conversation_id != record.conversation_id
            or frozen.model_name != current.model_name
            or frozen.model_api_mode != current.model_api_mode
            or frozen.api_address_fingerprint != current.api_address_fingerprint
            or frozen.tokenizer_profile != current.tokenizer_profile
            or frozen.resolution_schema != current.resolution_schema
            or frozen.intent_schema != current.intent_schema
        ):
            raise UnderstandingError(
                "understanding_incompatible", understanding_id=record.understanding_id
            )

    def view(self, record, status=None):
        status = status or record.status
        if record.final_json:
            view = UnderstandingView.model_validate_json(
                __import__("json").dumps(record.final_json)
            )
            return view.model_copy(
                update={
                    "status": status,
                    "route_ready": status == "ready",
                    "pending_requests": view.pending_requests
                    if status in TERMINAL
                    else [],
                    "followup_ids": view.followup_ids
                    | {"clarification_child": record.clarification_child_id},
                    "error_code": record.error_code,
                    "retryable": record.retryable,
                }
            )
        config = record.snapshot.config
        return UnderstandingView(
            schema_version=1,
            understanding_id=record.understanding_id,
            conversation_id=record.conversation_id,
            status=status,
            original_text=record.original_text,
            resolution=record.resolution.output if record.resolution else None,
            primary_request=None,
            branch=None,
            route_ready=False,
            pending_requests=[],
            clarification_question=None,
            context_refs=record.snapshot.source_fingerprints,
            versions={
                "resolution": config.resolution_version,
                "intents": config.intent_version,
                "mapping": config.mapping_version,
                "schema": 1,
            },
            stage_results={
                "resolution": record.resolution.model_dump(mode="json")
                if record.resolution
                else None,
                "intents": record.classification.model_dump(mode="json")
                if record.classification
                else None,
            },
            followup_ids={
                "parent": record.clarifies_understanding_id,
                "clarification_child": record.clarification_child_id,
                "continued_task": record.continue_task_id,
            },
            error_code=record.error_code,
            retryable=record.retryable,
        )

    def get(self, conversation_id, id):
        record = self.record(conversation_id, id)
        status = record.status
        if status == "running":
            try:
                with self.checkpoint_factory(conversation_id="understanding:" + id):
                    status = "interrupted"
            except CheckpointError as error:
                if error.code != "conversation_busy":
                    raise UnderstandingError(error.code, True, id) from None
        return self.view(record, status)

    def list_tasks(self, conversation_id, status="pending", cursor=None, limit=50):
        with self.session_factory() as session:
            return UnderstandingRepository(session, self.owner_id).list_tasks(
                conversation_id, status, cursor, limit
            )

    def resume(self, conversation_id, id):
        return self.run(conversation_id, id)

    def run(self, conversation_id, id):
        # LangChain merges implicit RunnableConfig from contextvars even when a
        # fresh config dict is supplied. This independent graph owns its entire
        # execution context, saver, thread and namespace.
        return Context().run(self._run, conversation_id, id)

    def _run(self, conversation_id, id):
        with self.session_factory() as session:
            repo = UnderstandingRepository(session, self.owner_id)
            repo.conversation(conversation_id, active=True)
            record = repo.get(conversation_id, id)
        if record.status in TERMINAL:
            return self.view(record)
        self.compatible(record)
        with understanding_execution(self, id) as execution:
            try:
                graph = build_understanding_graph(
                    UnderstandingNodes(self, execution),
                    execution.checkpoint_session.saver,
                )
                config = {
                    "configurable": {
                        "thread_id": "understanding:" + id,
                        "checkpoint_ns": "",
                    }
                }
                saved = graph.get_state(config)
                if saved.values:
                    values = saved.values
                    if (
                        type(values.get("state_schema_version")) is not int
                        or values.get("understanding_id") != id
                        or values.get("conversation_id") != conversation_id
                        or values.get("context_hash") != record.snapshot.snapshot_hash
                        or values.get("state_schema_version") != 1
                    ):
                        raise UnderstandingError(
                            "understanding_incompatible", understanding_id=id
                        )
                initial = {
                    "understanding_id": id,
                    "conversation_id": conversation_id,
                    "state_schema_version": 1,
                    "context_hash": record.snapshot.snapshot_hash,
                    "phase": "created",
                    "resolution": None,
                    "classification": None,
                    "selection": None,
                    "result_status": None,
                }
                durable = execution.checkpoint_session.saver.get_tuple(config)
                durable_end = (
                    durable is not None
                    and durable.checkpoint.get("channel_values", {}).get("phase")
                    == "end"
                )
                if not durable_end or saved.next:
                    graph.invoke(
                        None if saved.values else initial, config, durability="sync"
                    )
                final = graph.get_state(config)
                execution.assert_owned()
                durable = execution.checkpoint_session.saver.get_tuple(config)
                if (
                    final.next
                    or final.values.get("phase") != "end"
                    or durable is None
                    or durable.checkpoint.get("channel_values", {}).get("phase")
                    != "end"
                ):
                    raise UnderstandingError("checkpoint_incomplete", True, id)
                self.fault("after_graph_end")
                with self.session_factory.begin() as session:
                    finished = UnderstandingRepository(session, self.owner_id).finish(
                        id,
                        execution.epoch,
                        execution.execution_owner_id,
                        final.values["result_status"],
                    )
                result = self.view(finished)
            except Exception as caught:  # noqa: BLE001 — redact storage/driver/SDK boundaries
                error = (
                    caught
                    if isinstance(caught, UnderstandingError)
                    else UnderstandingError(
                        "checkpoint_failed"
                        if isinstance(caught, CheckpointError)
                        else "understanding_store_failed"
                        if isinstance(caught, SQLAlchemyError)
                        else "understanding_interrupted",
                        True,
                        id,
                    )
                )
                if error.understanding_id is None:
                    error = UnderstandingError(
                        error.code, error.retryable, id, error.related_id
                    )
                try:
                    with self.session_factory.begin() as session:
                        UnderstandingRepository(session, self.owner_id).record_error(
                            id, execution.epoch, execution.execution_owner_id, error
                        )
                except (UnderstandingError, SQLAlchemyError):
                    pass  # Lost/replaced owner must never overwrite a successor.
                raise error from None
        return result
