"""Exact retry results for the existing SQLite business transactions."""

from sqlalchemy import text

from resolv.repositories.conversation import ConversationRepository, fingerprint
from resolv.repositories.tool_execution import ToolExecutionRepository
from resolv.schemas.conversation import ConversationError
from resolv.tools.executor import ToolExecutionResult
from resolv.tools.registry import ToolEffect, get_registered_tool


class WriteExecutionService:
    def __init__(self, conversation_session_factory, business_session_factory):
        self.conversation_session_factory = conversation_session_factory
        self.business_session_factory = business_session_factory

    def execute(self, action_id, execution):
        execution.assert_owned()
        # No MySQL operation occurs inside the SQLite transaction.
        with self.conversation_session_factory() as session:
            repo = ConversationRepository(session, execution.service.owner_id)
            action = repo.action(action_id)
            run = repo.run(action.run_id)
            if (
                action.run_id != execution.run_id
                or action.effect != "write"
                or action.approval_status != "approved"
            ):
                raise ConversationError("approval_required", run_id=execution.run_id)
            registered = get_registered_tool(action.tool_name)
            if (
                registered.effect != ToolEffect.WRITE
                or registered.transaction_handler is None
                or fingerprint(action.arguments) != action.arguments_fingerprint
            ):
                raise ConversationError("action_conflict", run_id=execution.run_id)
            validated = registered.tool.args_schema.model_validate(
                action.arguments
            ).model_dump()
            if fingerprint(validated) != action.arguments_fingerprint:
                raise ConversationError("action_conflict", run_id=execution.run_id)
            if run.orchestration_version == "routed-v2":
                from sqlalchemy import select

                from resolv.models.conversation import ConversationAction
                from resolv.schemas.branch_execution import parse_business_config
                from resolv.services.business_policy import (
                    build_business_policy,
                    validate_business_call,
                )
                from resolv.services.conversation_routing import load_primary

                service = execution.service
                if (
                    action.tool_name != "cancel_order"
                    or action.model_step
                    >= parse_business_config(
                        run.business_config
                    ).max_agent_steps
                ):
                    raise ConversationError(
                        "business_scope_violation", run_id=run.run_id
                    )
                record = service.understanding.record(
                    run.conversation_id, run.understanding_id
                )
                service.understanding.compatible(record)
                primary = load_primary(
                    run,
                    service.understanding.view(record),
                    snapshot_hash=record.snapshot.snapshot_hash,
                )
                observations = [
                    ToolExecutionResult(a.tool_call_id, a.tool_name, a.result)
                    for a in session.scalars(
                        select(ConversationAction)
                        .where(
                            ConversationAction.run_id == run.run_id,
                            ConversationAction.execution_status == "completed",
                        )
                        .order_by(
                            ConversationAction.model_step,
                            ConversationAction.tool_call_id,
                        )
                    )
                ]
                validate_business_call(
                    {
                        "id": action.tool_call_id,
                        "name": action.tool_name,
                        "args": action.arguments,
                    },
                    build_business_policy(primary),
                    observations=observations,
                )
        try:
            with self.business_session_factory() as session:
                if session.bind.dialect.name != "sqlite":
                    raise ConversationError(
                        "tool_execution_failed", True, execution.run_id
                    )
                session.execute(text("BEGIN IMMEDIATE"))
                try:
                    receipts = ToolExecutionRepository(session)
                    receipt = receipts.get(action_id)
                    if receipt:
                        if (
                            receipt.tool_name != action.tool_name
                            or receipt.arguments_fingerprint
                            != action.arguments_fingerprint
                            or receipt.run_id != action.run_id
                            or receipt.conversation_id != run.conversation_id
                        ):
                            raise ConversationError(
                                "action_conflict", run_id=action.run_id
                            )
                        output = receipt.result
                    else:
                        output = registered.transaction_handler(session, **validated)
                        receipts.add(action, output, run.conversation_id)
                    session.commit()
                except BaseException:
                    session.rollback()
                    raise
        except ConversationError:
            raise
        except Exception:  # noqa: BLE001 — business/serialization/commit errors all roll back
            raise ConversationError(
                "tool_execution_failed", True, execution.run_id
            ) from None
        execution.assert_owned()
        if run.orchestration_version == "routed-v2":
            execution.service.fault("after_sqlite_commit")
        return ToolExecutionResult(action.tool_call_id, action.tool_name, output)
