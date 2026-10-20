"""Durable remote reads: commit preparation, release SQL, then dispatch once."""

from sqlalchemy import select

from resolv.models.logistics import ConversationLogisticsAttempt as Attempt
from resolv.repositories.conversation import ConversationRepository, fingerprint
from resolv.repositories.logistics import LogisticsAttemptRepository
from resolv.schemas.conversation import ConversationError
from resolv.schemas.logistics import (
    LogisticsBackendProfile,
    LogisticsFailure,
    ShipmentEnvelope,
    canonical_json_bytes,
)
from resolv.schemas.logistics_execution import project_logistics_observation
from resolv.services.logistics_api_client import validate_envelope_identity
from resolv.services.logistics_client import LogisticsFactsClient
from resolv.tools.executor import ToolExecutionResult

RETRYABLE = {
    "logistics_timeout", "logistics_transport_failed", "logistics_service_failed",
    "logistics_result_unknown", "logistics_store_failed",
}


class LogisticsReadService:
    def __init__(self, session_factory, client: LogisticsFactsClient, profile: LogisticsBackendProfile, *, fault_hook=None):
        self.session_factory, self.client, self.profile = session_factory, client, profile
        self.fault_hook = fault_hook

    def _hook(self, stage, attempt):
        if self.fault_hook:
            self.fault_hook(stage, attempt)

    def _load(self, session, action_id, execution):
        repo = LogisticsAttemptRepository(session, execution.service.owner_id)
        action, run = repo._owned_action(action_id, execution.epoch)
        if run.run_id != execution.run_id:
            raise ConversationError("business_scope_violation", run_id=execution.run_id)
        config = run.business_config or {}
        try:
            frozen = LogisticsBackendProfile.model_validate_json(canonical_json_bytes(config["logistics_backend"]))
        except (KeyError, ValueError, TypeError):
            raise LogisticsFailure("logistics_configuration_conflict") from None
        if (
            type(config.get("config_version")) is not int or config["config_version"] != 2
            or frozen != self.profile or getattr(self.client, "profile", self.profile) != frozen
            or set(action.arguments) != {"tracking_no"}
            or action.arguments_fingerprint != fingerprint(action.arguments)
        ):
            raise LogisticsFailure("logistics_configuration_conflict")
        return repo, action, fingerprint(config), action.arguments["tracking_no"]

    def _cached(self, repo, action, config_hash, tracking):
        saved = repo.find_completed(action.run_id, config_hash, tracking)
        if saved is None:
            if action.execution_status == "completed":
                raise LogisticsFailure("logistics_store_failed", True)
            return None
        try:
            value = ShipmentEnvelope.model_validate_json(canonical_json_bytes(saved.envelope))
            validate_envelope_identity(value, self.profile, tracking, saved.attempt_id)
            projection = project_logistics_observation(value, attempt_id=saved.attempt_id)
            if saved.result_hash != fingerprint(saved.envelope) or saved.projection != projection:
                raise ValueError("saved result mismatch")
            if action.execution_status == "completed" and action.result != projection:
                raise ValueError("saved action mismatch")
        except (ValueError, TypeError, LogisticsFailure):
            raise LogisticsFailure("logistics_store_failed", True) from None
        return projection

    def _reconcile(self, action_id, execution):
        """Only a committed, validated action proves completion; never redispatch."""
        with self.session_factory.begin() as session:
            repo, action, config_hash, tracking = self._load(session, action_id, execution)
            if action.execution_status != "completed":
                return None
            projection = self._cached(repo, action, config_hash, tracking)
            return ToolExecutionResult(action.tool_call_id, action.tool_name, projection)

    def read(self, action_id, execution, *, explicit_resume=False):
        try:
            return self._read(action_id, execution, explicit_resume=explicit_resume)
        except ConversationError:
            raise
        except LogisticsFailure as error:
            raise ConversationError(error.code, error.retryable, execution.run_id) from None
        except Exception:  # noqa: BLE001 -- no private SQL/provider details at HTTP boundaries
            try:
                confirmed = self._reconcile(action_id, execution)
                if confirmed:
                    return confirmed
            except ConversationError:
                raise
            except Exception:  # noqa: BLE001 -- unconfirmed storage must stop
                raise ConversationError("logistics_store_failed", True, execution.run_id) from None
            raise ConversationError("logistics_store_failed", True, execution.run_id) from None

    def _read(self, action_id, execution, *, explicit_resume):
        execution.assert_owned()
        blocked = None
        with self.session_factory.begin() as session:
            repo, action, config_hash, tracking = self._load(session, action_id, execution)
            cached = self._cached(repo, action, config_hash, tracking)
            tool_call_id = action.tool_call_id
            if cached is not None:
                ConversationRepository(session, execution.service.owner_id).save_action_result(action_id, execution.epoch, cached)
                return ToolExecutionResult(tool_call_id, "query_logistics", cached)
            previous = session.scalars(select(Attempt).where(
                Attempt.action_id == action_id,
            ).order_by(Attempt.attempt_no).with_for_update()).all()
            for old in previous:
                if old.state == "prepared":
                    repo.mark_unknown(old.attempt_id, execution.epoch)
            if previous:
                error = previous[-1].error_code or "logistics_result_unknown"
                if not explicit_resume or error not in RETRYABLE:
                    blocked = LogisticsFailure(error, error in RETRYABLE)
            if blocked is None:
                attempt = repo.prepare(action_id, execution.epoch, config_hash=config_hash, tracking_no=tracking)
        # Raising after the transaction preserves the unknown audit transition.
        if blocked:
            raise blocked
        self._hook("after_attempt_prepared", attempt)
        execution.assert_owned()
        try:
            envelope = self.client.query(tracking, request_id=attempt.attempt_id)
        except LogisticsFailure as error:
            with self.session_factory.begin() as session:
                LogisticsAttemptRepository(session, execution.service.owner_id).fail(attempt.attempt_id, execution.epoch, error)
            raise
        self._hook("after_remote_result", attempt)
        execution.assert_owned()
        try:
            if not isinstance(envelope, ShipmentEnvelope):
                raise TypeError("typed envelope required")
            envelope = ShipmentEnvelope.model_validate_json(canonical_json_bytes(envelope.model_dump(mode="json")))
            validate_envelope_identity(envelope, self.profile, tracking, attempt.attempt_id)
        except (ValueError, TypeError, LogisticsFailure):
            error = LogisticsFailure("logistics_result_invalid")
            with self.session_factory.begin() as session:
                LogisticsAttemptRepository(session, execution.service.owner_id).fail(attempt.attempt_id, execution.epoch, error)
            raise error from None
        projection = project_logistics_observation(envelope, attempt_id=attempt.attempt_id)
        try:
            with self.session_factory.begin() as session:
                _, _, current_hash, current_tracking = self._load(session, action_id, execution)
                if current_hash != attempt.config_hash or current_tracking != tracking:
                    raise LogisticsFailure("logistics_configuration_conflict")
                LogisticsAttemptRepository(session, execution.service.owner_id).complete(attempt.attempt_id, execution.epoch, envelope, projection)
        except (ConversationError, LogisticsFailure):
            raise
        except Exception:  # noqa: BLE001 -- commit may have succeeded; reconcile before returning
            confirmed = self._reconcile(action_id, execution)
            if confirmed is None:
                raise LogisticsFailure("logistics_store_failed", True) from None
        self._hook("after_result_commit", attempt)
        return ToolExecutionResult(tool_call_id, "query_logistics", projection)
