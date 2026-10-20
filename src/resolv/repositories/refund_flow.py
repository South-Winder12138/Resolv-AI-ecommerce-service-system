"""Write-once refund stages, tied to immutable primary identity and run epoch."""

from pydantic import ValidationError
from sqlalchemy import exists, select, update

from resolv.models.conversation import Conversation, ConversationRun
from resolv.models.refund_flow import ConversationRefundOffer, ConversationRefundRun
from resolv.models.understanding import (
    ConversationUnderstanding,
    ConversationUnderstandingRequest,
)
from resolv.repositories.conversation import ConversationRepository
from resolv.repositories.knowledge_branch import same_task_identity
from resolv.schemas.branch_execution import PrimaryExecution
from resolv.schemas.knowledge_branch import FrozenJSON, canonical_json
from resolv.schemas.refund import RefundFlowError, RefundObservation
from resolv.schemas.refund_flow import (
    PreparedRefundQuery,
    RefundBindingStage,
    RefundDecision,
    RefundModelAudit,
    RefundModelResponse,
    RefundProfile,
    RefundReadAudit,
    RefundRetrievalStage,
    RefundRunRecord,
)
from resolv.schemas.refund_submission import RefundOfferRecord
from resolv.schemas.understanding import UnderstandingRequestRecord
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.understanding_rules import INTENT_BRANCHES_V1

STAGES = {
    "binding": RefundBindingStage,
    "query": PreparedRefundQuery,
    "retrieval": RefundRetrievalStage,
    "decision": RefundDecision,
}


def checked(target, value):
    return target.model_validate_json(canonical_json(value))


class RefundFlowRepository:
    def __init__(self, session, owner_id):
        self.session, self.owner_id = session, owner_id

    def _row(self, run_id):
        return self.session.scalar(
            select(ConversationRefundRun)
            .join(Conversation)
            .where(
                ConversationRefundRun.run_id == run_id,
                Conversation.owner_id == self.owner_id,
            )
            .execution_options(populate_existing=True)
        )

    def _primary(self, run, primary):
        p = checked(PrimaryExecution, primary)
        task = self.session.get(
            ConversationUnderstandingRequest, p.primary.task_id, populate_existing=True
        )
        u = (
            self.session.get(
                ConversationUnderstanding, run.understanding_id, populate_existing=True
            )
            if run.understanding_id
            else None
        )
        if (
            run.orchestration_version != "routed-v4"
            or run.state_schema_version != 4
            or run.mode != "agent"
            or p.conversation_id != run.conversation_id
            or p.understanding_id != run.understanding_id
            or p.primary.branch != "refund_flow"
            or INTENT_BRANCHES_V1.get(p.primary.intent) != "refund_flow"
            or task is None
            or u is None
            or u.conversation_id != run.conversation_id
            or u.status != "ready"
            or u.snapshot.get("snapshot_hash") != p.snapshot_hash
        ):
            raise RefundFlowError("refund_identity_conflict")
        value = UnderstandingRequestRecord.model_validate(
            {k: getattr(task, k) for k in UnderstandingRequestRecord.model_fields}
        )
        if not same_task_identity(value, p.primary):
            raise RefundFlowError("refund_identity_conflict")
        return p

    def _record(self, row):
        try:
            run = ConversationRepository(self.session, self.owner_id).run_row(
                row.run_id
            )
            p = self._primary(run, row.primary_json)
            if (
                row.owner_id != self.owner_id
                or row.conversation_id != run.conversation_id
                or row.understanding_id != run.understanding_id
                or row.primary_task_id != p.primary.task_id
                or row.primary_hash != canonical_hash(row.primary_json)
                or row.profile_hash != canonical_hash(row.profile_json)
                or row.profile_json != run.refund_config
            ):
                raise RefundFlowError("refund_identity_conflict")
            values, hashes, previous = {}, [], row.primary_hash
            if not set(row.stage_data) <= STAGES.keys():
                raise RefundFlowError("refund_stage_hash_conflict")
            gap = False
            for name, target in STAGES.items():
                saved = row.stage_data.get(name)
                if saved is None:
                    gap = True
                    continue
                if (
                    gap
                    or set(saved) != {"payload", "input_hash", "output_hash"}
                    or saved["input_hash"] != previous
                    or canonical_hash(saved["payload"]) != saved["output_hash"]
                ):
                    raise RefundFlowError("refund_stage_hash_conflict")
                values[name] = checked(target, saved["payload"])
                hashes.append((name, previous, saved["output_hash"]))
                previous = saved["output_hash"]
            b = values.pop("binding", None)
            if (
                b
                and b.binding
                and (
                    b.binding.primary_hash != row.primary_hash
                    or b.binding.understanding_id != row.understanding_id
                    or b.binding.primary_task_id != row.primary_task_id
                    or b.binding.order_source
                    != row.profile_json["adapter"]["namespace"]
                )
            ):
                raise RefundFlowError("refund_binding_conflict")
            requests, observations = [], []
            for entry in row.model_requests:
                if canonical_hash(entry["payload"]) != entry["hash"]:
                    raise RefundFlowError("refund_model_hash_conflict")
                response = checked(RefundModelResponse, entry["payload"])
                if response.model_step != len(requests) + 1:
                    raise RefundFlowError("refund_model_sequence_conflict")
                requests.append(response)
            seen = set()
            for entry in row.observations:
                obs = checked(RefundObservation, entry["payload"])
                step = entry["model_step"]
                self._validate_observation(
                    b.binding if b else None, requests, step, obs
                )
                if (
                    obs.observation_id in seen
                    or canonical_hash(entry["payload"]) != entry["hash"]
                ):
                    raise RefundFlowError("refund_observation_conflict")
                seen.add(obs.observation_id)
                observations.append(obs)
            for audit in row.audits:
                if "read_audit" in audit and (
                    set(audit) != {"read_audit", "hash"}
                    or canonical_hash(audit["read_audit"]) != audit["hash"]
                ):
                    raise RefundFlowError("refund_read_audit_conflict")
            return RefundRunRecord(
                run_id=row.run_id,
                owner_id=row.owner_id,
                conversation_id=row.conversation_id,
                understanding_id=row.understanding_id,
                primary_task_id=row.primary_task_id,
                primary_hash=row.primary_hash,
                primary_json=FrozenJSON.from_value(row.primary_json),
                profile=checked(RefundProfile, row.profile_json),
                binding=b.binding if b else None,
                request=b.request if b else None,
                scope_reply=b.scope_reply if b else None,
                **values,
                model_requests=tuple(requests),
                observations=tuple(observations),
                audits=tuple(
                    checked(RefundModelAudit, a)
                    for a in row.audits
                    if "read_audit" not in a
                ),
                read_audits=tuple(
                    checked(RefundReadAudit, a["read_audit"])
                    for a in row.audits
                    if "read_audit" in a
                ),
                stage_hashes=tuple(hashes),
                error_code=row.error_code,
                retryable=row.retryable,
                step_limit=row.step_limit,
            )
        except (
            ValidationError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            raise RefundFlowError("refund_snapshot_invalid") from exc

    def get(self, run_id):
        row = self._row(run_id)
        return self._record(row) if row else None

    def _owned(self, run_id, epoch):
        run = ConversationRepository(self.session, self.owner_id).owned(run_id, epoch)
        row = self._row(run_id)
        if row is None:
            raise RefundFlowError("refund_run_missing")
        self._record(row)
        return run, row

    def _update(self, row, epoch, **values):
        fence = exists(
            select(ConversationRun.run_id)
            .join(Conversation)
            .where(
                ConversationRun.run_id == row.run_id,
                ConversationRun.execution_epoch == epoch,
                ConversationRun.execution_owner_id.is_not(None),
                Conversation.current_run_id == row.run_id,
                Conversation.owner_id == self.owner_id,
            )
        )
        result = self.session.execute(
            update(ConversationRefundRun)
            .where(
                ConversationRefundRun.run_id == row.run_id,
                ConversationRefundRun.owner_id == self.owner_id,
                ConversationRefundRun.revision == row.revision,
                fence,
            )
            .values(**values, revision=row.revision + 1)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise RefundFlowError("refund_execution_conflict", True)
        self.session.expire(row)
        return self._record(row)

    def create(self, run_id, epoch, primary, profile):
        run = ConversationRepository(self.session, self.owner_id).owned(run_id, epoch)
        p = self._primary(run, primary.model_dump(mode="json"))
        profile = checked(RefundProfile, profile.model_dump(mode="json"))
        if run.refund_config != profile.model_dump(mode="json"):
            raise RefundFlowError("refund_profile_conflict")
        row = self._row(run_id)
        if row is None:
            row = ConversationRefundRun(
                run_id=run_id,
                owner_id=self.owner_id,
                conversation_id=run.conversation_id,
                understanding_id=run.understanding_id,
                primary_task_id=p.primary.task_id,
                primary_json=p.model_dump(mode="json"),
                primary_hash=canonical_hash(p.model_dump(mode="json")),
                profile_json=profile.model_dump(mode="json"),
                profile_hash=profile.fingerprint,
            )
            self.session.add(row)
            self.session.flush()
        if (
            row.primary_json != p.model_dump(mode="json")
            or row.profile_hash != profile.fingerprint
        ):
            raise RefundFlowError("refund_identity_conflict")
        return self._record(row)

    def save_stage(self, run_id, epoch, stage, payload, *, expected_input_hash):
        _, row = self._owned(run_id, epoch)
        if stage not in STAGES:
            raise RefundFlowError("refund_stage_invalid")
        position = list(STAGES).index(stage)
        names = list(STAGES)
        if any(n not in row.stage_data for n in names[:position]):
            raise RefundFlowError("refund_stage_predecessor_missing")
        previous = (
            row.primary_hash
            if position == 0
            else row.stage_data[names[position - 1]]["output_hash"]
        )
        if expected_input_hash != previous:
            raise RefundFlowError("refund_stage_hash_conflict")
        value = checked(STAGES[stage], payload.model_dump(mode="json")).model_dump(
            mode="json"
        )
        entry = {
            "payload": value,
            "input_hash": previous,
            "output_hash": canonical_hash(value),
        }
        if stage in row.stage_data:
            if row.stage_data[stage] != entry:
                raise RefundFlowError("refund_stage_conflict")
            return self._record(row)
        return self._update(row, epoch, stage_data={**row.stage_data, stage: entry})

    def save_model_request(self, run_id, epoch, response):
        _, row = self._owned(run_id, epoch)
        response = checked(RefundModelResponse, response.model_dump(mode="json"))
        value = response.model_dump(mode="json")
        entry = {"payload": value, "hash": canonical_hash(value)}
        index = response.model_step - 1
        if index < len(row.model_requests):
            if row.model_requests[index] != entry:
                raise RefundFlowError("refund_model_request_conflict")
            return self._record(row)
        if (
            index != len(row.model_requests)
            or response.audit.profile_fingerprint
            != self._record(row).profile.decision_model.fingerprint
        ):
            raise RefundFlowError("refund_model_sequence_conflict")
        previous = {
            c.id: (c.name, c.args)
            for r in self._record(row).model_requests
            for c in r.tool_calls
        }
        if any(
            c.id in previous and previous[c.id] != (c.name, c.args)
            for c in response.tool_calls
        ):
            raise RefundFlowError("refund_model_call_conflict", True)
        return self._update(row, epoch, model_requests=[*row.model_requests, entry])

    @staticmethod
    def _validate_observation(binding, requests, step, observation):
        if type(step) is not int or step < 1 or step > len(requests) or binding is None:
            raise RefundFlowError("refund_observation_conflict")
        call = next(
            (
                c
                for c in requests[step - 1].tool_calls
                if c.id == observation.tool_call_id
            ),
            None,
        )
        if (
            call is None
            or call.name != observation.tool_name
            or canonical_hash(call.args.value) != observation.arguments_hash
            or observation.binding_hash != binding.binding_hash
            or observation.source != binding.order_source
        ):
            raise RefundFlowError("refund_observation_conflict")

    def save_observation(self, run_id, epoch, model_step, observation):
        _, row = self._owned(run_id, epoch)
        obs = checked(RefundObservation, observation.model_dump(mode="json"))
        record = self._record(row)
        self._validate_observation(
            record.binding, record.model_requests, model_step, obs
        )
        value = obs.model_dump(mode="json")
        entry = {
            "model_step": model_step,
            "payload": value,
            "hash": canonical_hash(value),
        }
        old = next(
            (
                e
                for e in row.observations
                if e["model_step"] == model_step
                and e["payload"]["tool_call_id"] == obs.tool_call_id
            ),
            None,
        )
        if old:
            if old != entry:
                raise RefundFlowError("refund_observation_conflict")
            return record
        return self._update(row, epoch, observations=[*row.observations, entry])

    def get_offer(self, offer_id):
        row = self.session.scalar(
            select(ConversationRefundOffer)
            .join(Conversation)
            .where(
                ConversationRefundOffer.offer_id == offer_id,
                Conversation.owner_id == self.owner_id,
            )
            .execution_options(populate_existing=True)
        )
        if row is None:
            return None
        offer = checked(RefundOfferRecord, row.payload)
        if (
            canonical_hash(row.payload) != row.payload_hash
            or offer.owner_id != self.owner_id
            or row.owner_id != self.owner_id
            or offer.conversation_id != row.conversation_id
            or offer.run_id != row.run_id
            or offer.offer_id != row.offer_id
        ):
            raise RefundFlowError("refund_offer_conflict")
        return offer.model_copy(
            update={"status": row.status, "invalid_reason": row.invalid_reason}
        )

    def put_offer(self, run_id, epoch, offer):
        _, row = self._owned(run_id, epoch)
        offer = checked(RefundOfferRecord, offer.model_dump(mode="json"))
        if (
            offer.run_id != run_id
            or offer.owner_id != self.owner_id
            or offer.conversation_id != row.conversation_id
            or offer.primary_task_id != row.primary_task_id
            or offer.binding.primary_hash != row.primary_hash
            or offer.profile.fingerprint != row.profile_hash
        ):
            raise RefundFlowError("refund_offer_conflict")
        old = self.session.scalar(
            select(ConversationRefundOffer).where(
                ConversationRefundOffer.run_id == run_id
            )
        )
        if old is not None:
            if old.payload != offer.model_dump(mode="json"):
                raise RefundFlowError("refund_offer_conflict")
            return self.get_offer(old.offer_id)
        self.session.add(
            ConversationRefundOffer(
                offer_id=offer.offer_id,
                run_id=run_id,
                owner_id=self.owner_id,
                conversation_id=row.conversation_id,
                payload=offer.model_dump(mode="json"),
                payload_hash=offer.fingerprint,
                status=offer.status,
                invalid_reason=offer.invalid_reason,
                created_at=offer.created_at,
                expires_at=offer.expires_at,
            )
        )
        self.session.flush()
        return self.get_offer(offer.offer_id)

    def record_audit(self, run_id, epoch, audit):
        _, row = self._owned(run_id, epoch)
        audit = checked(RefundModelAudit, audit.model_dump(mode="json"))
        profile = self._record(row).profile
        expected = {
            "refund_query": profile.query_model.fingerprint,
            "refund_decision": profile.decision_model.fingerprint,
            "refund_summary": profile.summary_profile.fingerprint,
        }[audit.purpose]
        if audit.profile_fingerprint != expected:
            raise RefundFlowError("refund_audit_profile_conflict")
        value = audit.model_dump(mode="json")
        old = next(
            (
                a
                for a in row.audits
                if "read_audit" not in a
                and (a["invocation_id"], a["outcome"])
                == (audit.invocation_id, audit.outcome)
            ),
            None,
        )
        if old is not None:
            if old != value:
                raise RefundFlowError("refund_audit_conflict")
            return self._record(row)
        return self._update(row, epoch, audits=[*row.audits, value])

    def record_read_audit(self, run_id, epoch, audit):
        _, row = self._owned(run_id, epoch)
        record = self._record(row)
        audit = checked(RefundReadAudit, audit.model_dump(mode="json"))
        if (
            record.binding is None
            or audit.binding_hash != record.binding.binding_hash
            or audit.profile_fingerprint != record.profile.adapter.fingerprint
        ):
            raise RefundFlowError("refund_read_audit_conflict")
        call = next(
            (
                c
                for response in record.model_requests
                if response.model_step == audit.model_step
                for c in response.tool_calls
                if c.id == audit.tool_call_id
            ),
            None,
        )
        if (
            call is None
            or call.name != audit.tool_name
            or audit.input_hash
            != canonical_hash(
                {"binding_hash": audit.binding_hash, "args": call.args.value}
            )
        ):
            raise RefundFlowError("refund_read_audit_conflict")
        value = audit.model_dump(mode="json")
        entry = {"read_audit": value, "hash": canonical_hash(value)}
        old = next(
            (
                a
                for a in row.audits
                if "read_audit" in a
                and (a["read_audit"]["invocation_id"], a["read_audit"]["outcome"])
                == (audit.invocation_id, audit.outcome)
            ),
            None,
        )
        if old is not None:
            if old != entry:
                raise RefundFlowError("refund_read_audit_conflict")
            return record
        return self._update(row, epoch, audits=[*row.audits, entry])

    def record_error(self, run_id, epoch, code, retryable):
        _, row = self._owned(run_id, epoch)
        return self._update(row, epoch, error_code=code, retryable=retryable)

    def mark_step_limit(self, run_id, epoch):
        _, row = self._owned(run_id, epoch)
        record = self._record(row)
        if (
            len(record.model_requests) != record.profile.max_agent_steps
            or not record.model_requests[-1].tool_calls
        ):
            raise RefundFlowError("refund_model_sequence_conflict")
        return self._update(
            row, epoch, step_limit=True, error_code=None, retryable=False
        )

    def offer_for_run(self, run_id):
        row = self.session.scalar(
            select(ConversationRefundOffer).where(
                ConversationRefundOffer.run_id == run_id,
                ConversationRefundOffer.owner_id == self.owner_id,
            )
        )
        return self.get_offer(row.offer_id) if row else None
