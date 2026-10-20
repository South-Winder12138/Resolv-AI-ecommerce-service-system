"""Named-lock execution capability; never a conversation run capability."""

from contextlib import contextmanager
from dataclasses import dataclass
from uuid import uuid4

from resolv.db.checkpoints import CheckpointError
from resolv.repositories.understanding import UnderstandingRepository
from resolv.schemas.understanding import UnderstandingError


@dataclass
class UnderstandingExecution:
    understanding_id: str
    execution_owner_id: str
    epoch: int
    checkpoint_session: object

    def assert_owned(self):
        try:
            self.checkpoint_session.assert_owned()
        except CheckpointError:
            raise UnderstandingError(
                "checkpoint_failed", True, self.understanding_id
            ) from None


@contextmanager
def understanding_execution(service, id):
    execution = None
    try:
        with service.checkpoint_factory(
            conversation_id="understanding:" + id
        ) as checkpoint:
            with service.session_factory.begin() as session:
                claimed = UnderstandingRepository(session, service.owner_id).claim(
                    id, uuid4().hex
                )
            execution = UnderstandingExecution(
                id, claimed.execution_owner_id, claimed.execution_epoch, checkpoint
            )
            try:
                yield execution
            finally:
                # A replaced executor's cleanup cannot touch the new epoch.
                try:
                    with service.session_factory.begin() as session:
                        UnderstandingRepository(session, service.owner_id).release(
                            id, execution.epoch, execution.execution_owner_id
                        )
                except UnderstandingError as error:
                    if error.code != "understanding_busy":
                        raise
    except CheckpointError as error:
        raise UnderstandingError(
            "understanding_busy" if error.code == "conversation_busy" else error.code,
            True,
            id,
        ) from None
