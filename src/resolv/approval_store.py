"""In-process HTTP approval IDs bound to one conversation session."""

from dataclasses import dataclass
from secrets import token_urlsafe
from threading import Lock

from resolv.agent_loop import AgentApprovalRequired
from resolv.approval import ApprovalRequest, SuspendedAgentState


class PendingApprovalExistsError(RuntimeError):
    """A session already has a pending or active Agent request."""


class ApprovalNotFoundError(LookupError):
    """No pending approval belongs to this ID and session."""


@dataclass(frozen=True)
class PendingHttpApproval:
    approval_id: str
    session_id: str
    original_question: str
    state: SuspendedAgentState
    request: ApprovalRequest


class ApprovalStore:
    """Keep one pending approval per session; consume IDs exactly once."""

    def __init__(self) -> None:
        self._records: dict[str, PendingHttpApproval] = {}
        self._by_session: dict[str, str] = {}
        self._active: set[str] = set()
        self._lock = Lock()

    def has_pending(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._by_session

    def claim_chat(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._by_session or session_id in self._active:
                raise PendingApprovalExistsError(session_id)
            self._active.add(session_id)

    def create(
        self, session_id: str, question: str, result: AgentApprovalRequired
    ) -> PendingHttpApproval:
        with self._lock:
            if session_id not in self._active or session_id in self._by_session:
                raise PendingApprovalExistsError(session_id)
            approval_id = f"apr_{token_urlsafe(24)}"
            record = PendingHttpApproval(
                approval_id, session_id, question, result.state, result.request
            )
            self._records[approval_id] = record
            self._by_session[session_id] = approval_id
            self._active.remove(session_id)
            return record

    def get(self, approval_id: str, session_id: str) -> PendingHttpApproval:
        with self._lock:
            record = self._records.get(approval_id)
            if record is None or record.session_id != session_id:
                raise ApprovalNotFoundError(approval_id)
            return record

    def consume(self, approval_id: str, session_id: str) -> PendingHttpApproval:
        with self._lock:
            record = self._records.get(approval_id)
            if record is None or record.session_id != session_id:
                raise ApprovalNotFoundError(approval_id)
            self._records.pop(approval_id)
            self._by_session.pop(session_id)
            self._active.add(session_id)
            return record

    def finish(self, session_id: str) -> None:
        with self._lock:
            self._active.discard(session_id)
