from threading import Lock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationStore:
    """Keep completed conversation turns in this process only."""

    def __init__(self) -> None:
        self._histories: dict[str, list[BaseMessage]] = {}
        self._lock = Lock()

    def get_history(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            return list(self._histories.get(session_id, []))

    def commit_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            self._histories.setdefault(session_id, []).extend(
                [HumanMessage(content=question), AIMessage(content=answer)]
            )

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._histories.pop(session_id, None)
