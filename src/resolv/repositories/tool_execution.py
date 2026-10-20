from resolv.models.tool_execution import ToolExecutionReceipt
from resolv.repositories.conversation import canonical_json


class ToolExecutionRepository:
    def __init__(self, session):
        self.session = session

    def get(self, action_id):
        return self.session.get(ToolExecutionReceipt, action_id)

    def add(self, action, result, conversation_id):
        canonical_json(result)
        row = ToolExecutionReceipt(
            action_id=action.action_id,
            conversation_id=conversation_id,
            run_id=action.run_id,
            tool_name=action.tool_name,
            arguments_fingerprint=action.arguments_fingerprint,
            result=result,
        )
        self.session.add(row)
        self.session.flush()
        return row
