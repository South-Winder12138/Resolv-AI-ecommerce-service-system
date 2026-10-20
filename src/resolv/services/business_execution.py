"""Read tools execute only against the explicitly supplied business store."""

from resolv.repositories.order import OrderRepository
from resolv.repositories.shipment import ShipmentRepository
from resolv.schemas.conversation import ConversationError
from resolv.services.business_policy import validate_business_call
from resolv.services.order import OrderService
from resolv.services.shipment import ShipmentService
from resolv.tools.executor import ToolExecutionResult


def execute_business_read(call, policy, *, session_factory, observations, logistics_service=None, action_id=None, execution=None):
    validate_business_call(call, policy, observations=observations)
    if (
        call["name"] not in {"query_order", "query_logistics"}
        or session_factory is None
    ):
        raise ConversationError("business_scope_violation")
    if call["name"] == "query_logistics" and logistics_service is not None:
        if action_id is None or execution is None:
            raise ConversationError("run_state_incompatible")
        explicit = execution.explicit_resume and action_id not in execution.logistics_resume_actions
        execution.logistics_resume_actions.add(action_id)
        return logistics_service.read(action_id, execution, explicit_resume=explicit)
    try:
        with session_factory() as session:
            if call["name"] == "query_order":
                output = OrderService(OrderRepository(session)).get_order(
                    call["args"]["order_id"]
                )
            else:
                output = ShipmentService(ShipmentRepository(session)).get_shipment(
                    call["args"]["tracking_no"]
                )
        return ToolExecutionResult(call["id"], call["name"], output)
    except ConversationError:
        raise
    except Exception:  # noqa: BLE001 — never expose database credentials/errors
        raise ConversationError("tool_execution_failed", True) from None
