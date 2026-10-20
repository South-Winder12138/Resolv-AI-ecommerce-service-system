"""Refund application rules and plain business results."""

from resolv.repositories.order import OrderRepository
from resolv.repositories.refund import RefundRepository

REFUNDABLE_STATUSES = frozenset({"shipped", "delivered"})


class RefundService:
    def __init__(self, orders: OrderRepository, refunds: RefundRepository) -> None:
        self.orders = orders
        self.refunds = refunds

    def query_refund_status(self, order_no: str) -> dict[str, object]:
        from resolv.services.refund_registration import (
            LOCAL_SOURCE,
            RefundRegistrationService,
        )

        order = self.orders.get_by_order_no(order_no)
        if order is None:
            return {"found": False, "order_id": order_no, "reason": "order_not_found"}
        refund = self.refunds.get_by_order_id(order.id)
        RefundRegistrationService(self.refunds.session)._consistent(LOCAL_SOURCE, order_no)
        if refund is None:
            return {"found": False, "order_id": order_no, "reason": "refund_not_found"}
        return {
            "found": True,
            "order_id": order_no,
            "status": refund.status,
            "reason": refund.reason,
        }

    def apply_refund(self, order_no: str, reason: str) -> dict[str, object]:
        from resolv.services.refund_registration import (
            RefundRegistrationService,
            begin_refund_write,
        )

        begin_refund_write(self.refunds.session)
        order = self.orders.get_by_order_no(order_no)
        if order is None:
            return {"success": False, "order_id": order_no, "reason": "order_not_found"}

        return RefundRegistrationService(self.refunds.session).register_legacy(order, reason)
