from resolv.repositories.order import OrderRepository


class OrderService:
    def __init__(self, repository: OrderRepository) -> None:
        self.repository = repository

    def get_order(self, order_no: str) -> dict[str, object]:
        order = self.repository.get_by_order_no(order_no)
        if order is None:
            return {"found": False, "order_id": order_no}
        return {
            "found": True,
            "order_id": order.order_no,
            "status": order.status,
            "tracking_no": order.shipment.tracking_no if order.shipment else None,
        }

    def cancel_order(self, order_no: str) -> dict[str, object]:
        order = self.repository.get_by_order_no(order_no)
        if order is None:
            return {"success": False, "order_id": order_no, "reason": "order_not_found"}
        if order.status == "cancelled":
            return {
                "success": True,
                "order_id": order_no,
                "status": "cancelled",
                "changed": False,
            }
        if order.status != "processing":
            return {
                "success": False,
                "order_id": order_no,
                "status": order.status,
                "reason": "order_not_cancellable",
            }
        self.repository.set_status(order, "cancelled")
        return {
            "success": True,
            "order_id": order_no,
            "status": "cancelled",
            "changed": True,
        }
