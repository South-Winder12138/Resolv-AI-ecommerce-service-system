from resolv.repositories.shipment import ShipmentRepository


class ShipmentService:
    def __init__(self, repository: ShipmentRepository) -> None:
        self.repository = repository

    def get_shipment(self, tracking_no: str) -> dict[str, object]:
        shipment = self.repository.get_by_tracking_no(tracking_no)
        if shipment is None:
            return {"found": False, "tracking_no": tracking_no}
        return {
            "found": True,
            "tracking_no": shipment.tracking_no,
            "status": shipment.status,
            "location": shipment.location,
        }
