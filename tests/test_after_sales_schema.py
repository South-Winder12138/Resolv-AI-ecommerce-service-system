import pytest
from pydantic import ValidationError

from resolv.schemas.after_sales import AfterSalesTicket, RequestType


def test_request_type_values() -> None:
    assert [item.value for item in RequestType] == [
        "refund",
        "exchange",
        "repair",
        "other",
    ]


def test_ticket_constructs_with_order_id() -> None:
    ticket = AfterSalesTicket(
        order_id="MH001", request_type=RequestType.REFUND, reason="到货损坏"
    )

    assert ticket.order_id == "MH001"
    assert ticket.request_type is RequestType.REFUND
    assert ticket.reason == "到货损坏"


def test_ticket_allows_missing_order_id() -> None:
    ticket = AfterSalesTicket(
        order_id=None, request_type=RequestType.EXCHANGE, reason="耳机无声"
    )

    assert ticket.order_id is None


def test_ticket_rejects_invalid_request_type() -> None:
    with pytest.raises(ValidationError, match="request_type"):
        AfterSalesTicket(order_id=None, request_type="cancel", reason="不想要了")
