from enum import StrEnum

from pydantic import BaseModel, Field


class RequestType(StrEnum):
    REFUND = "refund"
    EXCHANGE = "exchange"
    REPAIR = "repair"
    OTHER = "other"


class AfterSalesTicket(BaseModel):
    order_id: str | None = Field(
        description="用户明确提供的订单号；未提供时为 null，不得猜测。"
    )
    request_type: RequestType = Field(description="用户提出的售后请求类型。")
    reason: str = Field(description="用户描述的售后原因，简洁概括。")
