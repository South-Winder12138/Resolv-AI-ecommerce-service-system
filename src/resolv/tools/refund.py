"""Refund Tools expose business arguments while keeping DB objects internal."""

from langchain_core.tools import tool

from resolv.core.config import DatabaseSettings
from resolv.db.engine import get_session_factory
from resolv.repositories.order import OrderRepository
from resolv.repositories.refund import RefundRepository
from resolv.services.refund import RefundService


def apply_refund_in_transaction(session, *, order_id, reason):
    service = RefundService(OrderRepository(session), RefundRepository(session))
    return service.apply_refund(order_id, reason)


@tool(parse_docstring=True)
def query_refund_status(order_id: str) -> dict[str, object]:
    """查询订单已有退款申请的状态；订单或申请不存在时返回明确的 found=false 原因。

    Args:
        order_id: 业务订单号，例如 MH001；不是数据库整数主键。
    """
    session_factory = get_session_factory(DatabaseSettings().database_url)
    with session_factory() as session:
        service = RefundService(OrderRepository(session), RefundRepository(session))
        return service.query_refund_status(order_id)


@tool(parse_docstring=True)
def apply_refund(order_id: str, reason: str) -> dict[str, object]:
    """提交退款申请并写入 pending 记录；不会执行资金退款，也不用于查询。

    Args:
        order_id: 业务订单号，例如 MH001；不是数据库整数主键。
        reason: 用户提出退款申请的原因。
    """
    session_factory = get_session_factory(DatabaseSettings().database_url)
    with session_factory.begin() as session:
        return apply_refund_in_transaction(session, order_id=order_id, reason=reason)
