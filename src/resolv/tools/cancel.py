from langchain_core.tools import tool

from resolv.core.config import DatabaseSettings
from resolv.db.engine import get_session_factory
from resolv.repositories.order import OrderRepository
from resolv.services.order import OrderService


def cancel_order_in_transaction(session, *, order_id):
    return OrderService(OrderRepository(session)).cancel_order(order_id)


@tool(parse_docstring=True)
def cancel_order(order_id: str) -> dict[str, object]:
    """取消订单，会修改订单状态；不用于查询。业务层决定订单能否取消。

    Args:
        order_id: 用户提供的订单号。
    """
    session_factory = get_session_factory(DatabaseSettings().database_url)
    with session_factory.begin() as session:
        return cancel_order_in_transaction(session, order_id=order_id)
