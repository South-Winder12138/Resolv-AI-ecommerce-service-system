from langchain_core.tools import tool

from resolv.core.config import DatabaseSettings
from resolv.db.engine import get_session_factory
from resolv.repositories.order import OrderRepository
from resolv.services.order import OrderService


@tool(parse_docstring=True)
def query_order(order_id: str) -> dict[str, object]:
    """根据订单号查询真实状态；用户询问具体订单时使用。订单不存在则返回 found=false。

    Args:
        order_id: 用户提供的订单号。
    """
    session_factory = get_session_factory(DatabaseSettings().database_url)
    with session_factory() as session:
        return OrderService(OrderRepository(session)).get_order(order_id)
