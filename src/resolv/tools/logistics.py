from langchain_core.tools import tool

from resolv.core.config import DatabaseSettings
from resolv.db.engine import get_session_factory
from resolv.repositories.shipment import ShipmentRepository
from resolv.services.shipment import ShipmentService


@tool(parse_docstring=True)
def query_logistics(tracking_no: str) -> dict[str, object]:
    """查询真实物流状态；用户询问具体物流时使用。记录不存在则返回 found=false。

    Args:
        tracking_no: 订单查询结果中提供的物流单号。
    """
    session_factory = get_session_factory(DatabaseSettings().database_url)
    with session_factory() as session:
        return ShipmentService(ShipmentRepository(session)).get_shipment(tracking_no)
