from langchain_core.messages import HumanMessage, SystemMessage

import resolv.after_sales as after_sales_module
from resolv.core.prompts import AFTER_SALES_EXTRACTION_PROMPT
from resolv.schemas.after_sales import AfterSalesTicket, RequestType


def test_extraction_prompt_has_roles_and_input() -> None:
    messages = AFTER_SALES_EXTRACTION_PROMPT.invoke(
        {"text": "订单 MH001 到货就是坏的，我想退款。"}
    ).to_messages()

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "不得编造订单号" in messages[0].content
    assert messages[1].content == "订单 MH001 到货就是坏的，我想退款。"


def test_extraction_uses_structured_model(monkeypatch) -> None:
    expected = AfterSalesTicket(
        order_id="MH001", request_type=RequestType.REFUND, reason="到货损坏"
    )

    class FakeStructuredModel:
        def invoke(self, messages):
            assert isinstance(messages[0], SystemMessage)
            assert isinstance(messages[1], HumanMessage)
            assert messages[1].content == "订单 MH001 到货就是坏的，我想退款。"
            return expected

    class FakeModel:
        def with_structured_output(self, schema):
            assert schema is AfterSalesTicket
            return FakeStructuredModel()

    monkeypatch.setattr(after_sales_module, "get_chat_model", lambda: FakeModel())

    result = after_sales_module.extract_after_sales_ticket(
        "订单 MH001 到货就是坏的，我想退款。"
    )

    assert result is expected
