from resolv.core.llm import get_chat_model
from resolv.core.prompts import AFTER_SALES_EXTRACTION_PROMPT
from resolv.schemas.after_sales import AfterSalesTicket


def extract_after_sales_ticket(text: str) -> AfterSalesTicket:
    """Extract one validated after-sales ticket from user text."""
    prompt_value = AFTER_SALES_EXTRACTION_PROMPT.invoke({"text": text})
    messages = prompt_value.to_messages()
    model = get_chat_model()
    structured_model = model.with_structured_output(AfterSalesTicket)
    return structured_model.invoke(messages)
