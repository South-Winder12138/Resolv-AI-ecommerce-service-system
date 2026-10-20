"""Make two real structured-output requests for the learning exercise."""

from pydantic import ValidationError

from resolv.after_sales import extract_after_sales_ticket
from resolv.core.config import Settings


def main() -> None:
    try:
        Settings()
    except ValidationError as exc:
        raise SystemExit(
            "Check OPENAI_API_KEY and OPENAI_BASE_URL in .env or the environment."
        ) from exc

    examples = (
        "订单 MH001 到货就是坏的，我想退款。",
        "昨天买的耳机左边没声音，我想换一个。",
    )
    for text in examples:
        result = extract_after_sales_ticket(text)

        print(f"input: {text}")
        print(f"type: {type(result).__name__}")
        print(f"order_id: {result.order_id}")
        print(f"request_type: {result.request_type.value}")
        print(f"reason: {result.reason}")


if __name__ == "__main__":
    main()
