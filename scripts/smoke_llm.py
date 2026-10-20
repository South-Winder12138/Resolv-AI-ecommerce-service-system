"""Make one real request to verify the OpenAI Responses API connection."""

from pydantic import ValidationError

from resolv.chat import chat


def main() -> None:
    try:
        answer = chat("我的订单 MH001 到哪里了？")
    except ValidationError as exc:
        raise SystemExit(
            "Check OPENAI_API_KEY and OPENAI_BASE_URL in .env or the environment."
        ) from exc

    print(answer)


if __name__ == "__main__":
    main()
