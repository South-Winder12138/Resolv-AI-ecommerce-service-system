from langchain_openai import ChatOpenAI

from resolv.core.config import Settings, SummaryModelSettings


def get_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    """Build the project's OpenAI chat model from validated settings."""
    settings = settings or Settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        use_responses_api=True,
        output_version="responses/v1",
    )


def get_summary_model(settings: SummaryModelSettings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
        use_responses_api=True,
        output_version="responses/v1",
        use_previous_response_id=False,
        max_tokens=settings.reserved_output_tokens,
        timeout=settings.timeout_seconds,
        truncation="disabled",
    )
