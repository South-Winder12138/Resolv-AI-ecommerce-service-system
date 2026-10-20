from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from resolv.core.config import Settings
from resolv.core.llm import get_chat_model


def test_chat_model_uses_responses_api() -> None:
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        openai_model="test-model",
        openai_base_url="https://gateway.example.com/v1",
        _env_file=None,
    )

    model = get_chat_model(settings)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "test-model"
    assert model.use_responses_api is True
    assert model.output_version == "responses/v1"
    assert model.openai_api_base == "https://gateway.example.com/v1"


def test_summary_model_has_independent_output_budget():
    from resolv.core.config import SummaryModelSettings
    from resolv.core.llm import get_summary_model

    settings = SummaryModelSettings(
        _env_file=None,
        model_name="fake-summary",
        api_key="summary-key",
        base_url="https://summary.invalid/v1",
        model_context_window_tokens=16000,
        reserved_output_tokens=1000,
    )
    model = get_summary_model(settings)
    assert model.model_name == "fake-summary"
    assert model.openai_api_base == "https://summary.invalid/v1"
    assert model.max_tokens == 1000
    assert model.use_previous_response_id is False
