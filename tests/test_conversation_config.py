import pytest
from pydantic import ValidationError

from resolv.core import config

URL = "mysql+pymysql://app:private_pw@127.0.0.1:3307/resolv_conversations?charset=utf8mb4"


def test_settings_need_no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = config.ConversationDatabaseSettings(
        _env_file=None, conversation_database_url=URL
    )
    assert settings.conversation_database_url.get_secret_value() == URL
    assert "private_pw" not in repr(settings)


@pytest.mark.parametrize("url", [
    "sqlite:///business.db", "mysql://app:private_pw@host/db?charset=utf8mb4",
    "mysql+pymysql://app@host/db?charset=utf8mb4",
    "mysql+pymysql://app:private_pw@host/?charset=utf8mb4",
    URL.replace("utf8mb4", "utf8"), URL + "&ssl=false",
    URL.replace(":3307", ":0"), "private_pw is not a URL",
])
def test_invalid_url_hides_secret(url):
    with pytest.raises(ValidationError) as raised:
        config.ConversationDatabaseSettings(_env_file=None, conversation_database_url=url)
    assert "private_pw" not in str(raised.value)


def test_context_window_and_profile_are_explicit():
    with pytest.raises(ValidationError):
        config.ContextSettings(_env_file=None)
    with pytest.raises(ValidationError):
        config.ContextSettings(_env_file=None, model_context_window_tokens=3000, tokenizer_profile="cl100k_base")
    with pytest.raises(ValidationError):
        config.ContextSettings(_env_file=None, model_context_window_tokens=16000, tokenizer_profile="guess-from-model-name")


def test_execution_settings_default_and_independent(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = config.ConversationExecutionSettings(_env_file=None)
    assert (settings.owner_id, settings.lease_seconds, settings.heartbeat_seconds) == ("local", 120, 30)
    assert settings.state_schema_version == 1


@pytest.mark.parametrize("values", [{"heartbeat_seconds": 0}, {"lease_seconds": 30}, {"state_schema_version": 2}, {"owner_id": ""}])
def test_execution_settings_reject_unsafe_values(values):
    with pytest.raises(ValidationError):
        config.ConversationExecutionSettings(_env_file=None, **values)
