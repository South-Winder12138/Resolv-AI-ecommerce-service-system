import pytest
from pydantic import ValidationError

from resolv.core import config


def settings(**kwargs):
    if not hasattr(config, "UnderstandingSettings"):
        pytest.fail("UnderstandingSettings missing")
    return config.UnderstandingSettings(_env_file=None, **kwargs)


def test_understanding_is_optional_without_credentials():
    value = settings()
    assert not value.enabled


def test_independent_calibrated_default_does_not_change_chat_budget():
    from resolv.core.config import ContextSettings
    from resolv.services.understanding_context import UnderstandingContext
    from tests.test_understanding_context import Counter
    from tests.understanding_fixtures import make_snapshot

    chat = ContextSettings(
        _env_file=None,
        model_context_window_tokens=256000,
        tokenizer_profile="o200k_base",
        provider_input_overhead_tokens=5000,
    )
    options = settings()
    counter = Counter()
    counter.count = lambda *_: 1000
    context = UnderstandingContext(
        chat,
        counter,
        model_name="fake",
        api_address_fingerprint="f" * 64,
        resolution_prompt="r",
        intent_prompt="i",
        overhead_override=options.provider_input_overhead_tokens,
    )
    snapshot = make_snapshot("查订单", config=context.config)
    assert context.messages(snapshot, "resolution")[1] == 7500
    assert chat.provider_input_overhead_tokens == 5000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"state_schema_version": 2},
        {"provider_input_overhead_tokens": -1},
    ],
)
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        settings(**kwargs)
