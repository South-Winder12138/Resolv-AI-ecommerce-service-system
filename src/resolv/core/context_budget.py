"""Explicit allocation for new runs; frozen historical settings are not rewritten."""

CONTEXT_BUDGET = {
    "app_max_input_tokens": 60000,
    "history_soft_budget_tokens": 30000,
    "recent_history_target_tokens": 10000,
    "summary_max_tokens": 4000,
    "knowledge_context_max_tokens": 15000,
    "reserved_output_tokens": 10000,
    "safety_margin_tokens": 5000,
}
SUMMARY_BUDGET = {
    "input_max_tokens": 60000,
    "summary_max_tokens": 4000,
    "reserved_output_tokens": 10000,
    "safety_margin_tokens": 5000,
}


def apply_context_budget(settings):
    return type(settings)(_env_file=None, **(settings.model_dump() | CONTEXT_BUDGET))


def apply_summary_budget(settings):
    return type(settings)(_env_file=None, **(settings.model_dump() | SUMMARY_BUDGET))


def model_budget():
    return {
        key: CONTEXT_BUDGET[key]
        for key in (
            "app_max_input_tokens",
            "reserved_output_tokens",
            "safety_margin_tokens",
            "recent_history_target_tokens",
        )
    }
