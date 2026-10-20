"""Explicit defaults only. Importing this module never reads a selector or .env."""

from resolv.schemas.knowledge_branch import KnowledgeBranchProfile
from resolv.schemas.knowledge_release import ReleaseSelection


def default_knowledge_branch_profile(
    *, query_model, check_model, answer_model, summary_profile
):
    return KnowledgeBranchProfile(
        selection=ReleaseSelection(status="disabled"),
        query_model=query_model,
        check_model=check_model,
        answer_model=answer_model,
        summary_profile=summary_profile,
    )


def runtime_knowledge_branch_profile(
    settings, *, understanding_config, business_config, summary_settings
):
    from resolv.core.knowledge_branch_model import make_knowledge_model_profile
    from resolv.schemas.knowledge_branch import FrozenJSON, KnowledgeModelProfile
    from resolv.services.business_summary import summary_profile
    from resolv.services.summarization import SUMMARY_PROMPT

    summary = summary_profile(summary_settings)
    profile = default_knowledge_branch_profile(
        query_model=make_knowledge_model_profile(
            settings, purpose="knowledge_query", reference_profile=understanding_config
        ),
        check_model=make_knowledge_model_profile(
            settings, purpose="knowledge_check", reference_profile=understanding_config
        ),
        answer_model=make_knowledge_model_profile(
            settings, purpose="knowledge_answer", reference_profile=business_config
        ),
        summary_profile=KnowledgeModelProfile(
            purpose="summary",
            model_name=summary_settings.model_name,
            api_address_fingerprint=summary["api_address_fingerprint"],
            model_api_mode=summary["model_api_mode"],
            tokenizer_profile=business_config.tokenizer_profile,
            model_context_window_tokens=summary_settings.model_context_window_tokens,
            app_max_input_tokens=summary_settings.input_max_tokens,
            provider_input_overhead_tokens=summary_settings.provider_input_overhead_tokens,
            reserved_output_tokens=summary_settings.reserved_output_tokens,
            safety_margin_tokens=summary_settings.safety_margin_tokens,
            timeout_seconds=summary_settings.timeout_seconds,
            streaming=True,
            enable_thinking=False
            if summary["model_api_mode"] == "chat_completions"
            else None,
            top_p=0.1 if summary["model_api_mode"] == "chat_completions" else None,
            system_prompt=SUMMARY_PROMPT,
            prompt_version="summary-v1",
        ),
    )
    return profile.model_copy(update={"summary_config": FrozenJSON.from_value(summary)})
