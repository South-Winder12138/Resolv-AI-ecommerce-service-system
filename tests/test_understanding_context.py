import importlib
from types import SimpleNamespace

import pytest

from resolv.core.config import ContextSettings
from resolv.repositories.understanding import source
from resolv.schemas.conversation import ConversationRecord
from resolv.schemas.understanding import (
    EvidenceRef,
    SnapshotSources,
    UnderstandingError,
    UnderstandingInput,
)
from tests.conversation_fixtures import make_action, make_summary, make_turn
from tests.understanding_fixtures import input_ref, make_pending_task, make_resolution


class Counter:
    def count_text(self, text):
        return len(text) // 4

    def count(self, messages, schemas):
        return sum(self.count_text(m.content) for m in messages)


def test_pending_candidates_are_complete_without_duplicate_source_payload():
    import json

    task = make_pending_task("a", "refund_return")
    h = harness(pending=[task])
    payload = json.loads(
        h.context.messages(h.snapshot, "resolution")[0][1].content.split("\n", 1)[1]
    )
    assert payload["pending_requests"] == [task.model_dump(mode="json")]
    source_ = next(s for s in payload["sources"] if s["kind"] == "task")
    assert "json_value" not in source_ and source_["value_in_pending_requests"]


def test_pending_catalog_avoids_repeating_ids_and_intents_as_evidence():
    import json

    task = make_pending_task("a" * 32, "refund_return")
    h = harness(pending=[task])
    payload = json.loads(
        h.context.messages(h.snapshot, "resolution")[0][1].content.split("\n", 1)[1]
    )
    refs = [r for r in payload["reference_catalog"] if r["kind"] == "task"]
    assert [r["json_path"] for r in refs] == [["text"]]
    assert payload["pending_requests"] == [task.model_dump(mode="json")]


def test_calibrated_budget_keeps_both_full_pending_candidates():
    from resolv.core.understanding_prompts import (
        INTENT_PROMPT_V1,
        RESOLUTION_PROMPT_V1,
    )
    from resolv.services.context import TokenCounter

    h = harness()
    h.context.counter = TokenCounter("o200k_base")
    h.context.config.provider_input_overhead_tokens = 6500
    h.context.config.schema_constraint_version = "source-catalog-v6"
    h.context.config.output_format = "json_schema"
    h.context.config.resolution_prompt = RESOLUTION_PROMPT_V1
    h.context.config.intent_prompt = INTENT_PROMPT_V1
    h.sources.pending_requests = [
        make_pending_task(
            "016fa471d0844d629a9b5b078145e3c2",
            "refund_return",
            understanding_id="cf67e7cbf4604147b2085815b85a8c8c",
        ),
        make_pending_task(
            "9e72bcf37b354411a9a08d07e32107d3",
            "refund_return",
            understanding_id="7a670172e17f47169602c699528a4458",
        ),
    ]
    h.sources.linked_inputs = {
        t.understanding_id: "查订单，退款" for t in h.sources.pending_requests
    }
    snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="new", text="继续退款")
    )
    assert len(snapshot.pending_requests) == 2
    assert h.context.messages(snapshot, "resolution")[1] <= 12000


def test_multiple_pending_transport_roundtrips_every_candidate_field():
    import json

    tasks = [
        make_pending_task("a", "refund_return"),
        make_pending_task("b", "refund_return"),
    ]
    h = harness(pending=tasks)
    payload = json.loads(
        h.context.messages(h.snapshot, "resolution")[0][1].content.split("\n", 1)[1]
    )
    table = payload["pending_requests"]
    decoded = [dict(zip(table["columns"], row, strict=True)) for row in table["rows"]]
    assert decoded == [t.model_dump(mode="json") for t in tasks]


def test_compact_model_sources_preserve_full_text_and_json_without_hashes():
    import json

    turn = make_turn(question="完整原文 MH001", answer="历史答复")
    h = harness([turn])
    messages, _ = h.context.messages(h.snapshot, "resolution")
    payload = json.loads(messages[1].content.split("\n", 1)[1])
    assert all("source_hash" not in s for s in payload["sources"])
    assert any(
        r["source_id"] == turn.user.message_id and r["quote"] == "完整原文 MH001"
        for r in payload["reference_catalog"]
    )
    assert any(
        s.text == "完整原文 MH001" and s.source_hash for s in h.snapshot.model_sources
    )
    assert '"title"' not in messages[0].content


def test_canonical_reference_catalog_has_exact_history_and_clause_spans():
    import json

    turn = make_turn(question="订单 MH001")
    h = harness([turn])
    h.snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="new", text="先查物流，再退款")
    )
    payload = json.loads(
        h.context.messages(h.snapshot, "resolution")[0][1].content.split("\n", 1)[1]
    )
    refs = payload["reference_catalog"]
    history = next(ref for ref in refs if ref["source_id"] == turn.user.message_id)
    assert history == {
        "kind": "message",
        "source_id": turn.user.message_id,
        "span": {"start": 0, "end": 8},
        "json_path": [],
        "quote": "订单 MH001",
    }
    assert any(
        ref["source_id"] == "current"
        and ref["quote"] == "再退款"
        and ref["span"] == {"start": 5, "end": 8}
        for ref in refs
    )
    assert payload["current_input_ref"]["quote"] == "先查物流，再退款"


def test_independent_reasoning_parameter_is_frozen():
    h = harness()
    assert getattr(h.snapshot.config, "reasoning_effort", None) == "low"


def test_qwen_wire_format_and_thinking_mode_are_frozen():
    h = harness()
    assert hasattr(h.context.config, "model_api_mode") and hasattr(
        h.context.config, "enable_thinking"
    )
    h.context.config.model_api_mode = "chat_completions"
    h.context.config.enable_thinking = False
    snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="qwen", text="查订单")
    )
    h.context.config.enable_thinking = True
    assert (
        snapshot.config.model_api_mode == "chat_completions"
        and snapshot.config.enable_thinking is False
    )


def test_api_schema_is_frozen_and_counted_without_prompt_duplication():
    h = harness()
    assert hasattr(h.context.config, "output_format")
    h.context.config.output_format = "json_schema"
    snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="schema", text="查订单")
    )
    h.context.config.output_format = "json_object"
    h.counter.count = lambda *_: 1000
    h.counter.count_text = lambda _: 75
    messages, estimate = h.context.messages(snapshot, "resolution")
    assert snapshot.config.output_format == "json_schema"
    assert estimate == 6075
    assert '"properties"' not in messages[0].content


def test_v3_uses_two_complete_request_variants_and_preserves_frozen_v2():
    h = harness()
    h.context.config.schema_constraint_version = "source-catalog-v3"
    snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="v3", text="先查物流，再退款")
    )
    schema = h.context.output_schema(snapshot, "intents", make_resolution())
    plain, priority = schema["$defs"]["ClassifiedRequest"]["anyOf"]
    assert plain["properties"]["explicit_priority"] == {"type": "null"}
    assert plain["properties"]["priority_evidence_refs"]["maxItems"] == 0
    assert priority["properties"]["explicit_priority"] == {
        "type": "integer",
        "minimum": 0,
    }
    assert (
        priority["properties"]["priority_evidence_refs"]["items"]["const"]["quote"]
        == snapshot.original_text
    )
    assert (
        set(plain["required"]) == set(priority["required"]) == set(plain["properties"])
    )
    h.context.config.schema_constraint_version = "source-catalog-v2"
    legacy = h.context.freeze(
        h.sources, UnderstandingInput(request_id="v2", text="查订单")
    )
    assert (
        "allOf"
        in h.context.output_schema(legacy, "intents", make_resolution())["$defs"][
            "ClassifiedRequest"
        ]
    )


def test_v4_evidence_schema_keeps_reference_fields_as_one_exact_tuple():
    from resolv.services.understanding_context import reference_catalog

    h = harness()
    h.context.config.schema_constraint_version = "source-catalog-v4"
    h.sources.linked_inputs = {"parent": "订单 MH001 退款，查订单 MH002 金额"}
    snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="v4", text="继续核实")
    )
    definition = h.context.output_schema(snapshot, "intents", make_resolution())[
        "$defs"
    ]["EvidenceRef"]
    assert definition == {
        "type": "object",
        "enum": reference_catalog(snapshot.model_sources, include_linked_clauses=True),
    }
    refs = definition["enum"]
    assert any(
        r["source_id"] == "parent" and r["quote"] == "订单 MH001 退款" for r in refs
    )
    for ref in refs:
        if ref["span"]:
            original = (
                snapshot.original_text
                if ref["source_id"] == "current"
                else snapshot.linked_inputs[ref["source_id"]]
            )
            assert original[ref["span"]["start"] : ref["span"]["end"]] == ref["quote"]


def test_v5_schema_separates_known_intent_from_unknown_goal_clarification():
    h = harness()
    h.context.config.schema_constraint_version = "source-catalog-v5"
    snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="v5", text="查退款进度")
    )
    schema = h.context.output_schema(snapshot, "intents", make_resolution())
    variants = schema["$defs"]["ClassifiedRequest"]["anyOf"]
    assert len(variants) == 4
    for variant in variants:
        properties = variant["properties"]
        if properties["intent"] == {"type": "null"}:
            assert properties["ambiguities"]["minItems"] == 1
        else:
            assert properties["ambiguities"]["maxItems"] == 0
    unresolved = make_resolution(
        unresolved_references=[
            {
                "span": {"start": 1, "end": 2},
                "candidate_refs": [],
                "question": "哪一个？",
            }
        ]
    )
    assert (
        len(
            h.context.output_schema(snapshot, "intents", unresolved)["$defs"][
                "ClassifiedRequest"
            ]["anyOf"]
        )
        == 2
    )


def test_v6_exposes_exact_selected_task_reference_without_changing_primary_order():
    import json

    h = harness(pending=[make_pending_task("selected", "refund_return")])
    h.context.config.schema_constraint_version = "source-catalog-v6"
    snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="v6", text="先查物流，再继续退款")
    )
    resolution = make_resolution(
        resolved_text=snapshot.original_text, continuation_selection="selected"
    )
    payload = json.loads(
        h.context.messages(snapshot, "intents", resolution)[0][1].content.split(
            "\n", 1
        )[1]
    )
    assert payload["required_continuation_ref"] == {
        "kind": "task",
        "source_id": "selected",
        "span": None,
        "json_path": ["text"],
        "quote": "退款",
    }
    assert payload["original_text"] == snapshot.original_text
    assert len(payload["pending_requests"]) == 1


def test_provider_schema_limits_sources_and_binding_indexes_to_frozen_input():
    h = harness([make_turn(question="订单 MH001")])
    assert hasattr(h.context.config, "schema_constraint_version")
    h.context.config.schema_constraint_version = "source-catalog-v1"
    h.context.config.output_format = "json_schema"
    snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="limits", text="查它")
    )
    schema = h.context.output_schema(snapshot, "resolution")
    props = schema["$defs"]["EvidenceRef"]["properties"]
    assert set(props["kind"]["enum"]) == {"input", "message"}
    assert set(props["source_id"]["enum"]) == {
        s.source_id for s in snapshot.model_sources
    }
    assert schema["properties"]["continuation_selection"] == {"type": "null"}
    schema = h.context.output_schema(snapshot, "intents", make_resolution())
    props = schema["$defs"]["EvidenceRef"]["properties"]
    assert props["kind"]["enum"] == ["input"]
    assert props["source_id"]["enum"] == ["current"]
    assert (
        schema["$defs"]["ClassifiedRequest"]["properties"]["binding_indexes"][
            "maxItems"
        ]
        == 0
    )


def test_frozen_v2_provider_schema_preserves_priority_and_exact_current_quotes():
    h = harness()
    h.context.config.schema_constraint_version = "source-catalog-v2"
    snapshot = h.context.freeze(
        h.sources,
        UnderstandingInput(request_id="v2", text="商家说‘已完成退款’，请核实退款进度"),
    )
    schema = h.context.output_schema(snapshot, "intents", make_resolution())
    condition = schema["$defs"]["ClassifiedRequest"]["allOf"][0]
    assert condition["then"]["properties"]["priority_evidence_refs"]["maxItems"] == 0
    priority = condition["else"]["properties"]["priority_evidence_refs"]
    assert (
        priority["minItems"] == 1
        and priority["items"]["const"]["quote"] == snapshot.original_text
    )
    assert (
        snapshot.original_text
        in schema["$defs"]["EvidenceRef"]["properties"]["quote"]["enum"]
    )
    h.context.config.schema_constraint_version = "source-catalog-v1"
    assert h.context.output_schema(snapshot, "intents", make_resolution()) == schema
    legacy = h.context.freeze(
        h.sources, UnderstandingInput(request_id="v1", text="查订单")
    )
    assert (
        "allOf"
        not in h.context.output_schema(legacy, "intents", make_resolution())["$defs"][
            "ClassifiedRequest"
        ]
    )


def harness(turns=(), summary=None, pending=(), parent=None):
    try:
        module = importlib.import_module("resolv.services.understanding_context")
    except ModuleNotFoundError:
        pytest.fail("UnderstandingContext missing")
    settings = ContextSettings(
        _env_file=None,
        model_context_window_tokens=256000,
        tokenizer_profile="o200k_base",
        provider_input_overhead_tokens=5000,
    )
    counter = Counter()
    context = module.UnderstandingContext(
        settings,
        counter,
        model_name="fake",
        api_address_fingerprint="f" * 64,
        resolution_prompt="resolve",
        intent_prompt="classify",
    )
    now = __import__("datetime").datetime.now(__import__("datetime").UTC)
    sources = SnapshotSources(
        conversation=ConversationRecord(
            conversation_id="c", thread_id="c", title="", created_at=now, updated_at=now
        ),
        turns=list(turns),
        summary=summary,
        pending_requests=list(pending),
        parent=parent,
    )
    return SimpleNamespace(
        context=context,
        counter=counter,
        sources=sources,
        snapshot=context.freeze(
            sources, UnderstandingInput(request_id="r", text="查它物流")
        ),
    )


def test_provider_overhead_is_inside_cap():
    h = harness()
    h.counter.count = lambda *_: 7000
    assert h.context.messages(h.snapshot, "resolution")[1] == 12000
    h.counter.count = lambda *_: 7001
    with pytest.raises(UnderstandingError, match="context_budget_exceeded"):
        h.context.messages(h.snapshot, "resolution")


def test_latest_large_whole_turn_is_kept_over_soft_target():
    latest = make_turn(seq=3, question="x" * 9000, answer="y" * 1000)
    h = harness([make_turn(question="old", answer="old"), latest])
    messages = [s.text for s in h.snapshot.model_sources if s.kind == "message"]
    assert messages == [latest.user.text, latest.assistant.text]
    assert h.snapshot.omitted_ranges[0].start == 1 and h.snapshot.cutoff_seq == 4


def test_latest_turn_that_exceeds_hard_limit_is_not_skipped():
    with pytest.raises(UnderstandingError, match="context_budget_exceeded"):
        harness([make_turn(question="small"), make_turn(seq=3, question="x" * 40000)])


def test_summary_covered_latest_turn_retains_original_evidence_for_pronoun():
    latest = make_turn(seq=3, question="订单 MH018", answer="当前仅讨论上述一个对象。")
    h = harness(
        [make_turn(question="older MH017"), latest], make_summary(covered_until_seq=4)
    )
    messages, estimate = h.context.messages(h.snapshot, "resolution")
    included = {s.source_id for s in h.snapshot.model_sources if s.kind == "message"}
    assert included == {latest.user.message_id, latest.assistant.message_id}
    assert "订单 MH018" in messages[1].content and estimate <= 12000
    assert h.snapshot.omitted_ranges[0].start == 1


def test_summary_covered_latest_turn_cannot_bypass_hard_input_limit():
    with pytest.raises(UnderstandingError, match="context_budget_exceeded"):
        harness([make_turn(question="x" * 40000)], make_summary(covered_until_seq=2))


def test_source_directory_has_omitted_originals_but_payload_does_not():
    turns = [
        make_turn(question="old-marker" + "x" * 7000),
        make_turn(seq=3, question="new" + "x" * 7000),
    ]
    h = harness(turns)
    payload = h.context.messages(h.snapshot, "resolution")[0][1].content
    assert "old-marker" not in payload and "new" in payload
    assert turns[0].user.message_id in {
        s.source_id for s in h.snapshot.source_fingerprints
    }


def test_two_pending_candidates_are_complete_and_not_chat_history():
    h = harness(
        pending=[
            make_pending_task("a", "refund_return"),
            make_pending_task("b", "refund_return"),
        ]
    )
    assert len(h.snapshot.pending_requests) == 2
    assert [s.kind for s in h.snapshot.model_sources] == ["input", "task", "task"]


def test_summary_and_completed_tool_observation_are_frozen():
    action = make_action(
        execution_status="completed",
        result={"order_id": "MH001"},
        completed_at=make_turn().user.created_at,
    )
    turn = make_turn(actions=[action])
    h = harness([turn], make_summary(covered_until_seq=0))
    observed = next(s for s in h.snapshot.model_sources if s.kind == "action")
    assert observed.json_value == {"order_id": "MH001"} and observed.observed_at
    assert h.snapshot.summary_version == 1
    assert h.snapshot.snapshot_hash
    assert "api_key" not in h.snapshot.config.model_dump()


def test_classification_output_is_recounted_without_all_history():
    h = harness([make_turn(question="HISTORY", answer="data")])
    resolution = make_resolution(resolved_text="x" * 32000)
    with pytest.raises(UnderstandingError, match="context_budget_exceeded"):
        h.context.messages(h.snapshot, "intents", resolution)
    payload = h.context.messages(h.snapshot, "intents", make_resolution())[0][1].content
    assert "HISTORY" not in payload and "resolved_text" in payload


def test_classification_carries_canonical_text_source_identity():
    import json

    h = harness()
    payload = json.loads(
        h.context.messages(h.snapshot, "intents", make_resolution())[0][
            1
        ].content.split("\n", 1)[1]
    )
    current = next(s for s in payload["sources"] if s["kind"] == "input")
    assert current["source_id"] == "current" and current["text"] == "查它物流"


def test_linked_originals_have_one_complete_catalog_copy():
    import json

    h = harness()
    h.sources.linked_inputs = {"old": "先查订单，另外退款"}
    snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="new", text="继续退款")
    )
    payload = json.loads(
        h.context.messages(snapshot, "resolution")[0][1].content.split("\n", 1)[1]
    )
    assert "linked_inputs" not in payload
    assert (
        next(r for r in payload["reference_catalog"] if r["source_id"] == "old")[
            "quote"
        ]
        == "先查订单，另外退款"
    )
    assert snapshot.linked_inputs == {"old": "先查订单，另外退款"}


def test_python_character_coordinate_guide_includes_combining_characters():
    h = harness()
    h.snapshot.original_text = "e\u0301它"
    import json

    payload = json.loads(
        h.context.messages(h.snapshot, "intents", make_resolution())[0][
            1
        ].content.split("\n", 1)[1]
    )
    assert payload["input_character_indexes"] == [[0, "e"], [1, "\u0301"], [2, "它"]]


def test_verify_rejects_wrong_python_span_and_unknown_source():
    h = harness()
    ref = EvidenceRef(**input_ref("它", 1))
    assert (
        h.context.verify(
            ref, h.snapshot, lambda _: pytest.fail("input should use snapshot")
        ).text
        == "查它物流"
    )
    wrong = ref.model_copy(update={"quote": "错"})
    with pytest.raises(UnderstandingError, match="source_invalid"):
        h.context.verify(wrong, h.snapshot, lambda _: None)
    ref = ref.model_copy(update={"kind": "message", "source_id": "foreign"})
    with pytest.raises(UnderstandingError, match="source_invalid"):
        h.context.verify(ref, h.snapshot, lambda _: pytest.fail("foreign lookup"))


def test_mutated_original_fails_frozen_fingerprint_check():
    turn = make_turn(question="订单 MH001")
    h = harness([turn])
    ref = EvidenceRef(
        kind="message",
        source_id=turn.user.message_id,
        span={"start": 3, "end": 8},
        json_path=[],
        quote="MH001",
    )
    bad = source(
        "message",
        turn.user.message_id,
        text="订单 MH002",
        observed_at=turn.user.created_at,
    )
    with pytest.raises(UnderstandingError, match="source_invalid"):
        h.context.verify(ref, h.snapshot, lambda _: bad)


def test_incomplete_turn_is_rejected_not_added_as_history():
    turn = make_turn()
    turn.assistant.run_id = "different"
    with pytest.raises(UnderstandingError, match="snapshot_invalid"):
        harness([turn])


def test_linked_pending_original_input_is_available_as_provenance():
    h = harness(pending=[make_pending_task("a", "refund_return")])
    # The repository provides the original input of explicitly linked pending records.
    h.sources.linked_inputs = {"old": "先查订单，另外退款"}
    frozen = h.context.freeze(
        h.sources, UnderstandingInput(request_id="r", text="继续退款")
    )
    assert frozen.linked_inputs == {"old": "先查订单，另外退款"}
    assert any(
        s.kind == "understanding" and s.text == "先查订单，另外退款"
        for s in frozen.model_sources
    )


def test_clarification_freezes_parent_not_newer_chat():
    h = harness([make_turn(question="旧订单 MH001")])
    parent = {
        "understanding_id": "p",
        "original_text": "查它物流",
        "snapshot": h.snapshot.model_dump(mode="json"),
        "final_json": None,
    }
    child = harness([make_turn(seq=3, question="后来 MH002")], parent=parent)
    assert child.snapshot.cutoff_seq == 2
    assert (
        "后来 MH002"
        not in child.context.messages(child.snapshot, "resolution")[0][1].content
    )
