"""Default memory behavior survives; explicit evidence replaces the old reserve."""

import pytest

from resolv.services.context import (
    ContextBudgetExceeded,
    ContextBuilder,
    knowledge_text,
)
from tests.conversation_fixtures import conversation_factory
from tests.test_summarization import memory_service, summary_harness

__all__ = ["conversation_factory", "memory_service", "summary_harness"]


def test_memory_default_still_reserves_legacy_knowledge_budget(memory_service):
    service, run, counter, _h = memory_service
    counter.total = 10000
    with (
        service.execution(run.run_id) as execution,
        pytest.raises(ContextBudgetExceeded),
    ):
        service.prepare_memory(run.run_id, execution)
    assert not service.get_run(run.run_id).memory_prepared


def test_explicit_evidence_has_no_second_legacy_reserve(memory_service):
    service, run, counter, h = memory_service
    counter.total = 10000

    class RecordingBuilder(ContextBuilder):
        def build(self, inputs):
            assert inputs.knowledge == [{"citations": [], "chunks": {}}]
            assert inputs.tool_schemas == []
            assert inputs.system_prompt == "知识专用提示"
            return super().build(inputs)

    base = service.builder_for_run(service.get_run(run.run_id))
    builder = RecordingBuilder(base.settings, counter)
    with service.execution(run.run_id) as execution:
        saved = service.prepare_memory(
            run.run_id,
            execution,
            context_builder=builder,
            system_prompt="知识专用提示",
            knowledge=[{"citations": [], "chunks": {}}],
            tool_schemas=[],
        )
    assert saved.memory_prepared and h.model.calls == 0


def test_full_serialized_evidence_can_make_memory_preparation_fail(memory_service):
    service, run, _counter, _ = memory_service
    evidence = [{"citations": ["metadata" * 2000], "chunks": {}}]
    assert len(knowledge_text(evidence)) > 12000
    with (
        service.execution(run.run_id) as execution,
        pytest.raises(ContextBudgetExceeded),
    ):
        service.prepare_memory(
            run.run_id, execution, knowledge=evidence, tool_schemas=[]
        )
    assert not service.get_run(run.run_id).memory_prepared
