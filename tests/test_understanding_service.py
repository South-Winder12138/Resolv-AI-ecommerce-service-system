import pytest
from langchain_core.messages import AIMessage

from resolv.models.conversation import Conversation, ConversationMessage
from resolv.schemas.understanding import UnderstandingError, UnderstandingInput
from tests.understanding_fixtures import input_ref, make_classification, make_resolution


def require(service):
    if service is None:
        pytest.fail("UnderstandingService missing")
    return service


def accept(service, request_id="req", text="查订单", **links):
    return service.accept(
        "c", UnderstandingInput(request_id=request_id, text=text, **links)
    )[0]


def test_understanding_uses_real_completed_context_without_mutating_chat(
    understanding_service,
):
    s = require(understanding_service)
    row = accept(s)
    assert row.snapshot.cutoff_seq == 2 and any(
        source.text == "订单 MH001" for source in row.snapshot.model_sources
    )
    view = s.run("c", row.understanding_id)
    assert view.status == "ready" and view.branch == "business" and view.route_ready
    with s.session_factory() as session:
        c = session.get(Conversation, "c")
        assert c.next_message_seq == 3 and c.current_run_id is None
        assert session.query(ConversationMessage).count() == 2
    assert s.list_tasks("c", "pending") == []


def test_technical_model_failure_never_becomes_other(understanding_service):
    s = require(understanding_service)
    s.resolver.model.client.replies = [RuntimeError("credential secret")]
    row = accept(s)
    with pytest.raises(UnderstandingError, match="model_unavailable"):
        s.run("c", row.understanding_id)
    view = s.get("c", row.understanding_id)
    assert view.status == "interrupted" and not view.route_ready and view.branch is None


def test_technical_failure_always_identifies_accepted_record(understanding_service):
    s = require(understanding_service)
    s.resolver.model.client.replies = [RuntimeError("transport")]
    row = accept(s)
    with pytest.raises(UnderstandingError) as caught:
        s.run("c", row.understanding_id)
    assert caught.value.understanding_id == row.understanding_id


def test_main_unresolved_reference_is_normal_clarification(understanding_service):
    s = require(understanding_service)
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text="查它物流",
                unresolved_references=[
                    {
                        "span": {"start": 1, "end": 2},
                        "candidate_refs": [],
                        "question": "哪笔订单？",
                    }
                ],
            ).model_dump_json()
        ),
        AIMessage(
            content=make_classification(["查它物流"], ["logistics"]).model_dump_json()
        ),
    ]
    row = accept(s, text="查它物流")
    view = s.run("c", row.understanding_id)
    assert (
        view.status == "needs_clarification"
        and view.branch == "business"
        and not view.route_ready
    )


def test_duplicate_accept_does_not_start_execution(understanding_service):
    s = require(understanding_service)
    first = accept(s)
    second, created = s.accept("c", UnderstandingInput(request_id="req", text="查订单"))
    assert not created and first.understanding_id == second.understanding_id
    assert not s.resolver.model.calls_by_stage


def test_clarification_uses_parent_snapshot_and_retains_old_pending_id(
    understanding_service,
):
    from resolv.schemas.understanding import EvidenceRef

    s = require(understanding_service)
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text="查它物流，退款",
                unresolved_references=[
                    {
                        "span": {"start": 1, "end": 2},
                        "candidate_refs": [],
                        "question": "哪笔订单？",
                    }
                ],
            ).model_dump_json()
        ),
        AIMessage(
            content=make_classification(
                ["查它物流", "退款"],
                ["logistics", "refund_return"],
                original_text="查它物流，退款",
            ).model_dump_json()
        ),
    ]
    parent = accept(s, text="查它物流，退款")
    s.run("c", parent.understanding_id)
    task = s.list_tasks("c")[0]
    resolution = make_resolution(resolved_text="查 MH001 物流，退款")
    c = make_classification(["MH001"], ["logistics"])
    c.requests[0].text = "查 MH001 物流"
    c.requests[0].evidence_refs += [
        EvidenceRef(
            kind="understanding",
            source_id=parent.understanding_id,
            span={"start": 0, "end": 4},
            json_path=[],
            quote="查它物流",
        )
    ]
    s.resolver.model.client.replies = [
        AIMessage(content=resolution.model_dump_json()),
        AIMessage(content=c.model_dump_json()),
    ]
    child = accept(
        s,
        request_id="child",
        text="MH001",
        clarifies_understanding_id=parent.understanding_id,
    )
    view = s.run("c", child.understanding_id)
    assert view.status == "ready" and view.branch == "business"
    assert s.list_tasks("c")[0].task_id == task.task_id
    assert s.get("c", parent.understanding_id).status == "needs_clarification"
    assert (
        s.get("c", parent.understanding_id).followup_ids["clarification_child"]
        == child.understanding_id
    )


def test_secondary_continuation_reuses_id_then_primary_selects_it(
    understanding_service,
):
    from resolv.schemas.understanding import EvidenceRef

    s = require(understanding_service)
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(resolved_text="查订单，退款").model_dump_json()
        ),
        AIMessage(
            content=make_classification(
                ["查订单", "退款"], ["order", "refund_return"]
            ).model_dump_json()
        ),
    ]
    row = accept(s, text="查订单，退款")
    s.run("c", row.understanding_id)
    task = s.list_tasks("c")[0]
    task_ref = EvidenceRef(
        kind="task", source_id=task.task_id, span=None, json_path=["text"], quote="退款"
    )
    text = "先查物流，再继续退款"
    classified = make_classification(
        ["查物流", "继续退款"], ["logistics", "refund_return"], original_text=text
    )
    classified.requests[1].evidence_refs.append(task_ref)
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text=text, continuation_selection=task.task_id
            ).model_dump_json()
        ),
        AIMessage(content=classified.model_dump_json()),
    ]
    next_ = accept(s, request_id="next", text=text)
    view = s.run("c", next_.understanding_id)
    assert (
        view.pending_requests[0].task_id == task.task_id
        and s.list_tasks("c")[0].queue_status == "pending"
    )
    classified = make_classification(["继续退款"], ["refund_return"])
    classified.requests[0].evidence_refs.append(task_ref)
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text="继续退款", continuation_selection=task.task_id
            ).model_dump_json()
        ),
        AIMessage(content=classified.model_dump_json()),
    ]
    last = accept(s, request_id="last", text="继续退款")
    s.run("c", last.understanding_id)
    assert s.list_tasks("c") == []
    selected = s.list_tasks("c", "selected")[0]
    assert (
        selected.task_id == task.task_id
        and selected.selected_by_understanding_id == last.understanding_id
    )


def test_selected_projection_is_hidden_until_child_durable_terminal(
    understanding_service,
):
    from resolv.schemas.understanding import EvidenceRef
    from tests.understanding_fixtures import fail_once

    s = require(understanding_service)
    text = "查物流，退款"
    s.resolver.model.client.replies = [
        AIMessage(content=make_resolution(resolved_text=text).model_dump_json()),
        AIMessage(
            content=make_classification(
                ["查物流", "退款"], ["logistics", "refund_return"], original_text=text
            ).model_dump_json()
        ),
    ]
    parent = accept(s, text=text)
    s.run("c", parent.understanding_id)
    task = s.list_tasks("c")[0]
    c = make_classification(["继续退款"], ["refund_return"])
    c.requests[0].evidence_refs.append(
        EvidenceRef(
            kind="task",
            source_id=task.task_id,
            span=None,
            json_path=["text"],
            quote=task.text,
        )
    )
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text="继续退款", continuation_selection=task.task_id
            ).model_dump_json()
        ),
        AIMessage(content=c.model_dump_json()),
    ]
    child = accept(s, request_id="continue", text="继续退款")
    s.fault_hook = fail_once("after_publish")
    with pytest.raises(UnderstandingError):
        s.run("c", child.understanding_id)
    assert s.list_tasks("c", "selected") == []
    s.fault_hook = None
    s.resume("c", child.understanding_id)
    selected = s.list_tasks("c", "selected")[0]
    assert selected.task_id == task.task_id and selected.text == "继续退款"


@pytest.mark.parametrize("old_intent", ["refund_return", None])
def test_continuation_primary_publishes_current_clarification(
    understanding_service, old_intent
):
    from resolv.schemas.understanding import Ambiguity, EvidenceRef

    s = require(understanding_service)
    text = "查物流，退它"
    c = make_classification(
        ["查物流", "退它"], ["logistics", old_intent], original_text=text
    )
    c.requests[1].ambiguities = [
        Ambiguity(kind="object", question="哪笔订单？", evidence_refs=[])
    ]
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text=text,
                unresolved_references=[
                    {
                        "span": {"start": 5, "end": 6},
                        "candidate_refs": [],
                        "question": "哪笔订单？",
                    }
                ],
            ).model_dump_json()
        ),
        AIMessage(content=c.model_dump_json()),
    ]
    parent = accept(s, text=text)
    s.run("c", parent.understanding_id)
    task = s.list_tasks("c")[0]
    original_parent = s.get("c", parent.understanding_id).model_dump_json()
    text = "继续退款，是订单 MH002"
    c = make_classification([text], ["refund_return"])
    c.requests[0].text = "订单 MH002 申请退款"
    c.requests[0].evidence_refs.append(
        EvidenceRef(
            kind="task",
            source_id=task.task_id,
            span=None,
            json_path=["text"],
            quote=task.text,
        )
    )
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text=text, continuation_selection=task.task_id
            ).model_dump_json()
        ),
        AIMessage(content=c.model_dump_json()),
    ]
    child = accept(s, request_id="continue", text=text)
    view = s.run("c", child.understanding_id)
    assert view.status == "ready" and view.route_ready
    assert view.primary_request.task_id == task.task_id
    assert view.primary_request.text == "订单 MH002 申请退款"
    assert view.primary_request.intent == "refund_return"
    assert view.primary_request.understanding_id == child.understanding_id
    assert view.primary_request.role == "primary"
    assert (
        not view.primary_request.ambiguities
        and not view.primary_request.unresolved_references
    )
    assert s.get("c", parent.understanding_id).model_dump_json() == original_parent
    assert s.get("c", child.understanding_id).primary_request == view.primary_request
    assert s.list_tasks("c", "selected")[0].text == "订单 MH002 申请退款"


def test_clarification_can_retain_parent_secondary_without_selecting_or_copying_it(
    understanding_service,
):
    from resolv.schemas.understanding import EvidenceRef

    s = require(understanding_service)
    text = "查它物流，退款"
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text=text,
                unresolved_references=[
                    {
                        "span": {"start": 1, "end": 2},
                        "candidate_refs": [],
                        "question": "哪笔订单？",
                    }
                ],
            ).model_dump_json()
        ),
        AIMessage(
            content=make_classification(
                ["查它物流", "退款"], ["logistics", "refund_return"]
            ).model_dump_json()
        ),
    ]
    parent = accept(s, text=text)
    s.run("c", parent.understanding_id)
    task = s.list_tasks("c")[0]
    classified = make_classification(
        ["退款", "查它物流"], ["refund_return", "logistics"], original_text=text
    )
    for request in classified.requests:
        original = request.evidence_refs[0]
        request.evidence_refs = [
            original.model_copy(
                update={"kind": "understanding", "source_id": parent.understanding_id}
            ),
            EvidenceRef(**input_ref("MH001")),
        ]
    classified.requests[0].evidence_refs.append(
        EvidenceRef(
            kind="task",
            source_id=task.task_id,
            span=None,
            json_path=["text"],
            quote="退款",
        )
    )
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text="查 MH001 物流，退款"
            ).model_dump_json()
        ),
        AIMessage(content=classified.model_dump_json()),
    ]
    child = accept(
        s,
        request_id="child",
        text="MH001",
        clarifies_understanding_id=parent.understanding_id,
    )
    view = s.run("c", child.understanding_id)
    assert view.primary_request.intent == "logistics" and view.status == "ready"
    assert [t.task_id for t in view.pending_requests] == [task.task_id]
    assert [t.task_id for t in s.list_tasks("c")] == [task.task_id]
    assert s.list_tasks("c")[0].queue_status == "pending"


def test_archived_conversation_reads_but_rejects_accept_and_resume(
    understanding_service,
):
    s = require(understanding_service)
    row = accept(s)
    s.run("c", row.understanding_id)
    with s.session_factory.begin() as session:
        session.get(Conversation, "c").status = "archived"
    assert s.get("c", row.understanding_id).status == "ready"
    with pytest.raises(UnderstandingError, match="conversation_archived"):
        s.resume("c", row.understanding_id)
    with pytest.raises(UnderstandingError, match="conversation_archived"):
        accept(s, "new")
