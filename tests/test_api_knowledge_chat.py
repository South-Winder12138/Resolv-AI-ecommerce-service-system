"""Actual HTTP serialization after a persisted offline knowledge run."""

from contextlib import contextmanager

from fastapi.testclient import TestClient

from tests.knowledge_branch_fixtures import knowledge_factory
from tests.test_knowledge_routing_graph import harness

__all__ = ["knowledge_factory"]


def test_http_retry_stream_and_refresh_share_v2_sources(
    knowledge_factory, tmp_path, monkeypatch
):
    import importlib

    main = importlib.import_module("resolv.main")

    h = harness(knowledge_factory, tmp_path)
    assert list(h.conversation.run(h.run.run_id))[-1].type == "done"

    @contextmanager
    def runtime():
        yield h.conversation

    monkeypatch.setattr(main, "routed_conversation_runtime", runtime)
    # The harness's first request was accepted under v2 before its fixture upgraded it;
    # use the completed replay endpoint to exercise the actual SSE projection.
    with TestClient(main.app) as client:
        result = client.get(f"/conversations/{h.run.conversation_id}")
        assert result.status_code == 200
        saved = next(r for r in result.json()["runs"] if r["run_id"] == h.run.run_id)
        assert saved["result_json"]["schema_version"] == 2
        assert saved["result_json"]["knowledge"]["used_citation_numbers"] == [1]
        stream = client.post(
            f"/conversations/{h.run.conversation_id}/runs/{h.run.run_id}/resume"
        )
        assert "event: message" in stream.text and '"schema_version":2' in stream.text
        from langchain_core.messages import AIMessage

        from tests.understanding_fixtures import make_classification, make_resolution

        question = "退货运费由谁承担？"
        h.conversation.understanding.resolver.model.client.replies.extend(
            [
                AIMessage(content=o.model_dump_json())
                for o in (
                    make_resolution(resolved_text=question),
                    make_classification([question], ["product_consultation"]),
                )
            ]
        )
        c = client.post("/conversations", json={}).json()
        reply = client.post(
            "/agent/chat",
            json={
                "session_id": c["conversation_id"],
                "request_id": "new-v3",
                "question": question,
            },
        )
        assert reply.status_code == 200, reply.text
        assert reply.json()["schema_version"] == 2 and reply.json()["knowledge"][
            "used_citation_numbers"
        ] == [1]
