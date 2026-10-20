"""Opt-in MySQL adapter gate. Fake model; no business DB or remote inference."""

import argparse
import json
import os
import traceback
from importlib.metadata import version
from typing import Annotated
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from sqlalchemy.engine import make_url
from typing_extensions import TypedDict

from resolv.core.config import ConversationDatabaseSettings
from resolv.db.checkpoints import (
    CheckpointError,
    _open_connection,
    checkpoint_session,
    setup_checkpoint_tables,
)


class GateState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: dict
    knowledge: list[dict]
    action: dict
    model_steps: int
    decision: bool


def _graph(saver):
    def model(state):
        return {
            "messages": [AIMessage(content="", tool_calls=[{
                "id": "fake-call", "name": "smoke_write", "args": {"order_id": "SMOKE001"},
            }])],
            "model_steps": state["model_steps"] + 1,
        }

    def stage(state, config):
        # The complete model response must already be durable before staging.
        root_config = {"configurable": {"thread_id": config["configurable"]["thread_id"], "checkpoint_ns": ""}}
        saved = saver.get_tuple(root_config).checkpoint["channel_values"]
        assert saved["model_steps"] == 1
        assert state["messages"][-1].tool_calls[0]["args"] == {"order_id": "SMOKE001"}
        return {"action": {"id": "fake-action", "arguments": {"order_id": "SMOKE001"}}}

    def await_approval(state):
        return {"decision": interrupt(state["action"])}

    def finish(state):
        assert state["decision"] is True
        return {"messages": [
            ToolMessage(content='{"executed":true}', tool_call_id="fake-call"),
            AIMessage(content="fake complete"),
        ]}

    graph = StateGraph(GateState)
    for name, node in [("model", model), ("stage", stage), ("await_approval", await_approval), ("finish", finish)]:
        graph.add_node(name, node)
    graph.add_edge(START, "model")
    graph.add_edge("model", "stage")
    graph.add_edge("stage", "await_approval")
    graph.add_edge("await_approval", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=saver)


def _evidence(scenario, **values):
    print(json.dumps({"scenario": scenario, "result": "PASS", **values}, ensure_ascii=False), flush=True)


def run_gate(settings, *, setup):
    database = make_url(settings.conversation_database_url.get_secret_value()).database
    versions = {name: version(name) for name in ("langgraph", "langgraph-checkpoint", "langgraph-checkpoint-mysql", "pymysql")}
    print(json.dumps({"database": database, "versions": versions}), flush=True)
    assert versions == {"langgraph": "1.2.12", "langgraph-checkpoint": "4.2.0", "langgraph-checkpoint-mysql": "3.0.0", "pymysql": "1.2.3"}
    namespace = "gate-" + uuid4().hex
    config = {"configurable": {"thread_id": namespace}}
    other_config = {"configurable": {"thread_id": namespace + "-other"}}
    if setup:
        setup_checkpoint_tables(settings)
        setup_checkpoint_tables(settings)
        _evidence("explicit_repeated_setup")
    initial = {
        "messages": [HumanMessage(content="fake request")],
        "summary": {"version": 2, "covered_until_seq": 4},
        "knowledge": [{"chunk_id": "smoke-chunk", "text": "完整政策", "source": "fake.md"}],
        "action": {}, "model_steps": 0, "decision": False,
    }
    with checkpoint_session(settings, conversation_id=namespace) as session:
        graph = _graph(session.saver)
        graph.invoke(initial, config, durability="sync")
        snapshot = graph.get_state(config)
        assert snapshot.next == ("await_approval",)
        assert snapshot.values["summary"] == initial["summary"]
        assert snapshot.values["knowledge"] == initial["knowledge"]
        assert isinstance(snapshot.values["messages"][0], HumanMessage)
        assert snapshot.values["messages"][-1].tool_calls[0]["id"] == "fake-call"
        saved = session.saver.get_tuple(config)
        session.saver.put_writes(saved.config, [("gate_marker", {"ok": True})], "fake-write-task")
        reread = session.saver.get_tuple(config)
        assert ("fake-write-task", "gate_marker", {"ok": True}) in reread.pending_writes
        assert session.saver.get_tuple(other_config) is None
        _evidence("sync_decision_message_json_put_writes_interrupt", thread_id=namespace, next=list(snapshot.next))
        try:
            with checkpoint_session(settings, conversation_id=namespace):
                raise AssertionError("second executor acquired live lock")
        except CheckpointError as exc:
            assert exc.code == "conversation_busy"
        session.assert_owned()
        _evidence("second_mysql_connection_cannot_takeover")
    with checkpoint_session(settings, conversation_id=namespace) as session:
        graph = _graph(session.saver)
        assert graph.get_state(config).next == ("await_approval",)
        restored = graph.invoke(Command(resume=True), config, durability="sync")
        assert restored["messages"][-1].content == "fake complete"
        assert restored["model_steps"] == 1
        assert graph.get_state(config).next == ()
        graph.invoke(initial, other_config, durability="sync")
        assert graph.get_state(other_config).next == ("await_approval",)
        assert graph.get_state(config).next == ()
        _evidence("closed_connection_rebuilt_graph_command_resume_thread_isolation")
    if setup:
        setup_checkpoint_tables(settings)
        with checkpoint_session(settings, conversation_id=namespace) as session:
            assert _graph(session.saver).get_state(config).values["messages"][-1].content == "fake complete"
        _evidence("setup_preserves_existing_checkpoint")
    with checkpoint_session(settings, conversation_id=namespace) as session:
        with session.serialized(), session._connection.cursor() as cursor:
            cursor.execute("SELECT CONNECTION_ID()")
            connection_id = cursor.fetchone()[0]
        killer = _open_connection(settings)
        try:
            with killer.cursor() as cursor:
                cursor.execute("KILL CONNECTION %s", (connection_id,))
        finally:
            killer.close()
        for operation in [session.assert_owned, lambda: session.saver.put_writes(saved.config, [("gate_marker", "must-not-save")], "lost-task")]:
            try:
                operation()
                raise AssertionError("lost connection continued execution")
            except CheckpointError:
                pass
        _evidence("killed_mysql_session_never_reconnects_or_writes")
    with checkpoint_session(settings, conversation_id=namespace) as session:
        assert all(task_id != "lost-task" for task_id, _, _ in session.saver.get_tuple(config).pending_writes)
    try:
        with checkpoint_session(settings, conversation_id=namespace):
            raise ValueError("fake node failure")
    except ValueError:
        pass
    with checkpoint_session(settings, conversation_id=namespace) as session:
        session.assert_owned()
    _evidence("failure_exit_releases_lock_and_closes_resources")
    print("MySQL checkpoint gate PASSED; resources closed.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-env", required=True)
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = ConversationDatabaseSettings(
            _env_file=None, conversation_database_url=os.environ.get(args.database_env, "")
        )
        database = make_url(settings.conversation_database_url.get_secret_value()).database
        if not database or not database.startswith("resolv_conversations_smoke_"):
            raise ValueError("smoke database required")
    except ValueError:
        raise SystemExit("Checkpoint smoke requires a valid isolated smoke database URL in the named environment variable.") from None
    try:
        run_gate(settings, setup=args.setup)
    except Exception as exc:  # noqa: BLE001 — safe diagnostics without credential-bearing exception text
        frames = traceback.extract_tb(exc.__traceback__)
        # Names/line numbers are safe; SQL, exception messages and local values are not.
        print(json.dumps({"gate": "FAILED", "exception_type": type(exc).__name__, "frames": [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in frames]}), flush=True)
        raise SystemExit("MySQL checkpoint gate FAILED; later slices must not proceed.") from None


if __name__ == "__main__":
    main()
