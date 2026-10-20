"""Preserve immutable floats across a JSON driver that changes a double by one ULP."""

import json
import math

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, event, select

from resolv.services.hybrid_fingerprints import canonical_hash
from tests.knowledge_branch_fixtures import knowledge_factory

__all__ = ["knowledge_factory"]


def lossy_native(value):
    if isinstance(value, float):
        return math.nextafter(value, -math.inf)
    if isinstance(value, list):
        return [lossy_native(v) for v in value]
    if isinstance(value, dict):
        return {k: lossy_native(v) for k, v in value.items()}
    return value


def test_exact_json_preserves_hash_while_native_double_changes(tmp_path):
    from resolv.db.exact_json import ExactJSON

    engine = create_engine(
        "sqlite:///" + str(tmp_path / "codec.db"),
        json_serializer=lambda value: json.dumps(lossy_native(value)),
    )
    table = Table(
        "codec",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("value", ExactJSON(none_as_null=True)),
    )
    table.metadata.create_all(engine)
    value = {
        "probability": 0.9919943477637045,
        "text": "原始证据\r\n",
        "nested": [-0.0, 1e-200, None],
    }
    assert canonical_hash(lossy_native(value)) != canonical_hash(value)
    try:
        with engine.begin() as connection:
            connection.execute(table.insert().values(id=1, value=value))
            restored = connection.scalar(select(table.c.value).where(table.c.id == 1))
            assert canonical_hash(restored) == canonical_hash(value)
            connection.execute(table.insert().values(id=2, value=None))
            assert (
                connection.exec_driver_sql(
                    "SELECT value IS NULL FROM codec WHERE id=2"
                ).scalar()
                == 1
            )
            legacy = json.dumps({"schema_version": 1, "branch": "business"})
            connection.exec_driver_sql(
                "INSERT INTO codec(id,value) VALUES (?,?)", (3, legacy)
            )
            assert connection.scalar(
                select(table.c.value).where(table.c.id == 3)
            ) == json.loads(legacy)
    finally:
        engine.dispose()


def test_reply_v1_storage_stays_native_and_v2_is_exact():
    from resolv.db.exact_json import ExactJSON

    codec = ExactJSON(only_reply_v2=True)
    old = {"schema_version": 1, "branch": "business"}
    assert codec.process_bind_param(old, None) == old
    new = {"schema_version": 2, "knowledge": {"score": 0.9919943477637045}}
    encoded = codec.process_bind_param(new, None)
    assert (
        encoded != new
        and codec.process_result_value(lossy_native(encoded), None) == new
    )
    assert codec.process_result_value(old, None) == old
    assert codec.process_bind_param(None, None) is None


def test_invalid_exact_json_envelope_is_rejected():
    from resolv.db.exact_json import ExactJSON

    codec = ExactJSON()
    encoded = codec.process_bind_param({"a": 1.0}, None)
    bad = {**encoded, "unexpected": "value"}
    with pytest.raises(ValueError):
        codec.process_result_value(bad, None)


@pytest.mark.parametrize("low", [False, True])
def test_real_graph_stages_reply_and_refresh_survive_native_json_float_coercion(
    knowledge_factory, tmp_path, low
):
    from tests.knowledge_branch_fixtures import hybrid_result
    from tests.test_knowledge_routing_graph import harness

    h = harness(
        knowledge_factory,
        tmp_path,
        hybrid_result=hybrid_result((-10.0,)) if low else None,
    )
    engine = knowledge_factory.kw["bind"]

    def coerce(conn, cursor, statement, parameters, context, executemany):
        def parameter(value):
            if isinstance(value, str) and value.startswith(("{", "[")):
                return json.dumps(lossy_native(json.loads(value)), ensure_ascii=False)
            return value

        if not executemany:
            if isinstance(parameters, tuple):
                parameters = tuple(parameter(v) for v in parameters)
            elif isinstance(parameters, dict):
                parameters = {k: parameter(v) for k, v in parameters.items()}
        return statement, parameters

    event.listen(engine, "before_cursor_execute", coerce, retval=True)
    try:
        events = list(h.conversation.run(h.run.run_id))
        assert events[-1].type == "done", [(e.type, e.data) for e in events]
        metadata = h.conversation.reply_metadata(h.run.run_id)
        detail = h.conversation.detail(h.run.conversation_id)
        assert (
            next(r.result_json for r in detail.runs if r.run_id == h.run.run_id)
            == metadata
        )
        if low:
            assert (
                metadata["reply_kind"] == "knowledge_fallback"
                and metadata["knowledge"]["citations"] == []
            )
        else:
            assert metadata["knowledge"]["citations"][0]["sigmoid_score"] == 1 / (
                1 + math.exp(-2.0)
            )
        assert "__resolv_exact_json__" not in json.dumps(metadata)
        assert (
            list(h.conversation.resume(h.run.conversation_id, h.run.run_id))[-1].type
            == "done"
        )

    finally:
        event.remove(engine, "before_cursor_execute", coerce)
