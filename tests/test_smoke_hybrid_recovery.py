import json

import pytest
from scripts import smoke_hybrid_recovery as module

from resolv.schemas.hybrid_knowledge import HybridRetrievalError


@pytest.mark.parametrize(
    "config",
    [
        {"database_prefix": "resolv_knowledge", "collection_prefix": "mhhybrid_gate_"},
        {
            "database_prefix": "mhhybrid_gate_",
            "collection_prefix": "resolv_knowledge_",
        },
        {
            "database_prefix": "mhhybrid_gate_",
            "collection_prefix": "mhhybrid_gate_",
            "database_name": "resolv_knowledge",
        },
    ],
)
def test_recovery_rejects_unowned_targets_before_writes(tmp_path, monkeypatch, config):
    sql_writes = milvus_writes = 0

    def forbidden(*a, **kw):
        nonlocal sql_writes, milvus_writes
        sql_writes += 1
        milvus_writes += 1
        pytest.fail("external write boundary reached")

    monkeypatch.setattr(module, "_run_owned_matrix", forbidden)
    with pytest.raises(HybridRetrievalError):
        module.run_recovery_matrix(config, tmp_path / "gate")
    assert sql_writes == milvus_writes == 0


def test_existing_private_manifest_must_match_requested_target(tmp_path, monkeypatch):
    target = tmp_path / "gate"
    target.mkdir()
    (target / "ownership.json").write_text(
        json.dumps(
            {
                "owner": "a" * 32,
                "database_name": "mhhybrid_gate_" + "a" * 32,
                "database_prefix": "mhhybrid_gate_",
                "collection_prefix": "mhhybrid_gate_",
                "version_bindings": {},
            }
        )
    )
    monkeypatch.setattr(
        module,
        "_run_owned_matrix",
        lambda *a, **kw: pytest.fail("write boundary reached"),
    )
    with pytest.raises(HybridRetrievalError):
        module.run_recovery_matrix(
            {
                "database_prefix": "mhhybrid_gate_",
                "collection_prefix": "mhhybrid_gate_",
                "database_name": "mhhybrid_gate_" + "b" * 32,
            },
            target,
        )


def test_fresh_target_collision_refused_before_create():
    writes = []

    class Admin:
        def scalar(self, *a, **kw):
            return "existing_database"

    class Client:
        def has_collection(self, **kw):
            return False

        def create_collection(self, **kw):
            writes.append(kw)

    with pytest.raises(HybridRetrievalError):
        module.assert_fresh_targets(
            Admin(),
            Client(),
            {"database_name": "mhhybrid_gate_" + "a" * 32, "collections": []},
        )
    assert writes == []


def test_unknown_collection_owner_is_refused():
    class Client:
        def describe_collection(self, **kw):
            return {"description": "unowned"}

    with pytest.raises(HybridRetrievalError):
        module.assert_owned_collection(Client(), "mhhybrid_gate_fake", "known-marker")


def test_child_fault_nonzero_is_required_and_no_success_json():
    assert module.validate_child_result(
        {"returncode": 73, "stdout": "", "stderr": "private"}, expect_fault=True
    )["fault_observed"]
    for result in (
        {"returncode": 0, "stdout": "{}"},
        {"returncode": 73, "stdout": '{"passed": true}'},
    ):
        with pytest.raises(HybridRetrievalError):
            module.validate_child_result(result, expect_fault=True)


def test_help_is_resource_free(monkeypatch, capsys):
    monkeypatch.setattr(
        module, "run_recovery_matrix", lambda *a, **kw: pytest.fail("created resources")
    )
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "--database-prefix" in capsys.readouterr().out


def test_command_failure_sanitized_no_success_json(tmp_path, monkeypatch, capsys):
    def failure(*a, **kw):
        raise RuntimeError("mysql://root:private@host source original text")

    monkeypatch.setattr(module, "run_recovery_matrix", failure)
    target = tmp_path / "gate"
    with pytest.raises(SystemExit):
        module.main(
            [
                "--database-prefix",
                "mhhybrid_gate_",
                "--collection-prefix",
                "mhhybrid_gate_",
                "--report-dir",
                str(target),
            ]
        )
    assert capsys.readouterr().out == ""
    data = (target / "command-failure.json").read_text()
    assert (
        "private" not in data and "original text" not in data and "mysql://" not in data
    )


def test_child_nonzero_without_reached_fault_marker_is_not_gate_evidence(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=73, stdout="", stderr=""),
    )
    with pytest.raises(HybridRetrievalError, match="gate_child_fault_marker_missing"):
        module._child(tmp_path, "gate_case", "before_upsert")


def test_private_manifest_cannot_claim_formal_collection(tmp_path, monkeypatch):
    target = tmp_path / "gate"
    target.mkdir()
    proof = {
        "owner": "a" * 32,
        "database_name": "mhhybrid_gate_" + "a" * 32,
        "database_prefix": "mhhybrid_gate_",
        "collection_prefix": "mhhybrid_gate_",
        "version_bindings": {},
        "collections": ["resolv_knowledge_v2"],
    }
    (target / "ownership.json").write_text(json.dumps(proof))
    monkeypatch.setattr(
        module,
        "_run_owned_matrix",
        lambda *a, **kw: pytest.fail("write boundary reached"),
    )
    with pytest.raises(HybridRetrievalError):
        module.run_recovery_matrix(
            {
                "database_prefix": "mhhybrid_gate_",
                "collection_prefix": "mhhybrid_gate_",
            },
            target,
        )


def test_resume_recognizes_only_recorded_intentional_ready_collection_loss():
    from resolv.schemas.hybrid_knowledge import HybridIndexProfile
    from resolv.services.hybrid_fingerprints import profile_hash

    p = HybridIndexProfile(collection_name="mhhybrid_gate_" + "a" * 32 + "_lost")
    owned = {
        "ready_loss_probe": {
            "version_id": "gate_lost",
            "collection": p.collection_name,
            "profile_hash": profile_hash(p),
        }
    }
    assert (
        module.skip_intentional_loss(owned, "gate_lost", p, collection_exists=False)
        is True
    )
    assert (
        module.skip_intentional_loss(owned, "gate_lost", p, collection_exists=True)
        is False
    )
    assert (
        module.skip_intentional_loss({}, "gate_lost", p, collection_exists=False)
        is False
    )
    owned["ready_loss_probe"]["profile_hash"] = "f" * 64
    with pytest.raises(HybridRetrievalError):
        module.skip_intentional_loss(owned, "gate_lost", p, collection_exists=False)


@pytest.mark.parametrize('case', ['empty', 'unscored', 'missing_tokens'])
def test_end_to_end_gate_requires_positive_matching_inference_evidence(case):
    from types import SimpleNamespace

    diagnostics = SimpleNamespace(dense_count=50, bm25_count=50, center_count=1,
        windows_after_merge=1, scored_count=1, selected_count=1,
        query_tokens=19, pair_tokens=(312,))
    answer = SimpleNamespace(diagnostics=diagnostics,
        windows=(SimpleNamespace(rerank_score=1.0, chunk_ids=('a',)),), chunks={'a':object()})
    if case == 'empty':
        answer.windows=(); answer.chunks={}
        diagnostics.center_count=diagnostics.windows_after_merge=diagnostics.scored_count=diagnostics.selected_count=0
        diagnostics.pair_tokens=()
    elif case == 'unscored': diagnostics.scored_count=0
    else: diagnostics.pair_tokens=()
    with pytest.raises(HybridRetrievalError):
        module.validate_end_to_end(answer)


def test_existing_native_collection_fingerprint_accepts_locked_protobuf_lists():
    from pymilvus.grpc_gen import schema_pb2

    fn=schema_pb2.FunctionSchema(input_field_names=['text'], input_field_ids=[105],
                               output_field_names=['bm25_vector'], output_field_ids=[107])
    class Existing:
        def describe_collection(self, **kw):
            return {'functions':[{'input_field_names':fn.input_field_names,
                'input_field_ids':fn.input_field_ids, 'output_field_names':fn.output_field_names,
                'output_field_ids':fn.output_field_ids}]}
        def get_collection_stats(self, **kw): return {'row_count':80}
    client=Existing()
    before=module._milvus_fingerprint(client,['existing_native_bm25'])
    assert before==module._milvus_fingerprint(client,['existing_native_bm25'])
    fn.output_field_ids[0]=108
    assert before!=module._milvus_fingerprint(client,['existing_native_bm25'])
