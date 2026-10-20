"""R3 must prove actual reached process exits, durable values and fences."""

import copy
import importlib

import pytest


def runner():
    assert importlib.util.find_spec("scripts.smoke_refund_flow_recovery"), (
        "missing actual refund recovery runner"
    )
    return importlib.import_module("scripts.smoke_refund_flow_recovery")


def report():
    m = runner()
    return {
        "windows": [
            {
                "window": w,
                "reached": True,
                "exit_code": 86,
                "resume_exit_code": 0,
                "saved_hashes_preserved": True,
                "saved_calls_reused": True,
                "assistant_count": 1,
                "offer_count": 1,
                "application_count": 1,
                "receipt_count": 1,
                "receipt_replayed": True,
                "original_run_preserved": True,
            }
            for w in m.WINDOWS
        ],
        "probes": dict.fromkeys(m.PROBES, True),
        "real_model_calls": 0,
        "formal_preserved": True,
    }


def test_recovery_requires_reached_exit_and_every_window():
    m = runner()
    assert len(m.WINDOWS) == len(set(m.WINDOWS)) == 28
    m.validate_recovery_report(report())
    for field, value in [
        ("reached", False),
        ("exit_code", 0),
        ("saved_hashes_preserved", False),
        ("saved_calls_reused", False),
        ("application_count", 2),
        ("receipt_replayed", False),
        ("original_run_preserved", False),
    ]:
        bad = report()
        bad["windows"][0][field] = value
        with pytest.raises(ValueError):
            m.validate_recovery_report(bad)
    bad = report()
    bad["windows"].pop()
    with pytest.raises(ValueError):
        m.validate_recovery_report(bad)


def test_recovery_requires_positive_and_negative_epoch_probes():
    m = runner()
    for field in m.PROBES:
        bad = report()
        bad["probes"][field] = False
        with pytest.raises(ValueError):
            m.validate_recovery_report(bad)
    bad = copy.deepcopy(report())
    bad["real_model_calls"] = 1
    with pytest.raises(ValueError):
        m.validate_recovery_report(bad)


@pytest.mark.parametrize("failed_snapshot", ["before", "after"])
def test_recovery_snapshot_failure_keeps_a_failed_receipt_without_private_errors(
    tmp_path, monkeypatch, failed_snapshot
):
    import json
    from contextlib import contextmanager

    m = runner()
    target = tmp_path / "recovery"
    target.mkdir()
    monkeypatch.setattr(m, "new_report_dir", lambda *args: target)
    snapshots, windows, owned = [], [], []
    baseline = {"formal": "unchanged"}

    def snapshot(**kwargs):
        phase = "after" if kwargs else "before"
        snapshots.append(phase)
        if phase == failed_snapshot:
            raise RuntimeError("private-provider-password-must-not-appear")
        return baseline

    @contextmanager
    def resources(_):
        owned.append(True)
        yield object()

    def window(_resources, _target, name, index, ownership):
        windows.append(name)
        return report()["windows"][index - 1]

    monkeypatch.setattr(m, "snapshot_formal_resources", snapshot)
    monkeypatch.setattr(m, "owned_refund_resources", resources)
    monkeypatch.setattr(m, "run_window", window)
    monkeypatch.setattr(m, "storage_probes", lambda *args: report()["probes"])
    result = m.run_recovery_matrix(ownership=tmp_path / "owner", report_dir=target)

    assert result["passed"] is False and result["formal_preserved"] is False
    assert result["error_code"] == "recovery_snapshot_failed"
    assert result["error_type"] == "RuntimeError"
    assert result["failure_stage"] == "formal_snapshot_" + failed_snapshot
    assert result["real_model_calls"] == 0
    assert json.loads((target / "report.json").read_text()) == result
    assert "private-provider-password" not in (target / "report.json").read_text()
    assert not (target / "preservation-after.json").exists()
    if failed_snapshot == "before":
        assert snapshots == ["before"] and owned == windows == []
        assert result["windows"] == [] and result["probes"] == {}
        assert not (target / "preservation-before.json").exists()
    else:
        assert snapshots == ["before", "after"] and owned == [True]
        assert windows == list(m.WINDOWS)
        assert all(result["probes"].values())
        assert json.loads((target / "preservation-before.json").read_text()) == baseline
    with pytest.raises(ValueError):
        m.validate_recovery_report(result)


def test_recovery_assembly_drives_actual_graph_and_pending_receipt(tmp_path):
    from types import SimpleNamespace

    from resolv.schemas.refund_submission import RefundSubmitInput
    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.refund_flow_fixtures import flow_case
    from tests.refund_submission_fixtures import LocalLocks

    m = runner()
    with flow_case(tmp_path) as h:
        resources = SimpleNamespace(
            conversation_factory=h.factory, business_factory=h.business
        )
        state = {
            "order_id": "R30001",
            "window": "assembly",
            "calls": str(tmp_path / "calls.jsonl"),
            "selector": str(tmp_path / "selector.json"),
        }
        service = m.make_service(
            resources,
            state,
            tmp_path / "deps",
            checkpoint_factory=MemoryCheckpoints(),
            local_locks=LocalLocks(),
        )
        c = service.create()
        run, _ = service.accept(
            c.conversation_id,
            "assembly",
            "订单R30001申请整单退款，因为商品未激活",
            "agent",
        )
        out = list(service.run(run.run_id))
        assert out[-1].type == "done", [(x.type, x.data) for x in out]
        saved = service.get_run(run.run_id)
        offer_id = saved.result_json["refund"]["offer_id"]
        assert offer_id
        view = service.refund_runtime.submissions.submit(
            c.conversation_id,
            offer_id,
            RefundSubmitInput(request_id="assembly-submit", confirm=True),
            {},
        )
        assert view.receipt.status == "pending"
        state.update(run_id=run.run_id, conversation_id=c.conversation_id)
        snap = m.snapshot(service, state)
        assert snap["receipt_count"] == snap["application_count"] == 1
        assert snap["hashes"]["decision"] and snap["observations"]
        service._fixture_cleanup.close()


def test_storage_probes_use_measured_boundaries(tmp_path):
    from types import SimpleNamespace

    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.refund_flow_fixtures import flow_case
    from tests.refund_submission_fixtures import LocalLocks

    m = runner()
    with flow_case(tmp_path) as h:
        resources = SimpleNamespace(
            conversation_factory=h.factory, business_factory=h.business
        )
        result = m.storage_probes(
            resources,
            tmp_path / "probes",
            checkpoint_factory=MemoryCheckpoints(),
            local_locks=LocalLocks(),
        )
        assert all(result.get(p) is True for p in m.PROBES)


def test_http_snapshot_requires_same_verified_commit_and_owned_export(tmp_path):
    m = runner()
    assert hasattr(m, "frontend_dist"), "missing exact exported frontend boundary"
    with pytest.raises(ValueError):
        m.frontend_dist(tmp_path / "report.json")


def test_review_fixed_frontend_requires_verified_same_source_export(
    tmp_path, monkeypatch
):
    import json
    from hashlib import sha256

    m = runner()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    root = tmp_path / m.WORKSPACE / "snapshot-review-fixed-01"
    frontend = root / "frontend/frontend"
    (frontend / "src").mkdir(parents=True)
    (frontend / "src/test.ts").write_text("verified source")
    (frontend / "dist").mkdir()
    (frontend / "dist/index.html").write_text("built source")
    commit, tree = "a" * 40, "b" * 40
    report = {
        "passed": True,
        "candidate_commit": commit,
        "tree": tree,
        "python": {"revision": commit, "tree": tree},
        "frontend": {
            "revision": commit,
            "tree": tree,
            "passed": True,
            "checks": dict.fromkeys(["test", "typecheck", "build", "lint"], 0),
            "source_fingerprints": {
                "frontend/src/test.ts": sha256(b"verified source").hexdigest()
            },
        },
    }
    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **kw: type("Result", (), {"returncode": 0})()
    )
    path = root / "report.json"
    path.write_text(json.dumps(report))
    assert m.frontend_dist(path)[0] == frontend / "dist"
    report["frontend"]["checks"]["build"] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        m.frontend_dist(path)
    report["frontend"]["checks"]["build"] = 0
    path.write_text(json.dumps(report))
    (frontend / "src/test.ts").write_text("changed source")
    with pytest.raises(ValueError):
        m.frontend_dist(path)


def test_snapshot_checks_source_but_not_generated_incremental_cache():
    m = runner()
    assert m.frontend_source_required("frontend/src/App.tsx")
    assert not m.frontend_source_required("frontend/tsconfig.app.tsbuildinfo")


@pytest.mark.parametrize("fail_publish", [False, True])
def test_owned_http_composition_serves_production_routes_without_real_models(
    tmp_path, fail_publish
):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.refund_flow_fixtures import flow_case
    from tests.refund_submission_fixtures import LocalLocks

    m = runner()
    with flow_case(tmp_path) as h:
        resources = SimpleNamespace(
            identity={"owner": "a" * 32},
            conversation_factory=h.factory,
            business_factory=h.business,
        )
        target = tmp_path / "http"
        target.mkdir()
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("verified export fixture")
        app, _nonce = m.make_http_app(
            resources,
            target,
            dist,
            checkpoint_factory=MemoryCheckpoints(),
            local_locks=LocalLocks(),
        )
        with TestClient(app) as client:
            c = client.post("/api/conversations").json()
            response = client.post(
                "/api/agent/chat",
                json={
                    "session_id": c["conversation_id"],
                    "request_id": "r",
                    "question": "订单R3UI001申请整单退款，因为商品未激活",
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["refund"]["offer_id"]
            offer = response.json()["refund"]["offer_id"]
            opened = client.get(
                f"/api/conversations/{c['conversation_id']}/refund-offers/{offer}"
            )
            assert opened.status_code == 200, opened.text
            assert client.get("/api/health").status_code == 200
            app.state.refund_wait_seconds = 5
            if fail_publish:
                assert (
                    client.post(
                        f"/__refund_fixture/{_nonce}/fail_publish_once"
                    ).status_code
                    == 200
                )
            submitted = client.post(
                f"/api/conversations/{c['conversation_id']}/refund-offers/{offer}/submit",
                json={"request_id": "http-submit", "confirm": True},
            )
            assert submitted.status_code in (200, 202), submitted.text
            command = submitted.json()["submission_id"]
            if fail_publish:
                assert submitted.json()["status"] == "interrupted", submitted.text
                receipt = submitted.json()["receipt"]
                assert receipt and submitted.json()["retryable"]
                resumed = client.post(
                    f"/api/conversations/{c['conversation_id']}/refund-submissions/{command}/resume",
                    json={},
                )
                assert resumed.json()["status"] == "completed", resumed.text
                assert resumed.json()["receipt"] == receipt
            app.state.refund_workers.close()
            replay = client.get(
                f"/api/conversations/{c['conversation_id']}/refund-submissions/{command}"
            )
            assert replay.json()["receipt"]["status"] == "pending", replay.text
            assert client.get("/").text == "verified export fixture"


def test_http_reassessment_runs_new_refund_in_same_conversation(tmp_path):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.refund_flow_fixtures import flow_case
    from tests.refund_submission_fixtures import LocalLocks

    m = runner()
    with flow_case(tmp_path) as h:
        resources = SimpleNamespace(
            identity={"owner": "a" * 32},
            conversation_factory=h.factory,
            business_factory=h.business,
        )
        target = tmp_path / "http"
        target.mkdir()
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("verified")
        app, _nonce = m.make_http_app(
            resources,
            target,
            dist,
            checkpoint_factory=MemoryCheckpoints(),
            local_locks=LocalLocks(),
        )
        with TestClient(app) as client:
            c = client.post("/api/conversations").json()["conversation_id"]
            first = client.post(
                "/api/agent/chat",
                json={
                    "session_id": c,
                    "request_id": "first",
                    "question": "订单R3UI002申请整单退款，因为商品未激活",
                },
            )
            assert first.status_code == 200, first.text
            offer = first.json()["refund"]["offer_id"]
            assert offer
            second = client.post(
                "/api/agent/chat",
                json={
                    "session_id": c,
                    "request_id": "second",
                    "question": "订单R3UI002申请整单退款，因为商品未激活且不再需要",
                    "reassesses_refund_offer_id": offer,
                },
            )
            assert second.status_code == 200, second.text
            assert second.json()["status"] == "completed", second.text
            assert second.json()["refund"]["offer_id"] != offer


def test_corrected_recovery_preserves_original_and_requires_source_change(
    tmp_path, monkeypatch
):
    import json

    m = runner()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    public = tmp_path / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    public.mkdir(parents=True)
    original = public / "R3.json"
    original.write_text(
        json.dumps(
            report()
            | {
                "gate": "R3",
                "passed": True,
                "candidate_commit": "a" * 40,
                "source_fingerprints": {"a": "a" * 64},
            }
        )
    )
    assert hasattr(m, "recovery_destination"), (
        "missing immutable correction report boundary"
    )
    destination, lineage = m.recovery_destination(original, "b" * 40, {"a": "b" * 64})
    assert destination.name == "R3-corrected.json" and lineage
    with pytest.raises(ValueError):
        m.recovery_destination(original, "a" * 40, {"a": "a" * 64})
    destination.write_text("{}")
    with pytest.raises(ValueError):
        m.recovery_destination(original, "b" * 40, {"a": "b" * 64})


def test_recovery_request_aliases_are_scoped_to_report():
    m = runner()
    assert hasattr(m, "submission_request_id"), "missing report-scoped request alias"
    a = m.submission_request_id({"scope": "a" * 12, "window": m.WINDOWS[0]})
    b = m.submission_request_id({"scope": "b" * 12, "window": m.WINDOWS[0]})
    assert (
        a != b
        and len(a) <= 128
        and a == m.submission_request_id({"scope": "a" * 12, "window": m.WINDOWS[0]})
    )


def test_failed_corrected_report_can_only_continue_with_changed_commit(
    tmp_path, monkeypatch
):
    import json

    m = runner()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    public = tmp_path / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    public.mkdir(parents=True)
    previous = public / "R3-corrected.json"
    previous.write_text(
        json.dumps(
            {
                "gate": "R3",
                "passed": False,
                "windows": [],
                "candidate_commit": "a" * 40,
                "source_fingerprints": {"runner": "a" * 64},
                "formal_preserved": True,
            }
        )
    )
    destination, lineage = m.recovery_destination(
        previous, "b" * 40, {"runner": "b" * 64}
    )
    assert destination.name == "R3-corrected-2.json" and lineage
