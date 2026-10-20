"""Immutable committed package and all-input preflight, without real services."""

import subprocess
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import func, select

from resolv.models.knowledge import KnowledgeDocument
from resolv.services import knowledge_release as release
from tests.test_knowledge_hybrid_repository import hybrid_factory

__all__ = ["hybrid_factory"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def git_release(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()

    git("init", "-q")
    for name in (*release.POLICY_PATHS, release.CONFIG_PATH):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    for name in release.SOURCE_TREES:
        path = root / name
        if "." not in Path(name).name:
            path = path / "fixture.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("committed fixture source")
    (root / "src/fixture.py").write_text("pass\n")
    git("add", "--", *release.POLICY_PATHS, release.CONFIG_PATH, *release.SOURCE_TREES)
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "policy",
    )
    monkeypatch.setattr(release, "ROOT", root)
    return root, git("rev-parse", "HEAD"), git


def package(git_release, label="package"):
    root, commit, _ = git_release
    out = root / release.WORKSPACE / label
    return out, release.prepare_release(commit, output_dir=out)


def test_package_uses_only_six_committed_sources_and_can_be_rebuilt(git_release):
    root, commit, _ = git_release
    original = (root / release.POLICY_PATHS[0]).read_bytes()
    (root / release.POLICY_PATHS[0]).write_text("uncommitted replacement")
    out, manifest = package(git_release)
    assert len(manifest.documents) == 6
    assert manifest.policy_commit == commit
    assert manifest.documents[0].raw_sha256 == sha256(original).hexdigest()
    assert all(
        d.source.startswith("knowledge/releases/knowledge-20261015-v1/")
        and not d.source.endswith("/refund.md")
        for d in manifest.documents
    )
    loaded, docs, corpus = release.load_release(out / "manifest.json")
    assert loaded == manifest and len(docs) == len(corpus.documents) == 6
    assert docs[0].text != "uncommitted replacement"
    _, second = package(git_release, "rebuilt")
    assert second == manifest
    with pytest.raises(ValueError):
        release.prepare_release(commit, output_dir=out)
    with pytest.raises(ValueError):
        release.prepare_release(commit, output_dir=root / "outside")


def test_package_detects_modified_bytes_or_manifest_mapping(git_release):
    out, manifest = package(git_release)
    (out / manifest.documents[-1].source).write_bytes(b"changed")
    with pytest.raises(ValueError):
        release.load_release(out / "manifest.json")


def test_package_rejects_symlink_blob_and_uncommitted_revision(git_release):
    root, _commit, git = git_release
    with pytest.raises(ValueError):
        release.prepare_release("HEAD", output_dir=root / release.WORKSPACE / "bad-ref")
    p = root / release.POLICY_PATHS[0]
    p.unlink()
    p.symlink_to("member-benefits.md")
    git("add", "--", release.POLICY_PATHS[0])
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "symlink",
    )
    with pytest.raises(ValueError):
        release.prepare_release(
            git("rev-parse", "HEAD"), output_dir=root / release.WORKSPACE / "symlink"
        )


def test_all_sources_preflight_before_first_write(git_release, hybrid_factory):
    from resolv.schemas.knowledge import KnowledgeIndexProfile
    from resolv.services.knowledge import KnowledgeIngestionService

    factory, _, _ = hybrid_factory
    out, manifest = package(git_release)
    _, docs, _ = release.load_release(out / "manifest.json")
    old = docs[-1].model_copy(update={"text": "conflicting existing content"})
    KnowledgeIngestionService(factory).stage_document(
        old,
        manifest.documents[-1].file_config,
        KnowledgeIndexProfile(model_revision=manifest.index_profile.model_revision),
    )

    class Store:
        def has_collection(self):
            return False

    before = None
    with factory() as session:
        before = session.scalar(select(func.count()).select_from(KnowledgeDocument))
    with pytest.raises(ValueError):
        release.preflight_release(factory, manifest, docs, Store())
    with factory() as session:
        assert (
            session.scalar(select(func.count()).select_from(KnowledgeDocument))
            == before
        )
        assert session.get(KnowledgeDocument, docs[0].document_id) is None


def test_unbound_existing_collection_rejected_before_staging(
    git_release, hybrid_factory
):
    factory, _, _ = hybrid_factory
    out, manifest = package(git_release)
    _, docs, _ = release.load_release(out / "manifest.json")

    class Store:
        def has_collection(self):
            return True

    with pytest.raises(ValueError, match="collection"):
        release.preflight_release(factory, manifest, docs, Store())


def test_incomplete_or_fabricated_acceptance_never_changes_selector(
    git_release, tmp_path
):
    import json

    out, _ = package(git_release)
    selector = tmp_path / "active.json"
    selector.write_text('{"enabled":false,"config_version":1}')
    original = selector.read_bytes()
    acceptance = out / "acceptance.json"
    acceptance.write_text(json.dumps({"passed": True}))
    with pytest.raises(ValueError):
        release.activate_release(
            out / "manifest.json",
            acceptance,
            selector_path=selector,
            expected_previous_hash=sha256(original).hexdigest(),
        )
    assert selector.read_bytes() == original


def test_atomic_selector_refuses_stale_or_unknown_old_state(tmp_path):
    path = tmp_path / "selector.json"
    release.write_selector(
        path, {"enabled": False, "config_version": 1}, expected_previous_hash="absent"
    )
    original = path.read_bytes()
    with pytest.raises(ValueError):
        release.write_selector(
            path,
            {"enabled": False, "config_version": 1},
            expected_previous_hash="absent",
        )
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        release.write_selector(
            path,
            {"enabled": False, "config_version": 1},
            expected_previous_hash="0" * 64,
        )
    release.write_selector(
        path,
        {"enabled": False, "config_version": 1},
        expected_previous_hash=sha256(original).hexdigest(),
    )
    path.write_text("{broken")
    with pytest.raises(ValueError):
        release.write_selector(
            path,
            {"enabled": False, "config_version": 1},
            expected_previous_hash=sha256(path.read_bytes()).hexdigest(),
        )


def test_existing_version_different_corpus_is_not_replaced(git_release, hybrid_factory):
    from resolv.repositories.knowledge_hybrid import HybridIndexRepository

    factory, _, previous_corpus = hybrid_factory
    out, manifest = package(git_release)
    _, docs, _ = release.load_release(out / "manifest.json")
    with factory.begin() as session:
        HybridIndexRepository(session).create_version(
            manifest.index_version, manifest.index_profile, previous_corpus
        )

    class Store:
        def has_collection(self):
            return True

    with pytest.raises(ValueError, match="version conflict"):
        release.preflight_release(factory, manifest, docs, Store())
    with factory() as session:
        assert session.get(KnowledgeDocument, docs[0].document_id) is None


def test_publish_preflights_last_conflict_before_staging_first(
    git_release, hybrid_factory, monkeypatch
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from pydantic import SecretStr

    from resolv.services.knowledge import KnowledgeIngestionService

    factory, _, _ = hybrid_factory
    out, manifest = package(git_release)
    _, docs, _ = release.load_release(out / "manifest.json")
    KnowledgeIngestionService(factory).stage_document(
        docs[-1].model_copy(update={"text": "conflict"}),
        manifest.documents[-1].file_config,
        release.staging_profile(manifest),
    )

    @contextmanager
    def owned(value):
        yield value

    class Client:
        def has_collection(self, **kwargs):
            return False

    monkeypatch.setattr(
        release,
        "KnowledgeDatabaseSettings",
        lambda: SimpleNamespace(
            knowledge_database_url=SecretStr(
                "mysql+pymysql://fixture@localhost/resolv_knowledge"
            )
        ),
    )
    monkeypatch.setattr(
        release, "knowledge_engine", lambda _: owned(factory.kw["bind"])
    )
    monkeypatch.setattr(release, "hybrid_index_lock", lambda *a: owned(None))
    monkeypatch.setattr(release, "milvus_client", lambda _: owned(Client()))
    monkeypatch.setattr(release, "MilvusSettings", lambda: None)
    monkeypatch.setattr(
        release,
        "database_identity",
        lambda *a: {"database": "resolv_knowledge", "server_fingerprint": "f" * 64},
    )
    monkeypatch.setattr(
        KnowledgeIngestionService,
        "stage_document",
        lambda *a, **kw: pytest.fail("publication staged before complete preflight"),
    )
    with pytest.raises(ValueError):
        release.publish_release(
            out / "manifest.json",
            expected_database="resolv_knowledge",
            version=manifest.index_version,
            collection=manifest.collection,
            report_dir=git_release[0] / release.WORKSPACE / "build-conflict",
        )
    assert not (
        git_release[0] / release.WORKSPACE / "build-conflict/receipt.json"
    ).exists()


def test_ready_check_failure_survives_cleanup_failure(monkeypatch):
    class Pool:
        def validate_selection(self, selection):
            raise ValueError("primary ready check failed")

        def close(self):
            raise OSError("cleanup failed")

    monkeypatch.setattr(release, "default_knowledge_retrievers", Pool)
    with pytest.raises(ValueError, match="primary ready check failed"):
        release.verify_ready_selection(None)


def acceptance_bundle(git_release, out, manifest, corpus):
    import json

    from resolv.schemas.knowledge_release import GateReceipt, ReleaseAcceptance

    _, commit, git = git_release
    reports = out / "reports"
    reports.mkdir()
    source = {
        name: sha256(release._git("show", commit + ":" + name)).hexdigest()
        for name in git("ls-tree", "-r", "--name-only", commit).splitlines()
    }
    receipts = []
    for gate in ReleaseAcceptance.required_gates:
        report = {"gate": gate, "passed": True, "source_fingerprints": source}
        if gate == "K1":
            report.update(
                business_tool_calls=0,
                calls=[
                    {
                        "purpose": purpose,
                        "tool_calls": 0,
                        "structure_valid": True,
                        "input_limit": 12000,
                        "output_limit": 2000,
                        "audit": {
                            "outcome": "succeeded",
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "input_estimate": 500,
                        },
                    }
                    for purpose in [
                        "knowledge_query",
                        "knowledge_check",
                        "knowledge_answer",
                    ]
                ],
            )
        elif gate == "K2":
            from tests.test_smoke_knowledge_branch import formal_report, positive_report

            isolated = positive_report()
            formal = formal_report()
            formal["release"] = {
                "profile_hash": release.profile_hash(manifest.index_profile),
                "manifest_hash": release.manifest_hash(corpus),
                "release_manifest_hash": manifest.fingerprint,
            }

            report.update(
                knowledge_scope="combined",
                real_services=True,
                business_tool_calls=0,
                reports=[isolated, formal],
                release=formal["release"],
            )
        elif gate == "K3":
            from tests.test_smoke_knowledge_branch_recovery import (
                report as recovery_report,
            )

            report.update(recovery_report())
            report["source_fingerprints"] = source
        elif gate == "review":
            review_doc = (
                release.ROOT
                / "docs/superpowers/plans/2026-10-15-resolv-knowledge-branch-review.md"
            )
            review_doc.parent.mkdir(parents=True, exist_ok=True)
            review_doc.write_text("synthetic-fixture-only " + commit)
            report.update(
                candidate_commit=commit,
                important_count=0,
                reviewer_id="synthetic-fixture-only",
                review_document_sha256=sha256(review_doc.read_bytes()).hexdigest(),
            )
        else:
            report.update(revision=commit, tree=git("rev-parse", commit + "^{tree}"))
            if gate == "snapshot_python":
                report.update(
                    checks={
                        key: 0
                        for key in (
                            "negative_probe",
                            "source_import",
                            "pytest",
                            "ruff_files",
                            "ruff",
                        )
                    },
                    expected_python_files=1,
                    ruff_discovered=1,
                    network_attempts=0,
                    negative_probe_attempts=2,
                    source_import_verified=True,
                    lint_inputs_verified=True,
                    pytest_summary="1 passed",
                )
            else:
                commands = [
                    ["npm", "test"],
                    ["npm", "run", "typecheck"],
                    ["npm", "run", "build"],
                    ["npm", "run", "lint"],
                ]
                report.update(
                    checks={c[-1]: 0 for c in commands},
                    steps=[{"command": c, "exit_code": 0} for c in commands],
                )
        raw = json.dumps(report).encode()
        (reports / (gate + ".json")).write_bytes(raw)
        receipts.append(
            GateReceipt(
                name=gate,
                passed=True,
                report_sha256=sha256(raw).hexdigest(),
                source_fingerprints=tuple(source.items()),
            )
        )
    acceptance = ReleaseAcceptance(
        candidate_commit=commit,
        policy_commit=commit,
        source_trees=tuple(
            (name, git("rev-parse", commit + ":" + name))
            for name in release.SOURCE_TREES
        ),
        release_id=manifest.release_id,
        index_version=manifest.index_version,
        profile_hash=release.profile_hash(manifest.index_profile),
        manifest_hash=release.manifest_hash(corpus),
        gate_receipts=tuple(receipts),
        review_important_count=0,
    )
    path = out / "acceptance.json"
    path.write_text(acceptance.model_dump_json())
    return path, acceptance


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "hash",
        "profile",
        "candidate",
        "code",
        "report-source",
        "fake-report",
        "important",
    ],
)
def test_acceptance_binds_gates_to_exact_candidate_and_release(git_release, mutation):
    import json

    out, manifest = package(git_release)
    _, _, corpus = release.load_release(out / "manifest.json")
    path, acceptance = acceptance_bundle(git_release, out, manifest, corpus)
    raw = json.loads(path.read_text())
    if mutation == "missing":
        raw["gate_receipts"].pop()
    elif mutation == "hash":
        raw["gate_receipts"][0]["report_sha256"] = "0" * 64
    elif mutation == "profile":
        raw["profile_hash"] = "0" * 64
    elif mutation == "candidate":
        raw["candidate_commit"] = "0" * 40
    elif mutation == "code":
        raw["source_trees"][0][1] = "0" * 40
    elif mutation == "report-source":
        raw["gate_receipts"][0]["source_fingerprints"][0][1] = "0" * 64
    elif mutation == "important":
        raw["review_important_count"] = 1
    elif mutation == "fake-report":
        gate = raw["gate_receipts"][0]["name"]
        report = out / "reports" / f"{gate}.json"
        fake = {
            "gate": gate,
            "passed": True,
            "source_fingerprints": dict(
                acceptance.gate_receipts[0].source_fingerprints
            ),
        }
        report.write_text(json.dumps(fake))
        raw["gate_receipts"][0]["report_sha256"] = sha256(
            report.read_bytes()
        ).hexdigest()
    path.write_text(json.dumps(raw))
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        release.verify_acceptance(manifest, corpus, path)


def test_activation_checks_resources_before_atomic_write_and_explicit_disable(
    git_release, monkeypatch
):
    out, manifest = package(git_release)
    _, _, corpus = release.load_release(out / "manifest.json")
    path, _ = acceptance_bundle(git_release, out, manifest, corpus)
    selector = git_release[0] / "knowledge/active-release.json"
    called = []

    def ready(selection):
        called.append(selection)

    monkeypatch.setattr(release, "verify_ready_selection", ready)
    selected = release.activate_release(
        out / "manifest.json",
        path,
        selector_path=selector,
        expected_previous_hash="absent",
    )
    assert called == [selected] and selected.status == "configured"
    release.disable_release(
        selector_path=selector,
        expected_previous_hash=sha256(selector.read_bytes()).hexdigest(),
    )
    assert release._Selector.model_validate_json(selector.read_bytes()).enabled is False


def test_k2_report_cannot_be_rebound_to_a_different_formal_manifest(git_release):
    import json

    out, manifest = package(git_release)
    _, _, corpus = release.load_release(out / "manifest.json")
    path, _ = acceptance_bundle(git_release, out, manifest, corpus)
    acceptance = json.loads(path.read_text())
    report_path = out / "reports/K2.json"
    report = json.loads(report_path.read_text())
    report["release"]["manifest_hash"] = "d" * 64
    for child in report["reports"]:
        if child["knowledge_scope"] == "formal-readonly":
            child["release"]["manifest_hash"] = "d" * 64
    report_path.write_text(json.dumps(report))
    next(r for r in acceptance["gate_receipts"] if r["name"] == "K2")[
        "report_sha256"
    ] = sha256(report_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(acceptance))
    with pytest.raises(ValueError):
        release.verify_acceptance(manifest, corpus, path)


@pytest.mark.parametrize(
    "gate",
    sorted(
        __import__(
            "resolv.schemas.knowledge_release", fromlist=["ReleaseAcceptance"]
        ).ReleaseAcceptance.required_gates
    ),
)
def test_acceptance_rejects_one_missing_mandatory_source_even_with_new_report_hash(
    git_release, gate
):
    import json

    out, manifest = package(git_release)
    _, _, corpus = release.load_release(out / "manifest.json")
    path, _ = acceptance_bundle(git_release, out, manifest, corpus)
    acceptance = json.loads(path.read_text())
    report_path = out / "reports" / (gate + ".json")
    report = json.loads(report_path.read_text())
    names = set(
        git_release[2]("ls-tree", "-r", "--name-only", git_release[1]).splitlines()
    )
    removed = min(release.required_gate_sources(gate, names))
    report["source_fingerprints"].pop(removed)
    raw = json.dumps(report).encode()
    report_path.write_bytes(raw)
    receipt = next(r for r in acceptance["gate_receipts"] if r["name"] == gate)
    receipt["source_fingerprints"] = list(report["source_fingerprints"].items())
    receipt["report_sha256"] = sha256(raw).hexdigest()
    path.write_text(json.dumps(acceptance))
    with pytest.raises(ValueError, match="Required gate sources"):
        release.verify_acceptance(manifest, corpus, path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("network_attempts", 1),
        ("negative_probe_attempts", 0),
        ("ruff_discovered", 0),
        ("source_import_verified", False),
        ("tree", "f" * 40),
    ],
)
def test_snapshot_proof_field_cannot_be_deleted_or_forged(git_release, field, value):
    import json
    from types import SimpleNamespace

    out, manifest = package(git_release)
    _, _, corpus = release.load_release(out / "manifest.json")
    acceptance_bundle(git_release, out, manifest, corpus)
    report = json.loads((out / "reports/snapshot_python.json").read_text())
    report[field] = value
    with pytest.raises(ValueError):
        release.validate_gate_report(
            "snapshot_python", report, SimpleNamespace(candidate_commit=git_release[1])
        )
