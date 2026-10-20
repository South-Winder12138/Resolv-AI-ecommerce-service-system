"""Explicit immutable Git publication. Importing or serving requests publishes nothing."""

import fcntl
import io
import json
import os
import re
import subprocess
import tomllib
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.engine import make_url

from resolv.core.config import KnowledgeDatabaseSettings, MilvusSettings
from resolv.db.engine import create_session_factory
from resolv.db.knowledge import knowledge_engine
from resolv.db.knowledge_hybrid import hybrid_index_lock
from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS, milvus_client
from resolv.knowledge import (
    chunk_knowledge_document,
    knowledge_index_hash,
    knowledge_ingest_hash,
)
from resolv.models.knowledge import KnowledgeDocument
from resolv.models.knowledge_hybrid import KnowledgeIndexVersion
from resolv.repositories.knowledge import KnowledgeRepository
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.repositories.knowledge_hybrid_milvus import HybridMilvusStore
from resolv.schemas.hybrid_knowledge import HybridIndexProfile, HybridSourceDocument
from resolv.schemas.knowledge import (
    Document,
    KnowledgeFileConfig,
    KnowledgeIndexProfile,
)
from resolv.schemas.knowledge_release import (
    ReleaseAcceptance,
    ReleaseBuildReceipt,
    ReleaseDocument,
    ReleaseManifest,
    ReleaseSelection,
)
from resolv.services.hybrid_embedding import HybridEmbeddingAdapter
from resolv.services.hybrid_fingerprints import manifest_hash, profile_hash
from resolv.services.hybrid_indexing import HybridKnowledgeIndexer
from resolv.services.hybrid_manifest import build_corpus_manifest
from resolv.services.knowledge import KnowledgeIngestionService
from resolv.services.knowledge_branch_runtime import (
    _Selector,
    default_knowledge_retrievers,
)

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ".superpowers/sdd/2026-10-15-resolv-knowledge-branch"
POLICY_PATHS = tuple(
    "knowledge/" + name + ".md"
    for name in (
        "product-faq",
        "product-specs",
        "returns-policy",
        "billing-shipping",
        "member-benefits",
        "after-sales-manual",
    )
)
CONFIG_PATH = "knowledge/file-config.formal.toml"
RELEASE_ID, INDEX_VERSION, COLLECTION = (
    "knowledge-20261015-v1",
    "knowledge_20261015_v1",
    "resolv_knowledge_20261015_v1",
)
SOURCE_TREES = (
    "src",
    "scripts",
    "tests",
    "migrations",
    "frontend",
    "pyproject.toml",
    "uv.lock",
)


def _git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL
    )


def _blob(revision, path):
    mode = _git("ls-tree", revision, "--", path).decode().split()
    if not mode or mode[0] != "100644" or mode[-1] != path:
        raise ValueError("Release source must be a committed regular file")
    return _git("show", revision + ":" + path)


def _decode(raw):
    # Identical UTF-8/BOM and universal-newline handling to load_document.
    return io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig").read()


def _committed_manifest(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An explicit full commit is required")
    commit = _git("rev-parse", "--verify", revision + "^{commit}").decode().strip()
    if commit != revision:
        raise ValueError("Commit identity mismatch")
    config = _blob(commit, CONFIG_PATH)
    payload = tomllib.loads(_decode(config))
    if set(payload) != {"files"} or set(payload["files"]) != set(POLICY_PATHS):
        raise ValueError("Exactly the six approved policy sources are required")
    blobs, documents = {}, []
    for path in POLICY_PATHS:
        raw = _blob(commit, path)
        source = f"knowledge/releases/{RELEASE_ID}/{Path(path).name}"
        blobs[source] = raw
        documents.append(
            ReleaseDocument(
                original_path=path,
                original_commit=commit,
                source=source,
                document_id=sha256(source.encode()).hexdigest(),
                raw_sha256=sha256(raw).hexdigest(),
                content_hash=sha256(_decode(raw).encode()).hexdigest(),
                file_config=KnowledgeFileConfig.model_validate(
                    payload["files"][path]
                ).model_dump(),
            )
        )
    return (
        ReleaseManifest(
            release_id=RELEASE_ID,
            index_version=INDEX_VERSION,
            collection=COLLECTION,
            policy_commit=commit,
            policy_tree=_git("rev-parse", commit + "^{tree}").decode().strip(),
            file_config_sha256=sha256(config).hexdigest(),
            index_profile=HybridIndexProfile(collection_name=COLLECTION),
            documents=tuple(documents),
        ),
        blobs,
        config,
    )


def new_output(path):
    path = Path(path).resolve()
    base = (ROOT / WORKSPACE).resolve()
    if path == base or not path.is_relative_to(base) or path.exists():
        raise ValueError("Output must be a new directory in the task workspace")
    return path


def prepare_release(revision: str, *, output_dir: Path) -> ReleaseManifest:
    out = new_output(output_dir)
    manifest, blobs, config = _committed_manifest(revision)
    out.mkdir(parents=True)
    for source, raw in blobs.items():
        p = out / source
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    (out / "file-config.formal.toml").write_bytes(config)
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def load_release(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = ReleaseManifest.model_validate_json(path.read_bytes())
    expected, blobs, config = _committed_manifest(manifest.policy_commit)
    if (
        manifest != expected
        or (path.parent / "file-config.formal.toml").read_bytes() != config
    ):
        raise ValueError("Release manifest differs from committed policy snapshot")
    documents = []
    for d in manifest.documents:
        p = path.parent / d.source
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to(path.parent)
            or p.read_bytes() != blobs[d.source]
        ):
            raise ValueError("Release bytes or source path mismatch")
        documents.append(
            Document(
                document_id=d.document_id,
                source=d.source,
                text=_decode(blobs[d.source]),
            )
        )
    return manifest, tuple(documents), release_corpus(manifest, documents)


def release_corpus(manifest, documents):
    sources = tuple(
        HybridSourceDocument(
            document=doc,
            ingest_hash=knowledge_ingest_hash(
                doc, spec.file_config, max_chars=spec.max_chars, overlap=spec.overlap
            ),
            chunks=tuple(
                chunk_knowledge_document(
                    doc,
                    spec.file_config,
                    max_chars=spec.max_chars,
                    overlap=spec.overlap,
                )
            ),
        )
        for spec, doc in zip(manifest.documents, documents, strict=True)
    )
    return build_corpus_manifest(
        sources, {d.source: d.file_config for d in manifest.documents}
    )


def staging_profile(manifest):
    return KnowledgeIndexProfile(
        model_revision=manifest.index_profile.model_revision,
        collection_name=manifest.collection,
    )


def preflight_release(factory, manifest, documents, store):
    corpus = release_corpus(manifest, documents)
    profile = staging_profile(manifest)
    with factory() as session:
        repo = KnowledgeRepository(session)
        for spec, doc in zip(manifest.documents, documents, strict=True):
            existing = repo.get_document(doc.document_id)
            collisions = session.scalars(
                select(KnowledgeDocument).where(KnowledgeDocument.source == doc.source)
            ).all()
            if any(row.document_id != doc.document_id for row in collisions):
                raise ValueError("Source identity conflict")
            if existing is not None:
                chunks = chunk_knowledge_document(
                    doc,
                    spec.file_config,
                    max_chars=spec.max_chars,
                    overlap=spec.overlap,
                )
                KnowledgeIngestionService._verify_existing(
                    existing,
                    repo.list_document_chunks(doc.document_id),
                    doc,
                    spec.file_config,
                    chunks,
                    knowledge_ingest_hash(
                        doc,
                        spec.file_config,
                        max_chars=spec.max_chars,
                        overlap=spec.overlap,
                    ),
                    {c.chunk_id: knowledge_index_hash(c, profile) for c in chunks},
                )
        version = session.get(KnowledgeIndexVersion, manifest.index_version)
        bound = session.scalar(
            select(KnowledgeIndexVersion.version_id).where(
                KnowledgeIndexVersion.collection_name == manifest.collection
            )
        )
        if bound is not None and bound != manifest.index_version:
            raise ValueError("Release collection is already bound")
        if version is not None:
            old = HybridIndexRepository(session).get_version(manifest.index_version)
            if old.profile != manifest.index_profile or old.manifest != corpus:
                raise ValueError("Immutable release version conflict")
    if store.has_collection() and version is None:
        raise ValueError("Refusing an existing unbound collection")
    return corpus


class _PreflightStore:
    def __init__(self, client, collection):
        self.client, self.collection = client, collection

    def has_collection(self):
        return self.client.has_collection(
            collection_name=self.collection, timeout=MILVUS_TIMEOUT_SECONDS
        )


def database_identity(engine, expected_database):
    if engine.dialect.name != "mysql" or engine.url.database != expected_database:
        raise ValueError("Explicit MySQL database mismatch")
    with engine.connect() as conn:
        database, server_uuid, version = conn.execute(
            text("SELECT DATABASE(), @@server_uuid, @@version")
        ).one()
    if database != expected_database:
        raise ValueError("Connected database mismatch")
    return {
        "database": database,
        "server_fingerprint": sha256(str(server_uuid).encode()).hexdigest(),
        "server_version": version,
    }


def publish_release(
    manifest_path, *, expected_database, version, collection, report_dir
):
    out = new_output(report_dir)
    manifest, documents, corpus = load_release(manifest_path)
    settings = KnowledgeDatabaseSettings()
    configured = make_url(settings.knowledge_database_url.get_secret_value()).database
    if (configured, version, collection) != (
        expected_database,
        manifest.index_version,
        manifest.collection,
    ) or expected_database != "resolv_knowledge":
        raise ValueError("Formal publication target mismatch")
    out.mkdir(parents=True)
    try:
        with ExitStack() as cleanup:
            engine = cleanup.enter_context(knowledge_engine(settings))
            identity = database_identity(engine, expected_database)
            factory = create_session_factory(engine)
            lock = cleanup.enter_context(hybrid_index_lock(engine, version))
            client = cleanup.enter_context(milvus_client(MilvusSettings()))
            preflight_release(
                factory, manifest, documents, _PreflightStore(client, collection)
            )
            # Only new versioned sources; never the old v2 pending sweep.
            ingestion = KnowledgeIngestionService(factory)
            for spec, doc in zip(manifest.documents, documents, strict=True):
                lock.assert_owned()
                ingestion.stage_document(
                    doc,
                    spec.file_config,
                    staging_profile(manifest),
                    max_chars=spec.max_chars,
                    overlap=spec.overlap,
                )
            embedding = HybridEmbeddingAdapter(manifest.index_profile)
            cleanup.callback(embedding.close)
            indexer = HybridKnowledgeIndexer(
                factory,
                embedding,
                HybridMilvusStore(client, manifest.index_profile),
                manifest.index_profile,
                lock,
            )
            indexer.prepare(version, corpus)
            result = indexer.run(version)
            receipt = ReleaseBuildReceipt(
                release_id=manifest.release_id,
                index_version=version,
                collection=collection,
                profile_hash=profile_hash(manifest.index_profile),
                manifest_hash=manifest_hash(corpus),
                release_manifest_hash=manifest.fingerprint,
                database=identity["database"],
                server_fingerprint=identity["server_fingerprint"],
                state=result.summary.state,
                expected_count=result.summary.expected_count,
                done_count=result.summary.done_count,
                pending_count=result.summary.pending_count,
            )
        selection = release_selection(manifest, receipt.manifest_hash)
        verify_ready_selection(selection)
        (out / "receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n")
        return receipt
    except Exception as exc:
        (out / "failure.json").write_text(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error_code": getattr(exc, "code", "publication_failed"),
                }
            )
        )
        raise


def release_selection(manifest, corpus_hash):
    return ReleaseSelection(
        status="configured",
        release_id=manifest.release_id,
        index_version=manifest.index_version,
        collection=manifest.collection,
        profile=manifest.index_profile,
        profile_hash=profile_hash(manifest.index_profile),
        manifest_hash=corpus_hash,
    )


def verify_ready_selection(selection):
    pool = default_knowledge_retrievers()
    primary = None
    try:
        return pool.validate_selection(selection)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            pool.close()
        except Exception as cleanup_error:
            if primary is None:
                raise
            primary.add_note(
                "Ready validation cleanup also failed: " + type(cleanup_error).__name__
            )


def selector_payload(selection):
    return {
        "enabled": True,
        **selection.model_dump(
            mode="json", exclude={"status", "profile", "error_code"}
        ),
    }


def write_selector(selector_path, value, *, expected_previous_hash):
    path = Path(selector_path).resolve()
    _Selector.model_validate_json(json.dumps(value))
    lock_dir = ROOT / WORKSPACE / "selector-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / (sha256(str(path).encode()).hexdigest() + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = path.read_bytes() if path.exists() else None
        actual = "absent" if previous is None else sha256(previous).hexdigest()
        if actual != expected_previous_hash:
            raise ValueError("Selector changed since explicit comparison")
        if previous is not None:
            _Selector.model_validate_json(previous)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
        try:
            with temp.open("x") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temp.unlink(missing_ok=True)


def verify_acceptance(manifest, corpus, acceptance_path):
    path = Path(acceptance_path).resolve()
    acceptance = ReleaseAcceptance.model_validate_json(path.read_bytes())
    acceptance.require_passed()
    if (
        acceptance.policy_commit,
        acceptance.release_id,
        acceptance.index_version,
        acceptance.profile_hash,
        acceptance.manifest_hash,
    ) != (
        manifest.policy_commit,
        manifest.release_id,
        manifest.index_version,
        profile_hash(manifest.index_profile),
        manifest_hash(corpus),
    ):
        raise ValueError("Acceptance release identity mismatch")
    trees = dict(acceptance.source_trees)
    if set(trees) != set(SOURCE_TREES):
        raise ValueError("Complete candidate code and dependency trees required")
    for name, tree in trees.items():
        if any(
            _git("rev-parse", rev + ":" + name).decode().strip() != tree
            for rev in (acceptance.candidate_commit, "HEAD")
        ):
            raise ValueError("Acceptance candidate code differs")
    candidate_files = set(
        _git("ls-tree", "-r", "--name-only", acceptance.candidate_commit)
        .decode()
        .splitlines()
    )
    for receipt in acceptance.gate_receipts:
        report_path = path.parent / "reports" / (receipt.name + ".json")
        raw = report_path.read_bytes()
        report = json.loads(raw)
        if (
            sha256(raw).hexdigest() != receipt.report_sha256
            or report.get("passed") is not True
            or report.get("gate") != receipt.name
        ):
            raise ValueError("Invalid gate receipt")
        fingerprints = dict(receipt.source_fingerprints)
        if fingerprints != report.get("source_fingerprints") or not fingerprints:
            raise ValueError("Gate sources mismatch")
        required = required_gate_sources(receipt.name, candidate_files)
        if not required or not required <= fingerprints.keys():
            raise ValueError("Required gate sources are missing")
        for name, digest in fingerprints.items():
            if name == "release_manifest":
                actual = manifest.fingerprint
            else:
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError("Gate source path invalid")
                actual = sha256(
                    _git("show", acceptance.candidate_commit + ":" + name)
                ).hexdigest()
            if actual != digest:
                raise ValueError("Gate source no longer matches candidate")
        validate_gate_report(receipt.name, report, acceptance)
        if receipt.name == "K2":
            binding = report.get("release", {})
            if (
                binding.get("profile_hash"),
                binding.get("manifest_hash"),
                binding.get("release_manifest_hash"),
            ) != (
                acceptance.profile_hash,
                acceptance.manifest_hash,
                manifest.fingerprint,
            ):
                raise ValueError("K2 formal release binding differs from acceptance")

    return acceptance


def required_gate_sources(name, candidates):
    python = {
        p
        for p in candidates
        if p.endswith(".py")
        and p.split("/", 1)[0] in {"src", "scripts", "tests", "migrations"}
    }
    meta = {"pyproject.toml", "uv.lock", "frontend/package-lock.json"}
    if name == "snapshot_python":
        wanted = python | meta
    elif name == "snapshot_frontend":
        wanted = {p for p in candidates if p.startswith("frontend/")}
    elif name == "review":
        wanted = (
            {
                p
                for p in candidates
                if p.split("/", 1)[0]
                in {"src", "scripts", "tests", "migrations", "frontend"}
            }
            | meta
            | set(POLICY_PATHS)
            | {CONFIG_PATH}
        )
    elif name == "K1":
        wanted = {
            "src/resolv/core/knowledge_branch_model.py",
            "src/resolv/core/knowledge_branch_prompts.py",
            "src/resolv/schemas/knowledge_branch.py",
            "scripts/smoke_knowledge_branch_model.py",
            "pyproject.toml",
            "uv.lock",
        }
    else:
        wanted = (
            {
                p
                for p in python
                if p.startswith(("src/", "migrations/"))
                or (p.startswith("scripts/") and "knowledge" in Path(p).name)
            }
            | meta
            | set(POLICY_PATHS)
            | {CONFIG_PATH}
        )
        wanted |= {
            "tests/knowledge_branch_fixtures.py",
            "tests/branch_fixtures.py",
            "tests/understanding_fixtures.py",
            "tests/test_context.py",
            "tests/test_understanding_context.py",
            "tests/conversation_fixtures.py",
            "tests/hybrid_fixtures.py",
        }
        if name == "K3":
            wanted.add("scripts/smoke_three_branch_recovery.py")
    return wanted & candidates


def validate_gate_report(name, report, acceptance):
    # Concrete producer validators tighten K2/K3 further in their gate module.
    if name == "K1":
        calls = report.get("calls", [])
        if (
            report.get("business_tool_calls") != 0
            or len(calls) != 3
            or {c.get("purpose") for c in calls}
            != {"knowledge_query", "knowledge_check", "knowledge_answer"}
        ):
            raise ValueError("Incomplete model transport evidence")
        for c in calls:
            a = c.get("audit", {})
            if (
                c.get("tool_calls") != 0
                or c.get("structure_valid") is not True
                or a.get("outcome") != "succeeded"
                or not (
                    type(a.get("input_tokens")) is int
                    and type(a.get("output_tokens")) is int
                    and 0
                    < a["input_tokens"]
                    <= a.get("input_estimate", -1)
                    <= c.get("input_limit", -1)
                    and 0 < a["output_tokens"] <= c.get("output_limit", -1)
                )
            ):
                raise ValueError("Missing model usage or budget evidence")
    elif name in {"snapshot_python", "snapshot_frontend"}:
        required = (
            {"negative_probe", "source_import", "pytest", "ruff_files", "ruff"}
            if name == "snapshot_python"
            else {"test", "typecheck", "build", "lint"}
        )
        if (
            report.get("revision") != acceptance.candidate_commit
            or set(report.get("checks", {})) != required
            or any(
                type(value) is not int or value != 0
                for value in report["checks"].values()
            )
        ):
            raise ValueError("Incomplete committed snapshot evidence")
        if (
            report.get("tree")
            != _git("rev-parse", acceptance.candidate_commit + "^{tree}")
            .decode()
            .strip()
        ):
            raise ValueError("Snapshot candidate tree differs")
        if name == "snapshot_python":
            expected = sum(
                p.endswith(".py")
                and p.split("/", 1)[0] in {"src", "scripts", "tests", "migrations"}
                for p in _git(
                    "ls-tree", "-r", "--name-only", acceptance.candidate_commit
                )
                .decode()
                .splitlines()
            )
            if (
                not expected
                or report.get("expected_python_files") != expected
                or report.get("ruff_discovered") != expected
                or report.get("network_attempts") != 0
                or report.get("negative_probe_attempts") != 2
                or report.get("source_import_verified") is not True
                or report.get("lint_inputs_verified") is not True
                or not re.search(
                    r"\b[1-9]\d* passed\b", report.get("pytest_summary", "")
                )
            ):
                raise ValueError("Incomplete Python isolation or discovery proof")
        else:
            commands = [
                ["npm", "test"],
                ["npm", "run", "typecheck"],
                ["npm", "run", "build"],
                ["npm", "run", "lint"],
            ]
            steps = report.get("steps", [])
            if [s.get("command") for s in steps] != commands or any(
                type(s.get("exit_code")) is not int or s["exit_code"] != 0
                for s in steps
            ):
                raise ValueError("Incomplete frontend command proof")
    elif name == "review":
        if (
            report.get("candidate_commit") != acceptance.candidate_commit
            or report.get("important_count") != 0
            or not report.get("reviewer_id")
            or not report.get("review_document_sha256")
        ):
            raise ValueError("Incomplete independent review evidence")
        path = (
            ROOT
            / "docs/superpowers/plans/2026-10-15-resolv-knowledge-branch-review.md"
        )
        try:
            raw = path.read_bytes()
            document = raw.decode("utf-8")
        except (OSError, UnicodeError):
            raise ValueError("Review document evidence unavailable") from None
        if (
            sha256(raw).hexdigest() != report["review_document_sha256"]
            or acceptance.candidate_commit not in document
            or report["reviewer_id"] not in document
        ):
            raise ValueError("Review document binding mismatch")
    elif name == "K2":
        from resolv.services.knowledge_gate_validation import validate_knowledge_gate

        if report.get("knowledge_scope") != "combined":
            raise ValueError("Both K2 scopes are required for activation")
        validate_knowledge_gate(report)
    elif name == "K3":
        from resolv.services.knowledge_gate_validation import validate_recovery_gate

        validate_recovery_gate(report)


def activate_release(
    manifest_path, acceptance_path, *, selector_path, expected_previous_hash
):
    manifest, _, corpus = load_release(manifest_path)
    verify_acceptance(manifest, corpus, acceptance_path)
    selection = release_selection(manifest, manifest_hash(corpus))
    verify_ready_selection(selection)
    write_selector(
        selector_path,
        selector_payload(selection),
        expected_previous_hash=expected_previous_hash,
    )
    return selection


def disable_release(*, selector_path, expected_previous_hash):
    write_selector(
        selector_path,
        {"enabled": False, "config_version": 1},
        expected_previous_hash=expected_previous_hash,
    )
    return ReleaseSelection(status="disabled")
