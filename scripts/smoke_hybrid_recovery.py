"""Gate C: owned MySQL/Milvus resources, real process recovery and local models."""

import argparse
import json
import logging
import math
import os
import re
import resource
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from dotenv import dotenv_values
from pymilvus import DataType
from sqlalchemy import func, inspect, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from resolv.core.config import KnowledgeDatabaseSettings, MilvusSettings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.knowledge_hybrid import consistent_read_factory, hybrid_index_lock
from resolv.db.milvus import milvus_client
from resolv.knowledge import (
    chunk_knowledge_document,
    knowledge_index_entry,
    knowledge_ingest_hash,
)
from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from resolv.models.knowledge_hybrid import KnowledgeChunkIndex
from resolv.models.order import utc_now
from resolv.repositories.knowledge import KnowledgeRepository
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.repositories.knowledge_hybrid_milvus import (
    HybridMilvusStore,
    collection_marker,
)
from resolv.repositories.knowledge_milvus import KnowledgeMilvusVectorStore
from resolv.schemas.hybrid_knowledge import (
    HybridCandidate,
    HybridIndexProfile,
    HybridRetrievalError,
    HybridRetrievalProfile,
    HybridRetrievalResult,
    HybridSourceDocument,
)
from resolv.schemas.knowledge import (
    Document,
    KnowledgeFileConfig,
    KnowledgeIndexProfile,
    KnowledgeIndexSnapshot,
)
from resolv.services.hybrid_authority import HybridAuthorityReader
from resolv.services.hybrid_embedding import HybridEmbeddingAdapter
from resolv.services.hybrid_fingerprints import (
    canonical_hash,
    manifest_hash,
    profile_hash,
)
from resolv.services.hybrid_indexing import HybridKnowledgeIndexer
from resolv.services.hybrid_manifest import build_corpus_manifest
from resolv.services.hybrid_ranking import fuse_rrf
from resolv.services.hybrid_reranker import LocalHybridReranker
from resolv.services.hybrid_retrieval import HybridKnowledgeRetriever

PREFIX = "mhhybrid_gate_"
ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "before_upsert",
    "after_upsert",
    "before_confirm_commit",
    "after_confirm_commit",
    "before_mark_ready",
)
LABELS = (*STAGES, "partial", "tail", "lost", "large_a", "large_b")
CONFIG = KnowledgeFileConfig(
    content_type="gate-synthetic-policy", key_clause_keywords=("不能",)
)


def validate_end_to_end(answer: HybridRetrievalResult) -> None:
    d = answer.diagnostics
    if d.dense_count != 50 or d.bm25_count != 50:
        raise HybridRetrievalError("gate_e2e_route_counts")
    if (
        not 1 <= d.center_count <= 50
        or not 1 <= d.windows_after_merge <= 50
        or d.scored_count != d.windows_after_merge
        or len(d.pair_tokens) != d.scored_count
        or not 1 <= d.query_tokens <= 512
        or any(not 1 <= n <= 3072 for n in d.pair_tokens)
        or not 1 <= len(answer.windows) <= 10
        or d.selected_count != len(answer.windows)
        or not 1 <= len(answer.chunks) <= 30
        or any(not math.isfinite(w.rerank_score) for w in answer.windows)
        or any(cid not in answer.chunks for w in answer.windows for cid in w.chunk_ids)
    ):
        raise HybridRetrievalError("gate_e2e_contract")

def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _safe_error(exc):
    causes = []
    current = exc
    while current is not None:
        original = getattr(current, "orig", None)
        code = (
            original.args[0]
            if original is not None
            and original.args
            and isinstance(original.args[0], int)
            else None
        )
        causes.append({"type": type(current).__name__, "sql_code": code})
        current = current.__cause__
    return {
        "passed": False,
        "code": getattr(exc, "code", "gate_c_dependency_failed"),
        "causes": causes,
        "frames": [
            {"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
            for f in traceback.extract_tb(exc.__traceback__)
        ],
    }


def assert_fresh_targets(admin, client, owned):
    if admin.scalar(
        text(
            "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=:name"
        ),
        {"name": owned["database_name"]},
    ):
        raise HybridRetrievalError("gate_existing_unowned_database")
    for name in owned["collections"]:
        if client.has_collection(collection_name=name, timeout=10):
            raise HybridRetrievalError("gate_existing_unowned_collection")


def assert_owned_collection(client, name, marker):
    if (
        not name.startswith(PREFIX)
        or client.describe_collection(collection_name=name, timeout=10).get(
            "description"
        )
        != marker
    ):
        raise HybridRetrievalError("gate_collection_owner_mismatch")


def validate_child_result(result, *, expect_fault):
    if expect_fault:
        if result["returncode"] != 73 or '"passed": true' in result["stdout"].lower():
            raise HybridRetrievalError("gate_fault_not_observed")
        return {"fault_observed": True, "returncode": 73, "success_json": False}
    if result["returncode"] != 0:
        raise HybridRetrievalError("gate_resume_child_failed")
    lines = [line for line in result["stdout"].splitlines() if line.startswith("{")]
    if len(lines) != 1:
        raise HybridRetrievalError("gate_child_result_missing")
    parsed = json.loads(lines[0])
    if parsed.get("passed") is not True:
        raise HybridRetrievalError("gate_child_failed")
    return parsed


def _validate_private_manifest(owned):
    owner = owned.get("owner", "")
    if not isinstance(owner, str) or re.fullmatch(r"[0-9a-f]{32}", owner) is None:
        raise HybridRetrievalError("gate_ownership_manifest_mismatch")
    allowed = {PREFIX + owner + "_" + label for label in (*LABELS, "v2")}
    collections = owned.get("collections", [])
    if (
        not isinstance(collections, list)
        or len(collections) != len(set(collections))
        or not set(collections).issubset(allowed)
    ):
        raise HybridRetrievalError("gate_ownership_manifest_mismatch")
    for label, binding in owned.get("version_bindings", {}).items():
        if (
            label not in LABELS
            or binding.get("version_id") != "gate_" + label
            or binding.get("collection") != PREFIX + owner + "_" + label
            or any(
                re.fullmatch(r"[0-9a-f]{64}", binding.get(field, "")) is None
                for field in ("profile_hash", "manifest_hash")
            )
        ):
            raise HybridRetrievalError("gate_ownership_manifest_mismatch")


def skip_intentional_loss(owned, version_id, profile, *, collection_exists):
    proof = owned.get("ready_loss_probe")
    if proof is None or collection_exists:
        return False
    if proof != {
        "version_id": version_id,
        "collection": profile.collection_name,
        "profile_hash": profile_hash(profile),
    }:
        raise HybridRetrievalError("gate_ready_loss_proof_mismatch")
    return True


def run_recovery_matrix(config, report_dir, *, fault_injector=None):
    report_dir = Path(report_dir)
    if (
        config.get("database_prefix") != PREFIX
        or config.get("collection_prefix") != PREFIX
    ):
        raise HybridRetrievalError("gate_unowned_prefix")
    requested = config.get("database_name")
    if (
        requested is not None
        and re.fullmatch(PREFIX + r"[0-9a-f]{32}", requested) is None
    ):
        raise HybridRetrievalError("gate_unowned_database_name")
    path = report_dir / "ownership.json"
    if path.exists():
        owned = json.loads(path.read_text())
        if (
            owned.get("database_prefix") != PREFIX
            or owned.get("collection_prefix") != PREFIX
            or owned.get("database_name") != PREFIX + owned.get("owner", "")
            or (requested is not None and requested != owned["database_name"])
        ):
            raise HybridRetrievalError("gate_ownership_manifest_mismatch")
        _validate_private_manifest(owned)
        resumed = True
    else:
        if report_dir.exists():
            raise HybridRetrievalError("gate_missing_ownership_manifest")
        report_dir.mkdir(parents=True)
        owner = uuid4().hex
        owned = {
            "owner": owner,
            "database_name": requested or PREFIX + owner,
            "database_prefix": PREFIX,
            "collection_prefix": PREFIX,
            "version_bindings": {},
            "collections": [],
        }
        if requested is not None:
            raise HybridRetrievalError("gate_explicit_unproven_target")
        _write(path, owned)
        resumed = False
    return _run_owned_matrix(
        config, report_dir, owned, resumed=resumed, fault_injector=fault_injector
    )


def _engine(url):
    if not isinstance(url, str):
        url = url.render_as_string(hide_password=False)
    return create_db_engine(
        url,
        hide_parameters=True,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
    )


def _upgrade(engine, revision):
    cfg = Config(str(ROOT / "alembic.knowledge.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations/knowledge"))
    with engine.begin() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, revision)


def _legacy_fingerprint(engine):
    inspector = inspect(engine)
    data = []
    with engine.connect() as c:
        for table in ("knowledge_documents", "knowledge_chunks"):
            schema = {
                "columns": inspector.get_columns(table),
                "checks": inspector.get_check_constraints(table),
                "foreign_keys": inspector.get_foreign_keys(table),
                "unique": inspector.get_unique_constraints(table),
            }
            rows = [
                dict(row)
                for row in c.execute(
                    text("SELECT * FROM " + table + " ORDER BY 1")
                ).mappings()
            ]
            data.append({"table": table, "schema": schema, "rows": rows})
    return canonical_hash(json.loads(json.dumps(data, default=str, ensure_ascii=False)))


def _milvus_fingerprint(client, names):
    data = []
    for n in sorted(names):
        schema = client.describe_collection(collection_name=n, timeout=10)
        for function in schema.get("functions", []):
            for key in ("input_field_names", "input_field_ids", "output_field_names", "output_field_ids"):
                if key in function:
                    function[key] = list(function[key])
        data.append({"name": n, "schema": schema,
                     "stats": client.get_collection_stats(collection_name=n, timeout=10)})
    return canonical_hash(data)


def _sources(owner):
    small = "".join(
        f"# 退货\n合成条款{i}：型号MX100未拆封可在七天内申请退货，已拆封不能无理由退货。\n"
        for i in range(4)
    )
    large = "".join(
        "# 退货政策\n"
        + (
            "型号MX100未拆封可在七天内申请退货，已拆封不能无理由退货，会员运费二十元。"
            * 12
        )
        + "\n"
        for _ in range(20)
    )
    result = []
    for name, raw, rules in (
        (f"synthetic-core-{owner}.md", small, (800, 100)),
        (f"synthetic-large-{owner}.md", large, (160, 20)),
    ):
        d = Document(
            document_id=sha256(name.encode()).hexdigest(), source=name, text=raw
        )
        result.append(
            (
                HybridSourceDocument(
                    document=d,
                    ingest_hash=knowledge_ingest_hash(
                        d, CONFIG, max_chars=rules[0], overlap=rules[1]
                    ),
                    chunks=tuple(
                        chunk_knowledge_document(
                            d, CONFIG, max_chars=rules[0], overlap=rules[1]
                        )
                    ),
                ),
                rules,
            )
        )
    return result


def _seed(factory, sources, legacy_profile):
    with factory.begin() as session:
        for src, _ in sources:
            d = KnowledgeDocument(
                document_id=src.document.document_id,
                source=src.document.source,
                raw_text=src.document.text,
                ingest_hash=src.ingest_hash,
                content_type=CONFIG.content_type,
            )
            d.chunks = [
                KnowledgeChunkRecord(
                    **c.model_dump(),
                    index_hash=knowledge_index_entry(c, legacy_profile).index_hash,
                )
                for c in src.chunks
            ]
            session.add(d)


def _profile(owned, label):
    return HybridIndexProfile(collection_name=PREFIX + owned["owner"] + "_" + label)


def _bind(factory, owned, path, label, manifest):
    p = _profile(owned, label)
    vid = "gate_" + label
    binding = {
        "version_id": vid,
        "collection": p.collection_name,
        "profile_hash": profile_hash(p),
        "manifest_hash": manifest_hash(manifest),
    }
    previous = owned["version_bindings"].get(label)
    if previous is not None and previous != binding:
        raise HybridRetrievalError("gate_version_binding_changed")
    if previous is None:
        owned["version_bindings"][label] = binding
        owned["collections"].append(p.collection_name)
        _write(path / "ownership.json", owned)
    with factory.begin() as session:
        HybridIndexRepository(session).create_version(vid, p, manifest)
    return vid, p


def _count(client, name):
    client.flush(collection_name=name, timeout=10)
    return int(
        client.query(
            collection_name=name,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
            timeout=10,
        )[0]["count(*)"]
    )


def _state(factory, vid):
    with factory() as session:
        return HybridIndexRepository(session).summary(vid)


def _child(report_dir, vid, stage=None):
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        "--report-dir",
        str(report_dir),
        "--_version",
        vid,
    ]
    if stage:
        args += ["--_fault-stage", stage]
    result = subprocess.run(
        args, capture_output=True, text=True, check=False, timeout=240
    )
    # Raw synthetic-only subprocess output stays private, never published.
    log = report_dir / (
        vid + "-" + (stage or "resume") + "-" + uuid4().hex[:8] + ".log"
    )
    log.write_text(result.stdout + "\n" + result.stderr)
    if stage and not (report_dir / (vid + "-" + stage + "-reached.json")).exists():
        raise HybridRetrievalError("gate_child_fault_marker_missing")
    return validate_child_result(
        {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
        expect_fault=stage is not None,
    )


def _child_main(args):
    private = json.loads((args.report_dir / "ownership.json").read_text())
    _validate_private_manifest(private)
    if private["database_name"] != PREFIX + private["owner"]:
        raise HybridRetrievalError("gate_child_target_invalid")
    formal = make_url(
        KnowledgeDatabaseSettings().knowledge_database_url.get_secret_value()
    )
    secret = dotenv_values(ROOT / ".env").get("MYSQL_ROOT_PASSWORD")
    target = formal.set(
        username="root", password=secret, database=private["database_name"]
    )

    class AckLossSession(Session):
        injected = False

        def commit(self):
            done = self.scalar(
                select(func.count())
                .select_from(KnowledgeChunkIndex)
                .where(
                    KnowledgeChunkIndex.index_version == args._version,
                    KnowledgeChunkIndex.index_status == "done",
                )
            )
            super().commit()
            if (
                args._fault_stage == "after_confirm_commit"
                and done
                and not AckLossSession.injected
            ):
                AckLossSession.injected = True
                reached("after_confirm_commit")
                raise OSError("Injected lost SQL commit acknowledgment")

    with ExitStack() as cleanup:
        engine = _engine(target)
        cleanup.callback(engine.dispose)
        with engine.connect() as c:
            if (
                c.scalar(text("SELECT owner FROM mhhybrid_gate_ownership"))
                != private["owner"]
            ):
                raise HybridRetrievalError("gate_child_owner_mismatch")
        factory = sessionmaker(
            bind=engine, class_=AckLossSession, expire_on_commit=False
        )
        with factory() as session:
            v = HybridIndexRepository(session).get_version(args._version)
        if not any(
            b["version_id"] == args._version
            and b["profile_hash"] == profile_hash(v.profile)
            and b["manifest_hash"] == manifest_hash(v.manifest)
            for b in private["version_bindings"].values()
        ):
            raise HybridRetrievalError("gate_child_binding_mismatch")
        lock = cleanup.enter_context(hybrid_index_lock(engine, args._version))
        embedding = HybridEmbeddingAdapter(v.profile)
        cleanup.callback(embedding.close)
        client = cleanup.enter_context(milvus_client(MilvusSettings()))
        store = HybridMilvusStore(client, v.profile)

        def reached(stage):
            _write(
                args.report_dir / (args._version + "-" + stage + "-reached.json"),
                {
                    "stage": stage,
                    "version_id": args._version,
                    "profile_hash": profile_hash(v.profile),
                    "manifest_hash": manifest_hash(v.manifest),
                },
            )

        def hook(stage):
            if stage == args._fault_stage and stage != "after_confirm_commit":
                reached(stage)
                os._exit(73)

        result = HybridKnowledgeIndexer(
            factory,
            embedding,
            store,
            v.profile,
            lock,
            fault_hook=hook if args._fault_stage else None,
        ).run(args._version)
    print(json.dumps({"passed": True, "build": result.model_dump(mode="json")}))


def _run_owned_matrix(config, report_dir, owned, *, resumed, fault_injector=None):
    formal_settings = KnowledgeDatabaseSettings()
    formal = make_url(formal_settings.knowledge_database_url.get_secret_value())
    secret = dotenv_values(ROOT / ".env").get("MYSQL_ROOT_PASSWORD")
    if not secret:
        raise HybridRetrievalError("gate_mysql_root_credential_missing")
    admin_url = formal.set(username="root", password=secret, database=None)
    target_url = admin_url.set(database=owned["database_name"])
    target_signature = canonical_hash(
        {
            "sql_host": admin_url.host,
            "sql_port": admin_url.port,
            "milvus": MilvusSettings().milvus_uri,
        }
    )
    if resumed and owned.get("target_signature") != target_signature:
        raise HybridRetrievalError("gate_service_target_mismatch")
    owned["target_signature"] = target_signature
    report = {
        "passed": False,
        "gate": "C",
        "database_name": owned["database_name"],
        "owner": owned["owner"],
        "checks": {},
        "process_matrix": [],
        "llm_calls": 0,
        "business_tool_calls": 0,
        "accuracy_evaluated": False,
    }

    def record(name, value):
        report["checks"][name] = value
        _write(report_dir / "progress.json", report)
        print("Gate C: " + name, file=sys.stderr, flush=True)
        if fault_injector is not None:
            fault_injector(name)

    with ExitStack() as cleanup:
        for name in ("pymilvus", "pymilvus.milvus_client"):
            logger = logging.getLogger(name)
            cleanup.callback(setattr, logger, "handlers", logger.handlers)
            cleanup.callback(setattr, logger, "propagate", logger.propagate)
            logger.handlers = [logging.NullHandler()]
            logger.propagate = False
        admin = _engine(admin_url)
        cleanup.callback(admin.dispose)
        formal_engine = _engine(formal)
        cleanup.callback(formal_engine.dispose)
        client = cleanup.enter_context(milvus_client(MilvusSettings()))
        if client.get_server_version(timeout=10).lstrip("v") != "3.0.2":
            raise HybridRetrievalError("gate_c_milvus_version")
        formal_before = _legacy_fingerprint(formal_engine)
        old_names = client.list_collections(timeout=10)
        # Exclude only probes proven by this private manifest when resuming.
        old_names = [n for n in old_names if n not in owned["collections"]]
        old_milvus = _milvus_fingerprint(client, old_names)
        sources = _sources(owned["owner"])
        legacy_collection = PREFIX + owned["owner"] + "_v2"
        legacy_profile = KnowledgeIndexProfile(
            model_revision="5617a9f61b028005a4858fdac845db406aefb181",
            collection_name=legacy_collection,
        )
        if not resumed:
            planned = {
                "database_name": owned["database_name"],
                "collections": [
                    legacy_collection,
                    *[
                        _profile(owned, x).collection_name
                        for x in (
                            *STAGES,
                            "partial",
                            "tail",
                            "lost",
                            "large_a",
                            "large_b",
                        )
                    ],
                ],
            }
            with admin.connect() as c:
                assert_fresh_targets(c, client, planned)
            _write(report_dir / "ownership.json", owned)
            with admin.begin() as c:
                c.exec_driver_sql(
                    "CREATE DATABASE `"
                    + owned["database_name"]
                    + "` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
                )
        engine = _engine(target_url)
        cleanup.callback(engine.dispose)
        factory = create_session_factory(engine)
        if not resumed:
            with engine.begin() as c:
                c.exec_driver_sql(
                    "CREATE TABLE mhhybrid_gate_ownership (owner VARCHAR(32) PRIMARY KEY) ENGINE=InnoDB"
                )
                c.execute(
                    text("INSERT INTO mhhybrid_gate_ownership(owner) VALUES(:owner)"),
                    {"owner": owned["owner"]},
                )
            _upgrade(engine, "20261012_0002")
            _seed(factory, sources, legacy_profile)
            before = _legacy_fingerprint(engine)
            tables_before = set(inspect(engine).get_table_names())
            _upgrade(engine, "head")
            after = _legacy_fingerprint(engine)
            _upgrade(engine, "head")
            added = set(inspect(engine).get_table_names()) - tables_before
            if (
                before != after
                or after != _legacy_fingerprint(engine)
                or added != {"knowledge_index_versions", "knowledge_chunk_indexes"}
            ):
                raise HybridRetrievalError("gate_migration_changed_legacy")
            owned["migration"] = {
                "before": before,
                "after": after,
                "added_tables": sorted(added),
                "repeated_unchanged": True,
            }
            _write(report_dir / "ownership.json", owned)
        else:
            with engine.connect() as c:
                if (
                    c.scalar(text("SELECT owner FROM mhhybrid_gate_ownership"))
                    != owned["owner"]
                ):
                    raise HybridRetrievalError("gate_database_owner_mismatch")
            for b in owned["version_bindings"].values():
                with factory() as session:
                    v = HybridIndexRepository(session).get_version(b["version_id"])
                if (
                    profile_hash(v.profile) != b["profile_hash"]
                    or manifest_hash(v.manifest) != b["manifest_hash"]
                ):
                    raise HybridRetrievalError("gate_resume_binding_changed")
        record("migration", owned["migration"])
        core, rules = sources[0]
        core_manifest = build_corpus_manifest(
            (core,),
            {core.document.source: CONFIG},
            max_chars=rules[0],
            overlap=rules[1],
        )
        large, rules = sources[1]
        large_manifest = build_corpus_manifest(
            (large,),
            {large.document.source: CONFIG},
            max_chars=rules[0],
            overlap=rules[1],
        )
        # Legacy v2 is built once using real BGE, then its source fields stay unchanged.
        model_profile = _profile(owned, "before_upsert")
        dense = HybridEmbeddingAdapter(model_profile)
        cleanup.callback(dense.close)
        if legacy_collection not in owned["collections"]:
            if client.has_collection(collection_name=legacy_collection, timeout=10):
                raise HybridRetrievalError("gate_unowned_v2_collision")
            owned["collections"].append(legacy_collection)
            _write(report_dir / "ownership.json", owned)
            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
                description="mhhybrid-owner:" + owned["owner"],
            )
            for name, capacity in KnowledgeMilvusVectorStore._varchar_limits.items():
                schema.add_field(
                    name,
                    DataType.VARCHAR,
                    max_length=capacity,
                    is_primary=name == "chunk_id",
                )
            for name, datatype in KnowledgeMilvusVectorStore._integer_fields.items():
                schema.add_field(name, datatype)
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
            client.create_collection(
                collection_name=legacy_collection, schema=schema, timeout=10
            )
            old_store = KnowledgeMilvusVectorStore(client, legacy_collection)
            old_store.ensure_collection(1024)
            entries = [knowledge_index_entry(c, legacy_profile) for c in core.chunks]
            if old_store.upsert(
                entries, dense.embed_documents(tuple(c.text for c in core.chunks))
            ) != len(entries):
                raise HybridRetrievalError("gate_v2_partial")
            with factory.begin() as session:
                snaps = [
                    KnowledgeIndexSnapshot(**c.model_dump(), index_hash=e.index_hash)
                    for c, e in zip(core.chunks, entries, strict=True)
                ]
                if KnowledgeRepository(session).confirm_indexed(
                    snaps, legacy_collection, utc_now()
                ) != len(entries):
                    raise HybridRetrievalError("gate_v2_confirm")
        assert_owned_collection(
            client, legacy_collection, "mhhybrid-owner:" + owned["owner"]
        )
        legacy_before = owned.get("legacy_before") or _legacy_fingerprint(engine)
        owned["legacy_before"] = legacy_before
        _write(report_dir / "ownership.json", owned)
        for stage in STAGES:
            vid, p = _bind(factory, owned, report_dir, stage, core_manifest)
            receipt_path = report_dir / (stage + "-fault.json")
            if not receipt_path.exists():
                fault = _child(report_dir, vid, stage)
                state = _state(factory, vid)
                if state.state != "building":
                    raise HybridRetrievalError("gate_fault_misreported_ready")
                fault.update(
                    {
                        "stage": stage,
                        "state": state.model_dump(),
                        "injection": "commit-boundary-ack-loss"
                        if stage == "after_confirm_commit"
                        else "process-exit-hook",
                    }
                )
                _write(receipt_path, fault)
            fault = json.loads(receipt_path.read_text())
            first = _child(report_dir, vid)
            second = _child(report_dir, vid)
            count = _count(client, p.collection_name)
            if (
                first["build"]["summary"]["state"] != "ready"
                or second["build"]["attempted_count"] != 0
                or count != len(core.chunks)
            ):
                raise HybridRetrievalError("gate_recovery_not_idempotent")
            if (
                stage == "after_confirm_commit"
                and first["build"]["attempted_count"] != 0
            ):
                raise HybridRetrievalError("gate_ack_loss_reupserted")
            report["process_matrix"].append(
                {
                    "fault": fault,
                    "resume": first,
                    "repeat": second,
                    "milvus_logical_rows": count,
                }
            )
            record("process_" + stage, {"passed": True, "milvus_logical_rows": count})
        # Real named-lock conflict and lost ownership, no SQLite lock substitution.
        with hybrid_index_lock(engine, "gate_lock_probe") as lock:
            try:
                with hybrid_index_lock(engine, "gate_lock_probe"):
                    raise HybridRetrievalError("gate_lock_conflict_bypassed")
            except HybridRetrievalError as exc:
                if exc.code == "gate_lock_conflict_bypassed":
                    raise
            lock.assert_owned()
        try:
            with hybrid_index_lock(engine, "gate_lock_lost") as lock:
                lock._connection.execute(
                    text("SELECT RELEASE_LOCK(:name)"), {"name": lock._name}
                )
                lock._connection.commit()
                lock.assert_owned()
            raise HybridRetrievalError("gate_lost_lock_bypassed")
        except HybridRetrievalError as exc:
            if exc.code == "gate_lost_lock_bypassed":
                raise
        record("real_named_locks", {"conflict_rejected": True, "lost_rejected": True})
        # Inject only the acknowledgment count; the underlying Milvus write is real.
        vid, p = _bind(factory, owned, report_dir, "partial", core_manifest)
        adapter = HybridMilvusStore(client, p)

        class Partial:
            profile = p
            collection_name = p.collection_name
            create_or_validate = adapter.create_or_validate
            validate_existing = adapter.validate_existing

            def upsert(self, entries, vectors):
                return adapter.upsert(entries, vectors) - 1

        class EmbeddingBinding:
            profile = p
            dimension = 1024

            def embed_documents(self, texts):
                return dense.embed_documents(texts)

        if _state(factory, vid).state != "ready":
            with hybrid_index_lock(engine, vid) as lock:
                try:
                    HybridKnowledgeIndexer(
                        factory, EmbeddingBinding(), Partial(), p, lock
                    ).run(vid)
                except HybridRetrievalError as exc:
                    if exc.code != "hybrid_partial_upsert":
                        raise
                else:
                    raise HybridRetrievalError("gate_partial_count_bypassed")
            if _state(factory, vid).done_count != 0:
                raise HybridRetrievalError("gate_partial_confirmed")
        record(
            "partial_receipt",
            {"real_upsert": True, "count_injection": True, "no_done_on_partial": True},
        )
        # Mutate an owned source after upsert, then restore exactly; confirmations remain pending.
        vid, p = _bind(factory, owned, report_dir, "tail", core_manifest)
        adapter = HybridMilvusStore(client, p)

        class TailEmbedding(EmbeddingBinding):
            profile = p

        with factory() as session:
            original = session.get(KnowledgeDocument, core.document.document_id)
            original_time = original.updated_at

        def mutate(stage):
            if stage == "after_upsert":
                with factory.begin() as session:
                    session.execute(
                        update(KnowledgeDocument)
                        .where(
                            KnowledgeDocument.document_id == core.document.document_id
                        )
                        .values(raw_text=core.document.text + " ")
                    )

        try:
            with hybrid_index_lock(engine, vid) as lock:
                HybridKnowledgeIndexer(
                    factory, TailEmbedding(), adapter, p, lock, fault_hook=mutate
                ).run(vid)
            raise HybridRetrievalError("gate_tail_guard_bypassed")
        except HybridRetrievalError as exc:
            if exc.code == "gate_tail_guard_bypassed":
                raise
        finally:
            with factory.begin() as session:
                session.execute(
                    update(KnowledgeDocument)
                    .where(KnowledgeDocument.document_id == core.document.document_id)
                    .values(raw_text=core.document.text, updated_at=original_time)
                )
        if _state(factory, vid).done_count != 0:
            raise HybridRetrievalError("gate_tail_guard_confirmed")
        record(
            "source_tail_guard", {"rejected": True, "restored_owned_source_bytes": True}
        )
        # Two generations of the complete synthetic corpus coexist without v2 mutation.
        built = {}
        for label in ("large_a", "large_b", "lost"):
            manifest = large_manifest if label != "lost" else core_manifest
            vid, p = _bind(factory, owned, report_dir, label, manifest)
            if label == "lost" and skip_intentional_loss(
                owned,
                vid,
                p,
                collection_exists=client.has_collection(
                    collection_name=p.collection_name, timeout=10
                ),
            ):
                if _state(factory, vid).state != "ready":
                    raise HybridRetrievalError("gate_lost_probe_state_changed")
                built[label] = (vid, p, None)
                continue
            model = HybridEmbeddingAdapter(p)
            try:
                with hybrid_index_lock(engine, vid) as lock:
                    result = HybridKnowledgeIndexer(
                        factory, model, HybridMilvusStore(client, p), p, lock
                    ).run(vid)
            finally:
                model.close()
            built[label] = (vid, p, result)
        vid, p, result = built["large_a"]
        with factory.begin() as session:
            repo = HybridIndexRepository(session)
            for changed in (p.model_copy(update={"passage_max_length": 1024}),):
                try:
                    repo.create_version(vid, changed, large_manifest)
                except HybridRetrievalError:
                    pass
                else:
                    raise HybridRetrievalError("gate_profile_conflict_bypassed")
            changed = large_manifest.model_copy(
                update={
                    "documents": (
                        large_manifest.documents[0].model_copy(
                            update={"document_hash": "f" * 64}
                        ),
                    )
                }
            )
            try:
                repo.create_version(vid, p, changed)
            except HybridRetrievalError:
                pass
            else:
                raise HybridRetrievalError("gate_manifest_conflict_bypassed")
        record(
            "independent_generations",
            {
                "v2_rows": len(core.chunks),
                "large_a": result.summary.model_dump(),
                "large_b": built["large_b"][2].summary.model_dump(),
                "conflicts_rejected": True,
            },
        )
        lost_vid, lost_p, _ = built["lost"]
        owned["ready_loss_probe"] = {
            "version_id": lost_vid,
            "collection": lost_p.collection_name,
            "profile_hash": profile_hash(lost_p),
        }
        _write(report_dir / "ownership.json", owned)
        if client.has_collection(collection_name=lost_p.collection_name, timeout=10):
            assert_owned_collection(
                client, lost_p.collection_name, collection_marker(lost_p)
            )
            client.drop_collection(collection_name=lost_p.collection_name, timeout=10)
        try:
            HybridMilvusStore(client, lost_p).validate_existing()
        except HybridRetrievalError:
            pass
        else:
            raise HybridRetrievalError("gate_lost_collection_bypassed")
        model = HybridEmbeddingAdapter(lost_p)
        try:
            with hybrid_index_lock(engine, lost_vid) as lock:
                try:
                    HybridKnowledgeIndexer(
                        factory, model, HybridMilvusStore(client, lost_p), lost_p, lock
                    ).run(lost_vid)
                except HybridRetrievalError:
                    pass
                else:
                    raise HybridRetrievalError("gate_ready_rebuilt")
        finally:
            model.close()
        if client.has_collection(collection_name=lost_p.collection_name, timeout=10):
            raise HybridRetrievalError("gate_ready_recreated")
        record(
            "ready_collection_loss",
            {
                "failed_without_rebuild": True,
                "sql_state": _state(factory, lost_vid).state,
            },
        )
        # Dual connection mutations between center and neighbor/document reads.
        target = large.chunks[2]
        neighbor = large.chunks[1]
        with factory() as session:
            confirmation = session.get(KnowledgeChunkIndex, (vid, target.chunk_id))
            candidate = HybridCandidate(
                index_version=vid,
                chunk_id=target.chunk_id,
                document_id=target.document_id,
                index_hash=confirmation.index_hash,
                route="dense",
                rank=1,
                raw_score=0.5,
            )
            doc_time = session.get(KnowledgeDocument, target.document_id).updated_at
            neighbor_time = session.get(
                KnowledgeChunkRecord, neighbor.chunk_id
            ).updated_at

        def concurrent(stage):
            with factory.begin() as session:
                session.execute(
                    update(KnowledgeDocument)
                    .where(KnowledgeDocument.document_id == target.document_id)
                    .values(raw_text=large.document.text + " ")
                )
                session.execute(
                    update(KnowledgeChunkRecord)
                    .where(KnowledgeChunkRecord.chunk_id == neighbor.chunk_id)
                    .values(answer=neighbor.answer + " ")
                )

        try:
            authority = HybridAuthorityReader(
                consistent_read_factory(engine), vid, p, fault_hook=concurrent
            )
            centers, chunks = authority.prepare(fuse_rrf((candidate,), ()))
            if (
                not centers
                or chunks.get(neighbor.chunk_id).chunk.answer != neighbor.answer
            ):
                raise HybridRetrievalError("gate_read_view_mixed")
            next_reader = HybridAuthorityReader(consistent_read_factory(engine), vid, p)
            if next_reader.prepare(fuse_rrf((candidate,), ()))[0]:
                raise HybridRetrievalError("gate_stale_document_accepted")
            with factory.begin() as session:
                session.execute(
                    update(KnowledgeDocument)
                    .where(KnowledgeDocument.document_id == target.document_id)
                    .values(raw_text=large.document.text, updated_at=doc_time)
                )
            centers, next_chunks = next_reader.prepare(fuse_rrf((candidate,), ()))
            if not centers or neighbor.chunk_id in next_chunks:
                raise HybridRetrievalError("gate_stale_neighbor_accepted")
        finally:
            with factory.begin() as session:
                session.execute(
                    update(KnowledgeDocument)
                    .where(KnowledgeDocument.document_id == target.document_id)
                    .values(raw_text=large.document.text, updated_at=doc_time)
                )
                session.execute(
                    update(KnowledgeChunkRecord)
                    .where(KnowledgeChunkRecord.chunk_id == neighbor.chunk_id)
                    .values(answer=neighbor.answer, updated_at=neighbor_time)
                )
        record(
            "real_repeatable_read",
            {
                "single_read_view": True,
                "next_document_rejected": True,
                "next_neighbor_rejected": True,
            },
        )
        # Full real query path, cold/warm runtime performance, no label judgments.
        load_started = time.perf_counter()
        model = HybridEmbeddingAdapter(p)
        ranker = None
        try:
            ranker = LocalHybridReranker(HybridRetrievalProfile())
            load_seconds = time.perf_counter() - load_started
            service = HybridKnowledgeRetriever(
                model,
                HybridMilvusStore(client, p),
                HybridAuthorityReader(consistent_read_factory(engine), vid, p),
                ranker,
                p,
                ranker.profile,
            )
            runs = []
            for mode in ("cold", "warm"):
                answer = service.retrieve("型号MX100已拆封是否支持七天无理由退货？")
                validate_end_to_end(answer)
                import torch

                torch.mps.synchronize()
                runs.append(
                    {
                        "mode": mode,
                        "diagnostics": answer.diagnostics.model_dump(),
                        "window_count": len(answer.windows),
                        "body_count": len(answer.chunks),
                        "references_closed": all(
                            cid in answer.chunks
                            for w in answer.windows
                            for cid in w.chunk_ids
                        ),
                        "finite_window_logits": True,
                        "peak_rss_bytes": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                    }
                )
            record(
                "real_end_to_end",
                {
                    "version": vid,
                    "profile": p.model_dump(),
                    "profile_hash": profile_hash(p),
                    "manifest_hash": manifest_hash(large_manifest),
                    "retrieval_profile": ranker.profile.model_dump(),
                    "model_load_seconds": load_seconds,
                    "runs": runs,
                },
            )
        finally:
            if ranker is not None:
                ranker.close()
            model.close()
        legacy_after = _legacy_fingerprint(engine)
        formal_after = _legacy_fingerprint(formal_engine)
        after_milvus = _milvus_fingerprint(client, old_names)
        if (
            legacy_before != legacy_after
            or formal_before != formal_after
            or old_milvus != after_milvus
        ):
            raise HybridRetrievalError("gate_original_resource_changed")
        record(
            "original_resources",
            {
                "task_legacy_before": legacy_before,
                "task_legacy_after": legacy_after,
                "formal_before": formal_before,
                "formal_after": formal_after,
                "old_milvus_before": old_milvus,
                "old_milvus_after": after_milvus,
            },
        )
        report["version_bindings"] = owned["version_bindings"]
        with admin.connect() as connection:
            report["mysql_version"] = connection.scalar(text("SELECT VERSION()"))
    report["passed"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-prefix")
    parser.add_argument("--collection-prefix")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_version", help=argparse.SUPPRESS)
    parser.add_argument("--_fault-stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._child:
            _child_main(args)
            return
        if not args.database_prefix or not args.collection_prefix:
            parser.error("database-prefix and collection-prefix are required")
        report = run_recovery_matrix(
            {
                "database_prefix": args.database_prefix,
                "collection_prefix": args.collection_prefix,
            },
            args.report_dir,
        )
        _write(args.report_dir / "result.json", report)
        print(json.dumps(report, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 -- sanitized metadata only, preserving raw failures privately
        args.report_dir.mkdir(parents=True, exist_ok=True)
        _write(args.report_dir / "command-failure.json", _safe_error(exc))
        if args._child:
            _write(
                args.report_dir / (args._version + "-command-failure.json"),
                _safe_error(exc),
            )
            raise SystemExit(73) from None
        raise SystemExit(
            "Gate C failed: " + getattr(exc, "code", "gate_c_dependency_failed")
        ) from None


if __name__ == "__main__":
    main()
