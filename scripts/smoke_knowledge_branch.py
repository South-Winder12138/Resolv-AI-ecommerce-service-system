"""K2: owned real MySQL/Milvus/BGE/MPS plus bounded real provider cases."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import resource
import time
from contextlib import ExitStack, contextmanager

from scripts.knowledge_gate_support import (
    CONVERSATION_DB,
    OWNERSHIP,
    begin_provider_case,
    formal_snapshot,
    initialize_resources,
    owned_resources,
    runtime_environment,
    save_json,
    source_fingerprints,
)
from sqlalchemy import func, select

from resolv.core.config import MilvusSettings
from resolv.db.engine import create_session_factory
from resolv.db.knowledge_hybrid import hybrid_index_lock
from resolv.db.milvus import milvus_client
from resolv.models.conversation import ConversationAction
from resolv.repositories.knowledge_branch import KnowledgeBranchRepository
from resolv.repositories.knowledge_hybrid_milvus import HybridMilvusStore
from resolv.schemas.hybrid_knowledge import HybridIndexProfile
from resolv.services.hybrid_embedding import HybridEmbeddingAdapter
from resolv.services.hybrid_fingerprints import (
    canonical_hash,
    manifest_hash,
    profile_hash,
)
from resolv.services.hybrid_indexing import HybridKnowledgeIndexer
from resolv.services.knowledge import KnowledgeIngestionService
from resolv.services.knowledge_gate_validation import (
    validate_formal_case,
    validate_knowledge_gate,
)
from resolv.services.knowledge_release import (
    ROOT,
    WORKSPACE,
    _PreflightStore,
    load_release,
    new_output,
    preflight_release,
    release_selection,
    selector_payload,
    staging_profile,
)
from resolv.services.routed_runtime import routed_conversation_runtime

CASES = ROOT / "tests/fixtures/knowledge_branch/formal_cases.json"


def build_isolated(manifest, documents, corpus, obj, engine):
    manifest = manifest.model_copy(
        update={
            "index_version": obj["collection"],
            "collection": obj["collection"],
            "index_profile": HybridIndexProfile(collection_name=obj["collection"]),
        }
    )
    with ExitStack() as cleanup:
        factory = create_session_factory(engine)
        lock = cleanup.enter_context(hybrid_index_lock(engine, manifest.index_version))
        client = cleanup.enter_context(milvus_client(MilvusSettings()))
        preflight_release(
            factory, manifest, documents, _PreflightStore(client, manifest.collection)
        )
        ingestion = KnowledgeIngestionService(factory)
        for doc, spec in zip(documents, manifest.documents, strict=True):
            ingestion.stage_document(
                doc,
                spec.file_config,
                staging_profile(manifest),
                max_chars=spec.max_chars,
                overlap=spec.overlap,
            )
        dense = HybridEmbeddingAdapter(manifest.index_profile)
        cleanup.callback(dense.close)
        indexer = HybridKnowledgeIndexer(
            factory,
            dense,
            HybridMilvusStore(client, manifest.index_profile),
            manifest.index_profile,
            lock,
        )
        indexer.prepare(manifest.index_version, corpus)
        build = indexer.run(manifest.index_version)
    return manifest, build.model_dump(mode="json")


@contextmanager
def capture_public_model_outputs(service, path, run_id):
    class RecordedClient:
        def __init__(self, base):
            self.base = base

        def __getattr__(self, name):
            return getattr(self.base, name)

        def invoke_json(self, purpose, messages, output_type, *, profile, audit_hook):
            invocation_ids = []

            def track(audit):
                audit_hook(audit)
                if audit.outcome == "started":
                    invocation_ids.append(audit.invocation_id)

            output = self.base.invoke_json(
                purpose, messages, output_type, profile=profile, audit_hook=track
            )
            value = {
                "run_id": run_id,
                "purpose": purpose,
                "invocation_id": invocation_ids[0] if invocation_ids else None,
                "profile_fingerprint": profile.fingerprint,
                "input_hash": canonical_hash(
                    [m.model_dump(mode="json") for m in messages]
                ),
                "output": output.model_dump(mode="json"),
            }
            with Path(path).open("a") as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            return output

    clients = service.knowledge_clients
    originals = dict(clients)
    try:
        for purpose in ("knowledge_query", "knowledge_check"):
            clients[purpose] = RecordedClient(originals[purpose])
        yield
    finally:
        clients.update(originals)


def prior_invalid_checks(out, case_id, run_id):
    ids, receipts = set(), []
    for path in sorted(Path(out).parent.glob("gate-k2-*/" + case_id + "-failure.json")):
        data = json.loads(path.read_text())
        if data.get("run_id") != run_id or data.get("error_code") != "self_check_error":
            continue
        checks = [
            a for a in data.get("audits", []) if a.get("purpose") == "knowledge_check"
        ]
        if checks:
            ids.add(checks[-1]["invocation_id"])
            from hashlib import sha256

            receipts.append(
                {
                    "report": path.name,
                    "report_directory": path.parent.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "invalid_check_invocation": checks[-1]["invocation_id"],
                }
            )
    return sorted(ids), receipts


def prior_failed_upstream(out, case_id, run_id):
    from hashlib import sha256

    ids, receipts = set(), []
    for path in sorted(Path(out).parent.glob("gate-k2-*/" + case_id + "-failure.json")):
        data = json.loads(path.read_text())
        if data.get("run_id") != run_id:
            continue
        for audit in data.get("audits", []):
            if audit.get("purpose") in {"resolution", "intents"} and audit.get(
                "outcome"
            ) in {"failed", "unknown"}:
                invocation = audit["invocation_id"]
                ids.add(invocation)
                receipts.append(
                    {
                        "report_directory": path.parent.name,
                        "report": path.name,
                        "sha256": sha256(path.read_bytes()).hexdigest(),
                        "failed_upstream_invocation": invocation,
                    }
                )
    return sorted(ids), receipts


def formal_resume_inventory(prior, owner):
    prior = Path(prior).resolve()
    root = (ROOT / WORKSPACE).resolve()
    if prior.parent != root or not prior.name.startswith("gate-k2-formal-"):
        raise ValueError("Formal resume requires an existing task report")
    if (
        json.loads((prior / "failure.json").read_text()).get("knowledge_scope")
        != "formal-readonly"
    ):
        raise ValueError("Formal resume report scope mismatch")
    result = {}
    for pattern in ("*-observed.json", "*-failure.json"):
        for path in prior.glob(pattern):
            case = json.loads(path.read_text())
            key, run_id = case["case_id"], case["run_id"]
            if key in result and result[key] != run_id:
                raise ValueError("Conflicting saved case identity")
            result[key] = run_id
    required = {
        s.split(":", 1)[1]
        for s in owner["provider_cases"]
        if s.startswith("formal-readonly:")
    }
    if set(result) != required:
        raise ValueError("Every journaled formal case must be resumed")
    return result


def run_case(service, obj, case, scope, out, *, resume_run=None):
    started = time.monotonic()
    if resume_run:
        from resolv.services.knowledge_branch_runtime import read_release_selection

        run = service.get_run(resume_run)
        selection = read_release_selection(service.release_selector_path)
        if (
            scope not in {"isolated", "formal-readonly"}
            or scope + ":" + case["case_id"] not in obj.get("provider_cases", [])
            or run.question != case["question"]
            or run.request_id != case["case_id"]
            or run.mode != "agent"
            or run.orchestration_version != "routed-v3"
            or not run.knowledge_config
            or run.knowledge_config["selection"] != selection.model_dump(mode="json")
        ):
            raise ValueError("Resume must match the existing journaled case")
        failed, _ = prior_failed_upstream(out, case["case_id"], run.run_id)
        if len(failed) >= 2 and run.status != "completed":
            raise ValueError("Controlled upstream recovery attempt budget exhausted")
        with service.session_factory() as session:
            prior = KnowledgeBranchRepository(session, service.owner_id).get(run.run_id)
        if (
            prior
            and prior.check is None
            and sum(
                a.purpose == "knowledge_check" and a.outcome != "started"
                for a in prior.audits
            )
            >= 3
        ):
            raise ValueError("Controlled selfcheck attempt budget exhausted")
        c = service.detail(run.conversation_id).conversation
        with capture_public_model_outputs(
            service, out / (case["case_id"] + "-public-model-outputs.jsonl"), run.run_id
        ):
            events = list(service.resume(run.conversation_id, run.run_id))
    else:
        begin_provider_case(obj, scope + ":" + case["case_id"])
        c = service.create()
        run, _ = service.accept(
            c.conversation_id, case["case_id"], case["question"], "agent"
        )
        with capture_public_model_outputs(
            service, out / (case["case_id"] + "-public-model-outputs.jsonl"), run.run_id
        ):
            events = list(service.run(run.run_id))
    saved = service.get_run(run.run_id)
    understanding = service.understanding.record(
        c.conversation_id, saved.understanding_id
    )
    audits = [
        dict(a.model_dump(mode="json"), purpose=a.stage)
        for a in understanding.invocations
        if a.outcome != "started"
    ]
    with service.session_factory() as session:
        record = KnowledgeBranchRepository(session, service.owner_id).get(run.run_id)
        tools = session.scalar(
            select(func.count())
            .select_from(ConversationAction)
            .where(ConversationAction.run_id == run.run_id)
        )
    if record:
        audits.extend(
            a.model_dump(mode="json") for a in record.audits if a.outcome != "started"
        )
    invalid_checks, failure_receipts = prior_invalid_checks(
        out, case["case_id"], run.run_id
    )
    failed_upstream, upstream_receipts = prior_failed_upstream(
        out, case["case_id"], run.run_id
    )
    result = {
        "case_id": case["case_id"],
        "question": case["question"],
        "expected": case["expected"],
        "run_id": run.run_id,
        "resumed_existing_run": bool(resume_run),
        "recovered_invalid_check_invocations": invalid_checks,
        "recovered_failed_upstream_invocations": failed_upstream,
        "recovery_failure_receipts": failure_receipts + upstream_receipts,
        "selfcheck_invocation_id": next(
            (
                a["invocation_id"]
                for a in reversed(audits)
                if a["purpose"] == "knowledge_check"
            ),
            None,
        ),
        "status": saved.status,
        "error_code": saved.error_code,
        "events": [e.type for e in events],
        "audits": audits,
        "business_tool_calls": tools,
        "transport_tools": 0,
        "knowledge_record_present": record is not None,
        "elapsed_seconds": time.monotonic() - started,
        "passed": False,
    }
    if saved.status != "completed":
        save_json(out / (case["case_id"] + "-failure.json"), result)
        from resolv.schemas.conversation import ConversationError

        raise ConversationError(
            saved.error_code or "knowledge_gate_failed", saved.retryable
        )
    meta = service.reply_metadata(run.run_id)
    detail = service.detail(c.conversation_id)
    answer = next(
        m.text for m in detail.messages if m.message_id == saved.answer_message_id
    )
    if next(r.result_json for r in detail.runs if r.run_id == run.run_id) != meta:
        raise ValueError("Refresh metadata mismatch")
    result.update(answer=answer, metadata=meta)
    if record and record.retrieval:
        b = record.retrieval.bundle
        d = record.retrieval.result.diagnostics
        result.update(
            prepared_query=record.query.model_dump(mode="json"),
            counts=d.model_dump(mode="json"),
            query_tokens=d.query_tokens,
            pair_tokens=list(d.pair_tokens),
            evidence={
                "evidence_hash": b.evidence_hash,
                "input_tokens": b.input_tokens,
                "budget": record.profile.evidence_max_tokens,
                "windows": [w.window_id for w in b.windows],
                "citations": {c.window_id: list(c.chunk_ids) for c in b.citations},
                "chunks": [c.chunk.chunk_id for c in b.chunks],
                "used_citation_numbers": list(record.answer.used_citation_numbers)
                if record.answer
                else [],
            },
            signals=record.check.signals.model_dump(mode="json"),
            selfcheck=record.check.self_check.model_dump(mode="json")
            if record.check.self_check
            else None,
            same_evidence=record.check.evidence_hash == b.evidence_hash
            and (
                record.answer is None
                or record.answer.evidence_hash == b.evidence_hash
                and record.answer_input.evidence_hash == b.evidence_hash
                and any(
                    b.serialized in m["content"]
                    for m in record.answer_input.messages.value
                )
            ),
        )
    save_json(out / (case["case_id"] + "-observed.json"), result)
    validate_formal_case(result, case)
    result["passed"] = True
    save_json(out / (case["case_id"] + ".json"), result)
    return result


def run_isolated_probes(service, obj, selection, out):
    from scripts.smoke_knowledge_branch_recovery import QUESTION, make_service

    from resolv.core.config import ConversationDatabaseSettings
    from resolv.models.knowledge import KnowledgeDocument
    from resolv.schemas.hybrid_knowledge import HybridRetrievalError

    pool = service.knowledge_retrievers
    if selection.collection != obj["collection"]:
        raise ValueError("Fault probes require exact owned knowledge resources")
    pool.validate_selection(selection)
    if pool._factory.kw["bind"].url.database != obj["knowledge_database"]:
        raise ValueError("Fault probes require exact owned knowledge resources")
    _, store, authority = pool._versions[selection.fingerprint]
    ranker = pool._ranker()
    query = QUESTION
    probes = {}
    empty = pool.retrieve(selection, query, bm25_text="zz" + obj["owner"] + "qz")
    probes["one_lane_empty"] = {
        "passed": empty.diagnostics.bm25_count == 0
        and empty.diagnostics.dense_count > 0
        and empty.diagnostics.scored_count > 0,
        "counts": empty.diagnostics.model_dump(mode="json"),
        "real_query": True,
    }
    for name, target, attr in [
        ("bm25_fault", store, "search_bm25"),
        ("sql_fault", authority, "fault_hook"),
        ("reranker_fault", ranker, "score"),
    ]:
        original = getattr(target, attr)

        def fail(*args, _probe=name, **kwargs):
            raise TimeoutError("Controlled isolated " + _probe)

        setattr(target, attr, fail)
        try:
            pool.retrieve(selection, query, bm25_text=query)
        except HybridRetrievalError as exc:
            probes[name] = {
                "passed": True,
                "error_code": exc.code,
                "partial_result_returned": False,
                "synthetic_fault": True,
            }
        else:
            raise ValueError("Fault produced a partial retrieval result: " + name)
        finally:
            setattr(target, attr, original)
    # Actual SQL data mismatch is injected only into the owned corpus, then restored.
    with pool._factory() as session:
        old = {
            row.document_id: row.raw_text
            for row in session.scalars(select(KnowledgeDocument))
        }
    try:
        with pool._factory.begin() as session:
            for did, raw in old.items():
                session.get(KnowledgeDocument, did).raw_text = (
                    raw + "\ncontrolled-isolated-mismatch"
                )
        invalid = pool.retrieve(selection, query, bm25_text=query)
        probes["corrupt_authority"] = {
            "passed": invalid.diagnostics.invalid_count > 0
            and invalid.diagnostics.center_count == 0
            and not invalid.windows,
            "counts": invalid.diagnostics.model_dump(mode="json"),
            "owned_sql_modified_and_restored": True,
        }
    finally:
        with pool._factory.begin() as session:
            for did, raw in old.items():
                session.get(KnowledgeDocument, did).raw_text = raw
    for name in ("low_score", "selfcheck_insufficient"):
        state = {
            "window": "knowledge_after_answer_saved",
            "calls": str(out / (name + "-fake-calls.jsonl")),
            "selector": str(out / (name + "-unused-fixture-selector.json")),
            "self_check_false": name == "selfcheck_insufficient",
        }
        fake = make_service(
            service.session_factory, ConversationDatabaseSettings(), state
        )
        fake.knowledge_profile = service.knowledge_profile
        fake.knowledge_retrievers = pool
        fake.release_selector_path = service.release_selector_path
        original = ranker.score
        if name == "low_score":
            ranker.score = lambda query, windows: tuple(-10.0 for _ in windows)
        try:
            c = fake.create()
            run, _ = fake.accept(c.conversation_id, name, QUESTION, "agent")
            events = list(fake.run(run.run_id))
            saved = fake.get_run(run.run_id)
            if saved.status != "completed":
                raise ValueError(
                    "Controlled gate run failed: "
                    + str([(e.type, e.data) for e in events])
                )
            with fake.session_factory() as session:
                record = KnowledgeBranchRepository(session, fake.owner_id).get(
                    run.run_id
                )
            calls = [
                json.loads(line)["purpose"]
                for line in Path(state["calls"]).read_text().splitlines()
            ]
            expected_check = 0 if name == "low_score" else 1
            meta = fake.reply_metadata(run.run_id)
            probes[name] = {
                "passed": meta["reply_kind"] == "knowledge_fallback"
                and calls.count("knowledge_check") == expected_check
                and calls.count("knowledge_answer") == 0
                and not meta["knowledge"]["citations"],
                "fake_inference": True,
                "accuracy_evidence": False,
                "counts": record.retrieval.result.diagnostics.model_dump(mode="json"),
                "signals": record.check.signals.model_dump(mode="json"),
                "selfcheck": record.check.self_check.model_dump(mode="json")
                if record.check.self_check
                else None,
                "calls": calls,
            }
        finally:
            ranker.score = original
    if not all(p["passed"] for p in probes.values()):
        raise ValueError("Isolated boundary probe failed")
    return probes


def run(args):
    started = time.monotonic()
    out = new_output(args.report_dir)
    out.mkdir(parents=True)
    report = {
        "gate": "K2",
        "passed": False,
        "real_services": True,
        "knowledge_scope": args.knowledge_scope,
        "business_tool_calls": 0,
        "source_fingerprints": source_fingerprints((str(CASES.relative_to(ROOT)),)),
        "cases": [],
    }
    try:
        manifest, documents, corpus = load_release(args.manifest)
        before = formal_snapshot()
        save_json(out / "preservation-before.json", before)
        if args.knowledge_scope == "isolated":
            if args.ownership_manifest:
                ownership = args.ownership_manifest
            else:
                initialize_resources(args.conversation_database, args.business_db)
                ownership = OWNERSHIP
        else:
            if not args.ownership_manifest:
                raise ValueError("Formal read-only scope requires owned chat resources")
            ownership = args.ownership_manifest
        with owned_resources(ownership) as (obj, engine, _):
            if (args.conversation_database, str(Path(args.business_db).resolve())) != (
                obj["conversation_database"],
                obj["business_db"],
            ):
                raise ValueError("Requested chat resources do not match owner")
            resume_runs = (
                formal_resume_inventory(args.resume_report_dir, obj)
                if getattr(args, "resume_report_dir", None)
                else {}
            )
            original_manifest = manifest
            if args.knowledge_scope == "isolated":
                manifest, report["isolated_build"] = build_isolated(
                    manifest, documents, corpus, obj, engine
                )
            report["release"] = {
                "release_id": manifest.release_id,
                "index_version": manifest.index_version,
                "collection": manifest.collection,
                "profile_hash": profile_hash(manifest.index_profile),
                "manifest_hash": manifest_hash(corpus),
                "release_manifest_hash": original_manifest.fingerprint,
                "policy_commit": manifest.policy_commit,
            }
            selection = release_selection(manifest, manifest_hash(corpus))
            selector = out / "selector.json"
            save_json(selector, selector_payload(selection))
            with (
                runtime_environment(
                    obj, formal=args.knowledge_scope == "formal-readonly"
                ),
                routed_conversation_runtime() as service,
            ):
                service.release_selector_path = selector
                pool = service.knowledge_retrievers
                original_ranker = pool._ranker
                forwards = []
                handles = []

                def ranker():
                    rank = original_ranker()
                    if not handles:

                        def observed(model, inputs, output):
                            parameter = next(model.parameters())
                            forwards.append(
                                {
                                    "device": parameter.device.type,
                                    "actual_device": str(parameter.device),
                                    "dtype": str(parameter.dtype),
                                    "rows": int(output.logits.shape[0]),
                                }
                            )

                        handles.append(rank._model.register_forward_hook(observed))
                    return rank

                pool._ranker = ranker
                try:
                    cases = json.loads(CASES.read_text())
                    if args.knowledge_scope == "isolated":
                        cases = cases[:1]
                    for case in cases:
                        report["cases"].append(
                            run_case(
                                service,
                                obj,
                                case,
                                args.knowledge_scope,
                                out,
                                resume_run=resume_runs.get(case["case_id"])
                                or args.resume_run,
                            )
                        )
                    if args.knowledge_scope == "isolated":
                        report["probes"] = run_isolated_probes(
                            service, obj, selection, out
                        )
                    import torch

                    torch.mps.synchronize()
                    dense = pool._dense._service._service._model
                    # FlagEmbedding's runner owns the actual transformer module.
                    module = getattr(dense, "model", dense)
                    parameter = next(module.parameters())
                    report["local_inference"] = {
                        "device": forwards[0]["device"] if forwards else None,
                        "dtype": forwards[0]["dtype"] if forwards else None,
                        "batch_size": pool.retrieval_profile.batch_size,
                        "forward_calls": len(forwards),
                        "forwards": forwards,
                        "model_revision": pool._reranker.snapshot.name,
                        "dense_revision": manifest.index_profile.model_revision,
                        "dense_dimension": pool._dense.dimension,
                        "dense_normalized": manifest.index_profile.normalize_embeddings,
                        "dense_device": str(parameter.device),
                        "dense_dtype": str(parameter.dtype),
                        "rss_peak_bytes": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                    }
                finally:
                    for handle in handles:
                        handle.remove()
        report["preservation"] = {"before": before, "after": formal_snapshot(before)}
        report["elapsed_seconds"] = time.monotonic() - started
        report["passed"] = True
        validate_knowledge_gate(report)
        save_json(out / "report.json", report)
        return report
    except Exception as exc:
        report["passed"] = False
        report["error_type"] = type(exc).__name__
        report["error_code"] = getattr(exc, "code", "knowledge_gate_failed")
        save_json(out / "failure.json", report)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--report-dir", required=True, type=Path)
    p.add_argument(
        "--knowledge-scope", required=True, choices=["isolated", "formal-readonly"]
    )
    p.add_argument("--conversation-database", default=CONVERSATION_DB)
    p.add_argument("--business-db", type=Path, default=ROOT / WORKSPACE / "business.db")
    p.add_argument("--ownership-manifest", type=Path)
    p.add_argument(
        "--resume-report-dir",
        type=Path,
        help="Continue exact journaled formal runs from a failed task report",
    )
    p.add_argument(
        "--resume-run",
        help="Resume only the journaled isolated case with identical frozen selection",
    )
    args = p.parse_args(argv)
    if args.resume_report_dir and (
        args.knowledge_scope != "formal-readonly"
        or not args.ownership_manifest
        or args.resume_run
    ):
        p.error(
            "--resume-report-dir requires formal-readonly scope and ownership manifest"
        )
    if args.resume_run and (
        args.knowledge_scope != "isolated" or not args.ownership_manifest
    ):
        p.error("--resume-run requires isolated scope and explicit ownership manifest")
    try:
        report = run(args)
    except Exception as exc:  # noqa: BLE001 -- record safe gate failure without credentials
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error_code": getattr(exc, "code", "knowledge_gate_failed"),
                }
            )
        )
        return 1
    print(
        json.dumps(
            {
                "passed": True,
                "cases": len(report["cases"]),
                "report": str(args.report_dir / "report.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
