"""Real MPS Gate B with synthetic pairs, lengths and performance, no labels."""

import argparse
import json
import math
import resource
import time
import traceback
from pathlib import Path

from resolv.schemas.hybrid_knowledge import (
    FusedHybridCandidate,
    HybridCandidate,
    HybridRetrievalError,
    HybridRetrievalProfile,
    PreparedWindow,
)
from resolv.services.hybrid_reranker import LocalHybridReranker


def _window(text, i):
    cid = f"{i:064x}"
    d = HybridCandidate(
        index_version="gate-b-v3",
        chunk_id=cid,
        document_id="d" * 64,
        index_hash="a" * 64,
        route="dense",
        rank=1,
        raw_score=0.1,
    )
    c = FusedHybridCandidate(
        index_version=d.index_version,
        chunk_id=cid,
        document_id=d.document_id,
        index_hash=d.index_hash,
        dense=d,
        rrf_score=1 / 61,
    )
    return PreparedWindow(
        window_id=f"{i + 100:064x}",
        document_id="d" * 64,
        section_instance_id="f" * 64,
        chunk_ids=(cid,),
        center_chunk_ids=(cid,),
        center_candidates=(c,),
        body_text=text,
        best_rrf=c.rrf_score,
    )


def _memory(torch):
    # macOS ru_maxrss is bytes; this Gate only supports the approved Mac/MPS target.
    return {
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
    }


def run_gate(profile: HybridRetrievalProfile) -> dict:
    start = time.perf_counter()
    reranker = LocalHybridReranker(profile)
    primary = None
    try:
        import torch

        torch.mps.synchronize()
        loaded = time.perf_counter() - start
        records = []
        query = "型号MX100已拆封可以七天无理由退货吗？"
        base = (
            "退货政策\n\n型号MX100未拆封可以在七天内申请退货，已拆封不支持无理由退货。"
        )
        for count in (1, 4, 50):
            windows = tuple(_window(base + f" 合成条款 {i}", i) for i in range(count))
            for mode in ("first", "warm"):
                started = time.perf_counter()
                scores = reranker.score(query, windows)
                torch.mps.synchronize()
                if len(scores) != count or not all(math.isfinite(x) for x in scores):
                    raise HybridRetrievalError("gate_b_invalid_scores")
                records.append(
                    {
                        "pairs": count,
                        "mode": mode,
                        "seconds": time.perf_counter() - started,
                        "query_tokens": reranker.query_tokens,
                        "pair_tokens": reranker.pair_tokens,
                        "score_count": len(scores),
                        "score_min": min(scores),
                        "score_max": max(scores),
                        **_memory(torch),
                    }
                )
        # Use the real tokenizer to construct the longest synthetic pair <=3072.
        tok = reranker._tokenizer
        lo, hi = 0, 4096
        while lo < hi:
            mid = (lo + hi + 1) // 2
            n = len(
                tok(query, "条款 " * mid, add_special_tokens=True, truncation=False)[
                    "input_ids"
                ]
            )
            if n <= profile.pair_max_tokens:
                lo = mid
            else:
                hi = mid - 1
        text = "条款 " * lo
        actual = len(
            tok(query, text, add_special_tokens=True, truncation=False)["input_ids"]
        )
        if not 3060 <= actual <= 3072:
            raise HybridRetrievalError("gate_b_boundary_fixture")
        for count in (1, 4, 50):
            windows = tuple(_window(text, i) for i in range(count))
            started = time.perf_counter()
            scores = reranker.score(query, windows)
            torch.mps.synchronize()
            if len(scores) != count or not all(math.isfinite(x) for x in scores):
                raise HybridRetrievalError("gate_b_invalid_scores")
            records.append(
                {
                    "pairs": count,
                    "mode": "near_limit",
                    "seconds": time.perf_counter() - started,
                    "query_tokens": reranker.query_tokens,
                    "pair_tokens": reranker.pair_tokens,
                    "score_count": len(scores),
                    "score_min": min(scores),
                    "score_max": max(scores),
                    **_memory(torch),
                }
            )
        return {
            "passed": True,
            "gate": "B",
            "profile": profile.model_dump(),
            "snapshot_revision": reranker.snapshot.name,
            "load_seconds": loaded,
            "records": records,
            "llm_calls": 0,
            "business_tool_calls": 0,
            "accuracy_evaluated": False,
        }
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            reranker.close()
        except Exception:
            if primary is None:
                raise
            primary.add_note("Gate B cleanup also failed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", required=True, choices=["mps"])
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.report_dir.exists():
        raise SystemExit("Gate B failed: gate_report_dir_exists")
    try:
        args.report_dir.mkdir(parents=True)
        report = run_gate(
            HybridRetrievalProfile(reranker_revision=args.revision, device=args.device)
        )
        (args.report_dir / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        print(json.dumps(report, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 -- safe metadata only; no dependency credential text
        args.report_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "passed": False,
            "gate": "B",
            "code": getattr(exc, "code", "gate_b_dependency_failed"),
            "type": type(exc).__name__,
            "frames": [
                {"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
                for f in traceback.extract_tb(exc.__traceback__)
            ],
        }
        (args.report_dir / "failure.json").write_text(json.dumps(failure, indent=2))
        raise SystemExit("Gate B failed: " + failure["code"]) from None


if __name__ == "__main__":
    main()
