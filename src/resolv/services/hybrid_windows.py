"""Same-section adjacent windows; identical windows merge, overlap stays separate."""

import math
from collections.abc import Mapping, Sequence

from resolv.schemas.hybrid_knowledge import (
    HybridEvidenceWindow,
    HybridRetrievalError,
    PreparedWindow,
    VerifiedHybridCenter,
    VerifiedHybridChunk,
)
from resolv.services.hybrid_fingerprints import canonical_hash

RULE = "title-boundaries-v1"


def build_windows(
    centers: Sequence[VerifiedHybridCenter], chunks: Mapping[str, VerifiedHybridChunk]
) -> tuple[PreparedWindow, ...]:
    try:
        if len(centers) > 50:
            raise ValueError("Center limit")
        positions = {}
        for cid, c in chunks.items():
            if cid != c.chunk.chunk_id:
                raise ValueError("Chunk key mismatch")
            position = (c.chunk.document_id, c.chunk.chunk_index)
            if position in positions:
                raise ValueError("Ambiguous position")
            positions[position] = c
        merged = {}
        seen = set()
        for center in centers:
            c = center.verified_chunk
            if c.chunk.chunk_id in seen:
                raise ValueError("Duplicate center")
            seen.add(c.chunk.chunk_id)
            if chunks.get(c.chunk.chunk_id) != c:
                raise ValueError("Center body mismatch")
            selected = [c]
            if c.expandable:
                for delta in (-1, 1):
                    neighbor = positions.get(
                        (c.chunk.document_id, c.chunk.chunk_index + delta)
                    )
                    if (
                        neighbor is not None
                        and neighbor.expandable
                        and neighbor.section_instance_id == c.section_instance_id
                        and neighbor.document_hash == c.document_hash
                    ):
                        selected.append(neighbor)
            selected.sort(key=lambda x: x.chunk.chunk_index)
            ids = tuple(x.chunk.chunk_id for x in selected)
            wid = canonical_hash(
                {
                    "rule": RULE,
                    "document_id": c.chunk.document_id,
                    "section_instance_id": c.section_instance_id,
                    "chunks": [(x.chunk.chunk_id, x.content_hash) for x in selected],
                }
            )
            if wid in merged:
                prior = merged[wid]
                merged[wid] = prior.model_copy(
                    update={
                        "center_chunk_ids": prior.center_chunk_ids
                        + (c.chunk.chunk_id,),
                        "center_candidates": prior.center_candidates
                        + (center.fused_candidate,),
                        "best_rrf": max(
                            prior.best_rrf, center.fused_candidate.rrf_score
                        ),
                    }
                )
            else:
                title = c.chunk.section_path
                bodies = "\n\n".join(
                    f"[chunk:{x.chunk.chunk_id} index:{x.chunk.chunk_index}]\n{x.chunk.answer}"
                    for x in selected
                )
                text = f"{title}\n\n{bodies}" if title else bodies
                merged[wid] = PreparedWindow(
                    window_id=wid,
                    document_id=c.chunk.document_id,
                    section_instance_id=c.section_instance_id,
                    chunk_ids=ids,
                    center_chunk_ids=(c.chunk.chunk_id,),
                    center_candidates=(center.fused_candidate,),
                    best_rrf=center.fused_candidate.rrf_score,
                    body_text=text,
                )
        return tuple(
            PreparedWindow.model_validate(w.model_dump()) for w in merged.values()
        )
    except Exception as exc:
        raise HybridRetrievalError("hybrid_window_contract_failed") from exc


def select_evidence(
    windows: Sequence[PreparedWindow],
    scores: Sequence[float],
    chunks: Mapping[str, VerifiedHybridChunk],
    *,
    limit: int = 10,
):
    try:
        if (
            type(limit) is not int
            or not 1 <= limit <= 10
            or len(windows) > 50
            or len(windows) != len(scores)
        ):
            raise ValueError("Evidence count mismatch")
        if any(isinstance(score, bool) or not math.isfinite(score) for score in scores):
            raise ValueError("Invalid logit")
        pairs = sorted(
            zip(windows, scores, strict=True),
            key=lambda pair: (-pair[1], -pair[0].best_rrf, pair[0].window_id),
        )[:limit]
        evidence = tuple(
            HybridEvidenceWindow(
                **w.model_dump(exclude={"body_text"}), rerank_score=float(score)
            )
            for w, score in pairs
        )
        bodies = {cid: chunks[cid] for w in evidence for cid in w.chunk_ids}
        return evidence, bodies
    except Exception as exc:
        raise HybridRetrievalError("hybrid_evidence_contract_failed") from exc
