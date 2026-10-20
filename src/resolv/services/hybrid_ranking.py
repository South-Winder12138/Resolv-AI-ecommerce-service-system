"""Equal-weight RRF on full index identities, retaining original route ranks."""
from collections.abc import Sequence

from resolv.schemas.hybrid_knowledge import (
    FusedHybridCandidate,
    HybridCandidate,
    HybridRetrievalError,
)


def fuse_rrf(dense: Sequence[HybridCandidate], bm25: Sequence[HybridCandidate], *, k: int = 60
             ) -> tuple[FusedHybridCandidate, ...]:
    if type(k) is not int or k != 60:
        raise HybridRetrievalError('invalid_rrf_config')
    routes = {}
    for route, values in (('dense', dense), ('bm25', bm25)):
        if len(values) > 50:
            raise HybridRetrievalError('invalid_candidate_count')
        unique = {}
        for value in values:
            if not isinstance(value, HybridCandidate) or value.route != route:
                raise HybridRetrievalError('invalid_candidate')
            # Revalidate even objects created through model_copy/model_construct.
            value = HybridCandidate.model_validate(value.model_dump())
            unique.setdefault(value.identity, value)
        routes[route] = unique
    fused = []
    for identity in routes['dense'].keys() | routes['bm25'].keys():
        d, b = routes['dense'].get(identity), routes['bm25'].get(identity)
        exemplar = d if d is not None else b
        if d is not None and b is not None and d.document_id != b.document_id:
            raise HybridRetrievalError('candidate_document_conflict')
        fused.append(FusedHybridCandidate(
            index_version=identity[0], chunk_id=identity[1], index_hash=identity[2],
            document_id=exemplar.document_id, dense=d, bm25=b,
            rrf_score=sum(1/(k+x.rank) for x in (d,b) if x is not None),
        ))
    return tuple(sorted(fused, key=lambda x: (-x.rrf_score, x.best_rank, x.chunk_id,
                                              x.index_version, x.index_hash)))
