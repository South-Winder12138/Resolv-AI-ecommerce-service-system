"""Explicit Gate A with synthetic inputs and an owned temporary collection."""
import argparse
import json
import logging
import traceback
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from resolv.core.config import MilvusSettings
from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS, milvus_client
from resolv.repositories.knowledge_hybrid_milvus import (
    HybridMilvusStore,
    collection_marker,
)
from resolv.schemas.hybrid_knowledge import (
    HybridIndexEntry,
    HybridIndexProfile,
    HybridManifestChunk,
    HybridRetrievalError,
)
from resolv.schemas.knowledge import KnowledgeChunk
from resolv.services.hybrid_fingerprints import (
    canonical_hash,
    content_hash,
    profile_hash,
)

PREFIX='mhhybrid_gate_'


def _baseline(client, collections):
    data=[]
    for name in sorted(collections):
        schema=client.describe_collection(collection_name=name,timeout=MILVUS_TIMEOUT_SECONDS)
        for function in schema.get('functions',[]):
            for key in ('input_field_names','input_field_ids','output_field_names','output_field_ids'):
                if key in function:
                    function[key]=list(function[key])
        stats=client.get_collection_stats(collection_name=name,timeout=MILVUS_TIMEOUT_SECONDS)
        data.append({'name':name,'schema':schema,'stats':stats})
    return canonical_hash(data)


def run_gate(client, *, collection_prefix: str, report_dir: Path) -> dict:
    if collection_prefix!=PREFIX: raise HybridRetrievalError('unowned_gate_prefix')
    if report_dir.exists(): raise HybridRetrievalError('gate_report_dir_exists')
    report_dir.mkdir(parents=True)
    owner=uuid4().hex; name=PREFIX+'a_'+owner
    profile=HybridIndexProfile(collection_name=name)
    owned={'owner':owner,'collection':name,'profile_hash':profile_hash(profile),'description':collection_marker(profile)}
    (report_dir/'ownership.json').write_text(json.dumps(owned,indent=2))
    if client.has_collection(collection_name=name,timeout=MILVUS_TIMEOUT_SECONDS):
        raise HybridRetrievalError('unowned_existing_collection')
    existing=client.list_collections(timeout=MILVUS_TIMEOUT_SECONDS)
    before=_baseline(client,existing)
    store=HybridMilvusStore(client,profile)
    created=False; primary=None
    try:
        store.create_or_validate(); created=True
        server=client.get_server_version(timeout=MILVUS_TIMEOUT_SECONDS)
        if server.lstrip('v')!='3.0.2' or version('pymilvus')!='3.0.2':
            raise HybridRetrievalError('gate_version_mismatch')
        doc=sha256(b'gate-a-synthetic.md').hexdigest()
        texts=('型号 MX100 七天内退货，未拆封可以申请。','型号 MX200 已拆封不能无理由退货。',
               '运费 20 元，会员免运费。','保修 12 个月，进水损坏不保修。')
        entries=[]; vectors=[]
        for i,text in enumerate(texts):
            chunk=KnowledgeChunk(chunk_id=sha256(f'{doc}:{i}'.encode()).hexdigest(),document_id=doc,
                chunk_index=i,source='gate-a-synthetic.md',text=text,answer=text,category='',questions='',
                section_path='',content_type='synthetic-policy',is_key_clause=0)
            m=HybridManifestChunk(chunk=chunk,content_hash=content_hash(chunk),section_instance_id='f'*64,expandable=False)
            entries.append(HybridIndexEntry(index_version='gate-a-v3',index_hash=canonical_hash(m.model_dump()),manifest_chunk=m))
            v=[0.0]*1024; v[i]=1.0; vectors.append(v)
        count=store.upsert(entries,vectors)
        if count!=4: raise HybridRetrievalError('gate_partial_upsert')
        client.flush(collection_name=name,timeout=MILVUS_TIMEOUT_SECONDS)
        d=store.search_dense(vectors[0],index_version='gate-a-v3')
        b=store.search_bm25('MX100 七天 未拆封 退货',index_version='gate-a-v3')
        if not d or not b: raise HybridRetrievalError('gate_rpc_empty')
        analyzer=client.run_analyzer(texts=['中文退货 MX100 20元 不支持已拆封退货'],
            collection_name=name,field_name='text',timeout=MILVUS_TIMEOUT_SECONDS)
        after=_baseline(client,existing)
        if before!=after: raise HybridRetrievalError('gate_legacy_changed')
        report={'passed':True,'gate':'A','sdk_version':version('pymilvus'),'server_version':server,
            'collection':name,'profile':profile.model_dump(),'profile_hash':profile_hash(profile),
            'upsert_count':count,'dense_count':len(d),'bm25_count':len(b),
            'dense_candidates':[x.model_dump() for x in d],'bm25_candidates':[x.model_dump() for x in b],
            'analyzer_tokens':[list(item.tokens) for item in analyzer],
            'baseline_before':before,'baseline_after':after,
            'legacy_collection_count':len(existing),'llm_calls':0,'business_tool_calls':0,
            'accuracy_evaluated':False}
    except BaseException as exc:
        primary=exc
        (report_dir/'failure.json').write_text(json.dumps({'passed':False,'gate':'A',
            'code':getattr(exc,'code','gate_a_dependency_failed'),'error_type':type(exc).__name__},indent=2))
        raise
    finally:
        if created:
            try:
                desc=client.describe_collection(collection_name=name,timeout=MILVUS_TIMEOUT_SECONDS)
                if desc.get('description')!=owned['description']:
                    raise HybridRetrievalError('gate_cleanup_ownership_mismatch')
                client.drop_collection(collection_name=name,timeout=MILVUS_TIMEOUT_SECONDS)
            except Exception as exc:
                if primary is None: raise HybridRetrievalError('gate_cleanup_failed') from exc
                primary.add_note('Owned Gate A cleanup also failed; resource retained.')
    report['owned_probe_removed']=True
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-prefix',required=True)
    parser.add_argument('--report-dir',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.report_dir.exists():
        raise SystemExit('Gate A failed: gate_report_dir_exists')
    try:
        with ExitStack() as cleanup:
            for namespace in ('pymilvus','pymilvus.milvus_client'):
                logger=logging.getLogger(namespace)
                cleanup.callback(setattr,logger,'handlers',logger.handlers)
                cleanup.callback(setattr,logger,'propagate',logger.propagate)
                logger.handlers=[logging.NullHandler()]; logger.propagate=False
            with milvus_client(MilvusSettings()) as client:
                report=run_gate(client,collection_prefix=args.collection_prefix,report_dir=args.report_dir)
        (args.report_dir/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 -- report safe codes, never arbitrary SDK credential text
        args.report_dir.mkdir(parents=True,exist_ok=True)
        failure={'passed':False,'gate':'A','code':getattr(exc,'code','gate_a_dependency_failed'),'error_type':type(exc).__name__}
        failure['frames'] = [{'file':Path(f.filename).name,'function':f.name,'line':f.lineno}
                             for f in traceback.extract_tb(exc.__traceback__)]
        (args.report_dir/'command-failure.json').write_text(json.dumps(failure,indent=2))
        raise SystemExit(f"Gate A failed: {failure['code']}") from None


if __name__=='__main__': main()
