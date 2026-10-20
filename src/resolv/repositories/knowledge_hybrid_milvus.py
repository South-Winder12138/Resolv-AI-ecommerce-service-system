"""Native Dense/BM25 on an explicit new collection, without SQL authority."""
import json
from collections.abc import Sequence

import numpy as np
from pymilvus import DataType, Function, FunctionType, MilvusClient

from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS
from resolv.schemas.hybrid_knowledge import (
    HybridCandidate,
    HybridIndexEntry,
    HybridIndexProfile,
    HybridRetrievalError,
    Version,
)
from resolv.services.hybrid_fingerprints import profile_hash

_FIELDS = {'chunk_id':64, 'document_id':64, 'index_version':128, 'index_hash':64, 'text':8192}
_OUTPUT = ['chunk_id','document_id','index_version','index_hash']


def collection_marker(profile: HybridIndexProfile) -> str:
    return 'resolv-hybrid-v3:' + profile_hash(profile)


def _schema(profile):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False,
                                        description=collection_marker(profile))
    for name, capacity in _FIELDS.items():
        options = {'enable_analyzer':True,'analyzer_params':{'type':profile.analyzer}} if name=='text' else {}
        schema.add_field(name, DataType.VARCHAR, is_primary=name=='chunk_id', max_length=capacity, **options)
    schema.add_field('dense_vector', DataType.FLOAT_VECTOR, dim=profile.dimension)
    schema.add_field('bm25_vector', DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name='text_bm25', function_type=FunctionType.BM25,
                                 input_field_names=['text'], output_field_names=['bm25_vector']))
    return schema


def _index_params(profile):
    params = MilvusClient.prepare_index_params()
    params.add_index(field_name='dense_vector', index_name='dense_cosine',
                     index_type=profile.dense_index_type, metric_type=profile.dense_metric)
    params.add_index(field_name='bm25_vector', index_name='bm25_sparse',
                     index_type=profile.sparse_index_type, metric_type=profile.sparse_metric,
                     params={'bm25_k1':profile.bm25_k1,'bm25_b':profile.bm25_b})
    return params


class HybridMilvusStore:
    def __init__(self, client, profile: HybridIndexProfile):
        self.client = client
        self.profile = HybridIndexProfile.model_validate(profile.model_dump())
        self.collection_name = self.profile.collection_name

    def create_or_validate(self) -> None:
        try:
            if not self.client.has_collection(collection_name=self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS):
                self.client.create_collection(collection_name=self.collection_name,
                    schema=_schema(self.profile), index_params=_index_params(self.profile),
                    consistency_level='Strong', timeout=MILVUS_TIMEOUT_SECONDS)
            self.validate_existing()
        except Exception as exc:
            raise HybridRetrievalError('hybrid_collection_setup_failed') from exc

    def validate_existing(self) -> None:
        try:
            desc = self.client.describe_collection(collection_name=self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS)
            if desc.get('description') != collection_marker(self.profile):
                raise ValueError('Profile marker mismatch')
            if desc['auto_id'] is not False or desc['enable_dynamic_field'] is not False:
                raise ValueError('Explicit schema required')
            fields = desc['fields']
            if len(fields)!=7 or {f['name'] for f in fields} != {*_FIELDS,'dense_vector','bm25_vector'}:
                raise ValueError('Schema fields mismatch')
            for field in fields:
                name = field['name']
                if bool(field.get('is_primary',False)) != (name=='chunk_id') or field.get('nullable',False):
                    raise ValueError('Primary/nullability mismatch')
                params = field.get('params',{})
                if name in _FIELDS:
                    if field['type'] != DataType.VARCHAR or int(params['max_length']) != _FIELDS[name]:
                        raise ValueError('VARCHAR mismatch')
                    if name=='text':
                        enabled = params.get('enable_analyzer',field.get('enable_analyzer'))
                        analyzer = params.get('analyzer_params',field.get('analyzer_params'))
                        if isinstance(analyzer,str):
                            analyzer = json.loads(analyzer)
                        if enabled not in (True,'true') or analyzer != {'type':self.profile.analyzer}:
                            raise ValueError('Analyzer mismatch')
                elif name=='dense_vector':
                    if field['type'] != DataType.FLOAT_VECTOR or int(params['dim']) != 1024:
                        raise ValueError('Dense mismatch')
                elif field['type'] != DataType.SPARSE_FLOAT_VECTOR:
                    raise ValueError('Sparse mismatch')
            functions = desc.get('functions',[])
            if len(functions)!=1:
                raise ValueError('BM25 function missing or ambiguous')
            fn=functions[0]
            if (fn['type'] != FunctionType.BM25 or fn['input_field_names'] != ['text']
                    or fn['output_field_names'] != ['bm25_vector'] or fn.get('params',{})):
                raise ValueError('BM25 function mismatch')
            names = self.client.list_indexes(collection_name=self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS)
            vector_indexes = {}
            for name in names:
                idx=self.client.describe_index(collection_name=self.collection_name,index_name=name,timeout=MILVUS_TIMEOUT_SECONDS)
                field=idx['field_name']
                if field in ('dense_vector','bm25_vector'):
                    if field in vector_indexes:
                        raise ValueError('Ambiguous vector index')
                    vector_indexes[field]=idx
            dense=vector_indexes['dense_vector']; sparse=vector_indexes['bm25_vector']
            if dense['index_type']!='FLAT' or dense['metric_type']!='COSINE':
                raise ValueError('Dense index mismatch')
            if sparse['index_type']!='SPARSE_INVERTED_INDEX' or sparse['metric_type']!='BM25':
                raise ValueError('Sparse index mismatch')
            pars=sparse.get('params',{})
            if isinstance(pars,str): pars=json.loads(pars)
            if (float(pars.get('bm25_k1',sparse.get('bm25_k1',-1)))!=1.2
                    or float(pars.get('bm25_b',sparse.get('bm25_b',-1)))!=0.75):
                raise ValueError('BM25 parameters mismatch')
            self.client.load_collection(collection_name=self.collection_name,timeout=MILVUS_TIMEOUT_SECONDS)
        except Exception as exc:
            raise HybridRetrievalError('hybrid_collection_incompatible') from exc

    def _vectors(self, vectors, count):
        try:
            values=np.asarray(vectors)
            if values.dtype.kind not in 'fiu' or values.shape != (count,1024):
                raise ValueError('Invalid dense vector')
            with np.errstate(over='raise',invalid='raise'):
                values=values.astype(np.float32)
            if not np.isfinite(values).all() or not np.any(values!=0,axis=1).all():
                raise ValueError('Dense vector must be finite/nonzero')
            return values.tolist()
        except Exception as exc:
            raise HybridRetrievalError('invalid_dense_vector') from exc

    def upsert(self, entries: Sequence[HybridIndexEntry], vectors: Sequence[Sequence[float]]) -> int:
        try:
            if len(entries)!=len(vectors): raise ValueError('Count mismatch')
            if not entries: return 0
            rows=[]; ids=set()
            for entry in entries:
                entry=HybridIndexEntry.model_validate(entry.model_dump())
                row=entry.milvus_projection()
                if row['chunk_id'] in ids: raise ValueError('Duplicate PK')
                ids.add(row['chunk_id'])
                for name,capacity in _FIELDS.items():
                    if not row[name].strip() or len(row[name].encode('utf-8'))>capacity:
                        raise ValueError('Invalid scalar capacity')
                rows.append(row)
            values=self._vectors(vectors,len(rows))
            self.validate_existing()
            for row,value in zip(rows,values,strict=True): row['dense_vector']=value
            res=self.client.upsert(collection_name=self.collection_name,data=rows,timeout=MILVUS_TIMEOUT_SECONDS)
            count=res['upsert_count']
            if type(count) is not int or not 0<=count<=len(rows): raise ValueError('Invalid upsert count')
            return count
        except Exception as exc:
            raise HybridRetrievalError('hybrid_upsert_failed') from exc

    def _search(self,data,route,index_version,limit):
        try:
            from pydantic import TypeAdapter
            TypeAdapter(Version).validate_python(index_version,strict=True)
            if type(limit) is not int or not 1<=limit<=50: raise ValueError('Invalid search limit')
            self.validate_existing()
            response=self.client.search(collection_name=self.collection_name,data=data,
                anns_field='dense_vector' if route=='dense' else 'bm25_vector',limit=limit,
                filter=f'index_version == "{index_version}"',output_fields=_OUTPUT,
                search_params={'metric_type':'COSINE' if route=='dense' else 'BM25','params':{}},
                consistency_level='Strong',timeout=MILVUS_TIMEOUT_SECONDS)
            if not isinstance(response,list) or len(response)!=1 or not isinstance(response[0],list) or len(response[0])>limit:
                raise ValueError('Invalid candidate container')
            result=[]
            for rank,raw in enumerate(response[0],1):
                entity=raw['entity']; pk=raw.get('chunk_id',raw.get('id'))
                if raw.get('id',pk)!=pk or entity.get('chunk_id',pk)!=pk or entity['index_version']!=index_version:
                    raise ValueError('Invalid candidate identity')
                result.append(HybridCandidate(index_version=entity['index_version'],chunk_id=pk,
                    document_id=entity['document_id'],index_hash=entity['index_hash'],route=route,rank=rank,raw_score=raw['distance']))
            return tuple(result)
        except Exception as exc:
            raise HybridRetrievalError('hybrid_'+route+'_search_failed') from exc

    def search_dense(self, vector: Sequence[float], *, index_version: str, limit: int=50) -> tuple[HybridCandidate,...]:
        return self._search(self._vectors([vector],1),'dense',index_version,limit)

    def search_bm25(self, query: str, *, index_version: str, limit: int=50) -> tuple[HybridCandidate,...]:
        if not isinstance(query,str) or not query.strip(): raise HybridRetrievalError('invalid_query')
        return self._search([query],'bm25',index_version,limit)
