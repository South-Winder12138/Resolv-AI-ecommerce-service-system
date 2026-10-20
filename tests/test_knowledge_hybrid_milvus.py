"""Locked SDK schema objects with RPC doubles; no live connection."""
from copy import deepcopy

import pytest
from pymilvus import DataType, MilvusClient

from resolv.repositories.knowledge_hybrid_milvus import HybridMilvusStore
from resolv.schemas.hybrid_knowledge import HybridIndexEntry, HybridRetrievalError
from tests.hybrid_fixtures import manifest_chunk, profile


class Client:
    create_schema = staticmethod(MilvusClient.create_schema)
    prepare_index_params = staticmethod(MilvusClient.prepare_index_params)

    def __init__(self):
        self.exists = False
        self.create_calls = self.drop_calls = 0
        self.schema = None
        self.indexes = {}
        self.events = []
        self.response = [[]]
        self.error = None
        self.count = 1

    def has_collection(self, **kw):
        return self.exists

    def create_collection(self, **kw):
        self.create_calls += 1
        self.schema = kw['schema'].to_dict()
        self.indexes = {i.to_dict()['index_name']:i.to_dict() for i in kw['index_params']}
        self.exists = True

    def describe_collection(self, **kw):
        if not self.exists:
            raise RuntimeError('missing')
        return deepcopy(self.schema)

    def list_indexes(self, **kw):
        return list(self.indexes)

    def describe_index(self, index_name, **kw):
        return deepcopy(self.indexes[index_name])

    def load_collection(self, **kw):
        self.events.append(('load',kw))

    def search(self, **kw):
        self.events.append(('search',kw))
        if self.error:
            raise self.error
        return deepcopy(self.response)

    def upsert(self, **kw):
        self.events.append(('upsert',kw))
        if self.error:
            raise self.error
        return {'upsert_count':self.count}


def existing():
    client = Client()
    store = HybridMilvusStore(client,profile())
    store.create_or_validate()
    client.create_calls = 0
    return client,store


def vector():
    return [1.0]+[0.0]*1023


def hit(**changes):
    data = {'id':'a'*64,'distance':-0.1,'entity':{
        'chunk_id':'a'*64,'document_id':'d'*64,'index_version':'fixture-v3','index_hash':'c'*64}}
    data.update(changes)
    return data


def test_search_never_creates_or_replaces_collection():
    client,store = existing()
    client.response = [[hit()]]
    d = store.search_dense(vector(),index_version='fixture-v3')
    b = store.search_bm25('原query',index_version='fixture-v3')
    assert client.create_calls == client.drop_calls == 0
    assert (d[0].route,b[0].route,d[0].rank,b[0].rank) == ('dense','bm25',1,1)
    assert d[0].raw_score == -0.1 and d[0].index_hash == 'c'*64
    requests = [kw for name,kw in client.events if name=='search']
    assert requests[-1]['data'] == ['原query']
    for req in requests:
        assert req['limit'] == 50 and req['timeout'] == 10 and req['consistency_level']=='Strong'
        assert set(req['output_fields']) == {'chunk_id','document_id','index_version','index_hash'}
        assert req['filter'] == 'index_version == "fixture-v3"'


def test_new_collection_has_exact_bm25_schema_function_and_indexes():
    client, _ = existing()
    fields={x['name']:x for x in client.schema['fields']}
    assert set(fields)=={'chunk_id','document_id','index_version','index_hash','text','dense_vector','bm25_vector'}
    assert fields['dense_vector']['type']==DataType.FLOAT_VECTOR
    assert fields['dense_vector']['params']['dim']==1024
    assert fields['text']['params']['enable_analyzer'] is True
    assert fields['text']['params']['analyzer_params']=='{"type":"chinese"}'
    assert fields['bm25_vector']['type']==DataType.SPARSE_FLOAT_VECTOR
    fn=client.schema['functions'][0]
    assert fn['input_field_names']==['text'] and fn['output_field_names']==['bm25_vector']
    assert int(fn['type'])==1
    assert client.indexes['dense_cosine']['index_type']=='FLAT'
    assert client.indexes['dense_cosine']['metric_type']=='COSINE'
    sparse=client.indexes['bm25_sparse']
    assert sparse['index_type']=='SPARSE_INVERTED_INDEX' and sparse['metric_type']=='BM25'
    assert (sparse['bm25_k1'], sparse['bm25_b'])==(1.2,0.75)


@pytest.mark.parametrize('mutation',['dimension','analyzer','function','metric','bm25','missing_index','dynamic'])
def test_incompatible_existing_is_rejected_without_mutation(mutation):
    client,_=existing()
    fields={x['name']:x for x in client.schema['fields']}
    if mutation=='dimension': fields['dense_vector']['params']['dim']=3
    elif mutation=='analyzer': fields['text']['params']['analyzer_params']='{"type":"standard"}'
    elif mutation=='function': client.schema['functions']=[]
    elif mutation=='metric': client.indexes['dense_cosine']['metric_type']='L2'
    elif mutation=='bm25': client.indexes['bm25_sparse']['bm25_b']=0.1
    elif mutation=='missing_index': client.indexes.pop('bm25_sparse')
    else: client.schema['enable_dynamic_field']=True
    with pytest.raises(HybridRetrievalError):
        HybridMilvusStore(client,profile()).create_or_validate()
    assert client.create_calls == client.drop_calls == 0


def test_missing_collection_is_not_recreated_by_read():
    client=Client()
    with pytest.raises(HybridRetrievalError):
        HybridMilvusStore(client,profile()).search_bm25('退款',index_version='fixture-v3')
    assert client.create_calls == client.drop_calls == 0


@pytest.mark.parametrize('marker',['missing','foreign','different_profile'])
@pytest.mark.parametrize('operation',['setup','dense','bm25','upsert'])
def test_profile_marker_mismatch_is_rejected_before_load_or_rpc(marker, operation):
    client,store=existing()
    client.events.clear()
    if marker=='missing':
        client.schema.pop('description')
    elif marker=='foreign':
        client.schema['description']='another-owner'
    else:
        store=HybridMilvusStore(client,profile().model_copy(update={'passage_max_length':1024}))
    entry=HybridIndexEntry(index_version='fixture-v3',index_hash='c'*64,manifest_chunk=manifest_chunk())
    with pytest.raises(HybridRetrievalError):
        if operation=='setup': store.create_or_validate()
        elif operation=='dense': store.search_dense(vector(),index_version='fixture-v3')
        elif operation=='bm25': store.search_bm25('退款',index_version='fixture-v3')
        else: store.upsert((entry,),(vector(),))
    assert client.events==[]
    assert client.create_calls==client.drop_calls==0


@pytest.mark.parametrize('bad',[[],{},[None],[[hit(distance=float('nan'))]],[[hit(distance=True)]],
                                 [[hit(id='short')]],[[hit()]*51]])
def test_bad_candidates_are_faults(bad):
    client,store=existing(); client.response=bad
    with pytest.raises(HybridRetrievalError):
        store.search_dense(vector(),index_version='fixture-v3')


def test_primary_mismatch_or_wrong_version_is_fault():
    client,store=existing()
    for changes in ({'chunk_id':'b'*64},{'index_version':'other'}):
        value=hit(); value['entity'].update(changes); client.response=[[value]]
        with pytest.raises(HybridRetrievalError):
            store.search_bm25('退款',index_version='fixture-v3')


def test_rpc_failure_is_not_empty_and_retains_chain():
    client,store=existing(); client.error=RuntimeError('private URI password')
    with pytest.raises(HybridRetrievalError) as caught:
        store.search_bm25('退款',index_version='fixture-v3')
    assert caught.value.__cause__ is client.error
    assert 'private' not in str(caught.value)


def test_upsert_projection_stable_key_and_whole_batch_count():
    client,store=existing()
    entry=HybridIndexEntry(index_version='fixture-v3',index_hash='c'*64,manifest_chunk=manifest_chunk())
    assert store.upsert((entry,),(vector(),))==1
    row=client.events[-1][1]['data'][0]
    assert set(row)=={'chunk_id','document_id','index_version','index_hash','text','dense_vector'}
    assert row['chunk_id']=='a'*64 and len(row['dense_vector'])==1024
    client.count=0
    assert store.upsert((entry,),(vector(),))==0
    for count in (True,-1,2,None):
        client.count=count
        with pytest.raises(HybridRetrievalError):
            store.upsert((entry,),(vector(),))


@pytest.mark.parametrize('bad',[[0.0]*1024,[1.0]*3,[True]*1024,[float('inf')]*1024])
def test_bad_dense_vectors_do_not_reach_rpc(bad):
    client,store=existing()
    with pytest.raises(HybridRetrievalError):
        store.search_dense(bad,index_version='fixture-v3')
    assert not any(name=='search' for name,_ in client.events)


def test_empty_routes_are_valid():
    _,store=existing()
    assert store.search_dense(vector(),index_version='fixture-v3')==()
    assert store.search_bm25('退款',index_version='fixture-v3')==()
