import pytest
from scripts.smoke_hybrid_milvus import run_gate

from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from tests.test_knowledge_hybrid_milvus import Client


def test_smoke_rejects_non_task_prefix_before_writes(tmp_path):
    client=Client()
    with pytest.raises(HybridRetrievalError):
        run_gate(client,collection_prefix='resolv_knowledge_',report_dir=tmp_path/'probe')
    assert client.create_calls==client.drop_calls==0


def test_smoke_rejects_existing_unowned_collection(tmp_path):
    client=Client(); client.exists=True
    client.list_collections=lambda **kw: []
    with pytest.raises(HybridRetrievalError):
        run_gate(client,collection_prefix='mhhybrid_gate_',report_dir=tmp_path/'probe')
    assert client.create_calls==client.drop_calls==0


def test_help_does_not_open_connection(monkeypatch,capsys):
    from scripts import smoke_hybrid_milvus as module
    monkeypatch.setattr(module,'milvus_client',lambda *a: pytest.fail('opened resource'))
    with pytest.raises(SystemExit) as caught:
        module.main(['--help'])
    assert caught.value.code==0
    assert '--collection-prefix' in capsys.readouterr().out


def test_command_failure_records_safe_location_without_credentials(tmp_path,monkeypatch,capsys):
    import json
    from contextlib import contextmanager

    from scripts import smoke_hybrid_milvus as module
    @contextmanager
    def owner(*args):
        yield object()
    def fail(*args,**kwargs):
        raise TypeError('private URI password')
    monkeypatch.setattr(module,'milvus_client',owner)
    monkeypatch.setattr(module,'run_gate',fail)
    target=tmp_path/'failure'
    with pytest.raises(SystemExit):
        module.main(['--collection-prefix','mhhybrid_gate_','--report-dir',str(target)])
    report=json.loads((target/'command-failure.json').read_text())
    assert report['frames'][-1]['function']=='fail'
    assert 'private' not in (target/'command-failure.json').read_text()
    assert capsys.readouterr().out==''


def test_real_sdk_analyzer_result_is_serialized_as_plain_tokens(tmp_path):
    import json
    from types import SimpleNamespace

    from pymilvus.client.types import AnalyzeResult
    class ProbeClient(Client):
        def list_collections(self,**kw): return []
        def get_server_version(self,**kw): return '3.0.2'
        def flush(self,**kw): pass
        def upsert(self,**kw):
            self.rows=kw['data']; return {'upsert_count':4}
        def search(self,**kw):
            row=self.rows[0]
            return [[{'id':row['chunk_id'],'distance':0.5,'entity':row}]]
        def run_analyzer(self,**kw):
            return [AnalyzeResult(SimpleNamespace(tokens=[SimpleNamespace(token='中文'),SimpleNamespace(token='MX100')]))]
        def drop_collection(self,**kw):
            self.drop_calls+=1; self.exists=False
    client=ProbeClient()
    report=run_gate(client,collection_prefix='mhhybrid_gate_',report_dir=tmp_path/'gate')
    encoded=json.dumps(report)
    assert json.loads(encoded)['analyzer_tokens']==[['中文','MX100']]
    assert client.drop_calls==1


def test_baseline_fingerprints_real_protobuf_function_containers_without_losing_fields():
    from pymilvus.grpc_gen import schema_pb2
    from scripts.smoke_hybrid_milvus import _baseline

    fn=schema_pb2.FunctionSchema(input_field_names=['text'], input_field_ids=[105],
                               output_field_names=['bm25_vector'], output_field_ids=[107])
    class Existing:
        def describe_collection(self, **kw):
            return {'functions':[{'input_field_names':fn.input_field_names,
                'input_field_ids':fn.input_field_ids, 'output_field_names':fn.output_field_names,
                'output_field_ids':fn.output_field_ids}]}
        def get_collection_stats(self, **kw): return {'row_count':80}
    client=Existing()
    before=_baseline(client,['existing_native_bm25'])
    assert before==_baseline(client,['existing_native_bm25'])
    fn.output_field_ids[0]=108
    assert before!=_baseline(client,['existing_native_bm25'])
