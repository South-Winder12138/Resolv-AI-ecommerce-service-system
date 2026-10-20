import importlib

import pytest


@pytest.mark.parametrize('name,args,filename', [
    ('smoke_hybrid_milvus', ['--collection-prefix','mhhybrid_gate_'], 'command-failure.json'),
    ('smoke_hybrid_reranker', ['--revision','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e','--device','mps'], 'failure.json'),
    ('smoke_hybrid_retrieval', ['--version','v3','--collection','mhhybrid_gate_fixture',
        '--reranker-revision','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e','--query','独立query'], 'failure.json'),
])
def test_existing_report_directory_preserves_original_failure_before_resources(
    tmp_path, monkeypatch, capsys, name, args, filename
):
    module=importlib.import_module('scripts.'+name)
    target=tmp_path/'previous-attempt'
    target.mkdir()
    original=b'{"passed": false, "code": "original_failure"}\n'
    (target/filename).write_bytes(original)
    boundary='milvus_client' if name=='smoke_hybrid_milvus' else ('run_gate' if name=='smoke_hybrid_reranker' else 'run')
    calls=[]

    def forbidden(*a, **kw):
        calls.append(True)
        raise AssertionError('resource boundary reached')

    monkeypatch.setattr(module,boundary,forbidden)
    with pytest.raises(SystemExit):
        module.main([*args,'--report-dir',str(target)])
    assert calls==[]
    assert capsys.readouterr().out==''
    assert {p.name:p.read_bytes() for p in target.iterdir()}=={filename:original}
