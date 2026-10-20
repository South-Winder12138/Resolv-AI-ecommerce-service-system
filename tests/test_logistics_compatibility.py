import inspect as pyinspect
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from resolv.db.engine import create_db_engine, create_session_factory
from resolv.schemas.conversation import ConversationError
from tests.branch_fixtures import branch_factory
from tests.test_logistics_contracts import api
from tests.test_logistics_mcp_server import profile
from tests.test_logistics_routing import harness

__all__ = ["branch_factory"]
ROOT = Path(__file__).resolve().parents[1]


def test_default_runtime_accepts_complete_0005_without_attempts(tmp_path):
    module = api("resolv.services.routed_runtime")
    assert hasattr(module, "validate_conversation_runtime_store"), "missing version-aware store validation"
    engine = create_db_engine(f"sqlite:///{tmp_path / 'old.db'}")
    factory = create_session_factory(engine)
    with engine.begin() as c:
        cfg = Config(str(ROOT / "alembic.conversation.ini"))
        cfg.attributes["connection"] = c
        command.upgrade(cfg, "20261016_0005")
        for name in ("checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"):
            c.exec_driver_sql(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY)")
    module.validate_conversation_runtime_store(factory)
    assert "conversation_logistics_attempts" not in inspect(engine).get_table_names()
    with pytest.raises(ConversationError, match="conversation_store_failed"):
        module.validate_conversation_runtime_store(factory, require_logistics=True)
    with engine.begin() as c:
        cfg.attributes["connection"] = c
        command.upgrade(cfg, "head")
    module.validate_conversation_runtime_store(factory, require_logistics=True)
    with engine.begin() as c:
        c.exec_driver_sql("DROP TABLE conversation_logistics_attempts")
    with pytest.raises(ConversationError, match="conversation_store_failed"):
        module.validate_conversation_runtime_store(factory)
    engine.dispose()


def test_v1_run_stays_local_under_new_runtime(branch_factory, tool_database):
    from resolv.schemas.branch_execution import BusinessRunConfig
    from resolv.services.conversation import ConversationService

    assert "logistics_read_service" in pyinspect.signature(ConversationService).parameters
    service, _, _, remote = harness(branch_factory, tool_database)
    service.business_run_config = BusinessRunConfig.model_validate({k: v for k, v in service.business_run_config.model_dump(mode="json").items() if k not in {"logistics_backend", "tool_policy_version"}} | {"config_version": 1, "prompt_version": "business-v3"})
    c = service.create()
    run, _ = service.accept(c.conversation_id, "v1", "查询订单 MH001 的物流", "agent")
    list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "completed"
    assert remote.calls == []
    assert not service.detail(c.conversation_id).messages[-1].text.startswith("【模拟物流】")


def test_profile_drift_rejects_resume_before_any_remote_read(branch_factory, tool_database):
    service, run, _, remote = harness(branch_factory, tool_database)
    remote.error = __import__("resolv.schemas.logistics", fromlist=["LogisticsFailure"]).LogisticsFailure("logistics_timeout", True)
    list(service.run(run.run_id))
    service.logistics_read_service.profile = profile().model_copy(update={"dataset_version": "changed"})
    events = list(service.resume(run.conversation_id, run.run_id))
    assert any(e.type == "error" and e.data["code"] == "run_state_incompatible" for e in events), events
    assert len(remote.calls) == 1
