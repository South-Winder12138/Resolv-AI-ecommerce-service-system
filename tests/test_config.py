import pytest
from pydantic import ValidationError

from resolv.core import config
from resolv.core.config import Settings


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert settings.openai_model == "test-model"
    assert settings.openai_base_url == "https://gateway.example.com/v1"


def test_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert Settings(_env_file=None).openai_model == "gpt-6-sol"


def test_missing_api_key_is_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")

    with pytest.raises(ValidationError, match="openai_api_key"):
        Settings(_env_file=None)


def test_missing_base_url_is_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValidationError, match="openai_base_url"):
        Settings(_env_file=None)


def test_milvus_default_does_not_require_openai(monkeypatch) -> None:
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "MILVUS_URI"):
        monkeypatch.delenv(name, raising=False)
    settings = config.MilvusSettings(_env_file=None)
    assert settings.milvus_uri == "http://127.0.0.1:19530"


def test_milvus_settings_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("MILVUS_URI", "  http://localhost:19531  ")
    settings = config.MilvusSettings(_env_file=None)
    assert settings.milvus_uri == "http://localhost:19531"


@pytest.mark.parametrize(
    "invalid_uri", ["", "   ", "./milvus.db", "http://", "http://localhost:invalid"]
)
def test_milvus_settings_rejects_invalid_uri(invalid_uri) -> None:
    with pytest.raises(ValidationError, match="milvus_uri"):
        config.MilvusSettings(milvus_uri=invalid_uri, _env_file=None)


def test_milvus_default_collection_independent(monkeypatch):
    for name in ("MILVUS_COLLECTION", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert (
        config.MilvusSettings(_env_file=None).milvus_collection
        == "resolv_knowledge_v1"
    )


def test_milvus_collection_environment(monkeypatch):
    monkeypatch.setenv("MILVUS_COLLECTION", "knowledge_v2")
    assert config.MilvusSettings(_env_file=None).milvus_collection == "knowledge_v2"


@pytest.mark.parametrize(
    "name", ["", " ", "a b", "a-b", "知识", "1knowledge", "a" * 256]
)
def test_milvus_collection_rejects_invalid_names(name):
    with pytest.raises(ValidationError):
        config.MilvusSettings(milvus_collection=name, _env_file=None)


EMBEDDING_ENV = (
    "BGE_M3_MODEL",
    "EMBEDDING_DEVICE",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_USE_FP16",
    "EMBEDDING_QUERY_MAX_LENGTH",
    "EMBEDDING_PASSAGE_MAX_LENGTH",
)


@pytest.fixture
def clean_embedding_env(monkeypatch):
    for name in (*EMBEDDING_ENV, "OPENAI_API_KEY", "OPENAI_BASE_URL", "MILVUS_URI"):
        monkeypatch.delenv(name, raising=False)


def test_embedding_defaults_are_cpu_fp32_without_external_settings(clean_embedding_env):
    settings = config.EmbeddingSettings(_env_file=None)
    assert settings.model_name == "BAAI/bge-m3"
    assert settings.device == "cpu"
    assert settings.batch_size == 8
    assert settings.use_fp16 is False
    assert settings.query_max_length == 512
    assert settings.passage_max_length == 512


def test_embedding_settings_reads_environment(clean_embedding_env, monkeypatch):
    values = (" /tmp/local-bge-m3 ", " mps ", "16", "true", "128", "2048")
    for name, value in zip(EMBEDDING_ENV, values, strict=True):
        monkeypatch.setenv(name, value)
    settings = config.EmbeddingSettings(_env_file=None)
    assert settings.model_name == "/tmp/local-bge-m3"
    assert settings.device == "mps"
    assert settings.batch_size == 16
    assert settings.use_fp16 is True
    assert settings.query_max_length == 128
    assert settings.passage_max_length == 2048


def test_embedding_settings_explicit_model_name_and_limits(clean_embedding_env):
    settings = config.EmbeddingSettings(
        model_name="./models/bge-m3",
        batch_size=1,
        query_max_length=1,
        passage_max_length=8192,
        _env_file=None,
    )
    assert settings.model_name == "./models/bge-m3"
    assert settings.batch_size == 1
    assert settings.query_max_length == 1
    assert settings.passage_max_length == 8192


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", ""),
        ("model_name", "  "),
        ("device", ""),
        ("device", " \t"),
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", 1.5),
        ("query_max_length", 0),
        ("query_max_length", -1),
        ("query_max_length", 8193),
        ("passage_max_length", 0),
        ("passage_max_length", -1),
        ("passage_max_length", 8193),
    ],
)
def test_embedding_settings_rejects_invalid_values(clean_embedding_env, field, value):
    with pytest.raises(ValidationError):
        config.EmbeddingSettings(**{field: value}, _env_file=None)


KNOWLEDGE_URI = "mysql+pymysql://knowledge:fake%40password@127.0.0.1:3307/resolv_knowledge?charset=utf8mb4"


def test_knowledge_database_is_independent_and_secret(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MILVUS_URI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///business.db")
    monkeypatch.setenv("KNOWLEDGE_DATABASE_URL", f"  {KNOWLEDGE_URI}  ")
    settings = config.KnowledgeDatabaseSettings(_env_file=None)
    assert settings.knowledge_database_url.get_secret_value() == KNOWLEDGE_URI
    assert "fake%40password" not in repr(settings)
    assert "fake%40password" not in settings.model_dump_json()


def test_knowledge_database_url_required_even_with_business_url(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", KNOWLEDGE_URI)
    with pytest.raises(ValidationError, match="knowledge_database_url"):
        config.KnowledgeDatabaseSettings(_env_file=None)


@pytest.mark.parametrize(
    "uri",
    [
        "", " ", "sqlite:///knowledge.db",
        "mysql+asyncmy://u:p@host/db?charset=utf8mb4",
        "mysql+pymysql://u:p@/db?charset=utf8mb4",
        "mysql+pymysql://host/db?charset=utf8mb4",
        "mysql+pymysql://u@host/db?charset=utf8mb4",
        "mysql+pymysql://u:p@host/?charset=utf8mb4",
        "mysql+pymysql://u:p@host:0/db?charset=utf8mb4",
        "mysql+pymysql://u:p@host:-1/db?charset=utf8mb4",
        "mysql+pymysql://u:p@host:65536/db?charset=utf8mb4",
        "mysql+pymysql://u:p@host:bad/db?charset=utf8mb4",
        "mysql+pymysql://u:p@host/db",
        "mysql+pymysql://u:p@host/db?charset=latin1",
    ],
)
def test_knowledge_database_rejects_invalid_url_without_exposing_input(uri):
    with pytest.raises(ValidationError) as captured:
        config.KnowledgeDatabaseSettings(knowledge_database_url=uri, _env_file=None)
    assert "input_value=" not in str(captured.value)


def test_knowledge_database_allows_driver_default_port():
    uri = "mysql+pymysql://u:p@localhost/db?charset=utf8mb4"
    assert config.KnowledgeDatabaseSettings(
        knowledge_database_url=uri, _env_file=None
    ).knowledge_database_url.get_secret_value() == uri


@pytest.mark.parametrize("option", [
    "port=bad", "database=other", "host=other", "user=root",
    "password=private_override", "init_command=CREATE%20TABLE%20unexpected(id%20INT)",
    "connect_timeout=999", "ssl_ca=unknown",
])
def test_knowledge_url_rejects_unsupported_driver_query_options(option):
    with pytest.raises(ValidationError) as captured:
        config.KnowledgeDatabaseSettings(
            knowledge_database_url=KNOWLEDGE_URI + "&" + option, _env_file=None
        )
    assert "input_value=" not in str(captured.value)
    assert "private_override" not in str(captured.value)
