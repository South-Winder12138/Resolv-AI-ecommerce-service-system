import importlib

import pytest

from resolv.schemas.understanding import UnderstandingError


@pytest.mark.parametrize(
    "url", ["sqlite:///local.db", "mysql+pymysql://a:b@localhost/resolv_conversations"]
)
def test_adversarial_smoke_rejects_nonisolated_database_before_model(url):
    try:
        mod = importlib.import_module("scripts.smoke_understanding_adversarial")
    except ModuleNotFoundError:
        pytest.fail("explicit adversarial smoke missing")
    with pytest.raises(UnderstandingError, match="isolated_mysql_required"):
        mod.isolated_database(url)
