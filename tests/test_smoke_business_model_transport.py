import importlib

import pytest

from tests.test_business_model import Stream, final


def test_transport_report_has_real_usage_and_no_execution_claim():
    try:
        module = importlib.import_module("scripts.smoke_business_model_transport")
    except ModuleNotFoundError:
        pytest.fail("explicit transport smoke missing")
    from tests.test_business_model import profile
    from tests.test_context import Counter

    report = module.run_probe(Stream([final()]), profile(), Counter(), question="你好")
    assert report["business_executions"] == 0 and report["actual_input"] == 100
    assert report["accuracy_evidence"] is False and "private thought" not in str(report)
