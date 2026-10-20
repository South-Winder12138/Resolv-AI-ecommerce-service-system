"""Release CLIs expose explicit reviewed targets; importing performs no I/O."""

import pytest
from scripts import (
    activate_knowledge_release,
    prepare_knowledge_release,
    publish_knowledge_release,
)


@pytest.mark.parametrize(
    "module",
    [prepare_knowledge_release, publish_knowledge_release, activate_knowledge_release],
)
def test_commands_require_explicit_arguments(module):
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code == 2
