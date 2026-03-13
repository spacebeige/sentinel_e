"""Legacy orchestrator diagnostic placeholder.

This module historically contained a manual script and is not part of the
maintained automated unit-test suite.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Legacy manual diagnostic module; automated coverage lives under backend/tests/.")


def test_legacy_orchestrator_placeholder():
    """Placeholder so pytest collects module cleanly without import-time side effects."""
    assert True
