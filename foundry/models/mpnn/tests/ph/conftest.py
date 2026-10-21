import pytest
from engine_imports import engine_module


@pytest.fixture(scope="module")
def engine():
    """The real engine module, imported with only the missing third-party deps stubbed."""
    with engine_module() as module:
        yield module
