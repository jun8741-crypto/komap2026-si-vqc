import pytest

from qsi import io


@pytest.fixture(scope="session")
def data_splits():
    return io.split_by_alloy(io.load_data())
