import pytest


@pytest.fixture(autouse=True)
def anyio_backend():
    return 'asyncio'
