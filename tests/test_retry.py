import asyncio
import pytest
from utils.retry import async_retry_call, retry_async


@pytest.mark.asyncio
async def test_async_retry_call_retries_and_succeeds(monkeypatch):
    """
    Simulate a flaky coroutine that fails twice then succeeds.
    Ensure async_retry_call retries and returns the final result.
    """
    attempts = {"count": 0}

    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return "ok"

    # Patch asyncio.sleep to avoid real delays during test
    async def _fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    result = await async_retry_call(lambda: flaky(), retries=3, base_backoff=0.01, max_backoff=0.02)
    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_retry_async_decorator(monkeypatch):
    """
    Ensure the decorator retries wrapped async function on failure.
    """
    calls = {"n": 0}

    async def _fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    @retry_async(retries=2, base_backoff=0.01, max_backoff=0.02)
    async def flaky_func(x):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("fail")
        return x * 2

    res = await flaky_func(3)
    assert res == 6
    assert calls["n"] == 2
