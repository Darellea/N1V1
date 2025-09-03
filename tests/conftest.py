# Ensure project root is on sys.path so pytest can import local packages when run
# from arbitrary working directories or when path resolution behaves unexpectedly.
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Discord Integration Test Fixtures
def pytest_configure(config):
    """Configure pytest with Discord test settings."""
    config.addinivalue_line("markers", "discord_live: mark test to use live Discord API")


@pytest.fixture(scope="session", autouse=True)
def mock_aiohttp_for_discord():
    """
    Mock aiohttp.ClientSession.post for Discord integration tests.

    By default, mocks all aiohttp POST requests to return successful Discord responses.
    Can be disabled by setting DISCORD_LIVE_TEST=true environment variable.

    Usage:
    - Normal testing: pytest tests/test_discord_integration.py (uses mocks)
    - Live testing: DISCORD_LIVE_TEST=true pytest tests/test_discord_integration.py (makes real API calls)
    - Live test specific: pytest -m discord_live tests/test_discord_integration.py (only runs live tests)
    """
    # Check if live testing is enabled
    live_test = os.getenv("DISCORD_LIVE_TEST", "false").lower() == "true"

    if live_test:
        # Skip mocking for live tests
        yield
        return

    # Mock aiohttp for testing
    import aiohttp

    original_post = aiohttp.ClientSession.post

    async def mock_post(self, url, **kwargs):
        """Mock POST method that returns successful Discord response."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}

        # Mock json() method to return Discord message ID
        mock_response.json = AsyncMock(return_value={"id": "1234567890"})
        mock_response.text = AsyncMock(return_value='{"id": "1234567890"}')
        mock_response.close = AsyncMock()

        return mock_response

    # Patch the post method
    aiohttp.ClientSession.post = mock_post

    yield

    # Restore original method
    aiohttp.ClientSession.post = original_post
