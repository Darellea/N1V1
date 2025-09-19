"""
Tests for secure secret management functionality.

Tests the integration with Vault/KMS, fallback mechanisms, and key rotation.
"""

import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock
from utils.security import (
    get_secret,
    rotate_key,
    SecureCredentialManager,
    VaultKeyManager,
    AWSKMSKeyManager,
    SecurityException
)


class TestSecretManager:
    """Test suite for secret management functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "security": {
                "vault": {
                    "enabled": True,
                    "url": "https://vault.example.com:8200",
                    "token": "test-token",
                    "mount_point": "secret"
                }
            }
        }

    @pytest.fixture
    def mock_env_dev(self):
        """Set up dev environment for testing."""
        with patch.dict(os.environ, {
            "ENV": "dev",
            "CRYPTOBOT_EXCHANGE_API_KEY": "test-exchange-key",
            "CRYPTOBOT_EXCHANGE_API_SECRET": "test-exchange-secret",
            "CRYPTOBOT_EXCHANGE_API_PASSPHRASE": "test-passphrase",
            "CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN": "test-discord-token",
            "CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID": "test-channel-id",
            "API_KEY": "test-api-key"
        }):
            yield

    @pytest.fixture
    def mock_env_live(self):
        """Set up live environment for testing."""
        with patch.dict(os.environ, {
            "ENV": "live"
        }, clear=True):
            yield

    @pytest.mark.asyncio
    async def test_get_secret_exchange_api_key_dev_mode(self, mock_env_dev):
        """Test getting exchange API key in dev mode (fallback to env)."""
        result = await get_secret("exchange_api_key")
        assert result == "test-exchange-key"

    @pytest.mark.asyncio
    async def test_get_secret_exchange_api_secret_dev_mode(self, mock_env_dev):
        """Test getting exchange API secret in dev mode."""
        result = await get_secret("exchange_api_secret")
        assert result == "test-exchange-secret"

    @pytest.mark.asyncio
    async def test_get_secret_discord_token_dev_mode(self, mock_env_dev):
        """Test getting Discord token in dev mode."""
        result = await get_secret("discord_token")
        assert result == "test-discord-token"

    @pytest.mark.asyncio
    async def test_get_secret_discord_channel_id_dev_mode(self, mock_env_dev):
        """Test getting Discord channel ID in dev mode."""
        result = await get_secret("discord_channel_id")
        assert result == "test-channel-id"

    @pytest.mark.asyncio
    async def test_get_secret_api_key_dev_mode(self, mock_env_dev):
        """Test getting API key in dev mode."""
        result = await get_secret("api_key")
        assert result == "test-api-key"

    @pytest.mark.asyncio
    async def test_get_secret_invalid_name(self):
        """Test getting secret with invalid name."""
        result = await get_secret("invalid_secret")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_secret_vault_success(self, mock_config):
        """Test getting secret from Vault successfully."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.get_secret = AsyncMock(return_value="vault-secret")
            mock_vault_class.return_value = mock_vault_instance

            with patch.dict(os.environ, {"ENV": "live"}):
                result = await get_secret("exchange_api_key")
                assert result == "vault-secret"
                mock_vault_instance.get_secret.assert_called_once_with("exchange", "api_key")

    @pytest.mark.asyncio
    async def test_get_secret_vault_failure_fallback_to_env(self, mock_config, mock_env_dev):
        """Test Vault failure falls back to environment in dev mode."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.get_secret = AsyncMock(return_value=None)
            mock_vault_class.return_value = mock_vault_instance

            result = await get_secret("exchange_api_key")
            assert result == "test-exchange-key"

    @pytest.mark.asyncio
    async def test_get_secret_live_mode_vault_failure_raises_exception(self, mock_config):
        """Test that Vault failure in live mode raises SecurityException."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.get_secret = AsyncMock(return_value=None)
            mock_vault_class.return_value = mock_vault_instance

            with patch.dict(os.environ, {"ENV": "live"}):
                with pytest.raises(SecurityException, match="Required secret 'exchange_api_key' not found"):
                    await get_secret("exchange_api_key")

    @pytest.mark.asyncio
    async def test_rotate_key_success(self, mock_config):
        """Test successful key rotation."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.rotate_key = AsyncMock(return_value=True)
            mock_vault_class.return_value = mock_vault_instance

            result = await rotate_key("exchange_api_key")
            assert result is True
            mock_vault_instance.rotate_key.assert_called_once_with("exchange", "api_key")

    @pytest.mark.asyncio
    async def test_rotate_key_failure(self, mock_config):
        """Test key rotation failure."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.rotate_key = AsyncMock(return_value=False)
            mock_vault_class.return_value = mock_vault_instance

            result = await rotate_key("exchange_api_key")
            assert result is False

    @pytest.mark.asyncio
    async def test_rotate_key_invalid_name(self):
        """Test rotating key with invalid name."""
        result = await rotate_key("invalid_secret")
        assert result is False


class TestSecureCredentialManager:
    """Test suite for SecureCredentialManager."""

    @pytest.fixture
    def mock_config_vault(self):
        """Mock config with Vault enabled."""
        return {
            "security": {
                "vault": {
                    "enabled": True,
                    "url": "https://vault.example.com:8200",
                    "token": "test-token"
                }
            }
        }

    @pytest.fixture
    def mock_config_kms(self):
        """Mock config with KMS enabled."""
        return {
            "security": {
                "kms": {
                    "enabled": True,
                    "region": "us-east-1"
                }
            }
        }

    def test_initialization_vault(self, mock_config_vault):
        """Test SecureCredentialManager initialization with Vault."""
        manager = SecureCredentialManager(mock_config_vault)
        assert isinstance(manager.key_manager, VaultKeyManager)

    def test_initialization_kms(self, mock_config_kms):
        """Test SecureCredentialManager initialization with KMS."""
        manager = SecureCredentialManager(mock_config_kms)
        assert isinstance(manager.key_manager, AWSKMSKeyManager)

    def test_initialization_no_secure_storage(self):
        """Test SecureCredentialManager initialization without secure storage."""
        config = {"security": {"vault": {"enabled": False}, "kms": {"enabled": False}}}
        manager = SecureCredentialManager(config)
        assert manager.key_manager is None

    @pytest.mark.asyncio
    async def test_get_credential_from_secure_storage(self, mock_config_vault):
        """Test getting credential from secure storage."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.get_secret = AsyncMock(return_value="secure-secret")
            mock_vault_class.return_value = mock_vault_instance

            manager = SecureCredentialManager(mock_config_vault)
            result = await manager.get_credential("exchange", "api_key")
            assert result == "secure-secret"

    @pytest.mark.asyncio
    async def test_get_credential_fallback_to_local(self, mock_config_vault):
        """Test fallback to local credentials when secure storage fails."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.get_secret = AsyncMock(return_value=None)
            mock_vault_class.return_value = mock_vault_instance

            manager = SecureCredentialManager(mock_config_vault)
            manager.local_credentials = {"exchange": {"api_key": "local-secret"}}

            result = await manager.get_credential("exchange", "api_key")
            assert result == "local-secret"

    @pytest.mark.asyncio
    async def test_rotate_key_secure_storage(self, mock_config_vault):
        """Test key rotation in secure storage."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.rotate_key = AsyncMock(return_value=True)
            mock_vault_class.return_value = mock_vault_instance

            manager = SecureCredentialManager(mock_config_vault)
            result = await manager.rotate_key("exchange", "api_key")
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_exchange(self, mock_config_vault):
        """Test credential validation for exchange."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.get_secret = AsyncMock(side_effect=lambda s, k: {
                ("exchange", "api_key"): "key",
                ("exchange", "api_secret"): "secret"
            }.get((s, k)))
            mock_vault_class.return_value = mock_vault_instance

            manager = SecureCredentialManager(mock_config_vault)
            result = await manager.validate_credentials()

            assert result["exchange"] is True

    @pytest.mark.asyncio
    async def test_validate_credentials_discord(self, mock_config_vault):
        """Test credential validation for Discord."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.get_secret = AsyncMock(side_effect=lambda s, k: {
                ("discord", "bot_token"): "token"
            }.get((s, k)))
            mock_vault_class.return_value = mock_vault_instance

            manager = SecureCredentialManager(mock_config_vault)
            result = await manager.validate_credentials()

            assert result["discord"] is True

    @pytest.mark.asyncio
    async def test_health_check_vault_up(self, mock_config_vault):
        """Test health check when Vault is up."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.health_check = AsyncMock(return_value=True)
            mock_vault_class.return_value = mock_vault_instance

            manager = SecureCredentialManager(mock_config_vault)
            result = await manager.health_check()

            assert result["key_manager"] is True

    @pytest.mark.asyncio
    async def test_health_check_vault_down(self, mock_config_vault):
        """Test health check when Vault is down."""
        with patch("utils.security.VaultKeyManager") as mock_vault_class:
            mock_vault_instance = MagicMock()
            mock_vault_instance.health_check = AsyncMock(return_value=False)
            mock_vault_class.return_value = mock_vault_instance

            manager = SecureCredentialManager(mock_config_vault)
            result = await manager.health_check()

            assert result["key_manager"] is False


class TestVaultKeyManager:
    """Test suite for VaultKeyManager."""

    @pytest.fixture
    def vault_manager(self):
        """Create VaultKeyManager instance."""
        return VaultKeyManager(
            vault_url="https://vault.example.com:8200",
            token="test-token"
        )

    @pytest.mark.asyncio
    async def test_get_secret_success(self, vault_manager):
        """Test successful secret retrieval from Vault."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": {
                "data": {
                    "test_key": "test_value"
                }
            }
        })

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session_class.return_value = mock_session

            result = await vault_manager.get_secret("test_path", "test_key")
            assert result == "test_value"

    @pytest.mark.asyncio
    async def test_get_secret_not_found(self, vault_manager):
        """Test secret not found in Vault."""
        mock_response = MagicMock()
        mock_response.status = 404

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session_class.return_value = mock_session

            result = await vault_manager.get_secret("test_path", "test_key")
            assert result is None

    @pytest.mark.asyncio
    async def test_store_secret_success(self, vault_manager):
        """Test successful secret storage in Vault."""
        mock_response = MagicMock()
        mock_response.status = 200

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session_class.return_value = mock_session

            result = await vault_manager.store_secret("test_path", "test_key", "test_value")
            assert result is True

    @pytest.mark.asyncio
    async def test_rotate_key(self, vault_manager):
        """Test key rotation in Vault."""
        with patch.object(vault_manager, "store_secret", new_callable=AsyncMock) as mock_store:
            mock_store.return_value = True
            result = await vault_manager.rotate_key("test_path", "test_key")
            assert result is True
            mock_store.assert_called_once()
            # Verify the new value is a valid token
            call_args = mock_store.call_args
            new_value = call_args[0][2]  # Third argument is the value
            assert len(new_value) > 0


class TestAWSKMSKeyManager:
    """Test suite for AWSKMSKeyManager."""

    @pytest.fixture
    def kms_manager(self):
        """Create AWSKMSKeyManager instance."""
        return AWSKMSKeyManager(region="us-east-1")

    @pytest.mark.asyncio
    async def test_get_secret_success(self, kms_manager):
        """Test successful secret retrieval from KMS."""
        mock_parameter = {"Value": "test-secret"}

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_ssm = MagicMock()
            mock_ssm.get_parameter.return_value = {"Parameter": mock_parameter}
            mock_session.client.return_value = mock_ssm
            mock_session_class.return_value = mock_session

            result = await kms_manager.get_secret("test", "key")
            assert result == "test-secret"

    @pytest.mark.asyncio
    async def test_store_secret_success(self, kms_manager):
        """Test successful secret storage in KMS."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_ssm = MagicMock()
            mock_session.client.return_value = mock_ssm
            mock_session_class.return_value = mock_session

            result = await kms_manager.store_secret("test", "key", "value")
            assert result is True
            mock_ssm.put_parameter.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_success(self, kms_manager):
        """Test successful KMS health check."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_kms = MagicMock()
            mock_kms.list_keys.return_value = {"Keys": []}
            mock_session.client.return_value = mock_kms
            mock_session_class.return_value = mock_session

            result = await kms_manager.health_check()
            assert result is True
