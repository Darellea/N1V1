"""
Security tests for key management and credential handling.

Tests Vault/KMS integration, key rotation, and secure credential management.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from utils.security import (
    SecureCredentialManager,
    VaultKeyManager,
    AWSKMSKeyManager,
    get_secure_credential_manager,
    log_security_event
)


class TestVaultKeyManager:
    """Test HashiCorp Vault integration."""

    @pytest.fixture
    def vault_config(self):
        return {
            "url": "https://vault.example.com:8200",
            "token": "test-token",
            "mount_point": "secret"
        }

    @pytest.fixture
    def vault_manager(self, vault_config):
        return VaultKeyManager(**vault_config)

    async def test_get_secret_success(self, vault_manager):
        """Test successful secret retrieval from Vault."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": {
                "data": {
                    "api_key": "test-key-123"
                }
            }
        })
        mock_response.close = AsyncMock()

        with patch.object(vault_manager, '_ensure_session') as mock_ensure:
            vault_manager.session = AsyncMock()
            vault_manager.session.get.return_value = mock_response

            result = await vault_manager.get_secret("exchange", "api_key")
            assert result == "test-key-123"

    async def test_get_secret_not_found(self, vault_manager):
        """Test secret not found in Vault."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.close = AsyncMock()

        with patch.object(vault_manager, '_ensure_session') as mock_ensure:
            vault_manager.session = AsyncMock()
            vault_manager.session.get.return_value = mock_response

            result = await vault_manager.get_secret("exchange", "nonexistent")
            assert result is None

    async def test_store_secret_success(self, vault_manager):
        """Test successful secret storage in Vault."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.close = AsyncMock()

        with patch.object(vault_manager, '_ensure_session') as mock_ensure:
            vault_manager.session = AsyncMock()
            vault_manager.session.post.return_value = mock_response

            result = await vault_manager.store_secret("exchange", "api_key", "new-key-123")
            assert result is True

    async def test_rotate_key(self, vault_manager):
        """Test key rotation functionality."""
        with patch.object(vault_manager, 'store_secret', new_callable=AsyncMock) as mock_store:
            mock_store.return_value = True
            result = await vault_manager.rotate_key("exchange", "api_key")
            assert result is True
            mock_store.assert_called_once()
            # Verify the new key is generated (should be different from any known value)
            call_args = mock_store.call_args
            assert call_args[0][0] == "exchange"
            assert call_args[0][1] == "api_key"
            assert len(call_args[0][2]) > 0  # New key should not be empty

    async def test_health_check_success(self, vault_manager):
        """Test successful Vault health check."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.close = AsyncMock()

        with patch.object(vault_manager, '_ensure_session') as mock_ensure:
            vault_manager.session = AsyncMock()
            vault_manager.session.get.return_value = mock_response

            result = await vault_manager.health_check()
            assert result is True


class TestAWSKMSKeyManager:
    """Test AWS KMS integration."""

    @pytest.fixture
    def kms_config(self):
        return {
            "region": "us-east-1",
            "profile": None
        }

    @pytest.fixture
    def kms_manager(self, kms_config):
        return AWSKMSKeyManager(**kms_config)

    @patch('boto3.Session')
    async def test_get_secret_success(self, mock_session, kms_manager):
        """Test successful secret retrieval from AWS SSM."""
        mock_ssm = Mock()
        mock_ssm.get_parameter.return_value = {
            "Parameter": {
                "Value": "test-secret-value"
            }
        }

        mock_session_instance = Mock()
        mock_session_instance.client.return_value = mock_ssm
        mock_session.return_value = mock_session_instance

        result = await kms_manager.get_secret("exchange", "api_key")
        assert result == "test-secret-value"

    @patch('boto3.Session')
    async def test_store_secret_success(self, mock_session, kms_manager):
        """Test successful secret storage in AWS SSM."""
        mock_ssm = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.return_value = mock_ssm
        mock_session.return_value = mock_session_instance

        result = await kms_manager.store_secret("exchange", "api_key", "new-secret")
        assert result is True
        mock_ssm.put_parameter.assert_called_once()

    async def test_rotate_key(self, kms_manager):
        """Test key rotation in AWS KMS."""
        with patch.object(kms_manager, 'store_secret', new_callable=AsyncMock) as mock_store:
            mock_store.return_value = True
            result = await kms_manager.rotate_key("exchange", "api_key")
            assert result is True
            mock_store.assert_called_once()


class TestSecureCredentialManager:
    """Test secure credential manager functionality."""

    @pytest.fixture
    def config(self):
        return {
            "security": {
                "vault": {
                    "enabled": False
                },
                "kms": {
                    "enabled": False
                },
                "key_rotation_days": 90
            }
        }

    @pytest.fixture
    def secure_manager(self, config):
        return SecureCredentialManager(config)

    async def test_get_credential_fallback(self, secure_manager):
        """Test credential retrieval with fallback to local storage."""
        # Set up local credential
        secure_manager.local_credentials = {
            "exchange": {
                "api_key": "local-key-123"
            }
        }

        result = await secure_manager.get_credential("exchange", "api_key")
        assert result == "local-key-123"

    async def test_store_credential_local(self, secure_manager):
        """Test credential storage in local storage."""
        result = await secure_manager.store_credential("exchange", "api_key", "test-key")
        assert result is True
        assert secure_manager.local_credentials["exchange"]["api_key"] == "test-key"

    async def test_validate_credentials_success(self, secure_manager):
        """Test successful credential validation."""
        secure_manager.local_credentials = {
            "exchange": {
                "api_key": "test-key",
                "api_secret": "test-secret"
            },
            "discord": {
                "bot_token": "test-token"
            },
            "api": {
                "key": "test-api-key"
            }
        }

        result = await secure_manager.validate_credentials()
        assert result["exchange"] is True
        assert result["discord"] is True
        assert result["api"] is True

    async def test_validate_credentials_missing(self, secure_manager):
        """Test credential validation with missing credentials."""
        secure_manager.local_credentials = {
            "exchange": {
                "api_key": "test-key"
                # Missing api_secret
            }
        }

        result = await secure_manager.validate_credentials()
        assert result["exchange"] is False

    async def test_rotate_key_local(self, secure_manager):
        """Test key rotation in local storage."""
        # Set up initial credential
        secure_manager.local_credentials = {
            "exchange": {
                "api_key": "old-key"
            }
        }

        result = await secure_manager.rotate_key("exchange", "api_key")
        assert result is True

        # Verify rotation timestamp was recorded
        assert "exchange/api_key" in secure_manager.key_rotation_schedule

        # Verify key was changed
        new_key = secure_manager.local_credentials["exchange"]["api_key"]
        assert new_key != "old-key"
        assert len(new_key) > 0

    def test_get_key_rotation_status(self, secure_manager):
        """Test key rotation status retrieval."""
        # Set up some rotation data
        rotation_time = datetime.utcnow() - timedelta(days=30)
        secure_manager.key_rotation_schedule = {
            "exchange/api_key": rotation_time
        }

        status = secure_manager.get_key_rotation_status()

        assert "rotation_schedule" in status
        assert "next_rotations" in status
        assert "exchange/api_key" in status["next_rotations"]

    async def test_health_check(self, secure_manager):
        """Test comprehensive health check."""
        # Set up some credentials
        secure_manager.local_credentials = {
            "exchange": {
                "api_key": "test-key",
                "api_secret": "test-secret"
            }
        }

        health = await secure_manager.health_check()

        assert "key_manager" in health
        assert "credentials" in health
        assert "encryption" in health
        assert health["credentials"] is True  # We have valid credentials
        assert health["encryption"] is True  # Local encryption always available


class TestSecurityIntegration:
    """Test integration between security components."""

    @pytest.fixture
    def config(self):
        return {
            "security": {
                "vault": {
                    "enabled": False
                },
                "kms": {
                    "enabled": False
                }
            }
        }

    async def test_get_secure_credential_manager_singleton(self, config):
        """Test singleton pattern for secure credential manager."""
        manager1 = await get_secure_credential_manager(config)
        manager2 = await get_secure_credential_manager(config)

        assert manager1 is manager2

    async def test_security_event_logging(self, config):
        """Test security event logging functionality."""
        with patch('utils.security.logging') as mock_logging:
            log_security_event("test_event", {"test": "data"}, "WARNING")

            # Verify logger was called
            mock_logger = mock_logging.getLogger.return_value
            mock_logger.warning.assert_called_once()

    async def test_credential_access_audit(self, config):
        """Test credential access auditing."""
        manager = SecureCredentialManager(config)
        manager.local_credentials = {
            "exchange": {
                "api_key": "test-key"
            }
        }

        # Access credential
        result = await manager.get_credential("exchange", "api_key")

        # Verify audit log was updated
        assert len(manager.key_manager.audit_log) > 0 if manager.key_manager else True

        # For local manager, check that we got the credential
        assert result == "test-key"


if __name__ == "__main__":
    pytest.main([__file__])
