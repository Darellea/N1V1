"""
core/execution/live_executor.py

Handles live order execution and exchange communication.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

import ccxt.async_support as ccxt
from ccxt.base.errors import ExchangeError, NetworkError

from core.api_protection import (
    get_default_circuit_breaker,
    get_default_rate_limiter,
    guarded_call,
)
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class LiveOrderExecutor:
    """Handles live order execution and exchange communication."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the LiveOrderExecutor.

        Args:
            config: Configuration dictionary with exchange settings
        """
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self._initialize_exchange()

    def _initialize_exchange(self) -> None:
        """Initialize the exchange connection.

        Prefer environment variables for sensitive credentials (backwards-compatible).
        Supported env vars (see .env.example):
          - CRYPTOBOT_EXCHANGE_API_KEY
          - CRYPTOBOT_EXCHANGE_API_SECRET
          - CRYPTOBOT_EXCHANGE_API_PASSPHRASE
        """
        exch_cfg = (
            self.config.get("exchange", {}) if isinstance(self.config, dict) else {}
        )
        api_key = os.getenv("CRYPTOBOT_EXCHANGE_API_KEY")
        api_secret = os.getenv("CRYPTOBOT_EXCHANGE_API_SECRET")
        api_pass = os.getenv("CRYPTOBOT_EXCHANGE_API_PASSPHRASE")

        exchange_config = {
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_pass,
            "enableRateLimit": True,
            "options": {
                "defaultType": exch_cfg.get("default_type", "spot"),
            },
        }
        exchange_name = (
            exch_cfg.get("name")
            or self.config.get("exchange", {}).get("name")
            or "binance"
        )
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class(exchange_config)

    async def _create_order_on_exchange(self, order_params: Dict[str, Any]) -> Dict:
        """
        Adapter that calls ccxt.create_order using a safe positional-args first strategy
        and falling back to keyword-args. This helps support exchanges with different
        create_order signatures and provides a single place to adapt our internal
        order dict to ccxt. All calls are protected by circuit breaker and rate limiter.
        """
        symbol = order_params.get("symbol")
        otype = order_params.get("type") or order_params.get("order_type") or "market"
        side = order_params.get("side")
        amount = order_params.get("amount") or order_params.get("size") or 0
        price = order_params.get("price", None)
        params = order_params.get("params", {}) or {}

        # Get protection instances
        circuit_breaker = get_default_circuit_breaker()
        rate_limiter = get_default_rate_limiter()

        # Try positional API first (common ccxt signature)
        try:
            return await guarded_call(
                self.exchange.create_order,
                symbol,
                otype,
                side,
                amount,
                price,
                params,
                circuit_breaker=circuit_breaker,
                rate_limiter=rate_limiter,
            )
        except TypeError:
            # Some adapters accept kwargs - try a kwargs call as a fallback
            try:
                return await guarded_call(
                    self.exchange.create_order,
                    symbol=symbol,
                    type=otype,
                    side=side,
                    amount=amount,
                    price=price,
                    params=params,
                    circuit_breaker=circuit_breaker,
                    rate_limiter=rate_limiter,
                )
            except Exception:
                # Re-raise original exception for upstream handling
                raise
        except Exception:
            # Propagate other exceptions (network/exchange errors) to caller
            raise

    async def execute_live_order(self, signal: Any) -> Dict[str, Any]:
        """
        Execute a live order on the exchange.

        This method is fully async-first and offloads any CPU-bound operations
        to thread pools to prevent blocking the event loop.

        Args:
            signal: Object providing order details.

        Returns:
            Dictionary containing order execution details.
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized for live trading")

        # Move order parameter processing to thread pool (CPU-bound)
        order_params = await asyncio.to_thread(self._prepare_order_params, signal)

        try:
            # Execute the order with timeout protection
            response = await asyncio.wait_for(
                self._create_order_on_exchange(order_params),
                timeout=30.0,  # 30 second timeout for order execution
            )
            return response

        except asyncio.TimeoutError:
            logger.error("Order execution timed out after 30 seconds")
            raise RuntimeError("Order execution timed out")
        except (NetworkError, ExchangeError) as e:
            logger.error(f"Exchange error during order execution: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during live order execution: {str(e)}")
            raise

    def _prepare_order_params(self, signal: Any) -> Dict[str, Any]:
        """
        Prepare order parameters from signal. This is CPU-bound work.

        Args:
            signal: Object providing order details

        Returns:
            Dictionary of order parameters
        """
        # Determine side from signal type if not explicitly provided
        side = None
        if isinstance(signal, dict):
            side = signal.get("side")
        else:
            side = getattr(signal, "side", None)

        # If side is not provided, map from signal_type
        if side is None and hasattr(signal, "signal_type"):
            from core.contracts import SignalType

            signal_type = getattr(signal, "signal_type", None)
            if signal_type == SignalType.ENTRY_LONG:
                side = "buy"
            elif signal_type == SignalType.ENTRY_SHORT:
                side = "sell"
            elif signal_type == SignalType.EXIT_LONG:
                side = "sell"
            elif signal_type == SignalType.EXIT_SHORT:
                side = "buy"

        return {
            "symbol": getattr(
                signal,
                "symbol",
                signal.get("symbol") if isinstance(signal, dict) else None,
            ),
            "type": getattr(getattr(signal, "order_type", None), "value", None)
            if not isinstance(signal, dict)
            else signal.get("order_type"),
            "side": side,
            "amount": float(
                getattr(
                    signal,
                    "amount",
                    signal.get("amount") if isinstance(signal, dict) else 0,
                )
            ),
            "price": float(
                getattr(
                    signal,
                    "price",
                    signal.get("price") if isinstance(signal, dict) else None,
                )
            )
            if getattr(
                signal,
                "price",
                signal.get("price", None) if isinstance(signal, dict) else None,
            )
            else None,
            "params": getattr(
                signal,
                "params",
                signal.get("params")
                if isinstance(signal, dict)
                else getattr(signal, "metadata", {}),
            )
            or {},
        }

    async def shutdown(self) -> None:
        """Cleanup exchange resources."""
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.error(f"Error during exchange shutdown: {str(e)}")
            finally:
                self.exchange = None
