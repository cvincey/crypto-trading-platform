"""Exchange registry for plugin-style exchange support."""

import logging
from typing import Any

from crypto.config.settings import get_settings
from crypto.core.registry import Registry
from crypto.exchanges.base import Exchange

logger = logging.getLogger(__name__)


class ExchangeRegistry(Registry[Exchange]):
    """
    Registry for exchange adapters.
    
    Allows registering exchange implementations and creating
    instances from configuration.
    
    Example:
        @exchange_registry.register("binance")
        class BinanceExchange(BaseExchange):
            ...
        
        # Create from config
        exchange = exchange_registry.create_from_config("binance_testnet")
    """

    def create_from_config(self, exchange_name: str) -> Exchange:
        """
        Create an exchange instance from configuration.
        
        Looks up the exchange in config/exchanges.yaml and
        creates an instance with the configured parameters.
        
        Args:
            exchange_name: Name of the exchange in config
            
        Returns:
            Configured Exchange instance
        """
        settings = get_settings()
        config = settings.get_exchange(exchange_name)
        
        if config is None:
            raise KeyError(
                f"Exchange '{exchange_name}' not found in config. "
                f"Available: {list(settings.exchanges.exchanges.keys())}"
            )

        adapter_name = config.adapter
        return self.create(
            adapter_name,
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.testnet,
            rate_limit=config.rate_limit,
            timeout=config.timeout,
        )


# Global exchange registry instance
exchange_registry = ExchangeRegistry("exchanges")


def get_exchange(name: str) -> Exchange:
    """
    Convenience function to get an exchange from config.
    
    Args:
        name: Exchange name in config
        
    Returns:
        Configured Exchange instance
    """
    return exchange_registry.create_from_config(name)
