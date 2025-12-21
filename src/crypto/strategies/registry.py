"""Strategy registry for plugin-style strategy support."""

import logging
from typing import Any

from crypto.config.settings import get_settings
from crypto.core.registry import Registry
from crypto.strategies.base import Strategy

logger = logging.getLogger(__name__)


class StrategyRegistry(Registry[Strategy]):
    """
    Registry for trading strategies.
    
    Allows registering strategy implementations and creating
    instances from configuration.
    
    Example:
        @strategy_registry.register("sma_crossover")
        class SMACrossoverStrategy(BaseStrategy):
            ...
        
        # Create from config
        strategy = strategy_registry.create_from_config("sma_crossover_btc")
    """

    def create_from_config(self, strategy_name: str) -> Strategy:
        """
        Create a strategy instance from configuration.
        
        Looks up the strategy in config/strategies.yaml and
        creates an instance with the configured parameters.
        
        Args:
            strategy_name: Name of the strategy in config
            
        Returns:
            Configured Strategy instance
        """
        settings = get_settings()
        config = settings.get_strategy(strategy_name)

        if config is None:
            raise KeyError(
                f"Strategy '{strategy_name}' not found in config. "
                f"Available: {settings.strategies.list_enabled()}"
            )

        strategy_type = config.type
        params = config.params

        logger.debug(f"Creating strategy '{strategy_name}' of type '{strategy_type}'")
        return self.create(strategy_type, **params)

    def list_available(self) -> list[dict[str, Any]]:
        """
        List all available strategy types with their metadata.
        
        Returns:
            List of strategy info dicts
        """
        return [
            {
                "name": name,
                **meta,
            }
            for name, meta in self.list_with_metadata().items()
        ]


# Global strategy registry instance
strategy_registry = StrategyRegistry("strategies")


def get_strategy(name: str) -> Strategy:
    """
    Convenience function to get a strategy from config.
    
    Args:
        name: Strategy name in config
        
    Returns:
        Configured Strategy instance
    """
    return strategy_registry.create_from_config(name)
