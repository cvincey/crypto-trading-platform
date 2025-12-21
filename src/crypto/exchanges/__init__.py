"""Exchange adapters module."""

from crypto.exchanges.base import Exchange
from crypto.exchanges.registry import exchange_registry

__all__ = [
    "Exchange",
    "exchange_registry",
]
