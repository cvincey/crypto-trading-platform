"""Generic registry pattern for extensible plugin architecture."""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Generic registry for plugin-style extensibility.
    
    Supports:
    - Decorator-based registration
    - Auto-discovery from directories
    - Creating instances from config with parameters
    
    Example:
        class MyRegistry(Registry[MyProtocol]):
            pass
        
        my_registry = MyRegistry("my_plugin")
        
        @my_registry.register("my_implementation")
        class MyImpl:
            def __init__(self, param: int = 10):
                self.param = param
    """

    def __init__(self, name: str):
        """Initialize registry with a name for logging."""
        self.name = name
        self._items: dict[str, type[T]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        *,
        description: str = "",
        **metadata: Any,
    ) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a class.
        
        Args:
            name: Unique name to register the class under
            description: Human-readable description
            **metadata: Additional metadata to store
            
        Returns:
            Decorator function
        """

        def decorator(klass: type[T]) -> type[T]:
            if name in self._items:
                logger.warning(
                    f"[{self.name}] Overwriting existing registration: {name}"
                )
            self._items[name] = klass
            self._metadata[name] = {
                "description": description,
                "class": klass.__name__,
                "module": klass.__module__,
                **metadata,
            }
            logger.debug(f"[{self.name}] Registered: {name} -> {klass.__name__}")
            return klass

        return decorator

    def get(self, name: str) -> type[T]:
        """
        Get a registered class by name.
        
        Args:
            name: The registered name
            
        Returns:
            The registered class
            
        Raises:
            KeyError: If name is not registered
        """
        if name not in self._items:
            available = ", ".join(self._items.keys())
            raise KeyError(
                f"[{self.name}] '{name}' not found. Available: {available}"
            )
        return self._items[name]

    def create(self, name: str, **params: Any) -> T:
        """
        Create an instance of a registered class with parameters.
        
        Args:
            name: The registered name
            **params: Parameters to pass to the constructor
            
        Returns:
            Instance of the registered class
        """
        klass = self.get(name)
        return klass(**params)

    def create_from_config(self, config: dict[str, Any]) -> T:
        """
        Create an instance from a config dict.
        
        Expected config format:
            {
                "type": "registered_name",
                "params": {"param1": value1, ...}
            }
            
        Args:
            config: Configuration dictionary
            
        Returns:
            Instance of the registered class
        """
        type_name = config.get("type")
        if not type_name:
            raise ValueError("Config must have 'type' field")
        params = config.get("params", {})
        return self.create(type_name, **params)

    def list(self) -> list[str]:
        """Get list of all registered names."""
        return list(self._items.keys())

    def list_with_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all registered items with their metadata."""
        return {name: self._metadata.get(name, {}) for name in self._items}

    def contains(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._items

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return self.contains(name)

    def __len__(self) -> int:
        """Get number of registered items."""
        return len(self._items)

    def discover_plugins(self, directory: Path | str) -> int:
        """
        Auto-discover and load plugins from a directory.
        
        Imports all .py files in the directory, which will trigger
        any @registry.register decorators.
        
        Args:
            directory: Path to directory containing plugin modules
            
        Returns:
            Number of modules loaded
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"[{self.name}] Plugin directory not found: {directory}")
            return 0

        loaded = 0
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                module_name = f"plugins.{directory.name}.{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    loaded += 1
                    logger.info(f"[{self.name}] Loaded plugin: {py_file.name}")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to load {py_file}: {e}")

        return loaded


def discover_all_plugins(plugin_dir: Path | str = "plugins") -> None:
    """
    Discover all plugins in the plugins directory.
    
    This imports all Python files in plugin subdirectories,
    triggering registration decorators.
    """
    plugin_path = Path(plugin_dir)
    if not plugin_path.exists():
        logger.info(f"No plugins directory found at {plugin_path}")
        return

    for subdir in plugin_path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("_"):
            for py_file in subdir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                try:
                    module_name = f"plugins.{subdir.name}.{py_file.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        logger.debug(f"Loaded plugin: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {py_file}: {e}")
