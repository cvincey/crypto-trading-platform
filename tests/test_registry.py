"""Tests for the registry system."""

import pytest

from crypto.core.registry import Registry


class DummyProtocol:
    """Dummy protocol for testing."""
    name: str


class TestRegistry:
    """Tests for Registry class."""

    def test_register_and_get(self):
        """Test registering and getting a class."""
        registry = Registry[DummyProtocol]("test")

        @registry.register("dummy")
        class DummyClass:
            name = "dummy"

        assert "dummy" in registry
        assert registry.get("dummy") == DummyClass

    def test_create_instance(self):
        """Test creating an instance from registry."""
        registry = Registry[DummyProtocol]("test")

        @registry.register("parameterized")
        class ParameterizedClass:
            def __init__(self, value: int = 10):
                self.value = value

        instance = registry.create("parameterized", value=42)
        assert instance.value == 42

    def test_create_from_config(self):
        """Test creating instance from config dict."""
        registry = Registry[DummyProtocol]("test")

        @registry.register("configurable")
        class ConfigurableClass:
            def __init__(self, param1: str = "default", param2: int = 0):
                self.param1 = param1
                self.param2 = param2

        config = {
            "type": "configurable",
            "params": {"param1": "custom", "param2": 100},
        }

        instance = registry.create_from_config(config)
        assert instance.param1 == "custom"
        assert instance.param2 == 100

    def test_get_not_found(self):
        """Test getting a non-existent class."""
        registry = Registry[DummyProtocol]("test")

        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_registered(self):
        """Test listing registered items."""
        registry = Registry[DummyProtocol]("test")

        @registry.register("item1")
        class Item1:
            pass

        @registry.register("item2")
        class Item2:
            pass

        items = registry.list()
        assert "item1" in items
        assert "item2" in items
        assert len(items) == 2

    def test_contains(self):
        """Test contains method."""
        registry = Registry[DummyProtocol]("test")

        @registry.register("exists")
        class ExistsClass:
            pass

        assert registry.contains("exists") is True
        assert registry.contains("not_exists") is False
        assert "exists" in registry

    def test_register_with_metadata(self):
        """Test registration with metadata."""
        registry = Registry[DummyProtocol]("test")

        @registry.register("with_meta", description="A test class", version="1.0")
        class MetaClass:
            pass

        metadata = registry.list_with_metadata()
        assert metadata["with_meta"]["description"] == "A test class"
        assert metadata["with_meta"]["version"] == "1.0"
