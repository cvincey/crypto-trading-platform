#!/usr/bin/env python3
"""
Run trading bot with web dashboard.

Starts both the multi-strategy trading bot and the monitoring dashboard.
"""

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR))


def load_module_from_file(name: str, filepath: Path):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def run_dashboard():
    """Run the FastAPI dashboard."""
    import uvicorn
    
    # Load dashboard module directly
    dashboard_module = load_module_from_file("dashboard", SCRIPT_DIR / "dashboard.py")
    app = dashboard_module.app
    
    port = int(os.environ.get("PORT", 8080))
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


async def run_trading():
    """Run the trading bot."""
    # Load trading module directly
    trading_module = load_module_from_file("run_all_strategies", SCRIPT_DIR / "run_all_strategies.py")
    await trading_module.main()


async def main():
    """Run both dashboard and trading concurrently."""
    print("🚀 Starting Trading Bot + Dashboard...")
    print(f"📊 Dashboard will be available on port {os.environ.get('PORT', 8080)}")
    print()
    
    # Run both concurrently
    await asyncio.gather(
        run_dashboard(),
        run_trading(),
    )


if __name__ == "__main__":
    asyncio.run(main())
