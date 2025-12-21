#!/usr/bin/env python3
"""
Run trading bot with web dashboard.

Starts both the multi-strategy trading bot and the monitoring dashboard.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def run_dashboard():
    """Run the FastAPI dashboard."""
    import uvicorn
    from scripts.dashboard import app
    
    port = int(os.environ.get("PORT", 8080))
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


async def run_trading():
    """Run the trading bot."""
    # Import here to avoid circular imports
    from scripts.run_all_strategies import main as trading_main
    await trading_main()


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
