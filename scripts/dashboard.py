#!/usr/bin/env python3
"""
Simple Trading Dashboard.

A lightweight web interface to monitor paper trading.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI(title="Crypto Trading Dashboard")

# State file path
STATE_FILE = Path("notes/test_logs/trading_state.json")


def load_state() -> dict:
    """Load trading state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "strategies": {},
        "trades": [],
        "last_update": None,
    }


def get_uptime(start_time: str | None) -> str:
    """Calculate uptime from start time."""
    if not start_time:
        return "N/A"
    try:
        start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - start
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m"
    except:
        return "N/A"


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    state = load_state()
    strategies = state.get("strategies", {})
    trades = state.get("trades", [])[-20:]  # Last 20 trades
    last_update = state.get("last_update", "Never")
    start_time = state.get("start_time")
    
    # Build strategy rows
    strategy_rows = ""
    for name, data in strategies.items():
        signal = data.get("signal", "HOLD")
        signal_class = "signal-hold"
        if signal == "BUY":
            signal_class = "signal-buy"
        elif signal == "SELL":
            signal_class = "signal-sell"
        
        strategy_rows += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{data.get('symbol', 'N/A')}</td>
            <td class="{signal_class}">{signal}</td>
            <td>{data.get('buffer_size', 0)} candles</td>
            <td>{data.get('last_price', 'N/A')}</td>
        </tr>
        """
    
    # Build trade rows
    trade_rows = ""
    if trades:
        for trade in reversed(trades):
            side_class = "trade-buy" if trade.get("side") == "BUY" else "trade-sell"
            trade_rows += f"""
            <tr>
                <td>{trade.get('timestamp', 'N/A')[:19]}</td>
                <td>{trade.get('strategy', 'N/A')}</td>
                <td>{trade.get('symbol', 'N/A')}</td>
                <td class="{side_class}">{trade.get('side', 'N/A')}</td>
                <td>${trade.get('price', 0):.2f}</td>
                <td>{trade.get('quantity', 0):.4f}</td>
            </tr>
            """
    else:
        trade_rows = '<tr><td colspan="6" style="text-align:center;color:#888;">No trades yet - waiting for signals...</td></tr>'
    
    # Calculate stats
    total_trades = len(state.get("trades", []))
    buy_trades = sum(1 for t in state.get("trades", []) if t.get("side") == "BUY")
    sell_trades = total_trades - buy_trades
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Crypto Trading Dashboard</title>
        <meta http-equiv="refresh" content="30">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #eee;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 10px;
            }}
            .subtitle {{
                text-align: center;
                color: #888;
                margin-bottom: 30px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #4ade80;
            }}
            .stat-label {{
                color: #888;
                margin-top: 5px;
            }}
            .card {{
                background: rgba(255,255,255,0.05);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .card h2 {{
                margin-top: 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                padding-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            th {{
                color: #888;
                font-weight: normal;
            }}
            .signal-hold {{ color: #888; }}
            .signal-buy {{ color: #4ade80; font-weight: bold; }}
            .signal-sell {{ color: #f87171; font-weight: bold; }}
            .trade-buy {{ color: #4ade80; }}
            .trade-sell {{ color: #f87171; }}
            .status-dot {{
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #4ade80;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
            .footer {{
                text-align: center;
                color: #666;
                margin-top: 30px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Crypto Trading Dashboard</h1>
            <p class="subtitle">
                <span class="status-dot"></span>
                Paper Trading Active | Last update: {last_update[:19] if last_update else 'Never'}
            </p>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{len(strategies)}</div>
                    <div class="stat-label">Active Strategies</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_trades}</div>
                    <div class="stat-label">Total Trades</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color:#4ade80">{buy_trades}</div>
                    <div class="stat-label">Buy Orders</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color:#f87171">{sell_trades}</div>
                    <div class="stat-label">Sell Orders</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{get_uptime(start_time)}</div>
                    <div class="stat-label">Uptime</div>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Strategy Status</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Symbol</th>
                            <th>Signal</th>
                            <th>Buffer</th>
                            <th>Last Price</th>
                        </tr>
                    </thead>
                    <tbody>
                        {strategy_rows if strategy_rows else '<tr><td colspan="5" style="text-align:center;color:#888;">Loading strategies...</td></tr>'}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>💰 Recent Trades</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Strategy</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Price</th>
                            <th>Quantity</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trade_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                Auto-refreshes every 30 seconds | Paper Trading Mode (No Real Money)
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/api/state")
async def get_state():
    """API endpoint for state."""
    return load_state()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
