#!/usr/bin/env python3
"""
Enhanced Trading Dashboard with Ratio Monitoring.

A web interface to monitor:
- Paper trading status
- Real-time ratio z-scores across all pairs
- Signal opportunities
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = FastAPI(title="Crypto Trading Dashboard")

# State file path
STATE_FILE = Path("notes/test_logs/trading_state.json")
RATIO_CONFIG_FILE = Path("config/research/ratio_universe.yaml")


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


def load_ratio_config() -> dict:
    """Load ratio universe configuration."""
    if RATIO_CONFIG_FILE.exists():
        with open(RATIO_CONFIG_FILE) as f:
            return yaml.safe_load(f).get("ratio_universe", {})
    return {}


def calculate_ratio_zscores(lookback: int = 72) -> list[dict]:
    """
    Calculate current z-scores for all ratio pairs.
    
    Returns list of dicts with pair info and current z-score.
    """
    try:
        from crypto.data.repository import DataRepository
        repo = DataRepository()
    except Exception:
        return []
    
    config = load_ratio_config()
    pairs = config.get("pairs", {})
    
    results = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=lookback * 2)
    
    for pair_name, pair_config in pairs.items():
        target = pair_config.get("target")
        reference = pair_config.get("reference")
        status = pair_config.get("status", "candidate")
        params = pair_config.get("params", {})
        lb = params.get("lookback", lookback)
        entry_threshold = params.get("entry_threshold", -1.2)
        
        try:
            target_data = repo.get_candles(target, "1h", start_date, end_date)
            ref_data = repo.get_candles(reference, "1h", start_date, end_date)
            
            if target_data.empty or ref_data.empty:
                continue
            
            # Align and calculate ratio
            common_idx = target_data.index.intersection(ref_data.index)
            if len(common_idx) < lb:
                continue
            
            target_close = target_data.loc[common_idx, "close"]
            ref_close = ref_data.loc[common_idx, "close"]
            
            ratio = target_close / ref_close
            ratio_mean = ratio.rolling(lb).mean()
            ratio_std = ratio.rolling(lb).std()
            z_score = (ratio - ratio_mean) / ratio_std
            
            current_z = float(z_score.iloc[-1]) if not pd.isna(z_score.iloc[-1]) else None
            current_ratio = float(ratio.iloc[-1])
            ratio_pct_change = float((ratio.iloc[-1] / ratio.iloc[-lb] - 1) * 100) if len(ratio) > lb else 0
            
            # Determine signal
            signal = "HOLD"
            if current_z is not None:
                if current_z < entry_threshold:
                    signal = "BUY"
                elif current_z > -entry_threshold:  # Symmetric
                    signal = "OVERBOUGHT"
            
            results.append({
                "pair_name": pair_name,
                "pair": f"{target[:-4]}/{reference[:-4]}",
                "target": target,
                "reference": reference,
                "status": status,
                "z_score": round(current_z, 2) if current_z else None,
                "ratio": round(current_ratio, 6),
                "ratio_pct_change": round(ratio_pct_change, 2),
                "entry_threshold": entry_threshold,
                "signal": signal,
            })
        except Exception as e:
            results.append({
                "pair_name": pair_name,
                "pair": f"{target[:-4]}/{reference[:-4]}",
                "status": "error",
                "error": str(e),
            })
    
    # Sort by z-score (most oversold first)
    results.sort(key=lambda x: x.get("z_score") or 999)
    
    return results


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
            
            <div class="card" style="text-align:center;">
                <a href="/ratios" style="color:#4ade80;text-decoration:none;font-size:1.2em;">
                    📊 View Ratio Universe Dashboard →
                </a>
            </div>
            
            <div class="footer">
                Auto-refreshes every 30 seconds | Paper Trading Mode (No Real Money)
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/ratios", response_class=HTMLResponse)
async def ratio_dashboard(request: Request):
    """Ratio monitoring dashboard - shows z-scores for all pairs."""
    ratio_data = calculate_ratio_zscores()
    
    # Separate by status
    deployed = [r for r in ratio_data if r.get("status") == "deployed"]
    candidates = [r for r in ratio_data if r.get("status") == "candidate"]
    retired = [r for r in ratio_data if r.get("status") == "retired"]
    
    def make_rows(pairs: list[dict]) -> str:
        rows = ""
        for r in pairs:
            z = r.get("z_score")
            signal = r.get("signal", "")
            
            # Color code z-score
            if z is None:
                z_class = "z-neutral"
                z_display = "N/A"
            elif z < -1.5:
                z_class = "z-extreme-low"
                z_display = f"{z:.2f}"
            elif z < -1.0:
                z_class = "z-low"
                z_display = f"{z:.2f}"
            elif z > 1.0:
                z_class = "z-high"
                z_display = f"{z:.2f}"
            else:
                z_class = "z-neutral"
                z_display = f"{z:.2f}"
            
            # Signal styling
            if signal == "BUY":
                signal_class = "signal-buy"
            elif signal == "OVERBOUGHT":
                signal_class = "signal-sell"
            else:
                signal_class = "signal-hold"
            
            # Ratio change arrow
            pct = r.get("ratio_pct_change", 0)
            if pct > 0:
                arrow = "↑"
                pct_class = "pct-up"
            elif pct < 0:
                arrow = "↓"
                pct_class = "pct-down"
            else:
                arrow = "→"
                pct_class = ""
            
            rows += f"""
            <tr>
                <td><strong>{r.get('pair', '?')}</strong></td>
                <td class="{z_class}">{z_display}</td>
                <td>{r.get('entry_threshold', -1.2)}</td>
                <td class="{signal_class}">{signal}</td>
                <td class="{pct_class}">{arrow} {abs(pct):.1f}%</td>
                <td>{r.get('ratio', 0):.6f}</td>
            </tr>
            """
        return rows
    
    deployed_rows = make_rows(deployed) or '<tr><td colspan="6" style="text-align:center;color:#888;">No deployed pairs</td></tr>'
    candidate_rows = make_rows(candidates) or '<tr><td colspan="6" style="text-align:center;color:#888;">No candidate pairs</td></tr>'
    
    # Count signals
    buy_signals = sum(1 for r in ratio_data if r.get("signal") == "BUY")
    oversold_count = sum(1 for r in ratio_data if (r.get("z_score") or 0) < -1.0)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>📊 Ratio Universe Dashboard</title>
        <meta http-equiv="refresh" content="60">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
                background: #0d1117;
                color: #c9d1d9;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h1 {{
                text-align: center;
                font-size: 2em;
                background: linear-gradient(135deg, #58a6ff 0%, #a371f7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 8px;
            }}
            .subtitle {{
                text-align: center;
                color: #8b949e;
                margin-bottom: 24px;
            }}
            .nav {{
                display: flex;
                justify-content: center;
                gap: 16px;
                margin-bottom: 24px;
            }}
            .nav a {{
                color: #58a6ff;
                text-decoration: none;
                padding: 8px 16px;
                border: 1px solid #30363d;
                border-radius: 6px;
            }}
            .nav a:hover {{ background: #21262d; }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }}
            .stat-card {{
                background: #161b22;
                border: 1px solid #30363d;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
            }}
            .stat-value.green {{ color: #3fb950; }}
            .stat-value.yellow {{ color: #d29922; }}
            .stat-value.blue {{ color: #58a6ff; }}
            .stat-label {{ color: #8b949e; margin-top: 4px; }}
            .card {{
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .card h2 {{
                margin: 0 0 16px 0;
                font-size: 1.2em;
                color: #c9d1d9;
            }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px 8px; text-align: left; }}
            th {{ color: #8b949e; font-weight: 500; border-bottom: 1px solid #30363d; }}
            td {{ border-bottom: 1px solid #21262d; }}
            tr:hover {{ background: #21262d; }}
            .z-extreme-low {{ color: #3fb950; font-weight: bold; background: rgba(63,185,80,0.1); padding: 4px 8px; border-radius: 4px; }}
            .z-low {{ color: #7ee787; }}
            .z-high {{ color: #f85149; }}
            .z-neutral {{ color: #8b949e; }}
            .signal-buy {{ color: #3fb950; font-weight: bold; }}
            .signal-sell {{ color: #f85149; }}
            .signal-hold {{ color: #8b949e; }}
            .pct-up {{ color: #3fb950; }}
            .pct-down {{ color: #f85149; }}
            .footer {{
                text-align: center;
                color: #484f58;
                margin-top: 24px;
                font-size: 0.85em;
            }}
            .legend {{
                display: flex;
                gap: 24px;
                justify-content: center;
                margin-bottom: 20px;
                font-size: 0.9em;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .legend-dot {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
            }}
            .dot-green {{ background: #3fb950; }}
            .dot-yellow {{ background: #d29922; }}
            .dot-gray {{ background: #484f58; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Ratio Universe Dashboard</h1>
            <p class="subtitle">Real-time z-scores for ratio mean reversion strategies</p>
            
            <div class="nav">
                <a href="/">🏠 Main Dashboard</a>
                <a href="/ratios">📊 Ratio Monitor</a>
                <a href="/api/ratios">🔌 API</a>
            </div>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot dot-green"></div>
                    <span>z &lt; -1.5 = Strong Buy</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot dot-yellow"></div>
                    <span>-1.5 &lt; z &lt; -1.0 = Watch</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot dot-gray"></div>
                    <span>z &gt; -1.0 = Neutral</span>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value blue">{len(ratio_data)}</div>
                    <div class="stat-label">Total Pairs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value green">{buy_signals}</div>
                    <div class="stat-label">Buy Signals</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value yellow">{oversold_count}</div>
                    <div class="stat-label">Oversold (z&lt;-1)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value blue">{len(deployed)}</div>
                    <div class="stat-label">Deployed</div>
                </div>
            </div>
            
            <div class="card">
                <h2>🚀 Deployed Pairs</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Pair</th>
                            <th>Z-Score</th>
                            <th>Threshold</th>
                            <th>Signal</th>
                            <th>Change (7d)</th>
                            <th>Ratio</th>
                        </tr>
                    </thead>
                    <tbody>
                        {deployed_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>🔍 Candidate Pairs</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Pair</th>
                            <th>Z-Score</th>
                            <th>Threshold</th>
                            <th>Signal</th>
                            <th>Change (7d)</th>
                            <th>Ratio</th>
                        </tr>
                    </thead>
                    <tbody>
                        {candidate_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                Auto-refreshes every 60 seconds | Data from last {72} hours | 
                Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/api/state")
async def get_state():
    """API endpoint for trading state."""
    return load_state()


@app.get("/api/ratios")
async def get_ratios():
    """API endpoint for ratio z-scores."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ratios": calculate_ratio_zscores(),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
