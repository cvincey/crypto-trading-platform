# Crypto Trading Platform

A **config-first, extensible** cryptocurrency trading platform for backtesting and live trading on Binance.

## Features

- **Config-First Architecture**: All strategies, backtests, and trading setups defined in YAML files
- **Extensible Plugin System**: Add custom strategies and exchanges without modifying core code
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more
- **Multiple Strategy Types**:
  - Technical indicator strategies (SMA crossover, MACD, etc.)
  - Statistical strategies (mean reversion, z-score)
  - Momentum strategies (breakout, trend following)
  - Machine learning strategies (gradient boosting, random forest)
- **Comprehensive Backtesting**: Performance metrics including Sharpe ratio, max drawdown, win rate
- **Live Trading**: Paper trading and real trading with risk management
- **TimescaleDB**: Optimized time-series storage for market data

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install the package
pip install -e ".[dev]"
```

### 2. Start TimescaleDB (Docker)

```bash
docker run -d --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=crypto \
  -e POSTGRES_DB=crypto \
  timescale/timescaledb:latest-pg16
```

### 3. Run Database Migrations

```bash
alembic upgrade head
```

### 4. Configure API Keys

Create a `.env` file or set environment variables:

```bash
export BINANCE_TESTNET_API_KEY="your_api_key"
export BINANCE_TESTNET_API_SECRET="your_api_secret"
```

### 5. Ingest Market Data

```bash
# Fetch 30 days of BTCUSDT hourly data
crypto ingest fetch --symbol BTCUSDT --interval 1h --days 30
```

### 6. Run a Backtest

```bash
# Run a quick backtest
crypto backtest quick sma_crossover_btc --symbol BTCUSDT --days 30

# Run a configured backtest
crypto backtest run btc_strategies_2024
```

### 7. Start Paper Trading

```bash
crypto trade start sma_crossover_btc --symbol BTCUSDT --paper
```

## Configuration

All configuration is in YAML files in the `config/` directory:

### `config/settings.yaml` - Global Settings

```yaml
database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  name: crypto

trading:
  default_commission: 0.001
  default_slippage: 0.0005
```

### `config/strategies.yaml` - Strategy Definitions

```yaml
strategies:
  sma_crossover_btc:
    type: sma_crossover
    params:
      fast_period: 10
      slow_period: 30
    symbols: [BTCUSDT]
    interval: 1h
```

### `config/backtests.yaml` - Backtest Configurations

```yaml
backtests:
  btc_strategies_2024:
    name: "BTC Strategy Comparison"
    strategies:
      - sma_crossover_btc
      - rsi_mean_reversion
    symbol: BTCUSDT
    start: "2024-01-01"
    end: "2024-12-01"
    initial_capital: 10000
```

## Adding Custom Strategies

1. Create a file in `plugins/strategies/` or `src/crypto/strategies/`:

```python
from crypto.strategies.registry import strategy_registry
from crypto.strategies.base import BaseStrategy

@strategy_registry.register("my_custom_strategy")
class MyCustomStrategy(BaseStrategy):
    name = "my_custom_strategy"
    
    def _setup(self, threshold: float = 0.5, **kwargs):
        self.threshold = threshold
    
    def generate_signals(self, candles):
        # Your strategy logic here
        signals = self.create_signal_series(candles.index)
        # ... set BUY/SELL signals
        return signals
```

2. Add configuration in `config/strategies.yaml`:

```yaml
strategies:
  my_custom:
    type: my_custom_strategy
    params:
      threshold: 0.7
    symbols: [BTCUSDT]
    interval: 1h
```

## CLI Commands

```bash
# Data ingestion
crypto ingest fetch --symbol BTCUSDT --interval 1h --days 30
crypto ingest status --symbol BTCUSDT --interval 1h

# Backtesting
crypto backtest run <backtest_name>
crypto backtest run-all
crypto backtest quick <strategy> --symbol BTCUSDT --days 30

# Strategies
crypto strategies list
crypto strategies show <strategy_name>

# Configuration
crypto config validate
crypto config show

# Trading
crypto trade start <strategy> --symbol BTCUSDT --paper
crypto trade start <strategy> --symbol BTCUSDT --live  # Real trading!
```

## Project Structure

```
crypto/
├── config/                    # YAML configuration files
│   ├── settings.yaml         # Global settings
│   ├── exchanges.yaml        # Exchange credentials
│   ├── strategies.yaml       # Strategy definitions
│   └── backtests.yaml        # Backtest configurations
├── src/crypto/
│   ├── config/               # Configuration loading
│   ├── core/                 # Core types and registry
│   ├── data/                 # Database and data ingestion
│   ├── exchanges/            # Exchange adapters
│   ├── indicators/           # Technical indicators
│   ├── strategies/           # Trading strategies
│   ├── backtesting/          # Backtesting engine
│   ├── trading/              # Live trading
│   └── cli.py                # CLI interface
├── plugins/                   # User plugins
└── tests/                    # Unit tests
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/crypto --cov-report=html
```

## License

MIT
