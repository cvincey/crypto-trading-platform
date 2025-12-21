---
name: Strategy Improvements and Additions
overview: Config-first implementation of stop-loss/take-profit, improved existing strategies with configurable filters, and 8 new strategies. All parameters driven by YAML configuration files.
todos:
  - id: config-settings-risk
    content: Add risk management defaults to config/settings.yaml
    status: completed
  - id: config-schemas
    content: Add Pydantic schemas for new config sections (risk, filters)
    status: completed
  - id: engine-risk
    content: Add stop-loss/take-profit/trailing-stop to BacktestEngine (reads from config)
    status: completed
  - id: base-strategy-risk
    content: Add optional risk parameters to BaseStrategy (from strategy config)
    status: completed
  - id: indicators-new
    content: Add ADX, OBV, Keltner, VWAP indicators
    status: completed
  - id: improve-technical
    content: Add configurable ADX filter and volume confirmation to existing technical strategies
    status: completed
  - id: improve-statistical
    content: Add configurable volume confirmation to statistical strategies
    status: completed
  - id: strategy-ensemble
    content: Create EnsembleVotingStrategy (references sub-strategies by config name)
    status: completed
  - id: strategy-regime
    content: Create RegimeAdaptiveStrategy with configurable thresholds
    status: completed
  - id: strategy-ml-enhanced
    content: Create MLEnhancedStrategy with configurable feature list
    status: completed
  - id: strategy-mtf
    content: Create MultiTimeframeStrategy with configurable intervals
    status: completed
  - id: strategy-squeeze
    content: Create VolatilitySqueezeStrategy with configurable bands
    status: completed
  - id: strategy-vwap
    content: Create VWAPReversionStrategy with configurable thresholds
    status: completed
  - id: strategy-rotation
    content: Create RelativeStrengthStrategy with configurable ranking params
    status: completed
  - id: strategy-rl
    content: Create RLDQNStrategy with configurable hyperparameters
    status: completed
  - id: config-strategies-update
    content: Add all new strategy configurations to strategies.yaml
    status: pending
  - id: deps-update
    content: Add RL dependencies to pyproject.toml
    status: completed
  - id: init-imports
    content: Update __init__.py to import new strategy modules
    status: completed
  - id: backtest-run
    content: Update and run backtest script on all strategies
    status: completed
---

# Strategy Improvements and New Strategies (Config-First)

All parameters are driven by YAML configuration. Code reads from config; no hardcoded values.

## Phase 1: Configuration Infrastructure

### 1.1 Update settings.yaml with Risk Management Defaults

Add to [config/settings.yaml](config/settings.yaml):

```yaml
trading:
  default_commission: 0.001
  default_slippage: 0.0005
  # NEW: Risk management defaults
  default_stop_loss_pct: 0.02      # 2% stop loss
  default_take_profit_pct: 0.04    # 4% take profit
  default_trailing_stop_pct: null  # disabled by default
  
  # NEW: Filter defaults
  adx_filter_threshold: 25         # ADX must be > 25 for trend strategies
  volume_filter_multiplier: 1.5    # Volume must be > 1.5x average
```



### 1.2 Update Config Schemas

Modify [src/crypto/config/schemas.py](src/crypto/config/schemas.py) to add:

- `RiskConfig` with stop_loss_pct, take_profit_pct, trailing_stop_pct
- `FilterConfig` with adx_threshold, volume_multiplier
- Update `StrategyConfig` to include optional risk overrides

### 1.3 Extend BacktestEngine with Risk Management

Modify [src/crypto/backtesting/engine.py](src/crypto/backtesting/engine.py):

- Read defaults from `get_settings().trading`
- Accept per-strategy overrides from strategy config
- Monitor positions each bar for SL/TP/trailing stop hits

### 1.4 Extend BaseStrategy with Config-Driven Risk

Modify [src/crypto/strategies/base.py](src/crypto/strategies/base.py):

- Add `stop_loss_pct`, `take_profit_pct`, `trailing_stop_pct` read from `_params`
- Add `use_adx_filter`, `adx_threshold`, `use_volume_filter`, `volume_multiplier`
- Engine checks `strategy.stop_loss_pct` before falling back to defaults

---

## Phase 2: Improve Existing Strategies (Config-Driven Filters)

### 2.1 Add ADX Filter to Trend Strategies

Modify [src/crypto/strategies/technical.py](src/crypto/strategies/technical.py):

- Add optional `use_adx_filter`, `adx_threshold` params to `_setup()`
- Only generate signals when ADX > threshold (if filter enabled)
- Affects: `SMACrossoverStrategy`, `EMACrossoverStrategy`, `MACDCrossoverStrategy`

Config example (strategies.yaml):

```yaml
sma_crossover_btc_filtered:
  type: sma_crossover
  params:
    fast_period: 10
    slow_period: 30
    use_adx_filter: true      # Enable ADX filter
    adx_threshold: 25         # Only trade when ADX > 25
    stop_loss_pct: 0.02       # Per-strategy SL override
```



### 2.2 Add Volume Confirmation

Add optional `use_volume_filter`, `volume_multiplier` params:

- [src/crypto/strategies/technical.py](src/crypto/strategies/technical.py): `BollingerBreakoutStrategy`
- [src/crypto/strategies/statistical.py](src/crypto/strategies/statistical.py): `RSIMeanReversionStrategy`

---

## Phase 3: New Strategies (All Config-Driven)

All parameters from strategies.yaml. Strategy classes read params via `self._params`.

### 3.1 Ensemble Strategy (new file: `ensemble.py`)

References sub-strategies by their config names:

```yaml
ensemble_consensus:
  type: ensemble_voting
  params:
    strategies: [sma_crossover_btc, rsi_mean_reversion, macd_crossover]
    min_agreement: 2           # BUY when 2+ agree
    mode: majority             # "majority" or "unanimous"
  symbols: [BTCUSDT]
  interval: 1h
  enabled: true
```



### 3.2 Market Regime Strategy (new file: `regime.py`)

Config-driven regime detection and strategy switching:

```yaml
regime_adaptive_btc:
  type: regime_adaptive
  params:
    adx_trending_threshold: 25
    volatility_lookback: 20
    trending_strategy: momentum_breakout
    ranging_strategy: rsi_mean_reversion
  symbols: [BTCUSDT]
  interval: 1h
  enabled: true
```



### 3.3 Enhanced ML Strategy (extend `ml.py`)

Configurable features and walk-forward validation:

```yaml
ml_enhanced_btc:
  type: ml_enhanced
  params:
    model: gradient_boosting
    features: [sma_20, rsi_14, macd, obv, atr_ratio, bb_width, volume_momentum]
    walk_forward: true
    train_window: 500
    test_window: 100
    retrain_frequency: 100
  symbols: [BTCUSDT]
  interval: 1h
  enabled: true
```



### 3.4 Multi-Timeframe Strategy (new file: `multi_timeframe.py`)

Configurable timeframe alignment:

```yaml
mtf_trend_btc:
  type: multi_timeframe
  params:
    base_interval: 1h
    higher_intervals: [4h, 1d]
    signal_strategy: sma_crossover
    require_alignment: true    # All TFs must agree
  symbols: [BTCUSDT]
  interval: 1h
  enabled: true
```



### 3.5 Volatility Squeeze Strategy (add to `technical.py`)

Configurable squeeze detection parameters:

```yaml
squeeze_breakout_btc:
  type: volatility_squeeze
  params:
    bb_period: 20
    bb_std: 2.0
    kc_period: 20
    kc_atr_mult: 1.5
    momentum_period: 12
  symbols: [BTCUSDT]
  interval: 1h
  enabled: true
```



### 3.6 Relative Strength Rotation (new file: `rotation.py`)

Configurable ranking and rotation:

```yaml
relative_strength_top5:
  type: relative_strength
  params:
    lookback_period: 20
    top_n: 5
    momentum_type: roc         # "roc" or "returns"
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT]  # Pool to rank
  interval: 1d
  enabled: true
```



### 3.7 VWAP Reversion Strategy (add to `statistical.py`)

Configurable VWAP deviation thresholds:

```yaml
vwap_reversion_btc:
  type: vwap_reversion
  params:
    vwap_period: 20
    entry_deviation: 0.02      # Enter when 2% from VWAP
    exit_deviation: 0.005      # Exit within 0.5%
  symbols: [BTCUSDT]
  interval: 1h
  enabled: true
```



### 3.8 Reinforcement Learning Strategy (new file: `rl.py`)

Configurable RL hyperparameters:

```yaml
rl_dqn_btc:
  type: rl_dqn
  params:
    algorithm: DQN             # DQN, PPO, or A2C
    learning_rate: 0.0001
    gamma: 0.99
    buffer_size: 100000
    train_episodes: 1000
    features: [sma_20, rsi_14, macd, atr]
    reward_type: sharpe        # "sharpe", "returns", or "sortino"
  symbols: [BTCUSDT]
  interval: 1h
  enabled: true
```

---

## Phase 4: New Indicators

Add to [src/crypto/indicators/base.py](src/crypto/indicators/base.py):

- `adx` - Average Directional Index (for trend strength filter)
- `obv` - On Balance Volume
- `keltner` - Keltner Channels (for squeeze detection)
- `vwap` - Volume Weighted Average Price

---

## Phase 5: Configuration Updates

### 5.1 Update strategies.yaml

Add entries in [config/strategies.yaml](config/strategies.yaml) for all new strategies with default parameters.

### 5.2 Update pyproject.toml

Add RL dependencies to [pyproject.toml](pyproject.toml):

```toml
"stable-baselines3>=2.0.0",
"gymnasium>=0.29.0",
```

---

## Phase 6: Update Backtest Script

Modify [scripts/backtest_top20.py](scripts/backtest_top20.py):

- Detect all registered strategies dynamically
- Handle multi-timeframe strategy data requirements
- Add summary statistics for new vs old strategies

---

## File Change Summary

| File | Action ||------|--------|| `config/settings.yaml` | Modify - add risk mgmt and filter defaults || `config/strategies.yaml` | Modify - add all new strategy configs || `src/crypto/config/schemas.py` | Modify - add RiskConfig, FilterConfig schemas || `src/crypto/config/settings.py` | Modify - expose new config fields || `src/crypto/backtesting/engine.py` | Modify - add SL/TP/trailing stop from config || `src/crypto/strategies/base.py` | Modify - add config-driven risk/filter params || `src/crypto/strategies/technical.py` | Modify - add ADX filter, volume, squeeze || `src/crypto/strategies/statistical.py` | Modify - add volume filter, VWAP || `src/crypto/strategies/ml.py` | Modify - add enhanced ML with more features || `src/crypto/strategies/ensemble.py` | Create - voting strategy (refs config names) || `src/crypto/strategies/regime.py` | Create - adaptive regime strategy || `src/crypto/strategies/multi_timeframe.py` | Create - MTF strategy || `src/crypto/strategies/rotation.py` | Create - relative strength rotation || `src/crypto/strategies/rl.py` | Create - RL DQN strategy || `src/crypto/indicators/base.py` | Modify - add ADX, OBV, Keltner, VWAP || `src/crypto/__init__.py` | Modify - import new strategy modules || `pyproject.toml` | Modify - add RL dependencies |