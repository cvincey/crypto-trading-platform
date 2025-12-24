---
name: Grid Optimization Valid Strategies
overview: Retire 7 underperforming strategies, run grid search on 5 valid strategies (4 legacy + basis_proxy), and document optimal parameters in research note 10.
todos:
  - id: retire-strategies
    content: "Mark 7 underperforming strategies as enabled: false in strategies.yaml"
    status: completed
  - id: add-grids
    content: Add hyperparameter grids for 5 valid strategies to optimization.yaml
    status: completed
  - id: create-script
    content: Create grid_search_validated.py with walk-forward scoring
    status: completed
  - id: run-grid-search
    content: Run grid search on all 5 strategies
    status: completed
  - id: write-note
    content: Write research note 10 with optimization results
    status: completed
---

# Grid Optimization for Valid Strategies

## 1. Mark Underperforming Strategies as Retired

In [`config/strategies.yaml`](config/strategies.yaml), set `enabled: false` for these 7 strategies:

- `volatility_mean_reversion` (line ~841)
- `volatility_breakout` (line ~858)
- `multi_asset_pair_trade` (line ~894)
- `dxy_correlation_proxy` (line ~909)
- `regime_volatility_switch` (line ~924)
- `momentum_quality` (line ~944)
- `trend_strength_filter` (line ~962)

## 2. Define Valid Strategies for Optimization

The 5 strategies to optimize:| Strategy | Source | Current OOS Sharpe ||----------|--------|-------------------|| `eth_btc_ratio_reversion` | Note 08 (paper trading) | 3.22 || `eth_btc_ratio_confirmed` | Note 08 | 1.87 || `volume_divergence` | Note 08 | 0.94 || `signal_confirmation_delay` | Note 08 | 0.16 || `basis_proxy` | Note 09 (new) | 0.63 |

## 3. Add Hyperparameter Grids to Config

Add to [`config/optimization.yaml`](config/optimization.yaml) under `hyperparameters:`:

```yaml
eth_btc_ratio_reversion:
  # Strategy params
  lookback: [96, 120, 168, 240]
  entry_threshold: [-2.5, -2.0, -1.5, -1.0]
  exit_threshold: [-0.3, -0.5, -0.7, -1.0]
  max_hold_hours: [24, 48, 72, 96]
  # Risk params
  stop_loss_pct: [0.03, 0.05, 0.07]
  take_profit_pct: [0.08, 0.10, 0.15]

eth_btc_ratio_confirmed:
  # Strategy params (same base as ratio_reversion + confirmation)
  lookback: [96, 120, 168]
  entry_threshold: [-2.0, -1.5, -1.0]
  exit_threshold: [-0.5, -0.7]
  max_hold_hours: [48, 72]
  confirmation_delay: [2, 3, 4]
  # Risk params
  stop_loss_pct: [0.03, 0.05, 0.07]
  take_profit_pct: [0.08, 0.10, 0.15]

volume_divergence:
  # Strategy params
  price_lookback: [24, 48, 72]
  volume_lookback: [24, 48, 72]
  price_threshold: [0.03, 0.05, 0.07]
  volume_decline_threshold: [0.2, 0.3, 0.4]
  # Risk params
  stop_loss_pct: [0.02, 0.03, 0.04]
  take_profit_pct: [0.04, 0.06, 0.08]

signal_confirmation_delay:
  # Strategy params
  confirmation_delay: [2, 3, 4, 6]
  require_consistent: [true]
  # Risk params
  stop_loss_pct: [0.03, 0.04, 0.05]
  take_profit_pct: [0.08, 0.10, 0.12]

basis_proxy:
  # Strategy params
  funding_lookback: [6, 9, 12, 18]
  entry_threshold: [-0.0005, -0.0003, -0.0002]
  exit_threshold: [0.0002, 0.0003, 0.0005]
  max_hold_hours: [48, 72, 96]
  # Risk params
  stop_loss_pct: [0.03, 0.04, 0.05]
  take_profit_pct: [0.08, 0.10, 0.12]
```

## 4. Create Grid Search Script

Create [`scripts/grid_search_validated.py`](scripts/grid_search_validated.py):

- Use walk-forward validation (not simple backtest) for realistic results
- Test on 3 symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Use 180 days of data
- Score by average OOS Sharpe across folds
- Track parameter stability (min/max Sharpe spread)

## 5. Run Grid Search

Combination counts (strategy params x risk params):

- `eth_btc_ratio_reversion`: 4x4x4x4 x 3x3 = 2,304 combos
- `eth_btc_ratio_confirmed`: 3x3x2x2x3 x 3x3 = 972 combos
- `volume_divergence`: 3x3x3x3 x 3x3 = 729 combos
- `signal_confirmation_delay`: 4x1 x 3x3 = 36 combos
- `basis_proxy`: 4x3x3x3 x 3x3 = 972 combos

Total: ~5,000 combinations x 3 symbols = ~15,000 testsTo keep runtime manageable (~30-60 min):

- Use quick backtest scoring for initial grid sweep
- Take top 20 combos per strategy to walk-forward validation
- Parallel processing where possible

## 6. Write Research Note 10

Create [`notes/research_notes/10-grid-optimization-results.md`](notes/research_notes/10-grid-optimization-results.md):

- Summary of best parameters per strategy
- Performance comparison: default vs optimized