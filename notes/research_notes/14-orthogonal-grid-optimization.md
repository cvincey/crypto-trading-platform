# Research Note 14: Orthogonal Strategies Grid Optimization

**Date:** 2024-12-22  
**Status:** Complete  
**Objective:** Optimize orthogonal strategies via grid search and deploy validated ones to paper trading

---

## Executive Summary

Grid search optimization of 5 orthogonal strategies yielded **4 viable candidates** for paper trading deployment. These strategies are designed to diversify PnL from the core ratio mean-reversion strategies by exploiting different market signals.

### Key Results

| Strategy | OOS Sharpe | Trades | Status |
|----------|------------|--------|--------|
| liquidity_vacuum_detector | **0.78** | 18 | ✅ DEPLOYED |
| btc_dominance_rotation | **0.58** | 111 | ✅ DEPLOYED |
| cross_sectional_momentum | **0.57** | 10 | ✅ DEPLOYED |
| funding_term_structure | **0.42** | 297 | ✅ DEPLOYED |
| gamma_mimic_breakout | 0.03 | 2 | ❌ RETIRED |
| correlation_regime_switch | 0.00 | 0 | ❌ RETIRED |

---

## Grid Search Methodology

### Configuration
- **Symbols:** BTCUSDT, ETHUSDT, SOLUSDT
- **Period:** 180 days
- **Walk-Forward:** 60-day train, 14-day test, bi-weekly steps
- **Workers:** 4 parallel processes
- **Total combinations tested:** 429

### Parameter Grids

**funding_term_structure** (48 combinations):
```
lookback_periods: [2, 3, 4, 5]
extreme_funding: [0.0003, 0.0004, 0.0005, 0.0006]
hold_period: [12, 24, 48]
```

**btc_dominance_rotation** (48 combinations):
```
risk_on_threshold: [-0.01, -0.015, -0.02, -0.025]
risk_off_threshold: [0.005, 0.01, 0.015, 0.02]
hold_period: [24, 48, 72]
```

**liquidity_vacuum_detector** (81 combinations):
```
range_percentile: [80, 85, 90]
volume_percentile: [30, 40, 50]
stall_threshold: [0.5, 0.6, 0.7]
hold_period: [8, 12, 24]
```

**gamma_mimic_breakout** (144 combinations):
```
min_compression_bars: [48, 72, 96, 120]
compression_percentile: [15, 20, 25, 30]
breakout_atr_mult: [1.5, 2.0, 2.5, 3.0]
hold_period: [48, 72, 96]
```

**correlation_regime_switch** (108 combinations):
```
high_correlation: [0.60, 0.65, 0.70]
low_correlation: [0.40, 0.45, 0.50]
stability_threshold: [0.08, 0.10, 0.12]
min_hold_period: [24, 48, 72]
```

---

## Optimized Parameters

### ✅ liquidity_vacuum_detector (Best Performer)

**Final Parameters:**
```yaml
range_percentile: 85      # Wide range (top 15%)
volume_percentile: 50     # Low volume (bottom 50%)
stall_threshold: 0.6      # Next bar < 60% of vacuum
hold_period: 12           # 12 hours
```

**Performance:**
- OOS Sharpe: 0.78
- Pass Rate: 100%
- Trades: 18 (over 180 days)
- Signal: Microstructure thin-book detection

**Diversification Value:** High - exploits order book dynamics, completely independent of price ratios.

---

### ✅ btc_dominance_rotation

**Final Parameters:**
```yaml
risk_on_threshold: -0.01   # Dominance falling → alt risk-on
risk_off_threshold: 0.005  # Dominance rising → risk-off
hold_period: 48            # 2 days
```

**Performance:**
- OOS Sharpe: 0.58
- Pass Rate: 67%
- Trades: 111 (over 180 days)
- Signal: Macro regime rotation

**Diversification Value:** High - trades on capital rotation, not individual asset performance.

---

### ✅ funding_term_structure

**Final Parameters:**
```yaml
lookback_periods: 2        # 2 funding periods (16h)
extreme_funding: 0.0005    # 0.05% = crowded
hold_period: 12            # 12 hours
use_contrarian: true
```

**Performance:**
- OOS Sharpe: 0.42
- Pass Rate: 67%
- Trades: 297 (over 180 days)
- Signal: Positioning/crowding

**Diversification Value:** Medium-High - positioning mean reversion, not price mean reversion.

---

### ✅ cross_sectional_momentum

**Parameters (unchanged from Note 13):**
```yaml
ranking_period: 168        # 7-day momentum
hold_period: 168           # 1 week
top_percentile: 0.20       # Long top 20%
bottom_percentile: 0.20    # Avoid bottom 20%
```

**Performance:**
- OOS Sharpe: 0.57
- Pass Rate: 67%
- Trades: 10 (over 180 days)
- Signal: Cross-sectional factor

**Diversification Value:** High - factor strategy, not pair-based.

---

## Retired Strategies

### ❌ gamma_mimic_breakout

**Reason:** Only 2 trades in 180 days with Sharpe 0.03. The compression-then-breakout signal is too rare to be practically useful. Grid search showed this is a fundamental design issue, not just parameter tuning.

**Best Grid Params (still insufficient):**
```yaml
min_compression_bars: 48
compression_percentile: 20
breakout_atr_mult: 2.0
hold_period: 48
```

### ❌ correlation_regime_switch

**Reason:** Zero trades even after extensive grid search. The correlation stability filter is too restrictive. Needs fundamental redesign - possibly as an overlay rather than standalone strategy.

**Best Grid Params (0 trades):**
```yaml
high_correlation: 0.65
low_correlation: 0.40
stability_threshold: 0.08
```

---

## Also Retired (from Note 13 Backtest)

| Strategy | OOS Sharpe | Issue |
|----------|------------|-------|
| crash_only_trend_filter | -0.16 | Defensive mode loses money |
| gap_reversion | -0.41 | Fading gaps not profitable |
| volatility_targeting_overlay | -0.63 | Poor as standalone (may work as overlay) |
| market_breadth_alt_participation | -3.00 | Fundamental issue with signal |

---

## Deployment Summary

### Files Updated

1. **config/creative_testing.yaml**
   - Updated all strategies with grid-optimized params
   - Added `status: validated` or `status: retired` tags
   - Set `enabled: false` for retired strategies

2. **config/strategies.yaml**
   - Added 4 new validated strategies with `_v1` suffix
   - Full configuration ready for production

3. **config/paper_trading.yaml**
   - Added orthogonal strategies section
   - Updated risk limits: max_positions 3→6, max_position_pct 25%→15%
   - All 4 strategies set to `enabled: true`

### Active Strategy Portfolio

| Category | Strategy | Sharpe | Status |
|----------|----------|--------|--------|
| **Ratio MR** | eth_btc_ratio_optimized | 9.54 | ✅ Active |
| **Ratio MR** | sol_btc_ratio | 6.54 | ✅ Active |
| **Ratio MR** | ltc_btc_ratio | 8.31 | ✅ Active |
| **Diversifier** | basis_proxy | 1.05 | ✅ Active |
| **Orthogonal** | liquidity_vacuum_detector | 0.78 | ✅ Active |
| **Orthogonal** | btc_dominance_rotation | 0.58 | ✅ Active |
| **Orthogonal** | cross_sectional_momentum | 0.57 | ✅ Active |
| **Orthogonal** | funding_term_structure | 0.42 | ✅ Active |

---

## Next Steps

1. **Monitor paper trading performance** of new orthogonal strategies
2. **Track correlation** between orthogonal and ratio strategies
3. **Consider redesign** of correlation_regime_switch as an overlay
4. **Evaluate volatility_targeting_overlay** as portfolio-level risk control

---

## Appendix: Grid Search Commands

```bash
# Run full grid search (all strategies)
python scripts/grid_search_orthogonal.py --workers 4

# Run single strategy
python scripts/grid_search_orthogonal.py --strategy liquidity_vacuum_detector --workers 4

# Run backtests after optimization
python scripts/run_orthogonal_testing.py
```

Results saved to:
- `notes/orthogonal_results/grid_search/best_params_*.json`
- `notes/orthogonal_results/grid_search/grid_search_*.json`
