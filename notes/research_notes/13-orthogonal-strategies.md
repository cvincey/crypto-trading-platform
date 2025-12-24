# Research Note 13: Orthogonal Diversification Strategies

**Date**: December 22, 2025  
**Objective**: Test strategies orthogonal to ratio mean reversion for portfolio diversification

## Executive Summary

Tested **13 hyper-creative strategies** designed to exploit different state variables than "alt underperformed BTC → mean revert". The goal was to find strategies whose PnL is **uncorrelated** with the existing ratio mean reversion edge.

| Metric | Value |
|--------|-------|
| Strategies Tested | 13 |
| Winners (Sharpe > 0.3) | 1 |
| Potential (Sharpe > 0) | 3 |
| Best OOS Sharpe | 0.59 (cross_sectional_momentum) |
| Most Stable | cross_sectional_momentum (100% pass rate) |

## Key Finding

**cross_sectional_momentum** is the clear winner with:
- OOS Sharpe: **0.59** (beats the 0.5 deployment threshold)
- Pass Rate: **100%** (passed on all 3 test symbols)
- Trades/Month: **5.2** (manageable frequency)
- Beats B&H: **100%** of the time

This strategy is **orthogonal** to ratio MR because it uses **rank-based momentum** (factor investing) rather than pair relationships.

## Strategy Rankings

| Rank | Strategy | OOS Sharpe | Pass Rate | Trades/Mo | Status |
|------|----------|------------|-----------|-----------|--------|
| 1 | cross_sectional_momentum | **0.59** | 100% | 5.2 | **DEPLOY** |
| 2 | funding_term_structure | 0.25 | 33% | 29.2 | OPTIMIZE |
| 3 | btc_dominance_rotation | 0.10 | 67% | 13.8 | OPTIMIZE |
| 4 | gamma_mimic_breakout | 0.00 | 0% | 0.0 | TUNE (no trades) |
| 5 | liquidity_vacuum_detector | 0.00 | 0% | 0.0 | TUNE (no trades) |
| 6 | correlation_regime_switch | 0.00 | 0% | 0.0 | TUNE (no trades) |
| 7 | liquidation_cluster_fade | 0.00 | 0% | 0.0 | NEEDS DATA |
| 8 | crash_only_trend_filter | -0.14 | 33% | 13.5 | RETIRE |
| 9 | stablecoin_liquidity_pulse | -0.22 | 33% | 13.8 | RETIRE |
| 10 | gap_reversion | -0.43 | 0% | 17.7 | RETIRE |
| 11 | volatility_targeting_overlay | -0.61 | 0% | 7.0 | RETIRE |
| 12 | funding_vol_interaction | -0.64 | 0% | 48.8 | RETIRE |
| 13 | market_breadth_alt_participation | -3.08 | 0% | 61.5 | RETIRE |

## Diversification Value Analysis

### What Worked (Orthogonal Signal Sources)

| Strategy | Signal Type | Why It Diversifies |
|----------|-------------|-------------------|
| cross_sectional_momentum | Rank-based factor | Uses **relative performance** across universe, not pair ratios |
| funding_term_structure | Positioning data | Uses **crowding signals**, not price patterns |
| btc_dominance_rotation | Market structure | Uses **rotation regime**, not mean reversion |

### What Failed (Reasons)

| Strategy | Issue | Root Cause |
|----------|-------|-----------|
| gamma_mimic_breakout | 0 trades | Compression threshold too strict (168 bars = 1 week minimum) |
| liquidity_vacuum_detector | 0 trades | Vacuum detection criteria rarely met |
| correlation_regime_switch | 0 trades | High correlation threshold (0.80) rarely sustained |
| market_breadth_alt_participation | Overtrading | 61.5 trades/month = churning losses |
| crash_only_trend_filter | Poor timing | Crash detection too late, recovery signal premature |

## Recommendations

### Immediate Actions

1. **Deploy cross_sectional_momentum**
   - Add to production config with current parameters
   - Monitor for correlation with eth_btc_ratio_reversion
   - Expected contribution: ~5% of portfolio allocation

2. **Grid Search on Tier 2 Potentials**
   - funding_term_structure: Tune `extreme_funding` (0.0003-0.0007) and `lookback_periods` (2-5)
   - btc_dominance_rotation: Tune `risk_on_threshold` (-0.02 to 0.00)

3. **Fix Zero-Trade Strategies**
   - gamma_mimic_breakout: Reduce `min_compression_bars` from 168 to 72
   - liquidity_vacuum_detector: Loosen percentile thresholds (range: 85, volume: 40)
   - correlation_regime_switch: Lower `high_correlation` to 0.70

### Parameter Tuning Priorities

```yaml
# High-priority tuning targets
gamma_mimic_breakout:
  min_compression_bars: [48, 72, 96, 120]  # Currently 168 (too strict)
  breakout_atr_mult: [2.0, 2.5, 3.0]

funding_term_structure:
  extreme_funding: [0.0003, 0.0004, 0.0005, 0.0006]
  lookback_periods: [2, 3, 4, 5]

btc_dominance_rotation:
  risk_on_threshold: [-0.02, -0.015, -0.01, -0.005]
  risk_off_threshold: [0.005, 0.01, 0.015, 0.02]
```

### Strategies to Retire (No Optimization Potential)

| Strategy | Reason |
|----------|--------|
| market_breadth_alt_participation | Fundamentally overtrading (-3.08 Sharpe) |
| funding_vol_interaction | Signal logic too complex for stable results |
| gap_reversion | Single-name dislocations too noisy |
| volatility_targeting_overlay | Position sizing overlay needs different implementation |
| stablecoin_liquidity_pulse | Weekly signal too slow for 1h timeframe |
| crash_only_trend_filter | Crash detection timing issues |

## Correlation with Ratio Strategies

To maximize diversification, the deployed orthogonal strategies should have **low correlation** with the existing ratio strategies. Based on signal sources:

| Orthogonal Strategy | Expected Correlation with Ratio MR |
|---------------------|-----------------------------------|
| cross_sectional_momentum | **Low** - Uses rank, not price levels |
| funding_term_structure | **Medium** - Funding correlates somewhat with price momentum |
| btc_dominance_rotation | **Low** - Dominance is independent of pair ratios |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Symbols | BTCUSDT, ETHUSDT, SOLUSDT |
| Period | 180 days (6 months) |
| Train Window | 1440 bars (60 days) |
| Test Window | 336 bars (14 days) |
| Step Size | 336 bars (bi-weekly rolls) |
| Walk-Forward Folds | ~5 |

## Next Steps

1. **Add to strategies.yaml**:
```yaml
cross_sectional_momentum_v1:
  type: cross_sectional_momentum
  params:
    ranking_period: 168
    hold_period: 168
    top_percentile: 0.20
    bottom_percentile: 0.20
    min_momentum: 0.0
  symbols: [SOLUSDT, AVAXUSDT, NEARUSDT, LINKUSDT, DOTUSDT, APTUSDT]
  interval: 1h
  stop_loss_pct: 0.05
  take_profit_pct: 0.15
  enabled: true
```

2. **Run correlation analysis** between cross_sectional_momentum and eth_btc_ratio_reversion on shared test period

3. **Full validation** on 8 symbols, 365 days for cross_sectional_momentum

---

*Generated by run_orthogonal_testing.py | 2025-12-22*
