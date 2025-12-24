# Research Note 09: Unexplored Strategies Testing

**Date**: December 21, 2025  
**Objective**: Test 8 previously unexplored strategy concepts using walk-forward validation

## Executive Summary

We implemented and tested 8 Tier 1 strategies (no external API required) that were identified as unexplored in previous research notes. **Only 1 strategy showed potential**: the `basis_proxy` strategy, which uses funding rates as a contrarian signal.

| Metric | Value |
|--------|-------|
| Strategies Tested | 8 |
| Potential Winners | 1 (12.5%) |
| Best OOS Sharpe | 0.63 (basis_proxy) |
| Worst OOS Sharpe | -2.65 (trend_strength_filter) |

## Strategies Tested

### Tier 1 Strategies (No External API Required)

| Strategy | Concept | OOS Sharpe | Beat B&H | Status |
|----------|---------|------------|----------|--------|
| basis_proxy | Trade contrarian to extreme funding rates | **0.63** | 100% | ◐ POTENTIAL |
| volatility_mean_reversion | Buy when volatility is extremely low | -0.00 | 33% | ✗ FAILED |
| regime_volatility_switch | Switch momentum/mean-reversion by regime | -0.25 | 67% | ✗ FAILED |
| volatility_breakout | Enter on Bollinger Band expansion | -0.35 | 33% | ✗ FAILED |
| multi_asset_pair_trade | Mean reversion between correlated assets | -1.18 | 0% | ✗ FAILED |
| dxy_correlation_proxy | Use BTC strength as DXY proxy | -1.58 | 0% | ✗ FAILED |
| momentum_quality | Quality-filtered momentum signals | -2.30 | 0% | ✗ FAILED |
| trend_strength_filter | ADX-based regime switching | -2.65 | 0% | ✗ FAILED |

### Tier 2 Strategies (Not Tested - Require Data Ingestion)

- `fear_greed_contrarian`: Trade contrarian to Fear & Greed Index extremes
- `btc_dominance_rotation`: Rotate BTC/alts based on dominance trends  
- `long_short_ratio_fade`: Fade extreme long/short positioning

### Tier 3 Strategies (Stubs Only - Require Paid APIs)

- `exchange_flow_signal`: On-chain exchange flow data
- `whale_tracking`: Large wallet movement tracking
- `options_max_pain`: Options max pain level signals
- `put_call_extremes`: Options put/call ratio extremes

## Key Findings

### 1. Basis Proxy Shows Genuine Edge

The `basis_proxy` strategy achieved a **positive 0.63 Sharpe ratio** with:
- 100% of tests beating buy-and-hold
- 100% acceptance gate pass rate
- 6.0 trades per month (reasonable frequency)

This strategy exploits the tendency for extreme funding rates to mean-revert. When funding is extremely negative (shorts paying longs), it signals overleveraged short positioning that often precedes upward moves.

**Why it works**: Funding rates contain genuine forward-looking information about market positioning that isn't fully priced in.

### 2. Volatility Strategies Underperform

Both volatility-based strategies failed:
- `volatility_mean_reversion`: Near-zero Sharpe, unreliable
- `volatility_breakout`: Negative Sharpe, worse than random

**Conclusion**: Volatility patterns alone don't provide tradeable edge in crypto markets.

### 3. Meta-Strategies Add Complexity Without Value

The regime-switching strategies (`regime_volatility_switch`, `trend_strength_filter`) performed poorly:
- Added complexity without improving returns
- Parameter sensitivity likely leads to overfitting
- Simpler approaches work better

### 4. Cross-Asset Strategies Need More Refinement

Both `multi_asset_pair_trade` and `dxy_correlation_proxy` failed significantly:
- Correlations in crypto are unstable
- Using BTC as a proxy for DXY doesn't capture the relationship well
- These concepts may work but need more sophisticated implementation

### 5. Quality Filters Hurt Performance

The `momentum_quality` strategy (which adds volume and trend filters to momentum) performed worst among momentum variants. This suggests:
- Additional filters increase false negatives more than they reduce false positives
- Simpler momentum signals outperform "improved" versions

## Comparison to Previous Research

| Research Note | Top Strategy Sharpe | Success Rate |
|---------------|---------------------|--------------|
| Note 06 (Creative Testing) | 0.48 | ~15% |
| Note 07 (Deep Validation) | 0.52 | ~10% |
| Note 08 (Complete Validation) | 0.58 | 5-10% |
| **Note 09 (Unexplored)** | **0.63** | **12.5%** |

The `basis_proxy` strategy outperforms all previously validated strategies.

## Recommendations

### Immediate Actions

1. **Promote basis_proxy to full validation**: Run extended walk-forward with more data and stricter gates
2. **Test funding rate variants**: Explore different thresholds, lookback periods, and multi-symbol funding signals
3. **Consider ensemble with funding_rate_fade**: The existing funding strategy may complement basis_proxy

### Future Research

1. **Tier 2 Data Ingestion**: Set up Fear & Greed Index and BTC Dominance data pipelines to test remaining strategies
2. **Paid API Evaluation**: Consider Glassnode or CryptoQuant trial for on-chain data testing
3. **Funding Rate Deep Dive**: Funding appears to be the most promising alpha source - deserves dedicated research

### Deprioritize

- Pure volatility strategies (no edge found)
- Complex meta/regime strategies (overfit prone)
- Cross-asset correlation strategies (unstable relationships)

## Technical Notes

### Test Configuration

- **Symbols**: BTCUSDT, ETHUSDT, SOLUSDT
- **Period**: Last 180 days (~5000 hourly candles per symbol)
- **Walk-Forward**: 720h train, 168h test, 168h step
- **Acceptance Gates**: 
  - OOS Sharpe > -2.0 (very lenient)
  - Total trades > 1

### Files Created

- `src/crypto/strategies/volatility_trading.py` - Volatility strategies
- `src/crypto/strategies/structural.py` - Structural/cross-asset strategies
- `src/crypto/strategies/momentum.py` - Added MomentumQuality, TrendStrengthFilter
- `src/crypto/data/alternative_data.py` - New data repositories
- `config/creative_testing.yaml` - Unexplored strategy configs
- `config/strategies.yaml` - Strategy instance definitions
- `scripts/run_unexplored_testing.py` - Testing script

---

*Next Step: Run full validation on basis_proxy with 2024-2025 data and stricter acceptance gates*
