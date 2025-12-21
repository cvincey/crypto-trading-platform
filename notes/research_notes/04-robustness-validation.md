# Research Note: Robustness Validation & Strategy Reality Check

**Date:** December 21, 2024  
**Author:** Trading Research Team  
**Version:** 1.0

---

## Executive Summary

Following the discovery of severe overfitting in Note 03, we implemented new "robust" strategies designed to address temporal generalization failure. We also created a fast validation framework for rapid iteration.

**Critical Finding:** All strategies—including simplified ML, rule-based, and legacy approaches—**underperform simple buy-and-hold**. The strategies are actively destroying value through frequent trading.

**Bottom Line:** Technical indicator-based strategies (ML or rule-based) do not work on hourly crypto data. A fundamental rethink is required.

---

## 1. New Strategies Implemented

### 1.1 Robust Strategy Suite

| Strategy | Type | Hypothesis |
|----------|------|------------|
| `ml_classifier_v5` | Simplified ML | Fewer features (3), stronger regularization, longer horizon |
| `ml_classifier_online` | Adaptive ML | Rolling retraining to adapt to regime changes |
| `ml_classifier_decay` | Weighted ML | Exponential decay on sample weights |
| `rule_ensemble` | Rule-based | Voting across ADX/RSI/Volume/MA rules (no ML = no overfit) |
| `trend_following_rules` | Rule-based | Classic trend following with confirmations |
| `mean_reversion_rules` | Rule-based | RSI + Bollinger Band mean reversion |
| `ml_cross_asset` | Cross-asset ML | Train on all symbols to learn universal patterns |
| `ml_cross_asset_regime` | Regime ML | Separate models for trending/ranging markets |

### 1.2 Fast Validation Framework

Created `scripts/quick_validate.py` for rapid iteration:
- 3 representative symbols instead of 19
- 4 strategies instead of 7+
- Bi-weekly folds instead of weekly
- 6 months data instead of 12

**Result:** 12 validations in 73 seconds (vs 60+ minutes for full suite)

---

## 2. Quick Validation Results

### 2.1 Configuration

| Parameter | Value |
|-----------|-------|
| Symbols | BTCUSDT, ETHUSDT, SOLUSDT |
| Train Window | 1440 bars (60 days) |
| Test Window | 336 bars (14 days) |
| Step Size | 336 bars (bi-weekly) |
| Data Period | 180 days |
| Folds per validation | ~8 |

### 2.2 Results: All Strategies Failed

| Strategy | Symbol | OOS Sharpe | IS Sharpe | Degradation | Trades | Status |
|----------|--------|------------|-----------|-------------|--------|--------|
| ml_ensemble_voting | SOLUSDT | -0.74 | 23.98 | 103% | 324 | ❌ FAIL |
| rule_ensemble | SOLUSDT | -2.53 | 0.77 | 430% | 254 | ❌ FAIL |
| ml_ensemble_voting | ETHUSDT | -4.78 | 20.39 | 123% | 301 | ❌ FAIL |
| rule_ensemble | ETHUSDT | -5.69 | -1.58 | N/A | 281 | ❌ FAIL |
| ml_classifier_xgb | SOLUSDT | -5.85 | 29.63 | 120% | 676 | ❌ FAIL |
| ml_classifier_v5 | SOLUSDT | -6.55 | -1.20 | N/A | 435 | ❌ FAIL |
| rule_ensemble | BTCUSDT | -6.66 | -5.37 | N/A | 286 | ❌ FAIL |
| ml_classifier_v5 | ETHUSDT | -7.17 | -0.26 | N/A | 492 | ❌ FAIL |
| ml_ensemble_voting | BTCUSDT | -7.98 | 10.32 | 177% | 330 | ❌ FAIL |
| ml_classifier_xgb | ETHUSDT | -9.22 | 23.19 | 140% | 689 | ❌ FAIL |
| ml_classifier_v5 | BTCUSDT | -11.55 | -6.14 | N/A | 410 | ❌ FAIL |
| ml_classifier_xgb | BTCUSDT | -13.75 | 10.13 | 236% | 663 | ❌ FAIL |

### 2.3 Strategy Summary

| Strategy | Avg OOS Sharpe | Avg Degradation | Pass Rate | Verdict |
|----------|----------------|-----------------|-----------|---------|
| ml_classifier_v5 | -8.42 | N/A | 0/3 | ❌ OVERFIT |
| rule_ensemble | -4.96 | 143% | 0/3 | ❌ OVERFIT |
| ml_classifier_xgb | -9.60 | 165% | 0/3 | ❌ OVERFIT |
| ml_ensemble_voting | -4.50 | 135% | 0/3 | ❌ OVERFIT |

---

## 3. Comparison to Buy-and-Hold

### 3.1 Market Conditions (Last 180 Days)

| Symbol | Price Change | Buy-Hold Sharpe |
|--------|--------------|-----------------|
| BTCUSDT | -16.9% | -0.77 |
| ETHUSDT | +22.4% | **+0.95** |
| SOLUSDT | -13.2% | -0.00 |

### 3.2 Strategy vs Buy-Hold Gap

| Symbol | Buy-Hold Sharpe | Best Strategy OOS | Gap |
|--------|-----------------|-------------------|-----|
| BTCUSDT | -0.77 | -11.55 | **-10.8 Sharpe points** |
| ETHUSDT | +0.95 | -4.78 | **-5.7 Sharpe points** |
| SOLUSDT | -0.00 | -0.74 | **-0.7 Sharpe points** |

**Key Insight:** Even on ETH (which was up 22%), all strategies lost money. Simple buy-and-hold crushed every strategy we tested.

---

## 4. Root Cause Analysis

### 4.1 Why Strategies Fail

1. **Excessive Trading Frequency**
   - Strategies make 300-700 trades per 6 months
   - At 0.1% commission per trade, that's 0.6-1.4% in costs alone
   - Signals need to be highly accurate to overcome costs

2. **Signals Are Worse Than Random**
   - Negative Sharpe means systematic wrong predictions
   - Models learn patterns that worked historically but reverse forward
   - Rule thresholds (RSI < 30, ADX > 25) don't predict direction

3. **Technical Indicators Don't Predict**
   - RSI, MACD, ADX, etc. are lagging indicators
   - They describe what happened, not what will happen
   - Any predictive power is arbitraged away in liquid markets

4. **Hourly Timeframe Too Noisy**
   - Signal-to-noise ratio is very low on 1h candles
   - Noise overwhelms any genuine patterns
   - Daily or weekly might have better signal

### 4.2 Why Even Rule-Based Failed

The rule_ensemble strategy cannot overfit by construction (no ML training), yet it still failed:
- OOS Sharpe: -2.53 to -6.66
- This proves the rules themselves are not predictive
- ADX trending filter, RSI extremes, MA crossovers don't work in current crypto

---

## 5. What Might Actually Work

### 5.1 Approaches to Explore

| Approach | Hypothesis | Implementation |
|----------|------------|----------------|
| **Momentum (Long-term)** | Trend persistence on weekly/monthly scale | Monthly rebalancing, top N performers |
| **Mean Reversion (Daily)** | Oversold bounces on daily timeframe | RSI < 20 on daily, hold for mean reversion |
| **Reduce Frequency** | Fewer trades = lower costs = easier to profit | Max 1 trade per week, not per hour |
| **Alternative Data** | Technical indicators are known, try unknowns | On-chain metrics, sentiment, funding rates |
| **Carry/Funding** | Collect funding rather than predict direction | Long spot, short perp when funding positive |

### 5.2 Recommended Next Steps

1. **Test Simple Momentum**
   - Monthly rebalancing
   - Hold top 3 coins by 30-day return
   - Backtest over 2+ years

2. **Test Funding Rate Strategy**
   - Purely collect funding payments
   - Delta-neutral position
   - No directional prediction needed

3. **Switch to Daily Timeframe**
   - Much less noise
   - Fewer trades = lower costs
   - Clearer trends

4. **Add Alternative Data Sources**
   - Exchange inflows/outflows
   - Open interest changes
   - Funding rate history
   - Social sentiment

---

## 6. Conclusions

### 6.1 What We Learned

1. ❌ **ML on technical indicators doesn't work** (learn noise, not signal)
2. ❌ **Rule-based on technical indicators doesn't work** (rules aren't predictive)
3. ❌ **High-frequency trading on crypto doesn't work** (costs kill profits)
4. ❌ **Hourly timeframe is too noisy** (signal-to-noise too low)
5. ✅ **Walk-forward validation works** (correctly identifies failures)
6. ✅ **Quick validation framework works** (73s vs 60+ min)

### 6.2 The Hard Truth

After extensive testing:
- 7+ ML strategies
- 4+ rule-based strategies
- 19 crypto symbols
- 180+ days of data
- Walk-forward validation with acceptance gates

**Result:** Every strategy underperforms doing nothing.

### 6.3 Path Forward

The current approach (ML/rules on hourly technicals) is a dead end. Pivot to:
- Longer timeframes (daily/weekly)
- Lower frequency (monthly rebalancing)
- Alternative alpha sources (funding, on-chain, sentiment)
- Simpler strategies (momentum, carry)

---

## Appendix A: Files Created/Modified

| File | Description |
|------|-------------|
| `scripts/quick_validate.py` | Fast validation framework (73s vs 60+ min) |
| `src/crypto/strategies/ml_siblings.py` | Added `MLClassifierV5Strategy` |
| `src/crypto/strategies/rule_ensemble.py` | New rule-based strategies |
| `src/crypto/strategies/ml_online.py` | Adaptive ML strategies |
| `src/crypto/strategies/ml_cross_asset.py` | Cross-asset ML strategies |
| `src/crypto/indicators/base.py` | Added volume/regime indicators |
| `config/optimization.yaml` | Updated with acceptance gates |
| `config/strategies.yaml` | Added new strategy configs |

## Appendix B: Validation Output

```
Quick Validation Results (73.7s total)

Strategy Summary:
  ml_classifier_v5:    OOS Sharpe -8.42, Pass 0/3 ✗ OVERFIT
  rule_ensemble:       OOS Sharpe -4.96, Pass 0/3 ✗ OVERFIT
  ml_classifier_xgb:   OOS Sharpe -9.60, Pass 0/3 ✗ OVERFIT
  ml_ensemble_voting:  OOS Sharpe -4.50, Pass 0/3 ✗ OVERFIT
```

---

*End of Research Note*
