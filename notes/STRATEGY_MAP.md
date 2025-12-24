# Complete Strategy Map

**Generated:** December 24, 2025  
**Total Registered Strategy Types:** 116  
**Total Strategy Configurations Evaluated:** 90+

---

## Lifecycle Taxonomy

All strategies use a clean, consistent tagging system:

```yaml
# STAGE (validation/deployment state - mutually exclusive)
stage: deployed | validated | experimental | retired

# ROLE (portfolio function - optional)
role: primary | backup | diversifier

# ENABLED (actual on/off switch)
enabled: true | false
```

| Stage | Meaning |
|-------|---------|
| `deployed` | Passed validation, currently running in paper/live |
| `validated` | Passed validation, ready to deploy but not yet running |
| `experimental` | Not yet validated through walk-forward testing |
| `retired` | Failed validation or deprecated |

| Role | Meaning |
|------|---------|
| `primary` | Core strategy, gets priority allocation |
| `backup` | Ready to replace primary if it underperforms |
| `diversifier` | Provides uncorrelated returns to core strategies |

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Registered Strategy Types** | 116 |
| **Strategy Files** | 28 |
| **Deployed** | 14 strategies |
| **Validated (Ready)** | 1 strategy |
| **Experimental** | 45+ strategies |
| **Retired (Failed)** | 30+ strategies |
| **Best Performer** | bnb_btc_ratio (Sharpe 23.94) |
| **Dominant Strategy Type** | Ratio Mean Reversion |

---

## Summary by Stage

| Stage | Count | Description |
|-------|-------|-------------|
| ✅ **deployed** | 14 | Currently running in paper trading |
| 📋 **validated** | 1 | Tested & validated, ready to deploy |
| 🟡 **experimental** | 45+ | Not yet validated through walk-forward |
| ❌ **retired** | 30+ | Failed validation or underperformed |

---

## ✅ DEPLOYED (stage: deployed)

### Ratio Mean Reversion (9 strategies, role: primary)

| Strategy | Symbol | Type | OOS Sharpe | Return | Win Rate | Trades/Mo |
|----------|--------|------|------------|--------|----------|-----------|
| **eth_btc_ratio_optimized** | ETHUSDT | ratio_reversion | **22.35** | 6.0% | - | 15 |
| **bnb_btc_ratio** | BNBUSDT | ratio_reversion | **23.94** | 439.2% | 85.9% | 15 |
| **ada_btc_ratio** | ADAUSDT | ratio_reversion | **19.74** | 2832.2% | 85.8% | 20 |
| **xrp_btc_ratio** | XRPUSDT | ratio_reversion | **16.10** | 1359.9% | 82.6% | 21 |
| **ltc_btc_ratio** | LTCUSDT | ratio_reversion | **16.29** | 4.8% | - | 32 |
| **dot_btc_ratio** | DOTUSDT | ratio_reversion | **14.74** | 992.4% | 79.0% | 19 |
| **sol_btc_ratio** | SOLUSDT | ratio_reversion | **14.4** | 4.7% | - | 39 |
| **link_btc_ratio** | LINKUSDT | ratio_reversion | **12.65** | 1510.3% | 77.0% | 26 |
| **near_btc_ratio** | NEARUSDT | ratio_reversion | **8.69** | 456.1% | 71.1% | 18 |

### Diversifier Strategies (5 strategies, role: diversifier)

| Strategy | Symbols | Type | OOS Sharpe | Trades/Mo |
|----------|---------|------|------------|-----------|
| **basis_proxy** | BTC/ETH/SOL | funding_mean_reversion | **1.05** | 3 |
| **liquidity_vacuum_detector** | BTC/ETH/SOL | microstructure | **0.78** | 3 |
| **btc_dominance_rotation** | Alts | regime_rotation | **0.58** | 18 |
| **cross_sectional_momentum** | 6 Alts | factor | **0.57** | 2 |
| **funding_term_structure** | BTC/ETH/SOL | positioning | **0.42** | 50 |

---

## 📋 VALIDATED (stage: validated, role: backup)

| Strategy | Symbol | Type | OOS Sharpe | Notes |
|----------|--------|------|------------|-------|
| **eth_btc_ratio_confirmed** | ETHUSDT | ratio_reversion_confirmed | **7.29** | Same as primary + 2h confirmation delay |

---

## 🟡 EXPERIMENTAL (stage: experimental)

### Technical Strategies (8)

| Strategy | Type | Symbols | Interval | Sharpe | Status |
|----------|------|---------|----------|--------|--------|
| sma_crossover_btc | sma_crossover | BTCUSDT | 1h | - | 🟡 Experimental |
| sma_crossover_eth | sma_crossover | ETHUSDT | 1h | - | 🟡 Experimental |
| sma_crossover_filtered | sma_crossover | BTCUSDT | 1h | - | 🟡 Experimental |
| rsi_mean_reversion | rsi_mean_reversion | BTC/ETH | 4h | - | 🟡 Experimental |
| macd_crossover | macd_crossover | BTCUSDT | 1h | - | 🟡 Experimental |
| bollinger_breakout | bollinger_breakout | BTC/ETH | 4h | - | 🟡 Experimental |
| momentum_breakout | momentum_breakout | BTC/ETH/SOL | 1d | - | 🟡 Experimental |
| squeeze_breakout_btc | volatility_squeeze | BTCUSDT | 1h | - | 🟡 Experimental |

### ML Strategies (12)

| Strategy | Type | Symbols | OOS Sharpe | Status |
|----------|------|---------|------------|--------|
| ml_classifier_btc | ml_classifier | BTCUSDT | Overfits | 🟡 Experimental |
| ml_classifier_v2 | ml_classifier_v2 | BTCUSDT | Overfits | 🟡 Experimental |
| ml_classifier_v3 | ml_classifier_v3 | BTCUSDT | Overfits | 🟡 Experimental |
| ml_classifier_v4 | ml_classifier_v4 | BTCUSDT | Overfits | 🟡 Experimental |
| ml_classifier_v5 | ml_classifier_v5 | Multiple | Overfits | 🟡 Experimental |
| ml_classifier_xgb | ml_classifier_xgb | BTCUSDT | Overfits | 🟡 Experimental |
| ml_classifier_hybrid | ml_classifier_hybrid | BTCUSDT | Overfits | 🟡 Experimental |
| ml_classifier_conservative | ml_classifier_conservative | BTCUSDT | Overfits | 🟡 Experimental |
| ml_classifier_online | ml_classifier_online | Multiple | Overfits | 🟡 Experimental |
| ml_classifier_decay | ml_classifier_decay | BTC/ETH | Overfits | 🟡 Experimental |
| ml_ensemble_voting | ml_ensemble_voting | BTCUSDT | Overfits | 🟡 Experimental |
| ml_enhanced_btc | ml_enhanced | BTCUSDT | Overfits | 🟡 Experimental |

### Ensemble & Meta Strategies (7)

| Strategy | Type | Symbols | Status |
|----------|------|---------|--------|
| vwap_reversion_btc | vwap_reversion | BTCUSDT | 🟡 Experimental |
| ensemble_consensus | ensemble_voting | BTCUSDT | 🟡 Experimental |
| regime_adaptive_btc | regime_adaptive | BTCUSDT | 🟡 Experimental |
| mtf_trend_btc | multi_timeframe | BTCUSDT | 🟡 Experimental |
| relative_strength_btc | relative_strength | BTCUSDT | 🟡 Experimental |
| momentum_ranking_btc | momentum_ranking | BTCUSDT | 🟡 Experimental |
| strategy_momentum | strategy_momentum | BTC/ETH | 🟡 Experimental |

### Cross-Symbol Strategies (6)

| Strategy | Type | Symbols | OOS Sharpe | Status |
|----------|------|---------|------------|--------|
| btc_lead_alt_follow | cross_symbol | SOL/AVAX/NEAR/DOT | - | 🟡 Experimental |
| correlation_breakdown | cross_symbol | ETH/SOL/BNB | - | 🟡 Experimental |
| correlation_breakdown_sol_avax | cross_symbol | SOLUSDT | - | 🟡 Experimental |
| sector_momentum_rotation | rotation | Multiple | - | 🟡 Experimental |
| btc_volatility_filter | filter | SOL/AVAX/NEAR | - | 🟡 Experimental |
| weekly_momentum | rotation | 6 coins | - | 🟡 Experimental |

### Calendar/Structural Strategies (4)

| Strategy | Type | Symbols | Status |
|----------|------|---------|--------|
| weekend_effect | calendar | BTC/ETH/SOL | 🟡 Experimental |
| hour_of_day_filter | calendar | BTC/ETH/SOL | 🟡 Experimental |
| month_end_rebalancing | calendar | 5 coins | 🟡 Experimental |
| signal_strength_filter | filter | BTC/ETH/SOL | 🟡 Experimental |

### Alternative Data Strategies (5)

| Strategy | Type | Symbols | OOS Sharpe | Status |
|----------|------|---------|------------|--------|
| funding_rate_fade | funding | BTC/ETH/SOL | - | 🟡 Experimental |
| funding_rate_carry | funding | BTC/ETH | - | 🟡 Experimental |
| buy_sell_imbalance | microstructure | BTC/ETH/SOL | - | 🟡 Experimental |
| volume_breakout_confirmation | volume | BTC/ETH/SOL | - | 🟡 Experimental |
| liquidation_cascade_fade_v1 | alternative_data | BTC/ETH/SOL | - | 🟡 Experimental |

### Meta Strategies (3)

| Strategy | Type | Symbols | Status |
|----------|------|---------|--------|
| regime_gate | meta | BTC/ETH/SOL | 🟡 Experimental |
| drawdown_pause | meta | BTC/ETH/SOL | 🟡 Experimental |
| regime_volatility_switch_v1 | regime | 5 coins | 🟡 Experimental |

### Rule-Based Strategies (4)

| Strategy | Type | Symbols | Status |
|----------|------|---------|--------|
| rule_ensemble | rule_based | 5 coins | 🟡 Experimental |
| trend_following_rules | rule_based | BTC/ETH/SOL | 🟡 Experimental |
| mean_reversion_rules | rule_based | BTC/ETH | 🟡 Experimental |
| ml_cross_asset | cross_asset | 5 coins | 🟡 Experimental |
| ml_cross_asset_regime | cross_asset | 3 coins | 🟡 Experimental |

---

## ❌ RETIRED (stage: retired)

### Failed Ratio Pairs (6)

| Strategy | Symbol | Type | OOS Sharpe | Retirement Reason |
|----------|--------|------|------------|-------------------|
| avax_btc_ratio | AVAXUSDT | ratio_reversion | 3.10 | Weakest ratio (+19% vs 160%+ for others) |
| apt_btc | APTUSDT | ratio_reversion | 0.48 | Negative OOS return (-1.4%) |
| sui_btc | SUIUSDT | ratio_reversion | -2.51 | Negative OOS return (-29.6%) |
| arb_btc | ARBUSDT | ratio_reversion | -1.55 | Negative OOS return (-21.3%) |
| sol_eth | SOLUSDT/ETH | ratio_reversion | -5.22 | ETH not a good reference |
| bnb_eth | BNBUSDT/ETH | ratio_reversion | -5.21 | ETH not a good reference |
| link_eth | LINKUSDT/ETH | ratio_reversion | 0.56→-0.33 | Not viable after optimization |

### Failed Orthogonal Strategies (6)

| Strategy | Type | OOS Sharpe | Retirement Reason |
|----------|------|------------|-------------------|
| gamma_mimic_breakout | volatility | 0.03 | Only 2 trades in 180 days |
| correlation_regime_switch | regime | 0.00 | Zero trades - filter too restrictive |
| crash_only_trend_filter | trend | -0.16 | Defensive mode loses money |
| gap_reversion | mean_reversion | -0.41 | Fading gaps not profitable |
| volatility_targeting_overlay | meta | -0.63 | Poor as standalone |
| market_breadth_alt_participation | breadth | -3.00 | Fundamental signal issue |

### Failed Unexplored Tier 1 Strategies (8)

| Strategy | Type | Avg OOS Sharpe | Retirement Reason |
|----------|------|----------------|-------------------|
| volatility_mean_reversion | volatility | -0.00 | No alpha after costs |
| volatility_breakout | volatility | -0.35 | Negative OOS Sharpe |
| multi_asset_pair_trade | pair_trading | -1.18 | Negative OOS across all symbols |
| dxy_correlation_proxy | macro | -1.58 | Proxy doesn't work |
| regime_volatility_switch | regime | -0.25 | Inconsistent across symbols |
| momentum_quality | momentum | -2.30 | Heavily negative |
| trend_strength_filter | trend | -2.65 | Heavily negative |

### Failed Frequency Reduction (2)

| Strategy | Type | OOS Sharpe | Retirement Reason |
|----------|------|------------|-------------------|
| signal_confirmation_delay | filter | 0.16 | Positive Sharpe but negative return (-0.1%) |
| volume_divergence | divergence | 1.47 | Only 2 trades in 180 days - too infrequent |

### Failed ML Strategies (All)

| Strategy Category | Count | Issue |
|-------------------|-------|-------|
| ml_classifier variants | 12 | All overfit - negative OOS Sharpe |
| ml_siblings | 6 | All overfit - walk-forward failed |
| rl_dqn_btc | 1 | Disabled - requires RL dependencies |

### Disabled Due to API Restrictions

| Strategy | Type | Issue |
|----------|------|-------|
| open_interest_divergence | alternative_data | Binance OI API returns 400 |
| liquidation_cascade_fade | alternative_data | Requires external liquidation data |

---

## Strategy Type Summary

### Registered Strategy Types: 116 Total

| Category | File | # Types | Examples |
|----------|------|---------|----------|
| **Technical** | technical.py | 7 | sma_crossover, rsi_mean_reversion, macd_crossover, bollinger_breakout |
| **ML Classifier** | ml.py, ml_siblings.py | 11 | ml_classifier, ml_classifier_v2-v5, ml_xgb, ml_hybrid |
| **ML Online** | ml_online.py | 2 | ml_classifier_online, ml_classifier_decay |
| **ML Cross-Asset** | ml_cross_asset.py | 2 | ml_cross_asset, ml_cross_asset_regime |
| **Cross-Symbol** | cross_symbol.py | 6 | btc_lead_alt_follow, eth_btc_ratio_reversion, correlation_breakdown |
| **Momentum** | momentum.py | 7 | momentum_breakout, cross_sectional_momentum, relative_strength |
| **Statistical** | statistical.py | 6 | ratio_reversion, z_score_reversion, cointegration |
| **Microstructure** | microstructure.py, microstructure_v2.py | 7 | volume_divergence, buy_sell_imbalance, liquidity_vacuum |
| **Volatility** | volatility_trading.py, volatility_v2.py | 6 | volatility_squeeze, vol_term_structure, squeeze_cascade |
| **Calendar** | calendar.py, calendar_v2.py | 6 | weekend_effect, hour_of_day_filter, asian_session_reversal |
| **Alternative Data** | alternative_data_strategies.py | 4 | funding_rate_fade, funding_rate_carry, open_interest_divergence |
| **Funding** | funding_v2.py | 2 | funding_arbitrage_proxy, funding_momentum |
| **Macro** | macro.py | 4 | btc_dominance_rotation, dominance_momentum, dxy_inverse |
| **Positioning** | positioning.py | 4 | long_short_ratio_fade, crowded_trade_detector |
| **Sentiment** | sentiment.py | 3 | fear_greed_divergence, fear_greed_extreme_fade |
| **Regime** | regime.py | 1 | regime_adaptive |
| **Rotation** | rotation.py | 2 | sector_momentum_rotation, weekly_momentum |
| **Multi-Timeframe** | multi_timeframe.py, multi_timeframe_v2.py | 4 | multi_timeframe, mtf_momentum_divergence |
| **Meta** | meta.py | 3 | regime_gate, drawdown_pause, strategy_momentum |
| **Frequency** | frequency.py | 3 | signal_confirmation_delay, signal_strength_filter |
| **Ensemble** | ensemble.py | 1 | ensemble_voting |
| **Rule-Based** | rule_ensemble.py | 3 | rule_ensemble, trend_following_rules, mean_reversion_rules |
| **Structural** | structural.py | 3 | basis_proxy, funding_term_structure |
| **Hyper-Creative** | hyper_creative.py | 8 | Various experimental |
| **Hyper-Creative Tier2** | hyper_creative_tier2.py | 5 | Various experimental |
| **Information Theoretic** | information_theoretic.py | 3 | entropy_collapse, hurst_regime, fractal_breakout |
| **RL** | rl.py | 1 | rl_dqn |
| **Hybrid** | hybrid.py | 1 | eth_btc_ratio_confirmed |

### By Deployment Status

| Type | Total | Deployed | Validated | Retired |
|------|-------|----------|-----------|---------|
| **ratio_reversion** | 16 | 9 | 6 | 7 |
| **ml_classifier** | 12 | 0 | 0 | 12 |
| **technical** | 7 | 0 | 0 | 0 |
| **cross_symbol** | 6 | 0 | 0 | 0 |
| **microstructure** | 7 | 1 | 0 | 1 |
| **orthogonal** | 8 | 4 | 0 | 4 |
| **calendar** | 6 | 0 | 0 | 0 |
| **alternative_data** | 6 | 1 | 0 | 2 |
| **meta** | 5 | 0 | 0 | 1 |
| **rule_based** | 4 | 0 | 0 | 0 |
| **unexplored** | 8 | 0 | 1 | 7 |
| **macro** | 4 | 1 | 0 | 0 |
| **momentum** | 7 | 1 | 0 | 0 |
| **Other** | 36+ | 0 | 0 | 0 |

---

## Top Performers by Sharpe Ratio

### All-Time Best (OOS Sharpe > 5.0)

| Rank | Strategy | Type | OOS Sharpe | Status |
|------|----------|------|------------|--------|
| 1 | bnb_btc_ratio | ratio_reversion | **23.94** | ✅ Paper Trading |
| 2 | ada_btc_ratio | ratio_reversion | **19.74** | ✅ Paper Trading |
| 3 | xrp_btc_ratio | ratio_reversion | **16.10** | ✅ Paper Trading |
| 4 | dot_btc_ratio | ratio_reversion | **14.74** | ✅ Paper Trading |
| 5 | link_btc_ratio | ratio_reversion | **12.65** | ✅ Paper Trading |
| 6 | eth_btc_ratio_optimized | ratio_reversion | **9.54** | ✅ Primary |
| 7 | near_btc_ratio | ratio_reversion | **8.69** | ✅ Paper Trading |
| 8 | ltc_btc_ratio | ratio_reversion | **8.31** | ✅ Deployed |
| 9 | eth_btc_ratio_confirmed | ratio_reversion | **7.29** | 🔄 Backup |
| 10 | sol_btc_ratio | ratio_reversion | **6.54** | ✅ Deployed |

### Best Non-Ratio Strategies

| Rank | Strategy | Type | OOS Sharpe | Status |
|------|----------|------|------------|--------|
| 1 | basis_proxy | funding | **1.05** | ✅ Deployed |
| 2 | liquidity_vacuum_detector | microstructure | **0.78** | ✅ Deployed |
| 3 | btc_dominance_rotation | regime | **0.58** | ✅ Deployed |
| 4 | cross_sectional_momentum | factor | **0.57** | ✅ Deployed |
| 5 | funding_term_structure | positioning | **0.42** | ✅ Deployed |

---

## Key Findings

### What Works ✅

1. **Ratio Mean Reversion** - Dominant strategy type with Sharpe 6-24
   - BTC as reference outperforms ETH as reference
   - Tier 1-2 alts work best (high liquidity, high correlation)
   - Optimized params: lookback=48-72, entry=-1.2 to -1.8, exit=-0.7 to -0.9

2. **Funding-Based Strategies** - Moderate but diversifying alpha
   - basis_proxy (Sharpe 1.05)
   - funding_term_structure (Sharpe 0.42)

3. **Microstructure Strategies** - Low frequency but profitable
   - liquidity_vacuum_detector (Sharpe 0.78)

4. **Factor Strategies** - Slow but steady
   - cross_sectional_momentum (Sharpe 0.57)

### What Doesn't Work ❌

1. **ML Classifiers** - All 12+ variants overfit
   - Walk-forward validation shows negative OOS Sharpe
   - Feature reduction, regularization don't fix it

2. **ETH as Reference** - All ETH-based ratio pairs fail
   - BTC is better "market beta" proxy

3. **Tier 3 Alts** - Higher volatility, lower liquidity
   - APT, SUI, ARB all failed
   - Only NEAR passed validation

4. **Volatility Strategies** - Mostly negative
   - volatility_breakout, volatility_mean_reversion both failed
   - Only volatility_squeeze (unvalidated) may work

5. **Trend Following** - Doesn't work in crypto
   - crash_only_trend_filter failed (-0.16 Sharpe)
   - SMA crossovers not validated

---

## Research Timeline

| Note | Date | Focus | Result |
|------|------|-------|--------|
| 01-04 | Dec 20 | ML strategies | All overfit |
| 05-06 | Dec 21 | Creative design | eth_btc_ratio wins |
| 07-08 | Dec 21 | Full validation | 5 strategies validated |
| 09 | Dec 21 | Unexplored | basis_proxy discovered |
| 10-11 | Dec 21 | Grid optimization | Final parameters |
| 12 | Dec 22 | Hyper-creative | 24 strategies implemented |
| 13 | Dec 22 | Orthogonal | 4 diversifiers found |
| 14 | Dec 22 | Grid optimization | 4 orthogonal optimized |
| 15-16 | Dec 24 | Ratio expansion | 6 new pairs validated |

---

*Last Updated: December 24, 2025*

