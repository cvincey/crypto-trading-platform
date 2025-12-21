# Research Plan: Creative Strategy Testing on Hourly Data

**Date:** December 21, 2024  
**Author:** Trading Research Team  
**Version:** 1.0

---

## Executive Summary

Following the findings in Notes 01-04, we know that:
- Technical indicators on hourly data don't predict direction
- Rule-based strategies fail even without overfitting
- High trading frequency destroys value through costs

This plan tests **creative alternative approaches** on hourly data to definitively determine whether any edge exists. We explore **new alpha sources**, **cross-symbol signals**, and **structural strategies** that might work where technical indicators failed.

---

## 1. Testing Philosophy

### 1.1 Key Constraints from Note 04

| Constraint | Implication for Testing |
|------------|------------------------|
| ~0.1% cost per trade | Strategies must win big OR trade rarely |
| Technical indicators don't predict | Need alternative signal sources |
| Hourly is noisy | Need higher signal strength OR longer holding periods |
| All strategies underperform B&H | New strategies must beat B&H, not just positive Sharpe |

### 1.2 New Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| OOS Sharpe | > 0.5 | Must be meaningfully positive |
| vs Buy-and-Hold | > 0 | Must beat doing nothing |
| Trades per month | < 20 | Limit cost drag |
| Max Drawdown | < 20% | Risk management |
| Win Rate | > 55% | Edge must be consistent |
| Profit Factor | > 1.3 | Wins must outweigh losses |

### 1.3 Validation Protocol

All strategies will be tested with:
1. **Walk-forward validation** (train 60d, test 14d, roll 7d)
2. **Out-of-sample symbols** (holdout coins not used in development)
3. **Comparison to buy-and-hold** on same period
4. **Cost sensitivity analysis** (test at 0.05%, 0.1%, 0.15% commission)

---

## 2. Strategy Categories to Test

### Overview

| Category | # Strategies | Key Hypothesis |
|----------|--------------|----------------|
| A. Cross-Symbol Signals | 5 | BTC/ETH lead, alts follow |
| B. Alternative Data | 4 | Non-price data has alpha |
| C. Structural/Carry | 3 | Collect premiums, not predict |
| D. Frequency Reduction | 3 | Hourly data, weekly trades |
| E. Meta-Strategies | 3 | Filter when to trade |
| F. Market Microstructure | 3 | Order flow & liquidity |
| **Total** | **21** | |

---

## 3. Category A: Cross-Symbol Signal Strategies

### Hypothesis
BTC and ETH move first; alts follow. Using leader signals to trade followers exploits structural information propagation.

### A1. `btc_lead_alt_follow`

**Signal:** BTC breaks above 24-bar (1-day) high  
**Action:** Buy ALT on next candle  
**Exit:** 24-72 hours later OR stop loss 4% OR take profit 12%  
**Hypothesis:** Information propagates from BTC to alts with lag

```yaml
btc_lead_alt_follow:
  type: btc_lead_alt_follow
  params:
    leader_symbol: BTCUSDT
    breakout_period: 24          # 24h high
    entry_delay: 1               # Enter 1 bar after BTC signal
    hold_period: 48              # Hold for 48 hours
    stop_loss_pct: 0.04
    take_profit_pct: 0.12
  symbols: [SOLUSDT, AVAXUSDT, NEARUSDT, DOTUSDT]
  interval: 1h
```

### A2. `eth_btc_ratio_reversion`

**Signal:** ETH/BTC ratio z-score < -2 (ETH underperforming)  
**Action:** Long ETH  
**Exit:** Ratio returns to mean (z > -0.5) OR 72 hours  
**Hypothesis:** ETH/BTC ratio mean-reverts over days

```yaml
eth_btc_ratio_reversion:
  type: eth_btc_ratio_reversion
  params:
    lookback: 168                # 7 days for z-score
    entry_threshold: -2.0
    exit_threshold: -0.5
    max_hold_hours: 72
  symbols: [ETHUSDT]
  interval: 1h
```

### A3. `correlation_breakdown`

**Signal:** 30-day correlation > 0.85, but 24h returns diverge by > 3%  
**Action:** Long the laggard  
**Exit:** Gap closes OR 48 hours  
**Hypothesis:** Correlated assets that diverge will reconverge

```yaml
correlation_breakdown:
  type: correlation_breakdown
  params:
    reference_symbol: BTCUSDT
    correlation_window: 720      # 30 days
    min_correlation: 0.85
    divergence_threshold: 0.03   # 3% divergence
    max_hold_hours: 48
  symbols: [ETHUSDT, SOLUSDT, BNBUSDT]
  interval: 1h
```

### A4. `sector_momentum_rotation`

**Signal:** Sector (L1s, DeFi, L2s) with highest 7-day momentum  
**Action:** Long top sector, hold for 7 days  
**Exit:** Rebalance weekly  
**Hypothesis:** Crypto capital rotates between narratives

```yaml
sector_momentum_rotation:
  type: sector_momentum_rotation
  params:
    sectors:
      L1: [SOLUSDT, AVAXUSDT, NEARUSDT, APTUSDT]
      DeFi: [UNIUSDT, AAVEUSDT, LINKUSDT]
      Major: [BTCUSDT, ETHUSDT]
    momentum_period: 168         # 7 days
    rebalance_period: 168        # Weekly
    top_n_sectors: 1
  symbols: [SOLUSDT, AVAXUSDT, NEARUSDT, APTUSDT, UNIUSDT, AAVEUSDT]
  interval: 1h
```

### A5. `btc_volatility_filter`

**Signal:** BTC ATR percentile < 70th (low volatility)  
**Action:** Allow trading; otherwise HOLD  
**Exit:** N/A (meta-filter)  
**Hypothesis:** Alt signals are noise during BTC volatility spikes

```yaml
btc_volatility_filter:
  type: btc_volatility_filter
  params:
    base_strategy: momentum_breakout
    btc_atr_period: 24
    btc_atr_percentile_window: 168
    max_volatility_percentile: 70
  symbols: [SOLUSDT, AVAXUSDT]
  interval: 1h
```

---

## 4. Category B: Alternative Data Strategies

### Hypothesis
Price-derived indicators are arbitraged away. External data sources may contain unexploited alpha.

### B1. `funding_rate_fade`

**Signal:** 8h funding rate > 0.05% (extreme positive)  
**Action:** Short bias (avoid longs, or short if enabled)  
**Exit:** Funding normalizes < 0.02%  
**Hypothesis:** Extreme funding = crowded trade = reversal

```yaml
funding_rate_fade:
  type: funding_rate_fade
  params:
    extreme_positive: 0.0005     # 0.05%
    extreme_negative: -0.0005
    normal_threshold: 0.0002
    action_mode: avoid_longs     # or 'short' if shorting enabled
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  interval: 1h
  data_sources:
    - binance_funding_rates
```

### B2. `funding_rate_carry`

**Signal:** Funding rate consistently positive (3-day average > 0.03%)  
**Action:** Long spot (collect implied carry)  
**Exit:** Funding turns negative  
**Hypothesis:** Positive funding = longs pay; being long spot captures this

```yaml
funding_rate_carry:
  type: funding_rate_carry
  params:
    funding_avg_period: 9        # 3 days of 8h funding
    min_avg_funding: 0.0003      # 0.03% avg
    exit_threshold: 0
  symbols: [BTCUSDT, ETHUSDT]
  interval: 1h
```

### B3. `open_interest_divergence`

**Signal:** Price up but OI down (weak rally)  
**Action:** Short or avoid longs  
**Exit:** OI confirms price direction  
**Hypothesis:** OI divergence signals unsustainable moves

```yaml
open_interest_divergence:
  type: open_interest_divergence
  params:
    price_lookback: 24           # 24h price change
    oi_lookback: 24              # 24h OI change
    divergence_threshold: 0.02   # 2% price up, OI flat/down
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  interval: 1h
  data_sources:
    - binance_open_interest
```

### B4. `liquidation_cascade_fade`

**Signal:** Large liquidation event detected (> $10M in 1h)  
**Action:** Fade the move (buy after long liquidations, sell after short liquidations)  
**Exit:** 12-24 hours later  
**Hypothesis:** Cascades overshoot; mean reversion follows

```yaml
liquidation_cascade_fade:
  type: liquidation_cascade_fade
  params:
    liquidation_threshold: 10_000_000  # $10M
    lookback_hours: 1
    fade_delay: 1                # Wait 1 bar after cascade
    hold_period: 24
  symbols: [BTCUSDT, ETHUSDT]
  interval: 1h
  data_sources:
    - coinglass_liquidations
```

---

## 5. Category C: Structural/Carry Strategies

### Hypothesis
Instead of predicting direction, exploit structural market features.

### C1. `weekend_effect`

**Signal:** Friday 20:00 UTC  
**Action:** Reduce exposure or go flat  
**Exit:** Monday 08:00 UTC  
**Hypothesis:** Weekend volatility is unrewarded; avoid it

```yaml
weekend_effect:
  type: weekend_effect
  params:
    exit_day: friday
    exit_hour: 20
    reentry_day: monday
    reentry_hour: 8
    base_strategy: ml_classifier_xgb
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  interval: 1h
```

### C2. `hour_of_day_filter`

**Signal:** Only trade during high-liquidity hours (14:00-22:00 UTC)  
**Action:** Filter signals outside these hours  
**Exit:** N/A (filter)  
**Hypothesis:** Low-liquidity hours have worse fills and more noise

```yaml
hour_of_day_filter:
  type: hour_of_day_filter
  params:
    active_hours: [14, 15, 16, 17, 18, 19, 20, 21, 22]
    base_strategy: momentum_breakout
  symbols: [BTCUSDT, ETHUSDT]
  interval: 1h
```

### C3. `month_end_rebalancing`

**Signal:** Last 3 days of month  
**Action:** Anticipate institutional flows; long momentum leaders  
**Exit:** First 3 days of new month  
**Hypothesis:** Institutions rebalance month-end

```yaml
month_end_rebalancing:
  type: month_end_rebalancing
  params:
    entry_days_before_eom: 3
    exit_days_after_eom: 3
    selection: momentum_leaders
    momentum_period: 168
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  interval: 1h
```

---

## 6. Category D: Frequency Reduction Strategies

### Hypothesis
Hourly data is fine, but we should trade less frequently to reduce costs.

### D1. `weekly_momentum_hourly_data`

**Signal:** Best performing coin over last 168 hours  
**Action:** Hold winner for next 168 hours  
**Exit:** Rebalance weekly  
**Hypothesis:** Momentum persists weekly; hourly data just for precise entry

```yaml
weekly_momentum:
  type: weekly_momentum
  params:
    momentum_period: 168
    hold_period: 168
    top_n: 3
    rebalance_at_hour: 0        # Rebalance at midnight UTC
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, NEARUSDT]
  interval: 1h
```

### D2. `signal_confirmation_delay`

**Signal:** Base strategy signal + confirmed by same signal 4 hours later  
**Action:** Enter only on confirmation  
**Exit:** Base strategy exit  
**Hypothesis:** False signals don't persist; requiring confirmation filters noise

```yaml
signal_confirmation_delay:
  type: signal_confirmation_delay
  params:
    base_strategy: rsi_mean_reversion
    confirmation_delay: 4        # 4 hours
    require_consistent: true
  symbols: [BTCUSDT, ETHUSDT]
  interval: 1h
```

### D3. `signal_strength_filter`

**Signal:** Only enter when signal strength is extreme (RSI < 20, not just < 30)  
**Action:** Trade only on extreme signals  
**Exit:** Mean reversion  
**Hypothesis:** Weak signals are noise; only extreme signals have edge

```yaml
signal_strength_filter:
  type: signal_strength_filter
  params:
    base_strategy: rsi_mean_reversion
    rsi_extreme_oversold: 20     # Instead of 30
    rsi_extreme_overbought: 80   # Instead of 70
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  interval: 1h
```

---

## 7. Category E: Meta-Strategies

### Hypothesis
Know when NOT to trade is as important as knowing when to trade.

### E1. `regime_gate`

**Signal:** Only allow trades when ADX > 30 (strong trend) OR ADX < 15 (range)  
**Action:** Filter signals in ambiguous regimes (ADX 15-30)  
**Exit:** N/A (filter)  
**Hypothesis:** Middle regimes are unpredictable; extremes are tradeable

```yaml
regime_gate:
  type: regime_gate
  params:
    trend_strategy: momentum_breakout
    range_strategy: rsi_mean_reversion
    adx_trend_threshold: 30
    adx_range_threshold: 15
    base_strategy: null          # Only trade in clear regimes
  symbols: [BTCUSDT, ETHUSDT]
  interval: 1h
```

### E2. `drawdown_pause`

**Signal:** Strategy equity curve drops 5%  
**Action:** Pause trading for 48 hours  
**Exit:** Resume after pause  
**Hypothesis:** Drawdowns cluster; pausing prevents cascading losses

```yaml
drawdown_pause:
  type: drawdown_pause
  params:
    base_strategy: ml_classifier_xgb
    pause_threshold: 0.05        # 5% drawdown
    pause_duration: 48           # 48 hours
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  interval: 1h
```

### E3. `strategy_momentum`

**Signal:** Track which strategy worked best in last 7 days  
**Action:** Only use the currently "hot" strategy  
**Exit:** Recompute weekly  
**Hypothesis:** Strategy performance persists short-term

```yaml
strategy_momentum:
  type: strategy_momentum
  params:
    strategies:
      - momentum_breakout
      - rsi_mean_reversion
      - ml_classifier_xgb
    evaluation_period: 168       # 7 days
    selection_count: 1           # Use top 1
  symbols: [BTCUSDT, ETHUSDT]
  interval: 1h
```

---

## 8. Category F: Market Microstructure Strategies

### Hypothesis
Volume and order flow contain information not captured by price indicators.

### F1. `volume_breakout_confirmation`

**Signal:** Price breakout + Volume > 2x average  
**Action:** Enter on confirmed breakout  
**Exit:** Volume-confirmed exit or stop loss  
**Hypothesis:** Volume confirms genuine breakouts; low-volume breakouts fail

```yaml
volume_breakout_confirmation:
  type: volume_breakout_confirmation
  params:
    breakout_period: 24
    volume_multiplier: 2.0
    volume_average_period: 48
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  interval: 1h
```

### F2. `volume_divergence`

**Signal:** Price making new highs but volume declining  
**Action:** Exit longs or prepare for reversal  
**Exit:** Volume confirms or reversal occurs  
**Hypothesis:** Divergence between price and volume signals exhaustion

```yaml
volume_divergence:
  type: volume_divergence
  params:
    price_lookback: 48
    volume_lookback: 48
    divergence_threshold: 0.3    # Volume down 30%+ while price up
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  interval: 1h
```

### F3. `buy_sell_imbalance`

**Signal:** Calculate buy/sell volume ratio from candle data  
**Action:** Trade in direction of imbalance when extreme  
**Exit:** Imbalance normalizes  
**Hypothesis:** Extreme imbalances predict short-term direction

```yaml
buy_sell_imbalance:
  type: buy_sell_imbalance
  params:
    # Estimate buy vol = close > open portion of range
    imbalance_lookback: 12
    buy_threshold: 0.7           # 70% buy volume
    sell_threshold: 0.3          # 30% buy volume (70% sell)
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  interval: 1h
```

---

## 9. Testing Plan Execution

### Phase 1: Quick Validation (Week 1)

**Goal:** Rapidly filter to promising candidates

| Step | Action | Duration |
|------|--------|----------|
| 1.1 | Implement all 21 strategies (stubs OK for data-dependent) | 2 days |
| 1.2 | Run quick_validate on 3 symbols, 6 months | 1 day |
| 1.3 | Filter to top 8 by OOS Sharpe | 0.5 day |

**Quick Validation Config:**
```yaml
quick_validation:
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  train_window: 1440             # 60 days
  test_window: 336               # 14 days
  step_size: 336                 # Bi-weekly
  days: 180
```

### Phase 2: Deep Validation (Week 2)

**Goal:** Rigorous walk-forward on promising candidates

| Step | Action | Duration |
|------|--------|----------|
| 2.1 | Full walk-forward on top 8 strategies | 2 days |
| 2.2 | 19 symbols, 12 months data | |
| 2.3 | Compare to buy-and-hold baseline | 0.5 day |

**Full Validation Config:**
```yaml
full_validation:
  symbols: all_19
  train_window: 2160             # 90 days
  test_window: 336               # 14 days
  step_size: 168                 # Weekly
  days: 365
```

### Phase 3: Robustness Testing (Week 3)

**Goal:** Stress test survivors

| Step | Action | Duration |
|------|--------|----------|
| 3.1 | Cost sensitivity (0.05%, 0.1%, 0.15%) | 1 day |
| 3.2 | Different time periods (2023 vs 2024) | 1 day |
| 3.3 | Symbol holdout testing | 1 day |
| 3.4 | Parameter sensitivity analysis | 1 day |

### Phase 4: Analysis & Conclusions (Week 4)

| Step | Action | Duration |
|------|--------|----------|
| 4.1 | Compile results into research note | 1 day |
| 4.2 | Identify any viable strategies | |
| 4.3 | Determine if hourly is fundamentally broken | |

---

## 10. Data Requirements

### 10.1 Currently Available

| Data | Source | Status |
|------|--------|--------|
| OHLCV hourly | Binance | ✅ Available |
| 19 symbols | TimescaleDB | ✅ Available |
| 12 months history | Ingested | ✅ Available |

### 10.2 New Data Needed

| Data | Source | Strategies Using | Priority |
|------|--------|------------------|----------|
| Funding rates | Binance Futures API | B1, B2 | 🔴 High |
| Open interest | Binance Futures API | B3 | 🔴 High |
| Liquidations | Coinglass API | B4 | 🟡 Medium |
| BTC dominance | CoinGecko API | (future) | 🟢 Low |

### 10.3 Data Ingestion Tasks

```python
# New ingestion scripts needed:
scripts/ingest_funding_rates.py    # Binance funding rate history
scripts/ingest_open_interest.py    # Binance OI history
scripts/ingest_liquidations.py     # Coinglass liquidation data (optional)
```

---

## 11. Success Criteria Summary

### Strategy Passes If:

| Criterion | Threshold |
|-----------|-----------|
| OOS Sharpe | > 0.5 |
| vs Buy-and-Hold gap | > 0 |
| Sharpe degradation | < 50% |
| Trades per month | < 20 |
| Max drawdown | < 20% |
| Pass rate across symbols | > 60% |

### Overall Research Succeeds If:

- At least 1 strategy passes all criteria
- OR we definitively prove hourly crypto is untradeable with current data

---

## 12. Implementation Checklist

### New Strategy Classes Needed

| Strategy | File | Complexity |
|----------|------|------------|
| `BTCLeadAltFollowStrategy` | cross_symbol.py | Medium |
| `ETHBTCRatioReversionStrategy` | cross_symbol.py | Medium |
| `CorrelationBreakdownStrategy` | cross_symbol.py | Medium |
| `SectorMomentumRotationStrategy` | cross_symbol.py | High |
| `BTCVolatilityFilterStrategy` | meta.py | Low |
| `FundingRateFadeStrategy` | alternative_data.py | Medium (needs data) |
| `FundingRateCarryStrategy` | alternative_data.py | Medium (needs data) |
| `OpenInterestDivergenceStrategy` | alternative_data.py | Medium (needs data) |
| `LiquidationCascadeFadeStrategy` | alternative_data.py | High (needs data) |
| `WeekendEffectStrategy` | calendar.py | Low |
| `HourOfDayFilterStrategy` | calendar.py | Low |
| `MonthEndRebalancingStrategy` | calendar.py | Low |
| `WeeklyMomentumStrategy` | frequency.py | Low |
| `SignalConfirmationDelayStrategy` | meta.py | Low |
| `SignalStrengthFilterStrategy` | meta.py | Low |
| `RegimeGateStrategy` | meta.py | Medium |
| `DrawdownPauseStrategy` | meta.py | Medium |
| `StrategyMomentumStrategy` | meta.py | High |
| `VolumeBreakoutConfirmationStrategy` | microstructure.py | Low |
| `VolumeDivergenceStrategy` | microstructure.py | Low |
| `BuySellImbalanceStrategy` | microstructure.py | Medium |

### Infrastructure Updates

| Component | Change | Priority |
|-----------|--------|----------|
| `BacktestRunner` | Support multi-symbol reference data | 🔴 High |
| `DataRepository` | Add funding rate / OI queries | 🔴 High |
| `quick_validate.py` | Add B&H comparison | 🟡 Medium |
| `walk_forward.py` | Track trades per month metric | 🟡 Medium |

---

## Appendix A: Hypothesis Priority Ranking

Based on research findings, ordered by probability of success:

| Rank | Category | Strategy | Rationale |
|------|----------|----------|-----------|
| 1 | B | funding_rate_carry | Not prediction, structural carry |
| 2 | D | weekly_momentum | Proven factor, low frequency |
| 3 | A | btc_lead_alt_follow | Structural relationship |
| 4 | A | correlation_breakdown | Cross-asset mean reversion |
| 5 | B | funding_rate_fade | External data, contrarian |
| 6 | E | drawdown_pause | Risk management, can't hurt |
| 7 | F | volume_breakout_confirmation | Filters false breakouts |
| 8 | C | weekend_effect | Calendar anomaly |
| 9 | D | signal_confirmation_delay | Reduces noise |
| 10 | B | open_interest_divergence | Position data signal |
| ... | ... | ... | ... |

---

*End of Research Plan*
