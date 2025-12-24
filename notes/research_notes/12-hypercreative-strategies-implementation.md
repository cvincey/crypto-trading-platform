# Research Note 12: Hyper-Creative Diversification Strategies - Implementation

**Date**: December 22, 2025  
**Objective**: Implement 24 out-of-the-box strategies orthogonal to ratio strategies for portfolio diversification

---

## Executive Summary

Implemented **24 hyper-creative trading strategies** across 2 tiers, designed to provide diversification from the dominant ratio strategies (ETH/BTC, SOL/BTC, LTC/BTC). These strategies exploit completely different market dynamics and signal sources.

**Status**: ✅ All implementations complete, ready for backtesting

---

## Implementation Summary

### Tier 1: OHLCV Only (16 Strategies)

No external data required - can be backtested immediately.

#### Category A: Information-Theoretic / Statistical Physics (3 strategies)

| Strategy | Description | Signal Source |
|----------|-------------|---------------|
| `entropy_collapse` | Shannon entropy of returns drops → regime change | Information theory |
| `hurst_regime` | Hurst exponent determines mean-reversion vs trending | Statistical physics |
| `fractal_breakout` | Fractal dimension drops → consolidation breakout | Fractal geometry |

**Implementation**: [`src/crypto/strategies/information_theoretic.py`](../../src/crypto/strategies/information_theoretic.py)

#### Category B: Microstructure / Volume Dynamics (4 strategies)

| Strategy | Description | Signal Source |
|----------|-------------|---------------|
| `volume_clock_momentum` | Momentum in volume-time, not clock-time | Liquidity adaptation |
| `liquidity_vacuum` | Volume spike → dry-up → breakout | Price discovery |
| `body_ratio_sequence` | 5+ indecision candles → decision candle | Candle patterns |
| `taker_imbalance_momentum` | Estimate buy/sell from (close-low)/(high-low) | Order flow proxy |

**Implementation**: [`src/crypto/strategies/microstructure_v2.py`](../../src/crypto/strategies/microstructure_v2.py)

#### Category C: Multi-Timeframe / Cross-Asset (3 strategies)

| Strategy | Description | Signal Source |
|----------|-------------|---------------|
| `mtf_momentum_divergence` | 1h positive but 4h negative = reversion | Timeframe disagreement |
| `leader_follower_rotation` | Detect BTC/ETH leadership, trade follower | Dynamic correlation |
| `sector_dispersion_trade` | High L1 dispersion → trade convergence/divergence | Sector rotation |

**Implementation**: [`src/crypto/strategies/multi_timeframe_v2.py`](../../src/crypto/strategies/multi_timeframe_v2.py)

#### Category D: Volatility Regimes (3 strategies)

| Strategy | Description | Signal Source |
|----------|-------------|---------------|
| `vol_term_structure` | 24h ATR / 7d ATR ratio signals vol regime | Volatility curve |
| `squeeze_cascade` | ATR + BB width both compressed → breakout | Double squeeze |
| `vol_of_vol_breakout` | Volatility of volatility spikes → regime change | Second-order vol |

**Implementation**: [`src/crypto/strategies/volatility_v2.py`](../../src/crypto/strategies/volatility_v2.py)

#### Category E: Calendar / Structural (3 strategies)

| Strategy | Description | Signal Source |
|----------|-------------|---------------|
| `asian_session_reversal` | Fade overnight moves at European open | Session patterns |
| `weekend_gap_fade` | Friday-Sunday gap > 2% tends to fill | Weekend inefficiency |
| `funding_hour_momentum` | Patterns around funding times (0/8/16 UTC) | Structural timing |

**Implementation**: [`src/crypto/strategies/calendar_v2.py`](../../src/crypto/strategies/calendar_v2.py)

---

### Tier 2: Alternative Data (8 Strategies)

Requires external data ingestion - infrastructure in place but data placeholders.

#### Category F: Sentiment / Positioning (3 strategies)

| Strategy | Description | Data Source |
|----------|-------------|-------------|
| `fear_greed_divergence` | Price ↑ but F&G ↓ = divergence trade | alternative.me |
| `fear_greed_extreme_fade` | F&G < 20 buy, > 80 sell | alternative.me |
| `long_short_ratio_fade` | L/S ratio > 2.0 or < 0.5 = crowded trade | Coinglass |

**Implementation**: [`src/crypto/strategies/sentiment.py`](../../src/crypto/strategies/sentiment.py)

**Note**: Strategies return empty signals with warnings until data is ingested.

#### Category G: Market Structure (3 strategies)

| Strategy | Description | Data Source |
|----------|-------------|-------------|
| `btc_dominance_rotation` | Dominance ↑ = BTC, ↓ = alts | CoinGecko |
| `dominance_momentum` | Dominance 7d change > 2% = regime shift | CoinGecko |
| `liquidation_cluster_magnet` | Price gravitates toward liq clusters | Coinglass |

**Implementation**: [`src/crypto/strategies/macro.py`](../../src/crypto/strategies/macro.py) & [`positioning.py`](../../src/crypto/strategies/positioning.py)

#### Category H: Enhanced Funding (2 strategies)

| Strategy | Description | Data Source |
|----------|-------------|-------------|
| `funding_arbitrage_proxy` | Cross-exchange funding divergence | Multi-exchange |
| `funding_momentum` | 3+ consecutive funding increases = trend | Binance |

**Implementation**: [`src/crypto/strategies/funding_v2.py`](../../src/crypto/strategies/funding_v2.py)

---

## Architecture

### File Structure

```
src/crypto/strategies/
├── information_theoretic.py   # NEW - Entropy, Hurst, Fractal
├── microstructure_v2.py        # NEW - Volume dynamics
├── multi_timeframe_v2.py       # NEW - Cross-TF, cross-asset
├── volatility_v2.py            # NEW - Vol regime strategies
├── calendar_v2.py              # NEW - Session/time strategies
├── sentiment.py                # NEW - Fear & Greed
├── positioning.py              # NEW - L/S ratio, liquidations
├── macro.py                    # NEW - BTC dominance
└── funding_v2.py               # NEW - Enhanced funding

config/creative_testing.yaml    # UPDATED - 24 strategy configs
scripts/run_hypercreative_testing.py  # NEW - Testing script
```

### Configuration

All 24 strategies configured in [`config/creative_testing.yaml`](../../config/creative_testing.yaml) under `hypercreative_strategies` section with:
- Strategy type and tier (tier1 / tier2)
- Default parameters
- Description and rationale

### Data Infrastructure

- **Tier 2 repositories already exist** in [`src/crypto/data/alternative_data.py`](../../src/crypto/data/alternative_data.py):
  - `FearGreedRepository`
  - `BTCDominanceRepository`
  - `LongShortRatioRepository`
  - `FundingRateRepository` (already used by existing strategies)

- **Data ingestion scripts** available:
  - [`scripts/ingest_alternative_data.py`](../../scripts/ingest_alternative_data.py) - Funding & OI
  - Need to extend for Fear & Greed, BTC Dominance, L/S Ratio

---

## Testing Script

Created [`scripts/run_hypercreative_testing.py`](../../scripts/run_hypercreative_testing.py) with:

### Usage

```bash
# Quick test: Tier 1 only, 3 symbols, 6 months
python scripts/run_hypercreative_testing.py --quick

# Standard test: Tier 1 only, 3 symbols, 6 months
python scripts/run_hypercreative_testing.py

# Full test: Tier 1 only, 50 symbols, 12 months
python scripts/run_hypercreative_testing.py --full

# Include Tier 2 (requires data ingestion first)
python scripts/run_hypercreative_testing.py --tier 2

# Full test with all tiers
python scripts/run_hypercreative_testing.py --full --tier 2
```

### Features

- Walk-forward validation on all strategies
- Acceptance gates (Sharpe > 0.5, trades > 10, degradation < 50%)
- Parallel testing across symbols
- Reference data loading for cross-asset strategies
- Rich progress display and results table
- JSON export of results

---

## Key Design Decisions

### 1. Orthogonality to Ratio Strategies

All strategies designed to be **uncorrelated** with existing ratio strategies:

| Ratio Strategy | Signal | Active When |
|----------------|--------|-------------|
| ETH/BTC ratio | Mean reversion | One asset underperforms |
| SOL/BTC ratio | Mean reversion | One asset underperforms |

| Hyper-Creative | Signal | Active When |
|----------------|--------|-------------|
| Entropy collapse | Information theory | Regime change pending |
| Vol term structure | Volatility curve | Vol regime shifts |
| Asian session | Time of day | Session transitions |
| Fear & Greed | Sentiment | Sentiment extremes |

**Expected correlation < 0.3** (to be validated in backtest).

### 2. Signal Diversity

Different classes of signals:

- **Mathematical**: Entropy, Hurst, Fractal dimension
- **Microstructure**: Volume-time, body ratios, order flow
- **Temporal**: Session effects, weekend gaps, funding times
- **Cross-asset**: Leadership, dispersion, correlation
- **Sentiment**: Fear & Greed, positioning, dominance

### 3. Tier Structure

**Tier 1 (OHLCV only)**:
- No dependencies
- Backtest immediately
- 16 strategies

**Tier 2 (Alternative data)**:
- Requires data ingestion
- Free API sources only
- 8 strategies (stubs with clear TODOs)

### 4. Placeholder Implementation for Tier 2

Tier 2 strategies return empty signals with warnings until data is available:

```python
logger.warning(
    f"{self.name}: Fear & Greed Index data not available. "
    "Run data ingestion first."
)
```

This allows:
- Code compilation and import
- Clear error messages during testing
- Easy activation once data is ingested

---

## Next Steps

### Immediate: Run Tier 1 Backtest

```bash
# Quick validation (2-3 hours runtime)
python scripts/run_hypercreative_testing.py --quick

# Full validation (12-24 hours runtime)
python scripts/run_hypercreative_testing.py --full
```

Expected outcome:
- 2-4 strategies with OOS Sharpe > 0.5
- ~12-15% success rate (based on historical creative testing)
- Low correlation with ratio strategies

### Data Ingestion for Tier 2

1. **Fear & Greed Index** (alternative.me)
   - Free API, daily data
   - Create ingestion script

2. **BTC Dominance** (CoinGecko)
   - Free API, daily data
   - Parse from total market cap API

3. **Long/Short Ratio** (Coinglass or Binance)
   - Free tier available
   - Hourly data

### Analysis Phase

After backtesting:

1. **Performance ranking**
   - Sort by OOS Sharpe
   - Filter by acceptance gates
   - Identify top 3-5 strategies

2. **Correlation analysis**
   - Calculate correlation matrix with ratio strategies
   - Ensure diversification (target < 0.3 correlation)
   - Analyze when strategies are active vs quiet

3. **Portfolio simulation**
   - Combine top hyper-creative with existing ratios
   - Test different allocation schemes
   - Measure portfolio Sharpe improvement

---

## Success Criteria

Based on historical testing (Notes 06-11):

| Metric | Target | Historical Range |
|--------|--------|-----------------|
| Strategies passing gates | 2-4 | 5-15% success rate |
| Top strategy OOS Sharpe | > 1.0 | 0.5 - 2.5 |
| Correlation with ratios | < 0.3 | Low correlation desired |
| Trade frequency | 5-30/month | Avoid too high/low |

**Overall Goal**: At least **2 strategies** with:
- OOS Sharpe > 1.0
- Correlation with ratio strategies < 0.3
- Pass rate > 50% across symbols
- Ready for portfolio inclusion

---

## Risk Assessment

### Implementation Risks

| Risk | Mitigation | Status |
|------|-----------|--------|
| Complex calculations (Hurst, Fractal) | Well-tested libraries (scipy) | ✓ Mitigated |
| Cross-asset data alignment | Common index alignment logic | ✓ Mitigated |
| Tier 2 data availability | Graceful fallback with warnings | ✓ Mitigated |
| Strategy registration | All imported in __init__.py | ✓ Mitigated |

### Strategy Risks

| Risk | Impact | Monitoring |
|------|--------|-----------|
| Mathematical strategies overfit | Medium | Check degradation % |
| Calendar effects disappear | Medium | Validate across time periods |
| Alternative data quality | High | Tier 2 only, test thoroughly |
| Cross-asset correlations unstable | Medium | Rolling correlation checks |

---

## Conclusion

Successfully implemented **24 hyper-creative strategies** designed for diversification:
- **16 Tier 1 strategies** ready for immediate backtesting
- **8 Tier 2 strategies** with infrastructure in place, pending data
- Testing script operational
- Configuration complete

**Next action**: Run `python scripts/run_hypercreative_testing.py --quick` to begin validation.

Expected timeline:
- Quick test: 2-3 hours
- Full test: 12-24 hours
- Analysis: 2-4 hours
- Research note: 2 hours

**Total time to results: ~48 hours**

---

*Implementation complete | December 22, 2025 | Research Note 12*
