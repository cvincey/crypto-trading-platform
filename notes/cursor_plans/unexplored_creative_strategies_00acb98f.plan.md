---
name: Unexplored Creative Strategies
overview: Implement 15+ unexplored creative strategies using config-first approach, backtest on 2024-2025 data for strategies that don't require external API keys, and write a summary research note (Note 09).
todos:
  - id: db-models
    content: Add database models for FearGreedIndex, MacroIndicator, ExchangeFlow
    status: completed
  - id: tier1-strategies
    content: Implement 8 Tier 1 strategies (vol, basis, pair trade, regime)
    status: completed
  - id: tier1-config
    content: Add Tier 1 strategy configs to creative_testing.yaml
    status: completed
  - id: tier1-backtest
    content: Run quick validation on Tier 1 strategies
    status: completed
  - id: tier2-ingestion
    content: Create ingestion scripts for Fear/Greed, BTC dominance, L/S ratio
    status: cancelled
  - id: tier2-fetch
    content: Fetch Tier 2 data from free APIs
    status: cancelled
  - id: tier2-strategies
    content: Implement 3 Tier 2 strategies (sentiment, dominance rotation)
    status: cancelled
  - id: tier2-backtest
    content: Run quick validation on Tier 2 strategies
    status: cancelled
  - id: tier3-stubs
    content: Create stub implementations for 4 Tier 3 strategies (on-chain, options)
    status: cancelled
  - id: full-validation
    content: Run full validation on all passing strategies (2024 + 2025)
    status: cancelled
  - id: research-note
    content: Write research note 09 with findings and recommendations
    status: completed
---

# Unexplored Creative Strategies Implementation

## Phase 1: Data Layer Updates

### 1.1 Database Schema Extensions

Add models for new data types in `src/crypto/data/models.py`:

- `FearGreedIndexModel` - Fear & Greed Index history
- `MacroIndicatorModel` - DXY, VIX, SPX correlation data  
- `ExchangeFlowModel` - Exchange inflow/outflow (future use)

### 1.2 Data Ingestion (No API Keys Required)

Create ingestion scripts for free data sources:| Data Source | API | Strategy Using It ||-------------|-----|-------------------|| Fear & Greed Index | alternative.me (free) | `fear_greed_contrarian` || BTC Dominance | CoinGecko (free tier) | `btc_dominance_rotation` || Long/Short Ratio | Binance Futures (free) | `sentiment_contrarian` |

### 1.3 API Keys Guidance (For Future Use)

Document required API keys for advanced strategies:

- **CryptoQuant** (on-chain): Exchange flows, whale tracking
- **Glassnode** (on-chain): MVRV, active addresses
- **Deribit/Laevitas** (options): Put/call ratio, max pain
- **TradingView** (macro): DXY, VIX real-time

---

## Phase 2: Strategy Implementation

### 2.1 New Strategy File Structure

Create new strategy modules in `src/crypto/strategies/`:

```javascript
strategies/
├── sentiment.py          # Fear/Greed, Long/Short ratio
├── macro_correlation.py  # DXY, VIX, SPX signals  
├── volatility_trading.py # Vol mean reversion, breakout
├── structural.py         # Basis trade, CME gaps
├── onchain.py           # Exchange flows (stub for API keys)
└── options_flow.py      # Put/call, max pain (stub for API keys)
```



### 2.2 Strategies to Implement (15 total)

**Tier 1 - No External Data Needed (Backtest Ready):**| Strategy | Type | Data Source ||----------|------|-------------|| `volatility_mean_reversion` | Vol Trading | OHLCV (ATR) || `volatility_breakout` | Vol Trading | OHLCV (ATR) || `basis_proxy` | Structural | Funding rate as proxy || `multi_asset_pair_trade` | Pair Trading | OHLCV (SOL/AVAX) || `dxy_correlation_proxy` | Macro | BTC as inverse proxy || `regime_volatility_switch` | Regime | OHLCV + ATR || `momentum_quality` | Momentum | OHLCV + volume || `trend_strength_filter` | Meta | OHLCV + ADX |**Tier 2 - Free API Data (Backtest After Ingestion):**| Strategy | Type | Data Source ||----------|------|-------------|| `fear_greed_contrarian` | Sentiment | alternative.me || `btc_dominance_rotation` | Rotation | CoinGecko || `long_short_ratio_fade` | Sentiment | Binance Futures |**Tier 3 - Stubs Only (Require Paid API Keys):**| Strategy | Type | Data Needed ||----------|------|-------------|| `exchange_flow_signal` | On-Chain | CryptoQuant/Glassnode || `whale_tracking` | On-Chain | CryptoQuant || `options_max_pain` | Options | Deribit/Laevitas || `put_call_extremes` | Options | Deribit/Laevitas |

### 2.3 Config-First Approach

Add all strategies to `config/creative_testing.yaml`:

```yaml
unexplored_strategies:
  tier1_no_api:
    - volatility_mean_reversion
    - volatility_breakout
    - basis_proxy
    - multi_asset_pair_trade
    # ... etc
  tier2_free_api:
    - fear_greed_contrarian
    - btc_dominance_rotation
    - long_short_ratio_fade
  tier3_paid_api:
    - exchange_flow_signal  # stub
    - whale_tracking        # stub
    # ... etc
```

---

## Phase 3: Data Fetching

### 3.1 Scripts to Create

- `scripts/ingest_fear_greed.py` - Fetch Fear & Greed Index history
- `scripts/ingest_btc_dominance.py` - Fetch BTC dominance from CoinGecko
- `scripts/ingest_long_short_ratio.py` - Fetch Binance long/short ratio

### 3.2 Execution Order

1. Run Tier 1 strategies immediately (no data fetch needed)
2. Fetch Tier 2 data (Fear/Greed, dominance, L/S ratio)
3. Run Tier 2 strategies after data ingestion

---

## Phase 4: Backtesting

### 4.1 Test Configuration

- **Symbols**: 19 crypto pairs (existing set)
- **Periods**: 2024 full year + 2025 YTD
- **Validation**: Walk-forward (90-day train, 14-day test)
- **Baseline**: Compare to buy-and-hold and `eth_btc_ratio_reversion`

### 4.2 Backtest Script

Create `scripts/run_unexplored_testing.py`:

- Phase 1: Quick validation (3 symbols, 180 days)
- Phase 2: Full validation (19 symbols, 365 days)
- Output: JSON results + summary tables

### 4.3 Metrics to Track

- OOS Sharpe ratio
- Beats buy-and-hold %
- Beats eth_btc_ratio_reversion %
- Trades per month
- Max drawdown
- Pass rate across symbols

---

## Phase 5: Research Note

### 5.1 Create `notes/research_notes/09-unexplored-strategies.md`

Structure:

1. **Executive Summary** - Top 3 findings
2. **Strategy Performance Table** - Ranked by OOS Sharpe
3. **Comparison vs Existing Winners** - vs eth_btc_ratio_reversion
4. **Key Recommendations** - What to deploy next
5. **API Keys Needed** - Guide for unlocking Tier 3 strategies

---

## Files to Create/Modify

| File | Action ||------|--------|| `src/crypto/data/models.py` | Add new data models || `src/crypto/strategies/sentiment.py` | NEW - Sentiment strategies || `src/crypto/strategies/macro_correlation.py` | NEW - Macro strategies || `src/crypto/strategies/volatility_trading.py` | NEW - Vol strategies || `src/crypto/strategies/structural.py` | NEW - Basis/gap strategies || `src/crypto/strategies/onchain.py` | NEW - On-chain stubs || `src/crypto/strategies/options_flow.py` | NEW - Options stubs || `config/creative_testing.yaml` | Add new strategy configs || `scripts/ingest_fear_greed.py` | NEW - Fear/Greed ingestion || `scripts/ingest_btc_dominance.py` | NEW - Dominance ingestion || `scripts/ingest_long_short_ratio.py` | NEW - L/S ratio ingestion || `scripts/run_unexplored_testing.py` | NEW - Backtest runner || `notes/research_notes/09-unexplored-strategies.md` | NEW - Results |---

## Execution Order

1. Update database models (schema migration if needed)
2. Implement Tier 1 strategies (8 strategies, no API)
3. Run quick validation on Tier 1
4. Create data ingestion scripts for Tier 2
5. Fetch Tier 2 data (Fear/Greed, dominance, L/S ratio)
6. Implement Tier 2 strategies (3 strategies)
7. Run quick validation on Tier 2
8. Implement Tier 3 stubs (4 strategies, no backtest)