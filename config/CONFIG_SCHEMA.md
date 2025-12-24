# Configuration Schema & Conventions

This document defines the structure, taxonomy, and conventions for all configuration files.

---

## Directory Structure

```
config/
├── CONFIG_SCHEMA.md          ← This file (conventions documentation)
│
├── ─────────────────────────────────────────────────────────────────
├── PRODUCTION (source of truth for deployment)
├── ─────────────────────────────────────────────────────────────────
├── strategies.yaml           ← Strategy TYPE definitions + default params
├── paper_trading.yaml        ← Deployment instances (what's running)
├── settings.yaml             ← Global settings (DB, intervals, etc.)
├── exchanges.yaml            ← Exchange connection config
│
├── ─────────────────────────────────────────────────────────────────
├── OPERATIONAL (testing infrastructure)
├── ─────────────────────────────────────────────────────────────────
├── backtests.yaml            ← Backtest run configurations
├── optimization.yaml         ← Parameter optimization settings
│
└── ─────────────────────────────────────────────────────────────────
    research/                 ← HISTORICAL (read-only after validation)
    ├── README.md             ← Documents research phases
    ├── ratio_universe.yaml   ← Ratio pair research & results
    └── creative_testing.yaml ← Creative strategy experiments
```

---

## Strategy Lifecycle Tags

Every strategy definition must include a `stage` tag indicating its validation state.

### `stage` (required)

| Value | Meaning | Next Step |
|-------|---------|-----------|
| `experimental` | Not yet validated through walk-forward testing | Run walk-forward validation |
| `validated` | Passed validation, ready to deploy | Add to `paper_trading.yaml` |
| `deployed` | Currently running in paper/live trading | Monitor performance |
| `retired` | Failed validation or deprecated | Document reason, disable |

### `role` (optional)

| Value | Meaning |
|-------|---------|
| `primary` | Core strategy, gets priority allocation |
| `backup` | Ready to replace primary if it underperforms |
| `diversifier` | Provides uncorrelated returns to core strategies |

### `enabled` (required in paper_trading.yaml)

| Value | Meaning |
|-------|---------|
| `true` | Strategy is actively trading |
| `false` | Strategy is configured but not running |

---

## File Responsibilities

### `strategies.yaml`
**Purpose:** Define strategy TYPES and their default parameters.

```yaml
strategies:
  eth_btc_ratio_reversion:
    stage: deployed
    role: primary
    type: eth_btc_ratio_reversion
    params:
      lookback: 48
      entry_threshold: -2.0
      # ... default params
    symbols: [ETHUSDT]
    interval: 1h
    stop_loss_pct: 0.03
    take_profit_pct: 0.08
    enabled: true
```

**Rules:**
- One entry per strategy type
- Include all parameters with their validated values
- Update `stage` when validation state changes

### `paper_trading.yaml`
**Purpose:** Configure DEPLOYMENT of strategies.

```yaml
paper_trading:
  strategies:
    eth_btc_ratio_optimized:
      stage: deployed
      role: primary
      type: eth_btc_ratio_reversion
      # ... params (can override strategies.yaml)
      enabled: true   # ← Actually trading
```

**Rules:**
- Contains full strategy definitions for deployment (currently not referencing strategies.yaml)
- `enabled: true/false` controls actual trading
- When updating params, update BOTH files for consistency

> 📝 **Note:** Currently `paper_trading.yaml` contains full inline definitions rather than
> referencing `strategies.yaml`. This is intentional for operational independence - 
> paper trading config shouldn't break if strategy definitions change. Future versions
> may implement inheritance/references for DRY compliance.

### `research/*.yaml`
**Purpose:** Document completed research (read-only).

**Rules:**
- Frozen after research phase is complete
- Contains validation results, not deployment config
- Referenced by research notes in `notes/research_notes/`

---

## Validation Metadata

When a strategy passes validation, include these fields:

```yaml
validation:
  sharpe_oos: 14.4          # Out-of-sample Sharpe ratio
  wf_sharpe: 12.64          # Walk-forward Sharpe
  wf_positive_folds: "17/18" # Positive folds in walk-forward
  oos_return: 4.7%          # Total OOS return
  win_rate: 74.5%           # Win rate
  trades_per_month: 39      # Trade frequency
  validated_date: "2024-12-24"
  grid_optimized: true      # Whether params were grid-searched
```

When a strategy is retired, include:

```yaml
retirement:
  reason: "Negative OOS return (-21.3%), Sharpe -1.55"
  tested_date: "2024-12-24"
```

---

## Conventions

1. **Single Source of Truth**: Each piece of config should have exactly one authoritative location.
2. **Stage Reflects Reality**: Update `stage` immediately when validation state changes.
3. **Document Decisions**: Use comments to explain non-obvious parameter choices.
4. **Research is Read-Only**: Don't modify research files for deployment changes.
5. **Enable Explicitly**: Use `enabled: true/false` to control what actually runs.

---

## Migration Checklist

When a strategy graduates from research to production:

- [ ] Add definition to `strategies.yaml` with validated params
- [ ] Set `stage: validated` initially
- [ ] Add deployment config to `paper_trading.yaml`
- [ ] Set `enabled: true` to start trading
- [ ] Update `stage: deployed` in both files
- [ ] Add validation metadata
- [ ] Leave research files unchanged (they document history)

