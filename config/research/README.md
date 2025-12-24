# Research Configuration Files

These files document completed research phases. They are **read-only after validation is complete** and serve as historical reference for what was tested and why decisions were made.

> ⚠️ **Do not modify these files for deployment changes.** Production configuration lives in the parent `config/` directory.

---

## Files

### `ratio_universe.yaml`
**Research Phase:** Notes 11, 14, 16  
**Status:** Frozen (Dec 24, 2024)

Documents the systematic evaluation of ratio mean reversion pairs:
- All candidate pairs (BTC reference, ETH reference)
- Validation results (OOS Sharpe, return, win rate)
- Parameter optimization history
- Retirement reasons for failed pairs

**Key findings:**
- BTC is better reference asset than ETH
- Optimal params cluster around: `lookback=48, entry=-1.8, exit=-0.9, hold=24-48h`
- 9 pairs deployed, 7 pairs retired

### `creative_testing.yaml`
**Research Phase:** Notes 09, 12, 13, 14  
**Status:** Frozen (Dec 24, 2024)

Documents creative strategy exploration:
- Phase 1 quick validation results
- Phase 2 deep validation winners
- Orthogonal strategy testing (diversifiers)
- Hyper-creative experiments

**Key findings:**
- Ratio MR dominates standalone alpha
- Diversifiers add value: `liquidity_vacuum_detector`, `cross_sectional_momentum`
- Most creative strategies failed validation

---

## Relationship to Production Config

```
config/
├── strategies.yaml      ← SOURCE OF TRUTH for strategy definitions
├── paper_trading.yaml   ← SOURCE OF TRUTH for deployment config
│
└── research/            ← HISTORICAL REFERENCE (read-only)
    ├── ratio_universe.yaml
    └── creative_testing.yaml
```

When a strategy graduates from research to production:
1. Add strategy definition to `strategies.yaml` with `stage: validated`
2. Add deployment config to `paper_trading.yaml` with `enabled: true`
3. Update `stage` to `deployed` in both files
4. **Do not modify** the research files - they document the validation process

---

## Research Note References

| Note | Topic | File |
|------|-------|------|
| 09 | Unexplored Strategies | creative_testing.yaml |
| 11 | Grid Search Optimization | ratio_universe.yaml |
| 12 | Diversification Strategies | creative_testing.yaml |
| 13 | Hyper-Creative Orthogonal | creative_testing.yaml |
| 14 | Orthogonal Grid Optimization | both |
| 16 | Ratio Universe Expansion | ratio_universe.yaml |

