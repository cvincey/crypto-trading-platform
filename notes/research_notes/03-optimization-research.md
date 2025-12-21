# Research Note: ML Strategy Optimization & Walk-Forward Validation

**Date:** December 21, 2024  
**Author:** Trading Research Team  
**Version:** 1.0

---

## Executive Summary

We implemented a comprehensive optimization suite to validate the ML sibling strategies developed in the previous research. **Walk-forward validation revealed severe overfitting** - all strategies showed 100%+ Sharpe degradation when tested out-of-sample in time.

**Critical Finding:** The previously reported Sharpe ratios of 25+ were artifacts of look-ahead bias. Real out-of-sample performance is **negative**.

---

## 1. Optimization Suite Implemented

### 1.1 New Infrastructure

| Component | File | Description |
|-----------|------|-------------|
| Config | `config/optimization.yaml` | Central optimization settings |
| Walk-Forward Engine | `src/crypto/backtesting/walk_forward.py` | Proper OOS testing |
| Walk-Forward Script | `scripts/walk_forward_validation.py` | Config-driven WF testing |
| Hyperparam Search | `scripts/grid_search_hyperparams.py` | Parameter optimization |
| Feature Importance | `scripts/feature_importance.py` | Feature analysis + SHAP |
| OOS Testing | `scripts/out_of_sample_test.py` | Holdout symbol testing |

### 1.2 Config Updates

- Extended `schemas.py` with optimization dataclasses
- Updated `settings.py` to load `optimization.yaml`
- Updated `grid_search_risk.py` to use config
- Added ML siblings evaluation to `backtests.yaml`

---

## 2. Walk-Forward Validation Results

### 2.1 Configuration

| Parameter | Value |
|-----------|-------|
| Train Window | 720 bars (30 days) |
| Test Window | 168 bars (7 days) |
| Step Size | 168 bars |
| Total Folds | 21 |
| Data Period | 180 days |

### 2.2 Results: Complete Overfitting Detected

| Strategy | In-Sample Sharpe | Out-of-Sample Sharpe | Degradation |
|----------|------------------|----------------------|-------------|
| ml_ensemble_voting | 28.89 | **-4.15** | 114% |
| ml_classifier_v4 | 20.08 | **-6.49** | 133% |
| ml_classifier_hybrid | 29.45 | **-6.69** | 123% |
| ml_classifier_xgb | 29.18 | **-7.53** | 126% |

**Key Observations:**

1. **All strategies have negative OOS Sharpe** - they lose money when tested forward in time
2. **Degradation exceeds 100%** - performance doesn't just drop, it reverses
3. **In-sample performance was illusory** - the 25+ Sharpe was entirely due to look-ahead bias
4. **The strategies memorized the training period** rather than learning generalizable patterns

---

## 3. Out-of-Sample Symbol Testing

Testing on holdout symbols (not used in development) showed **different results**:

| Strategy | Holdout Symbol Sharpe | Dev Symbol Sharpe | Difference |
|----------|----------------------|-------------------|------------|
| ml_classifier_xgb | 30.29 | 30.26 | +0.03 |
| ml_classifier_hybrid | 28.58 | 29.17 | -0.59 |
| ml_ensemble_voting | 23.00 | 25.98 | -2.98 |

**Interpretation:**

- Strategies generalize well to **new symbols** (different coins)
- Strategies do NOT generalize to **new time periods** (future data)
- This means the models learned **cross-asset patterns** but overfit to **temporal patterns**

---

## 4. Feature Importance Analysis

Top features by model importance (GradientBoosting):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | volume_momentum | 0.1075 |
| 2 | adx | 0.1028 |
| 3 | ema_12 | 0.0953 |
| 4 | rsi_14 | 0.0881 |
| 5 | macd_hist | 0.0866 |

**Observations:**

- Volume-based features (volume_momentum) are most important
- Trend indicators (adx, ema_12) are second most important
- Feature importances are relatively balanced (no single dominant feature)

---

## 5. Root Cause Analysis

### 5.1 Why Did Overfitting Occur?

1. **Training on Test Data**
   - Original backtests trained on 80% of data, tested on 20%
   - But the test period was at the END of the data
   - Model "knew" future patterns during training

2. **Feature Leakage**
   - Some features may contain future information
   - Normalized features (price vs SMA) can leak information about future movements

3. **Market Regime Shift**
   - 2024 market conditions may be unique
   - Models learned patterns specific to 2024 that won't repeat

4. **Too Many Features**
   - 10-13 features for a binary classification problem
   - High dimensionality increases overfitting risk

### 5.2 Why Symbol Generalization Works but Time Generalization Fails

- **Symbols share market structure** - all crypto pairs move together
- **Time periods have unique patterns** - each period has specific events/trends
- The model learned "this is what crypto looks like in 2024" not "this predicts direction"

---

## 6. Recommendations

### 6.1 Immediate Actions

1. **Do NOT deploy any ML sibling strategies in production**
   - All strategies are severely overfit
   - Expected live performance is negative

2. **Use Walk-Forward as the Primary Validation**
   - Standard train/test split is insufficient
   - Only trust walk-forward OOS results

### 6.2 Strategy Improvements

1. **Reduce Feature Complexity**
   - Use only top 3-5 features (volume_momentum, adx, rsi_14)
   - Remove highly correlated features

2. **Increase Regularization**
   - Lower max_depth (try 3-4 instead of 6)
   - Increase min_samples_leaf
   - Add L1/L2 regularization

3. **Use Larger Train Windows**
   - Current: 720 bars (30 days)
   - Try: 2160 bars (90 days) or more

4. **Add More Robust Targets**
   - Instead of 1-bar direction, predict 10-bar direction
   - Use risk-adjusted return instead of raw direction

5. **Consider Simpler Models**
   - Linear models may generalize better
   - Rule-based strategies with ML confirmation

### 6.3 Alternative Approaches

1. **Ensemble of Simple Rules**
   - Combine technical indicators with voting
   - Less prone to overfitting

2. **Online Learning**
   - Retrain daily/weekly on recent data
   - Adapt to changing market conditions

3. **Anomaly Detection Instead of Direction Prediction**
   - Predict unusual conditions rather than direction
   - May have better generalization

---

## 7. Files Created/Modified

### 7.1 New Files

| File | Purpose |
|------|---------|
| `config/optimization.yaml` | Optimization settings |
| `src/crypto/backtesting/walk_forward.py` | Walk-forward engine |
| `scripts/walk_forward_validation.py` | WF validation script |
| `scripts/grid_search_hyperparams.py` | Hyperparam optimization |
| `scripts/feature_importance.py` | Feature analysis |
| `scripts/out_of_sample_test.py` | OOS testing |

### 7.2 Modified Files

| File | Changes |
|------|---------|
| `src/crypto/config/schemas.py` | Added optimization dataclasses |
| `src/crypto/config/settings.py` | Added optimization loader |
| `scripts/grid_search_risk.py` | Config-driven, ML siblings support |
| `config/backtests.yaml` | Added ML evaluation configs |

### 7.3 Output Files

| File | Contents |
|------|----------|
| `notes/optimization_results/walk_forward_results.json` | Full WF results |
| `notes/optimization_results/walk_forward_summary.json` | WF summary |
| `notes/optimization_results/feature_importance.json` | Feature rankings |
| `notes/optimization_results/out_of_sample.json` | OOS test results |

---

## 8. Conclusions

### 8.1 Key Takeaways

1. **Walk-forward validation is essential** - standard backtesting gives misleading results
2. **High Sharpe ratios should be suspicious** - 25+ Sharpe is almost certainly overfit
3. **Symbol generalization ≠ Time generalization** - testing on new coins is not sufficient
4. **ML for trading is hard** - the signal-to-noise ratio in financial markets is very low

### 8.2 The Path Forward

The ML classifier approach shows promise in generalizing across assets, but current implementations are too complex and overfit to temporal patterns. Future work should focus on:

1. Simpler models with fewer features
2. Longer prediction horizons (less noise)
3. Online learning to adapt to market changes
4. Ensemble approaches combining rule-based and ML signals

### 8.3 What the Original Results Actually Showed

The original backtests (Sharpe 25+) demonstrated:
- **The model learned the training period perfectly** (not useful)
- **Cross-asset patterns exist** (potentially useful)
- **Standard backtesting is unreliable for ML** (important lesson)

---

## Appendix A: Walk-Forward Validation Explained

```
|<-- Train Window -->|<-- Test -->|
|-------- 720 -------|--- 168 ----|

Step 1: Train on bars 0-720, test on 720-888
Step 2: Train on bars 168-888, test on 888-1056
...
Step N: Train on bars (N-1)*168 to (N-1)*168+720, test next 168 bars

Final OOS performance = average of all test periods
```

This ensures:
- Model never sees future data during training
- Results reflect realistic forward performance
- Overfitting is detected (high train, low test)

---

## Appendix B: Command Reference

```bash
# Run walk-forward validation
python scripts/walk_forward_validation.py

# Run hyperparameter grid search
python scripts/grid_search_hyperparams.py

# Run feature importance analysis
python scripts/feature_importance.py

# Run out-of-sample testing
python scripts/out_of_sample_test.py

# Run risk parameter grid search
python scripts/grid_search_risk.py
```

---

*End of Research Note*
