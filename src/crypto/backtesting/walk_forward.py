"""Walk-forward validation engine for realistic backtesting."""

import logging
import operator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

import numpy as np
import pandas as pd

from crypto.backtesting.engine import BacktestEngine, BacktestResult
from crypto.backtesting.metrics import PerformanceMetrics, calculate_metrics
from crypto.backtesting.portfolio import Portfolio
from crypto.config.schemas import WalkForwardConfig, ValidationGateCriterion
from crypto.core.types import Signal
from crypto.strategies.base import Strategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Acceptance Gate Logic
# =============================================================================


OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "gt": operator.gt,
    "lt": operator.lt,
    "eq": operator.eq,
    "gte": operator.ge,
    "lte": operator.le,
}


@dataclass
class AcceptanceGate:
    """Single acceptance gate criterion."""
    
    name: str
    metric: str
    operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    threshold: float
    
    def check(self, value: float) -> bool:
        """Check if the value passes this gate."""
        op_func = OPERATORS.get(self.operator)
        if op_func is None:
            raise ValueError(f"Unknown operator: {self.operator}")
        return op_func(value, self.threshold)
    
    @classmethod
    def from_config(cls, config: ValidationGateCriterion) -> "AcceptanceGate":
        """Create from config object."""
        return cls(
            name=config.name,
            metric=config.metric,
            operator=config.operator,
            threshold=config.threshold,
        )


@dataclass
class AcceptanceResult:
    """Result of checking acceptance gates."""
    
    passed: bool
    gates_passed: list[str] = field(default_factory=list)
    gates_failed: list[str] = field(default_factory=list)
    details: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "details": self.details,
        }


def check_acceptance_gates(
    result: "WalkForwardResult",
    gates: list[AcceptanceGate],
    reject_all_negative_folds: bool = True,
) -> AcceptanceResult:
    """
    Check if a walk-forward result passes all acceptance gates.
    
    Args:
        result: WalkForwardResult to check
        gates: List of AcceptanceGate criteria
        reject_all_negative_folds: If True, reject if ALL folds have negative OOS Sharpe
        
    Returns:
        AcceptanceResult with pass/fail status and details
    """
    gates_passed = []
    gates_failed = []
    details = {}
    
    # Get metrics from result
    metrics = {
        "oos_sharpe": result.oos_sharpe,
        "oos_return_pct": result.oos_return_pct,
        "oos_win_rate": result.oos_win_rate,
        "oos_total_trades": result.oos_total_trades,
        "oos_max_dd": result.oos_max_dd,
        "is_sharpe": result.is_sharpe,
        "sharpe_degradation": result.sharpe_degradation,
        "return_degradation": result.return_degradation,
    }
    
    for gate in gates:
        value = metrics.get(gate.metric)
        if value is None:
            logger.warning(f"Metric '{gate.metric}' not found in result")
            gates_failed.append(gate.name)
            details[gate.name] = {
                "metric": gate.metric,
                "value": None,
                "threshold": gate.threshold,
                "operator": gate.operator,
                "passed": False,
                "reason": "metric not found",
            }
            continue
        
        passed = gate.check(value)
        if passed:
            gates_passed.append(gate.name)
        else:
            gates_failed.append(gate.name)
        
        details[gate.name] = {
            "metric": gate.metric,
            "value": value,
            "threshold": gate.threshold,
            "operator": gate.operator,
            "passed": passed,
        }
    
    # Check fail-safe: all folds negative
    all_folds_negative = all(f.test_sharpe < 0 for f in result.folds) if result.folds else True
    if reject_all_negative_folds and all_folds_negative:
        gates_failed.append("all_folds_positive")
        details["all_folds_positive"] = {
            "metric": "fold_sharpe",
            "value": "all negative",
            "threshold": "> 0 for at least one fold",
            "passed": False,
        }
    
    overall_passed = len(gates_failed) == 0
    
    return AcceptanceResult(
        passed=overall_passed,
        gates_passed=gates_passed,
        gates_failed=gates_failed,
        details=details,
    )


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""
    
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    
    # In-sample (train) metrics
    train_return_pct: float = 0.0
    train_sharpe: float = 0.0
    train_win_rate: float = 0.0
    
    # Out-of-sample (test) metrics
    test_return_pct: float = 0.0
    test_sharpe: float = 0.0
    test_win_rate: float = 0.0
    test_trades: int = 0
    test_max_dd: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    
    strategy_name: str
    symbol: str
    interval: str
    
    # Configuration
    train_window: int
    test_window: int
    step_size: int
    total_folds: int
    
    # Individual fold results
    folds: list[WalkForwardFold] = field(default_factory=list)
    
    # Aggregated out-of-sample metrics
    oos_return_pct: float = 0.0
    oos_sharpe: float = 0.0
    oos_win_rate: float = 0.0
    oos_total_trades: int = 0
    oos_max_dd: float = 0.0
    
    # In-sample vs out-of-sample comparison
    is_return_pct: float = 0.0
    is_sharpe: float = 0.0
    
    # Overfitting indicator (IS vs OOS difference)
    sharpe_degradation: float = 0.0  # How much Sharpe drops OOS
    return_degradation: float = 0.0  # How much return drops OOS
    
    # Acceptance gate result (populated after checking)
    acceptance_result: AcceptanceResult | None = None
    
    def check_acceptance(
        self,
        gates: list[AcceptanceGate],
        reject_all_negative_folds: bool = True,
    ) -> AcceptanceResult:
        """Check if this result passes acceptance gates."""
        self.acceptance_result = check_acceptance_gates(
            self, gates, reject_all_negative_folds
        )
        return self.acceptance_result
    
    @property
    def passed_validation(self) -> bool:
        """Return True if passed acceptance gates (or not yet checked)."""
        if self.acceptance_result is None:
            return True  # Not checked yet
        return self.acceptance_result.passed
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "interval": self.interval,
            "train_window": self.train_window,
            "test_window": self.test_window,
            "step_size": self.step_size,
            "total_folds": self.total_folds,
            "oos_return_pct": self.oos_return_pct,
            "oos_sharpe": self.oos_sharpe,
            "oos_win_rate": self.oos_win_rate,
            "oos_total_trades": self.oos_total_trades,
            "oos_max_dd": self.oos_max_dd,
            "is_return_pct": self.is_return_pct,
            "is_sharpe": self.is_sharpe,
            "sharpe_degradation": self.sharpe_degradation,
            "return_degradation": self.return_degradation,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train_start": f.train_start.isoformat(),
                    "train_end": f.train_end.isoformat(),
                    "test_start": f.test_start.isoformat(),
                    "test_end": f.test_end.isoformat(),
                    "train_samples": f.train_samples,
                    "test_samples": f.test_samples,
                    "train_return_pct": f.train_return_pct,
                    "train_sharpe": f.train_sharpe,
                    "test_return_pct": f.test_return_pct,
                    "test_sharpe": f.test_sharpe,
                    "test_trades": f.test_trades,
                }
                for f in self.folds
            ],
        }
        
        if self.acceptance_result is not None:
            result["acceptance"] = self.acceptance_result.to_dict()
        
        return result


class WalkForwardEngine:
    """
    Walk-forward validation engine for proper out-of-sample testing.
    
    Walk-forward validation:
    1. Split data into rolling train/test windows
    2. Train on window, generate signals on test period only
    3. Roll forward and repeat
    4. Aggregate all out-of-sample periods for realistic performance
    
    This addresses look-ahead bias in standard backtesting.
    """

    def __init__(
        self,
        train_window: int = 720,
        test_window: int = 168,
        step_size: int = 168,
        min_train_samples: int = 500,
        initial_capital: Decimal = Decimal("10000"),
        commission: Decimal = Decimal("0.001"),
        slippage: Decimal = Decimal("0.0005"),
    ):
        """
        Initialize walk-forward engine.
        
        Args:
            train_window: Number of bars for training
            test_window: Number of bars for testing
            step_size: How many bars to roll forward each fold
            min_train_samples: Minimum training samples required
            initial_capital: Starting capital per fold
            commission: Commission rate
            slippage: Slippage rate
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_samples = min_train_samples
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    @classmethod
    def from_config(cls, config: WalkForwardConfig) -> "WalkForwardEngine":
        """Create engine from configuration."""
        return cls(
            train_window=config.train_window,
            test_window=config.test_window,
            step_size=config.step_size,
            min_train_samples=config.min_train_samples,
        )

    def run(
        self,
        strategy_name: str,
        candles: pd.DataFrame,
        symbol: str = "UNKNOWN",
        interval: str = "1h",
        strategy_params: dict[str, Any] | None = None,
        reference_data: dict[str, pd.DataFrame] | None = None,
        funding_data: dict[str, pd.Series] | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation on a strategy.
        
        Args:
            strategy_name: Name of the registered strategy
            candles: Full OHLCV DataFrame
            symbol: Trading pair symbol
            interval: Candle interval
            strategy_params: Optional parameter overrides
            reference_data: Dict of reference symbol candles for cross-symbol strategies
            funding_data: Dict of funding rate series for alternative data strategies
            
        Returns:
            WalkForwardResult with aggregated metrics
        """
        self._reference_data = reference_data or {}
        self._funding_data = funding_data or {}
        self._current_symbol = symbol
        if len(candles) < self.train_window + self.test_window:
            raise ValueError(
                f"Insufficient data: {len(candles)} bars, "
                f"need at least {self.train_window + self.test_window}"
            )

        # Calculate number of folds
        data_len = len(candles)
        available_for_folds = data_len - self.train_window
        num_folds = max(1, available_for_folds // self.step_size)

        logger.info(
            f"Walk-forward: {strategy_name} on {symbol}, "
            f"{num_folds} folds, train={self.train_window}, test={self.test_window}"
        )

        folds: list[WalkForwardFold] = []
        all_oos_returns: list[float] = []
        all_oos_trades: list[int] = []
        
        for fold_id in range(num_folds):
            train_start_idx = fold_id * self.step_size
            train_end_idx = train_start_idx + self.train_window
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + self.test_window, data_len)

            if test_end_idx <= test_start_idx:
                break

            # Split data
            train_candles = candles.iloc[train_start_idx:train_end_idx].copy()
            test_candles = candles.iloc[test_start_idx:test_end_idx].copy()

            if len(train_candles) < self.min_train_samples:
                logger.warning(f"Fold {fold_id}: insufficient training data")
                continue

            # Create fold result
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_candles.index[0],
                train_end=train_candles.index[-1],
                test_start=test_candles.index[0],
                test_end=test_candles.index[-1],
                train_samples=len(train_candles),
                test_samples=len(test_candles),
            )

            try:
                # Create fresh strategy for this fold
                strategy = strategy_registry.create(strategy_name, **(strategy_params or {}))
                
                # Set up reference data for cross-symbol strategies
                self._setup_reference_data(strategy)

                # Train on train data (strategy trains internally)
                # Run backtest on train data to get in-sample metrics
                train_engine = BacktestEngine(
                    initial_capital=self.initial_capital,
                    commission=self.commission,
                    slippage=self.slippage,
                )
                train_result = train_engine.run(strategy, train_candles, symbol, interval)
                
                fold.train_return_pct = train_result.metrics.total_return_pct
                fold.train_sharpe = train_result.metrics.sharpe_ratio
                fold.train_win_rate = train_result.metrics.win_rate

                # Create NEW fresh strategy for test (to simulate real forward test)
                # The strategy will train on test_candles when generate_signals is called
                # But we need to pass the training data to establish the model state
                
                # For proper walk-forward, we need to:
                # 1. Train on train_candles
                # 2. Apply to test_candles WITHOUT retraining
                
                # Combine train+test data and mark where test starts
                combined = pd.concat([train_candles, test_candles])
                
                # Get signals for combined data (strategy trains on first portion)
                test_strategy = strategy_registry.create(strategy_name, **(strategy_params or {}))
                
                # Set up reference data for cross-symbol strategies
                self._setup_reference_data(test_strategy)
                
                all_signals = test_strategy.generate_signals(combined)
                
                # Extract only test period signals
                test_signals = all_signals.loc[test_candles.index]
                
                # Run backtest on test data with pre-generated signals
                test_result = self._run_with_signals(
                    test_candles, test_signals, symbol, interval
                )
                
                fold.test_return_pct = test_result.metrics.total_return_pct
                fold.test_sharpe = test_result.metrics.sharpe_ratio
                fold.test_win_rate = test_result.metrics.win_rate
                fold.test_trades = test_result.metrics.total_trades
                fold.test_max_dd = test_result.metrics.max_drawdown

                folds.append(fold)
                all_oos_returns.append(fold.test_return_pct)
                all_oos_trades.append(fold.test_trades)

            except Exception as e:
                logger.error(f"Fold {fold_id} failed: {e}")
                continue

        if not folds:
            raise ValueError("No successful folds")

        # Aggregate results
        result = WalkForwardResult(
            strategy_name=strategy_name,
            symbol=symbol,
            interval=interval,
            train_window=self.train_window,
            test_window=self.test_window,
            step_size=self.step_size,
            total_folds=len(folds),
            folds=folds,
        )

        # Calculate aggregated OOS metrics
        result.oos_return_pct = sum(f.test_return_pct for f in folds) / len(folds)
        result.oos_sharpe = sum(f.test_sharpe for f in folds) / len(folds)
        result.oos_win_rate = sum(f.test_win_rate for f in folds) / len(folds)
        result.oos_total_trades = sum(f.test_trades for f in folds)
        result.oos_max_dd = max(f.test_max_dd for f in folds)

        # Calculate aggregated IS metrics
        result.is_return_pct = sum(f.train_return_pct for f in folds) / len(folds)
        result.is_sharpe = sum(f.train_sharpe for f in folds) / len(folds)

        # Calculate degradation (overfitting indicator)
        if result.is_sharpe > 0:
            result.sharpe_degradation = (
                (result.is_sharpe - result.oos_sharpe) / result.is_sharpe * 100
            )
        if result.is_return_pct > 0:
            result.return_degradation = (
                (result.is_return_pct - result.oos_return_pct) / result.is_return_pct * 100
            )

        logger.info(
            f"Walk-forward complete: IS Sharpe={result.is_sharpe:.2f}, "
            f"OOS Sharpe={result.oos_sharpe:.2f}, "
            f"Degradation={result.sharpe_degradation:.1f}%"
        )

        return result

    def _setup_reference_data(self, strategy: Strategy) -> None:
        """
        Set up reference data for cross-symbol and alternative data strategies.
        
        Args:
            strategy: Strategy instance to set up
        """
        # Check if strategy is a cross-symbol strategy
        try:
            from crypto.strategies.cross_symbol_base import CrossSymbolBaseStrategy
            
            if isinstance(strategy, CrossSymbolBaseStrategy) and self._reference_data:
                for symbol, candles in self._reference_data.items():
                    strategy.set_reference_data(symbol, candles)
        except ImportError:
            pass  # Cross-symbol base not available
        
        # Check if strategy is an alternative data strategy
        try:
            from crypto.strategies.alternative_data_strategies import AlternativeDataBaseStrategy
            
            if isinstance(strategy, AlternativeDataBaseStrategy) and self._funding_data:
                # Get funding data for the current symbol being tested
                if hasattr(self, '_current_symbol') and self._current_symbol in self._funding_data:
                    strategy.set_funding_data(self._funding_data[self._current_symbol])
        except ImportError:
            pass  # Alternative data strategies not available

    def _run_with_signals(
        self,
        candles: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
        interval: str,
    ) -> BacktestResult:
        """Run backtest with pre-computed signals."""
        from crypto.config.settings import get_settings
        
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_rate=self.commission,
            slippage_rate=self.slippage,
        )

        for i, (timestamp, row) in enumerate(candles.iterrows()):
            price = Decimal(str(row["close"]))
            signal = signals.iloc[i] if i < len(signals) else Signal.HOLD

            portfolio.execute_signal(
                signal=signal,
                symbol=symbol,
                price=price,
                timestamp=timestamp,
            )
            portfolio.update_equity(
                timestamp=timestamp,
                prices={symbol: price},
            )

        # Get periods per year for Sharpe calculation
        periods_per_year = BacktestEngine._get_periods_per_year(interval)
        metrics = calculate_metrics(portfolio, periods_per_year=periods_per_year)

        return BacktestResult(
            backtest_id="wf",
            strategy_name="walk_forward",
            symbol=symbol,
            interval=interval,
            start_date=candles.index[0],
            end_date=candles.index[-1],
            initial_capital=self.initial_capital,
            commission=self.commission,
            metrics=metrics,
            portfolio=portfolio,
            signals=signals,
        )


def compare_walk_forward_results(
    results: list[WalkForwardResult],
) -> pd.DataFrame:
    """
    Compare multiple walk-forward results.
    
    Args:
        results: List of WalkForwardResult objects
        
    Returns:
        DataFrame with comparison
    """
    rows = []
    for r in results:
        rows.append({
            "strategy": r.strategy_name,
            "symbol": r.symbol,
            "folds": r.total_folds,
            "oos_return_pct": r.oos_return_pct,
            "oos_sharpe": r.oos_sharpe,
            "oos_win_rate": r.oos_win_rate,
            "oos_trades": r.oos_total_trades,
            "oos_max_dd": r.oos_max_dd,
            "is_sharpe": r.is_sharpe,
            "sharpe_degradation": r.sharpe_degradation,
            "return_degradation": r.return_degradation,
        })

    df = pd.DataFrame(rows)
    df.sort_values("oos_sharpe", ascending=False, inplace=True)
    return df
