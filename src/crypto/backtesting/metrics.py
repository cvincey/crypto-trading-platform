"""Performance metrics for backtesting."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from crypto.backtesting.portfolio import Portfolio, PortfolioState


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a backtest."""

    # Basic returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # in periods
    avg_drawdown: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Other
    total_commission: float = 0.0
    exposure_time: float = 0.0  # % of time in market
    
    # Raw data for custom analysis
    equity_curve: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    drawdowns: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding raw data)."""
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "avg_drawdown": self.avg_drawdown,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_trade": self.avg_trade,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "total_commission": self.total_commission,
            "exposure_time": self.exposure_time,
        }


def calculate_metrics(
    portfolio: Portfolio,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,  # Trading days for daily data
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        portfolio: Portfolio with completed backtest
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        periods_per_year: Number of periods per year (252 for daily, 365*24 for hourly)
        
    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics()

    if not portfolio.equity_curve:
        return metrics

    # Extract equity curve
    equity = [float(state.total_value) for state in portfolio.equity_curve]
    metrics.equity_curve = equity

    if len(equity) < 2:
        return metrics

    equity_series = pd.Series(equity)
    initial_capital = float(portfolio.initial_capital)
    final_value = equity[-1]

    # ==========================================================================
    # Basic Returns
    # ==========================================================================

    metrics.total_return = final_value - initial_capital
    metrics.total_return_pct = (metrics.total_return / initial_capital) * 100

    # Calculate period returns
    returns = equity_series.pct_change().dropna()
    metrics.returns = returns.tolist()

    if len(returns) > 0:
        # Annualized return
        total_periods = len(equity)
        years = total_periods / periods_per_year
        if years > 0:
            cagr = (final_value / initial_capital) ** (1 / years) - 1
            metrics.annualized_return = cagr * 100

    # ==========================================================================
    # Risk Metrics
    # ==========================================================================

    if len(returns) > 1:
        metrics.volatility = returns.std() * 100
        metrics.annualized_volatility = metrics.volatility * np.sqrt(periods_per_year)

        # Sharpe Ratio
        mean_return = returns.mean()
        std_return = returns.std()
        risk_free_per_period = risk_free_rate / periods_per_year

        if std_return > 0:
            sharpe = (mean_return - risk_free_per_period) / std_return
            metrics.sharpe_ratio = sharpe * np.sqrt(periods_per_year)

        # Sortino Ratio (uses downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            if downside_std > 0:
                sortino = (mean_return - risk_free_per_period) / downside_std
                metrics.sortino_ratio = sortino * np.sqrt(periods_per_year)

    # ==========================================================================
    # Drawdown Analysis
    # ==========================================================================

    # Calculate drawdowns
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    metrics.drawdowns = drawdown.tolist()

    metrics.max_drawdown = abs(drawdown.min()) * 100
    metrics.avg_drawdown = abs(drawdown[drawdown < 0].mean()) * 100 if len(drawdown[drawdown < 0]) > 0 else 0

    # Drawdown duration
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        # Find consecutive drawdown periods
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)
        
        max_duration = 0
        current_duration = 0
        for i, is_dd in enumerate(in_drawdown):
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        metrics.max_drawdown_duration = max_duration

    # Calmar Ratio
    if metrics.max_drawdown > 0 and metrics.annualized_return != 0:
        metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

    # ==========================================================================
    # Trade Statistics
    # ==========================================================================

    trades = portfolio.trades
    metrics.total_trades = len(trades)
    metrics.total_commission = float(portfolio.get_total_commission())

    if len(trades) >= 2:
        # Calculate trade PnL
        trade_pnls = []
        buy_price = None
        buy_quantity = None

        for trade in trades:
            if trade.side.value == "BUY":
                buy_price = float(trade.price)
                buy_quantity = float(trade.quantity)
            elif trade.side.value == "SELL" and buy_price is not None:
                sell_price = float(trade.price)
                pnl = (sell_price - buy_price) * buy_quantity
                pnl -= float(trade.commission)  # Subtract commission
                trade_pnls.append(pnl)
                buy_price = None
                buy_quantity = None

        if trade_pnls:
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]

            metrics.winning_trades = len(wins)
            metrics.losing_trades = len(losses)
            
            if len(trade_pnls) > 0:
                metrics.win_rate = (len(wins) / len(trade_pnls)) * 100
                metrics.avg_trade = np.mean(trade_pnls)

            if wins:
                metrics.avg_win = np.mean(wins)
                metrics.largest_win = max(wins)

            if losses:
                metrics.avg_loss = np.mean(losses)
                metrics.largest_loss = min(losses)

            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses

    # ==========================================================================
    # Exposure Time
    # ==========================================================================

    # Calculate what percentage of time we had positions
    position_states = [
        1 if state.position_value > 0 else 0
        for state in portfolio.equity_curve
    ]
    if position_states:
        metrics.exposure_time = (sum(position_states) / len(position_states)) * 100

    return metrics


def compare_strategies(
    results: dict[str, PerformanceMetrics],
) -> pd.DataFrame:
    """
    Compare multiple strategy results.
    
    Args:
        results: Dict mapping strategy name to metrics
        
    Returns:
        DataFrame with comparison
    """
    rows = []
    for name, metrics in results.items():
        row = {"strategy": name, **metrics.to_dict()}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index("strategy", inplace=True)
    
    return df


def generate_report(metrics: PerformanceMetrics, strategy_name: str = "") -> str:
    """
    Generate a text report of performance metrics.
    
    Args:
        metrics: Performance metrics
        strategy_name: Optional strategy name
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"BACKTEST REPORT: {strategy_name}" if strategy_name else "BACKTEST REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("RETURNS")
    lines.append("-" * 40)
    lines.append(f"Total Return:        ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
    lines.append(f"Annualized Return:   {metrics.annualized_return:.2f}%")
    lines.append("")

    lines.append("RISK METRICS")
    lines.append("-" * 40)
    lines.append(f"Volatility (Ann.):   {metrics.annualized_volatility:.2f}%")
    lines.append(f"Sharpe Ratio:        {metrics.sharpe_ratio:.2f}")
    lines.append(f"Sortino Ratio:       {metrics.sortino_ratio:.2f}")
    lines.append(f"Calmar Ratio:        {metrics.calmar_ratio:.2f}")
    lines.append("")

    lines.append("DRAWDOWN")
    lines.append("-" * 40)
    lines.append(f"Max Drawdown:        {metrics.max_drawdown:.2f}%")
    lines.append(f"Max DD Duration:     {metrics.max_drawdown_duration} periods")
    lines.append(f"Avg Drawdown:        {metrics.avg_drawdown:.2f}%")
    lines.append("")

    lines.append("TRADE STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total Trades:        {metrics.total_trades}")
    lines.append(f"Winning Trades:      {metrics.winning_trades}")
    lines.append(f"Losing Trades:       {metrics.losing_trades}")
    lines.append(f"Win Rate:            {metrics.win_rate:.2f}%")
    lines.append(f"Profit Factor:       {metrics.profit_factor:.2f}")
    lines.append("")

    lines.append("TRADE METRICS")
    lines.append("-" * 40)
    lines.append(f"Avg Trade:           ${metrics.avg_trade:,.2f}")
    lines.append(f"Avg Win:             ${metrics.avg_win:,.2f}")
    lines.append(f"Avg Loss:            ${metrics.avg_loss:,.2f}")
    lines.append(f"Largest Win:         ${metrics.largest_win:,.2f}")
    lines.append(f"Largest Loss:        ${metrics.largest_loss:,.2f}")
    lines.append("")

    lines.append("OTHER")
    lines.append("-" * 40)
    lines.append(f"Total Commission:    ${metrics.total_commission:,.2f}")
    lines.append(f"Exposure Time:       {metrics.exposure_time:.2f}%")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
