"""
High-Performance Vectorized Forex Momentum Backtesting Engine

This module provides a comprehensive backtesting framework optimized for forex momentum strategies.
It combines pandas preprocessing with numba-compiled execution for maximum performance.

Key Features:
- Vectorized portfolio simulation for forex momentum strategies  
- Numba-optimized performance-critical calculations
- Comprehensive performance metrics and risk analysis
- Support for multiple lookback periods and strategy parameters
- Weekly rebalancing with configurable leverage
- Clean equity curve generation and analysis

Author: Quantitative Trading System
Date: 2025-08-02
"""

import pandas as pd
import numpy as np
import numba
from numba import jit, prange
import warnings
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress numba warnings
warnings.filterwarnings('ignore', category=numba.NumbaWarning)


class BacktestEngine:
    """
    High-performance vectorized backtesting engine for forex momentum strategies.
    
    This class provides a complete backtesting framework that separates pandas
    preprocessing from numba-optimized execution loops for maximum performance.
    """
    
    def __init__(self, 
                 price_data_path: str = None,
                 signals_data_path: str = None,
                 initial_capital: float = 100000.0,
                 leverage: float = 1.0,
                 rebalance_frequency: str = 'weekly'):
        """
        Initialize the backtesting engine.
        
        Parameters:
        -----------
        price_data_path : str
            Path to the forex price data parquet file
        signals_data_path : str  
            Path to the trading signals parquet file
        initial_capital : float
            Starting portfolio value in USD
        leverage : float
            Portfolio leverage (1.0 = no leverage)
        rebalance_frequency : str
            Rebalancing frequency ('daily', 'weekly', 'monthly')
        """
        self.price_data_path = price_data_path
        self.signals_data_path = signals_data_path
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.rebalance_frequency = rebalance_frequency
            
        # Data containers
        self.price_data = None
        self.signals_data = None
        self.returns_data = None
        self.currency_pairs = []
        
        # Results containers
        self.portfolio_returns = None
        self.portfolio_value = None
        self.positions = None
        self.trades = None
        self.performance_metrics = None
        
        logger.info(f"BacktestEngine initialized with {initial_capital:,.0f} starting capital")
        
    
    def load_data(self) -> None:
        """Load price and signals data from parquet files."""
        logger.info("Loading price and signals data...")
        
        # Load price data
        if self.price_data_path:
            self.price_data = pd.read_parquet(self.price_data_path)
            logger.info(f"Loaded price data: {self.price_data.shape}")
        
        # Load signals data
        if self.signals_data_path:
            self.signals_data = pd.read_parquet(self.signals_data_path)
            logger.info(f"Loaded signals data: {self.signals_data.shape}")
        
        # Extract currency pairs from signals data
        if self.signals_data is not None:
            self.currency_pairs = [col.replace('_weight', '') for col in self.signals_data.columns]
            logger.info(f"Detected {len(self.currency_pairs)} currency pairs")
        
        # Validate data alignment
        if self.price_data is not None and self.signals_data is not None:
            self._validate_data_alignment()
    
    def _validate_data_alignment(self) -> None:
        """Validate that price and signals data are properly aligned."""
        # Check date ranges
        price_start, price_end = self.price_data.index.min(), self.price_data.index.max()
        signals_start, signals_end = self.signals_data.index.min(), self.signals_data.index.max()
        
        logger.info(f"Price data range: {price_start} to {price_end}")
        logger.info(f"Signals data range: {signals_start} to {signals_end}")
        
        # Check for missing currency pairs in price data
        missing_pairs = []
        for pair in self.currency_pairs:
            close_col = f"{pair}_Close"
            if close_col not in self.price_data.columns:
                missing_pairs.append(pair)
        
        if missing_pairs:
            logger.warning(f"Missing price data for pairs: {missing_pairs}")
            # Remove missing pairs from analysis
            self.currency_pairs = [p for p in self.currency_pairs if p not in missing_pairs]
            self.signals_data = self.signals_data[[f"{p}_weight" for p in self.currency_pairs]]
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns for all currency pairs.
        
        Returns:
        --------
        pd.DataFrame
            Daily returns for each currency pair
        """
        logger.info("Calculating daily returns...")
        
        returns_dict = {}
        for pair in self.currency_pairs:
            close_col = f"{pair}_Close"
            if close_col in self.price_data.columns:
                prices = self.price_data[close_col]
                returns_dict[pair] = prices.pct_change()
        
        self.returns_data = pd.DataFrame(returns_dict, index=self.price_data.index)
        
        # Handle missing values (forward fill first, then drop remaining)
        self.returns_data = self.returns_data.ffill().dropna()
        
        logger.info(f"Calculated returns for {len(self.currency_pairs)} pairs")
        return self.returns_data
    
    def run_backtest(self, 
                     price_data: pd.DataFrame, 
                     signals_data: pd.DataFrame,
                     start_date: str = None, 
                     end_date: str = None,
                     verbose: bool = True) -> Dict:
        """
        Run the complete backtesting simulation.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Forex price data
        signals_data : pd.DataFrame
            Trading signals data
        start_date : str
            Start date for backtesting (YYYY-MM-DD format)
        end_date : str
            End date for backtesting (YYYY-MM-DD format)
        verbose : bool
            Whether to print progress information
            
        Returns:
        --------
        Dict
            Complete backtest results including metrics and data
        """
        start_time = time.time()
        
        if verbose:
            logger.info("Starting backtesting simulation...")
        
        # Store provided data
        self.price_data = price_data
        self.signals_data = signals_data
        
        # Extract currency pairs from signals data
        if self.signals_data is not None:
            self.currency_pairs = [col.replace('_weight', '') for col in self.signals_data.columns]
            logger.info(f"Detected {len(self.currency_pairs)} currency pairs")
        
        # Validate data alignment
        if self.price_data is not None and self.signals_data is not None:
            self._validate_data_alignment()
        
        # Calculate returns if not already done
        if self.returns_data is None:
            self.calculate_returns()
        
        # Filter data by date range if specified
        if start_date or end_date:
            self._filter_data_by_date(start_date, end_date)
        
        # Align data
        aligned_data = self._align_data()
        returns_matrix = aligned_data['returns']
        signals_matrix = aligned_data['signals']
        dates = aligned_data['dates']
        
        if verbose:
            logger.info(f"Running simulation on {len(dates)} trading days")
        
        # Run numba-optimized backtesting simulation
        results = self._run_simulation_numba(
            returns_matrix=returns_matrix,
            signals_matrix=signals_matrix,
            dates=dates,
            initial_capital=self.initial_capital,
            leverage=self.leverage,
            rebalance_frequency=self.rebalance_frequency
        )
        
        # Store results
        self.portfolio_returns = pd.Series(results['portfolio_returns'], index=dates)
        self.portfolio_value = pd.Series(results['portfolio_value'], index=dates)
        self.positions = pd.DataFrame(results['positions'], 
                                    index=dates, 
                                    columns=self.currency_pairs)
        
        # Calculate performance metrics
        self.performance_metrics = self.calculate_performance_metrics()
        
        # Package complete results
        backtest_results = {
            'portfolio_returns': self.portfolio_returns,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions,
            'performance_metrics': self.performance_metrics,
            'currency_pairs': self.currency_pairs,
            'simulation_time': time.time() - start_time
        }
        
        if verbose:
            logger.info(f"Backtesting completed in {backtest_results['simulation_time']:.2f} seconds")
            self._print_performance_summary()
        
        return backtest_results
    
    def _filter_data_by_date(self, start_date: str, end_date: str) -> None:
        """Filter data by specified date range."""
        if start_date:
            start_date = pd.to_datetime(start_date)
            self.returns_data = self.returns_data[self.returns_data.index >= start_date]
            self.signals_data = self.signals_data[self.signals_data.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            self.returns_data = self.returns_data[self.returns_data.index <= end_date]
            self.signals_data = self.signals_data[self.signals_data.index <= end_date]
    
    def _align_data(self) -> Dict[str, np.ndarray]:
        """Align returns and signals data for numba processing."""
        # Get common dates
        common_dates = self.returns_data.index.intersection(self.signals_data.index)
        
        # Align returns data
        returns_aligned = self.returns_data.loc[common_dates, self.currency_pairs]
        
        # Align signals data  
        signals_cols = [f"{pair}_weight" for pair in self.currency_pairs]
        signals_aligned = self.signals_data.loc[common_dates, signals_cols]
        
        # Convert to numpy arrays for numba
        returns_matrix = returns_aligned.values.astype(np.float64)
        signals_matrix = signals_aligned.values.astype(np.float64)
        
        # Handle NaN values (replace with 0)
        returns_matrix = np.nan_to_num(returns_matrix, nan=0.0)
        signals_matrix = np.nan_to_num(signals_matrix, nan=0.0)
        
        return {
            'returns': returns_matrix,
            'signals': signals_matrix,
            'dates': common_dates
        }
    
    def _run_simulation_numba(self, 
                             returns_matrix: np.ndarray,
                             signals_matrix: np.ndarray,
                             dates: pd.DatetimeIndex,
                             initial_capital: float,
                             leverage: float,
                             rebalance_frequency: str) -> Dict:
        """Run the main backtesting simulation using numba optimization."""
        
        # Determine rebalancing frequency
        rebalance_freq_days = self._get_rebalance_frequency_days(rebalance_frequency)
        
        # Run numba-optimized simulation
        portfolio_returns, portfolio_value, positions = simulate_portfolio_numba(
            returns_matrix,
            signals_matrix,
            initial_capital,
            leverage,
            rebalance_freq_days
        )
        
        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_value': portfolio_value,
            'positions': positions
        }
    
    def _get_rebalance_frequency_days(self, frequency: str) -> int:
        """Convert rebalancing frequency to number of days."""
        freq_map = {
            'daily': 1,
            'weekly': 5,  # 5 business days
            'monthly': 22  # ~22 business days per month
        }
        return freq_map.get(frequency.lower(), 5)
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if self.portfolio_returns is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        logger.info("Calculating performance metrics...")
        
        returns = self.portfolio_returns.dropna()
        
        # Basic return metrics
        total_return = (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Risk-adjusted returns
        sharpe_ratio = (annualized_return - 0.02) / annualized_vol  # Assuming 2% risk-free rate
        sortino_ratio = (annualized_return - 0.02) / (returns[returns < 0].std() * np.sqrt(252))
        
        # Drawdown analysis
        dd_results = self._calculate_drawdown_metrics(self.portfolio_value)
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf
        
        # Distribution metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        var_95 = np.percentile(returns, 5)
        
        # Trading metrics
        trades_summary = self._calculate_trading_metrics()
        
        # Monthly performance
        monthly_performance = self._calculate_monthly_performance()
        
        metrics = {
            # Return Metrics
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': annualized_return * 100,
            'Daily Volatility (%)': daily_vol * 100,
            'Annualized Volatility (%)': annualized_vol * 100,
            
            # Risk-Adjusted Metrics
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': annualized_return / dd_results['max_drawdown'] if dd_results['max_drawdown'] != 0 else np.inf,
            
            # Drawdown Metrics
            'Maximum Drawdown (%)': dd_results['max_drawdown'] * 100,
            'Average Drawdown (%)': dd_results['avg_drawdown'] * 100,
            'Drawdown Duration (Days)': dd_results['max_duration'],
            'Recovery Time (Days)': dd_results['recovery_time'],
            
            # Distribution Metrics
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Value at Risk 95% (%)': var_95 * 100,
            
            # Trading Metrics
            'Win Rate (%)': win_rate * 100,
            'Profit Factor': profit_factor,
            'Number of Trades': trades_summary['num_trades'],
            'Average Trade (%)': trades_summary['avg_trade'] * 100,
            
            # Portfolio Metrics
            'Starting Value': self.portfolio_value.iloc[0],
            'Ending Value': self.portfolio_value.iloc[-1],
            'Trading Days': len(returns),
            
            # Monthly Performance
            'Monthly Returns': monthly_performance
        }
        
        return metrics
    
    def _calculate_drawdown_metrics(self, portfolio_value: pd.Series) -> Dict:
        """Calculate detailed drawdown metrics."""
        # Calculate running maximum
        running_max = portfolio_value.cummax()
        
        # Calculate drawdown
        drawdown = (portfolio_value - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown (only negative periods)
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Drawdown duration analysis
        in_drawdown = drawdown < -0.001  # 0.1% threshold
        
        # Find drawdown periods
        dd_periods = []
        start_dd = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_dd is None:
                start_dd = i
            elif not is_dd and start_dd is not None:
                dd_periods.append(i - start_dd)
                start_dd = None
        
        # Handle case where we end in drawdown
        if start_dd is not None:
            dd_periods.append(len(in_drawdown) - start_dd)
        
        max_duration = max(dd_periods) if dd_periods else 0
        
        # Recovery time (time to new high after max drawdown)
        max_dd_idx = drawdown.idxmin()
        recovery_idx = portfolio_value[max_dd_idx:].idxmax()
        recovery_time = len(portfolio_value[max_dd_idx:recovery_idx]) if recovery_idx != max_dd_idx else 0
        
        return {
            'max_drawdown': abs(max_drawdown),
            'avg_drawdown': abs(avg_drawdown),
            'max_duration': max_duration,
            'recovery_time': recovery_time,
            'drawdown_series': drawdown
        }
    
    def _calculate_trading_metrics(self) -> Dict:
        """Calculate trading-specific metrics."""
        if self.positions is None:
            return {'num_trades': 0, 'avg_trade': 0}
        
        # Calculate position changes (new trades)
        position_changes = self.positions.diff().abs().sum(axis=1)
        num_trades = (position_changes > 0.01).sum()  # Threshold for meaningful position change
        
        # Average trade return (approximation)
        non_zero_returns = self.portfolio_returns[self.portfolio_returns != 0]
        avg_trade = non_zero_returns.mean() if len(non_zero_returns) > 0 else 0
        
        return {
            'num_trades': num_trades,
            'avg_trade': avg_trade
        }
    
    def _calculate_monthly_performance(self) -> Dict:
        """Calculate monthly performance breakdown."""
        if self.portfolio_returns is None:
            return {}
        
        # Resample to monthly frequency
        monthly_returns = (1 + self.portfolio_returns).resample('ME').prod() - 1
        
        # Create year-month breakdown
        monthly_dict = {}
        for date, ret in monthly_returns.items():
            year = date.year
            month = date.strftime('%b')
            if year not in monthly_dict:
                monthly_dict[year] = {}
            monthly_dict[year][month] = ret * 100
        
        return monthly_dict
    
    def _print_performance_summary(self) -> None:
        """Print a summary of key performance metrics."""
        if self.performance_metrics is None:
            return
        
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        
        # Key metrics
        key_metrics = [
            'Total Return (%)',
            'Annualized Return (%)',
            'Annualized Volatility (%)',
            'Sharpe Ratio',
            'Maximum Drawdown (%)',
            'Win Rate (%)',
            'Number of Trades'
        ]
        
        for metric in key_metrics:
            if metric in self.performance_metrics:
                value = self.performance_metrics[metric]
                if isinstance(value, float):
                    print(f"{metric:<25}: {value:>10.2f}")
                else:
                    print(f"{metric:<25}: {value:>10}")
        
        print("="*60)
    
    def export_results(self, output_dir: str = "backtest_results") -> None:
        """Export backtesting results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export portfolio data
        if self.portfolio_returns is not None:
            self.portfolio_returns.to_csv(output_path / f"portfolio_returns_{timestamp}.csv")
            self.portfolio_value.to_csv(output_path / f"portfolio_value_{timestamp}.csv")
        
        # Export positions
        if self.positions is not None:
            self.positions.to_csv(output_path / f"positions_{timestamp}.csv")
        
        # Export performance metrics
        if self.performance_metrics is not None:
            with open(output_path / f"performance_metrics_{timestamp}.json", 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                metrics_json = {}
                for k, v in self.performance_metrics.items():
                    if isinstance(v, (np.integer, np.floating)):
                        metrics_json[k] = float(v)
                    elif isinstance(v, dict):
                        metrics_json[k] = v
                    else:
                        metrics_json[k] = v
                json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Results exported to {output_path}")


@jit(nopython=True, parallel=False)
def simulate_portfolio_numba(returns_matrix: np.ndarray,
                           signals_matrix: np.ndarray,
                           initial_capital: float,
                           leverage: float,
                           rebalance_freq: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized portfolio simulation function.
    
    Parameters:
    -----------
    returns_matrix : np.ndarray
        Daily returns matrix (days x assets)
    signals_matrix : np.ndarray
        Trading signals matrix (days x assets)
    initial_capital : float
        Starting portfolio value
    leverage : float
        Portfolio leverage
    rebalance_freq : int
        Rebalancing frequency in days
        
    Returns:
    --------
    Tuple of portfolio returns, values, and positions
    """
    n_days, n_assets = returns_matrix.shape
    
    # Initialize output arrays
    portfolio_returns = np.zeros(n_days)
    portfolio_value = np.zeros(n_days)
    positions = np.zeros((n_days, n_assets))
    
    # Initialize portfolio
    current_value = initial_capital
    current_positions = np.zeros(n_assets)
    
    portfolio_value[0] = current_value
    
    for day in range(n_days):
        # Check if it's a rebalancing day
        is_rebalance_day = (day % rebalance_freq == 0) or (day == 0)
        
        if is_rebalance_day and day < n_days - 1:  # Don't rebalance on last day
            # Get target positions from signals
            target_positions = signals_matrix[day] * leverage
            
            # Update positions
            current_positions = target_positions.copy()
        
        # Store current positions
        positions[day] = current_positions.copy()
        
        # Calculate daily portfolio return if not first day
        if day > 0:
            # Calculate return from positions
            daily_return = np.sum(current_positions * returns_matrix[day])
            portfolio_returns[day] = daily_return
            
            # Update portfolio value
            current_value = current_value * (1 + daily_return)
        
        portfolio_value[day] = current_value
    
    return portfolio_returns, portfolio_value, positions


def create_visualizations(backtest_results: Dict, output_dir: str = "backtest_results") -> None:
    """
    Create comprehensive visualization plots for backtest results.
    
    Parameters:
    -----------
    backtest_results : Dict
        Complete backtest results from BacktestEngine
    output_dir : str
        Directory to save visualization plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    portfolio_value = backtest_results['portfolio_value']
    portfolio_returns = backtest_results['portfolio_returns']
    performance_metrics = backtest_results['performance_metrics']
    positions = backtest_results['positions']
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Equity curve with drawdowns
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Portfolio value
    ax1.plot(portfolio_value.index, portfolio_value.values, linewidth=2, color='blue')
    ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Drawdown
    drawdown_series = performance_metrics.get('Monthly Returns', {})
    if 'drawdown_series' in performance_metrics:
        dd_series = performance_metrics['drawdown_series']
        ax2.fill_between(dd_series.index, dd_series.values * 100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(dd_series.index, dd_series.values * 100, color='red', linewidth=1)
    
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'equity_curve_drawdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly returns heatmap
    monthly_returns = performance_metrics.get('Monthly Returns', {})
    if monthly_returns:
        # Convert to DataFrame for heatmap
        years = sorted(monthly_returns.keys())
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        heatmap_data = []
        for year in years:
            year_data = []
            for month in months:
                value = monthly_returns[year].get(month, np.nan)
                year_data.append(value)
            heatmap_data.append(year_data)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=years, columns=months)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'})
        plt.title('Monthly Returns Heatmap', fontsize=16, fontweight='bold')
        plt.ylabel('Year', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'monthly_returns_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Return distribution
    returns_clean = portfolio_returns.dropna()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(returns_clean * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(returns_clean.mean() * 100, color='red', linestyle='--', 
               label=f'Mean: {returns_clean.mean()*100:.2f}%')
    ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Daily Return (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(returns_clean, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'returns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Rolling performance metrics
    window = 252  # 1 year
    if len(portfolio_returns) > window:
        rolling_sharpe = (portfolio_returns.rolling(window).mean() * 252) / \
                        (portfolio_returns.rolling(window).std() * np.sqrt(252))
        
        rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Rolling Sharpe ratio
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='green')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Rolling Sharpe Ratio (1-Year Window)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Sharpe Ratio', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax2.plot(rolling_vol.index, rolling_vol.values * 100, linewidth=2, color='orange')
        ax2.set_title('Rolling Volatility (1-Year Window)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Annualized Volatility (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'rolling_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Position allocation over time
    if positions is not None and len(positions.columns) <= 10:  # Only if reasonable number of assets
        plt.figure(figsize=(15, 8))
        
        # Create stacked area plot for positions
        positive_positions = positions.where(positions >= 0, 0)
        negative_positions = positions.where(positions < 0, 0)
        
        # Plot positive positions
        ax = positive_positions.plot.area(stacked=True, alpha=0.7, figsize=(15, 8))
        
        # Plot negative positions
        negative_positions.plot.area(stacked=True, alpha=0.7, ax=ax)
        
        plt.title('Position Allocation Over Time', fontsize=16, fontweight='bold')
        plt.ylabel('Position Weight', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'position_allocation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualizations saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    engine = BacktestEngine(
        price_data_path="data/processed/forex_synchronized_data.parquet",
        signals_data_path="signals_output/momentum_signals_30d.parquet",
        initial_capital=100000.0,
        leverage=1.0,
        rebalance_frequency='weekly'
    )
    
    # Run backtest
    results = engine.run_backtest(verbose=True)
    
    # Create visualizations
    create_visualizations(results)
    
    # Export results
    engine.export_results()