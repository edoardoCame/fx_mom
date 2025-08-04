"""
Combined Momentum Strategy Module for Forex and Commodities

This module implements a top/ranking momentum strategy that combines forex and commodity data,
scaling commodities to have similar magnitude as forex for fair comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class CombinedMomentumStrategy:
    """
    Combined momentum strategy that ranks both forex and commodity assets
    based on their momentum signals and selects top/bottom performers.
    """
    
    def __init__(self, data_path: str = "data"):
        """
        Initialize the combined momentum strategy.
        
        Parameters:
        -----------
        data_path : str
            Path to the data directory containing forex and commodity data
        """
        self.data_path = Path(data_path)
        self.forex_data = None
        self.commodity_data = None
        self.combined_data = None
        self.scaled_data = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load forex and commodity data from parquet files.
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Forex and commodity dataframes
        """
        # Load forex data
        forex_path = self.data_path / "forex_extended_data.parquet"
        if not forex_path.exists():
            raise FileNotFoundError(f"Forex data not found at {forex_path}")
        
        self.forex_data = pd.read_parquet(forex_path)
        print(f"✅ Loaded forex data: {self.forex_data.shape}")
        
        # Load commodity data
        commodity_path = self.data_path / "commodities_extended_data.parquet"
        if not commodity_path.exists():
            raise FileNotFoundError(f"Commodity data not found at {commodity_path}")
            
        self.commodity_data = pd.read_parquet(commodity_path)
        print(f"✅ Loaded commodity data: {self.commodity_data.shape}")
        
        return self.forex_data, self.commodity_data
    
    def extract_close_prices(self) -> pd.DataFrame:
        """
        Extract close prices from both forex and commodity data.
        
        Returns:
        --------
        pd.DataFrame
            Combined close prices for all assets
        """
        if self.forex_data is None or self.commodity_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Extract forex close prices (assuming columns like EURUSD_Close)
        forex_close = {}
        for col in self.forex_data.columns:
            if col.endswith('_Close'):
                asset_name = col.replace('_Close', '')
                forex_close[asset_name] = self.forex_data[col]
        
        # Extract commodity close prices (assuming columns like GOLD_Close)
        commodity_close = {}
        for col in self.commodity_data.columns:
            if col.endswith('_Close'):
                asset_name = col.replace('_Close', '')
                commodity_close[asset_name] = self.commodity_data[col]
        
        # Combine into single DataFrame
        all_close = {**forex_close, **commodity_close}
        self.combined_data = pd.DataFrame(all_close)
        
        # Align dates and fill missing values
        self.combined_data = self.combined_data.dropna(how='all')
        self.combined_data = self.combined_data.fillna(method='ffill')
        
        print(f"✅ Combined close prices: {self.combined_data.shape}")
        print(f"📅 Date range: {self.combined_data.index.min().date()} to {self.combined_data.index.max().date()}")
        
        return self.combined_data
    
    def calculate_volatility_scaling(self, lookback_days: int = 252) -> pd.Series:
        """
        Calculate volatility-based scaling factors to normalize asset magnitudes.
        
        Parameters:
        -----------
        lookback_days : int
            Number of days to calculate rolling volatility
            
        Returns:
        --------
        pd.Series
            Scaling factors for each asset
        """
        if self.combined_data is None:
            raise ValueError("Combined data not available. Call extract_close_prices() first.")
        
        # Calculate daily returns
        returns = self.combined_data.pct_change().dropna()
        
        # Calculate rolling volatility (annualized)
        volatility = returns.rolling(window=lookback_days).std() * np.sqrt(252)
        
        # Use median volatility as target (typically around 0.10-0.15 for forex)
        target_volatility = volatility.median().median()
        
        # Calculate scaling factors
        current_vol = volatility.iloc[-1]  # Use most recent volatility
        scaling_factors = target_volatility / current_vol
        
        # Handle any infinite or NaN values
        scaling_factors = scaling_factors.fillna(1.0)
        scaling_factors = scaling_factors.replace([np.inf, -np.inf], 1.0)
        
        print(f"📊 Target volatility: {target_volatility:.4f}")
        print(f"📊 Scaling factors range: {scaling_factors.min():.4f} to {scaling_factors.max():.4f}")
        
        return scaling_factors
    
    def apply_scaling(self, scaling_factors: pd.Series) -> pd.DataFrame:
        """
        Apply scaling factors to normalize asset price movements.
        
        Parameters:
        -----------
        scaling_factors : pd.Series
            Scaling factors for each asset
            
        Returns:
        --------
        pd.DataFrame
            Scaled price data
        """
        if self.combined_data is None:
            raise ValueError("Combined data not available. Call extract_close_prices() first.")
        
        # Apply scaling to returns rather than prices to avoid look-ahead bias
        returns = self.combined_data.pct_change().dropna()
        scaled_returns = returns * scaling_factors
        
        # Reconstruct scaled prices starting from 100 for all assets
        self.scaled_data = pd.DataFrame(index=returns.index, columns=returns.columns)
        self.scaled_data.iloc[0] = 100  # Start all assets at 100
        
        for i in range(1, len(self.scaled_data)):
            self.scaled_data.iloc[i] = self.scaled_data.iloc[i-1] * (1 + scaled_returns.iloc[i])
        
        print(f"✅ Applied scaling to {len(scaling_factors)} assets")
        
        return self.scaled_data
    
    def calculate_momentum_scores(self, lookback_days: int = 21, 
                                min_periods: int = 15) -> pd.DataFrame:
        """
        Calculate momentum scores for all assets without look-ahead bias.
        
        Parameters:
        -----------
        lookback_days : int
            Number of days to look back for momentum calculation
        min_periods : int
            Minimum number of periods required for momentum calculation
            
        Returns:
        --------
        pd.DataFrame
            Momentum scores for each asset and date
        """
        if self.scaled_data is None:
            raise ValueError("Scaled data not available. Apply scaling first.")
        
        # Calculate momentum as percentage change over lookback period
        momentum_scores = self.scaled_data.pct_change(periods=lookback_days)
        
        # Only keep scores where we have enough data points
        momentum_scores = momentum_scores.dropna(how='all')
        
        print(f"✅ Calculated momentum scores with {lookback_days}-day lookback")
        print(f"📊 Momentum scores shape: {momentum_scores.shape}")
        
        return momentum_scores
    
    def generate_friday_signals(self, momentum_scores: pd.DataFrame, 
                              top_n: int = 5, bottom_n: int = 5) -> pd.DataFrame:
        """
        Generate trading signals every Friday based on momentum rankings.
        
        Parameters:
        -----------
        momentum_scores : pd.DataFrame
            Momentum scores for all assets
        top_n : int
            Number of top performers to go long
        bottom_n : int
            Number of bottom performers to go short
            
        Returns:
        --------
        pd.DataFrame
            Trading signals (+1 for long, -1 for short, 0 for neutral)
        """
        # Filter for Fridays only (weekday = 4)
        friday_scores = momentum_scores[momentum_scores.index.weekday == 4].copy()
        
        if len(friday_scores) == 0:
            print("⚠️ No Friday data found in momentum scores")
            return pd.DataFrame()
        
        signals = pd.DataFrame(0, index=friday_scores.index, columns=friday_scores.columns)
        
        for date in friday_scores.index:
            # Get momentum scores for this Friday
            scores = friday_scores.loc[date].dropna()
            
            if len(scores) < (top_n + bottom_n):
                print(f"⚠️ Not enough valid scores on {date.date()}: {len(scores)} available")
                continue
            
            # Rank assets by momentum (highest to lowest)
            ranked_assets = scores.sort_values(ascending=False)
            
            # Select top performers (long)
            top_assets = ranked_assets.head(top_n).index
            signals.loc[date, top_assets] = 1
            
            # Select bottom performers (short)
            bottom_assets = ranked_assets.tail(bottom_n).index
            signals.loc[date, bottom_assets] = -1
        
        print(f"✅ Generated signals for {len(signals)} Fridays")
        print(f"📊 Signal summary:")
        print(f"   Long positions: {(signals == 1).sum().sum()}")
        print(f"   Short positions: {(signals == -1).sum().sum()}")
        
        return signals
    
    def backtest_strategy(self, signals: pd.DataFrame, 
                         holding_period_days: int = 5) -> Dict:
        """
        Backtest the momentum strategy with given signals.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Trading signals from generate_friday_signals
        holding_period_days : int
            Number of days to hold positions (default 5 for Friday to Friday)
            
        Returns:
        --------
        Dict
            Backtest results including returns, performance metrics
        """
        if self.scaled_data is None:
            raise ValueError("Scaled data not available.")
        
        # Calculate daily returns
        returns = self.scaled_data.pct_change().dropna()
        
        # Initialize portfolio returns
        portfolio_returns = []
        positions = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        
        # Forward-fill signals to create position matrix
        for i, signal_date in enumerate(signals.index):
            # Find the next signal date or end of data
            if i < len(signals.index) - 1:
                next_signal_date = signals.index[i + 1]
                end_date = min(next_signal_date, 
                              signal_date + pd.Timedelta(days=holding_period_days))
            else:
                end_date = signal_date + pd.Timedelta(days=holding_period_days)
            
            # Apply positions from signal_date to end_date
            mask = (positions.index > signal_date) & (positions.index <= end_date)
            positions.loc[mask] = signals.loc[signal_date].values
        
        # Calculate portfolio returns
        daily_portfolio_returns = (positions.shift(1) * returns).sum(axis=1)
        daily_portfolio_returns = daily_portfolio_returns.dropna()
        
        # Calculate performance metrics
        total_return = (1 + daily_portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(daily_portfolio_returns)) - 1
        volatility = daily_portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + daily_portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            'daily_returns': daily_portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'positions': positions
        }
        
        print(f"\n📊 Backtest Results:")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Annualized Return: {annualized_return:.2%}")
        print(f"   Volatility: {volatility:.2%}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {max_drawdown:.2%}")
        
        return results
    
    def run_full_strategy(self, momentum_lookback: int = 21, 
                         top_n: int = 5, bottom_n: int = 5,
                         volatility_lookback: int = 252) -> Dict:
        """
        Run the complete combined momentum strategy pipeline.
        
        Parameters:
        -----------
        momentum_lookback : int
            Days to look back for momentum calculation
        top_n : int
            Number of top performers to go long
        bottom_n : int
            Number of bottom performers to go short
        volatility_lookback : int
            Days to look back for volatility scaling calculation
            
        Returns:
        --------
        Dict
            Complete strategy results
        """
        print("🚀 Running Combined Momentum Strategy...")
        
        # 1. Load data
        self.load_data()
        
        # 2. Extract close prices
        self.extract_close_prices()
        
        # 3. Calculate scaling factors
        scaling_factors = self.calculate_volatility_scaling(volatility_lookback)
        
        # 4. Apply scaling
        self.apply_scaling(scaling_factors)
        
        # 5. Calculate momentum scores
        momentum_scores = self.calculate_momentum_scores(momentum_lookback)
        
        # 6. Generate Friday signals
        signals = self.generate_friday_signals(momentum_scores, top_n, bottom_n)
        
        # 7. Backtest strategy
        results = self.backtest_strategy(signals)
        
        # Add additional info to results
        results.update({
            'signals': signals,
            'momentum_scores': momentum_scores,
            'scaling_factors': scaling_factors,
            'combined_data': self.combined_data,
            'scaled_data': self.scaled_data
        })
        
        print("✅ Combined Momentum Strategy completed!")
        
        return results
