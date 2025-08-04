"""
Signal Loading Module for Forex Momentum Trading System

This module provides functions to load and validate trading signals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

def load_signals(signals_path: str, validate: bool = True) -> pd.DataFrame:
    """
    Load trading signals from parquet file.
    
    Parameters:
    -----------
    signals_path : str
        Path to the signals parquet file
    validate : bool
        Whether to perform signal validation
        
    Returns:
    --------
    pd.DataFrame
        Trading signals with columns for each currency pair
    """
    # Load signals
    signals = pd.read_parquet(signals_path)
    
    if validate:
        # Basic validation
        if signals.empty:
            raise ValueError("Signals file is empty")
            
        if signals.index.duplicated().any():
            raise ValueError("Duplicate dates found in signals")
            
        # Check signal values are valid (-0.2, 0.0, 0.2)
        valid_values = {-0.2, 0.0, 0.2}
        unique_values = set(signals.values.flatten())
        invalid_values = unique_values - valid_values
        
        if invalid_values:
            print(f"Warning: Invalid signal values found: {invalid_values}")
    
    print(f"✓ Loaded trading signals: {signals.shape[0]} days, {signals.shape[1]} pairs")
    print(f"✓ Date range: {signals.index.min().date()} to {signals.index.max().date()}")
    
    return signals

def get_available_lookback_periods(signals_dir: str = "signals") -> List[int]:
    """
    Get available lookback periods from signals directory.
    
    Parameters:
    -----------
    signals_dir : str
        Directory containing signal files
        
    Returns:
    --------
    List[int]
        Available lookback periods in days
    """
    signals_path = Path(signals_dir)
    lookback_periods = []
    
    for file in signals_path.glob("momentum_signals_*d.parquet"):
        # Extract lookback period from filename
        filename = file.stem
        lookback_str = filename.replace("momentum_signals_", "").replace("d", "")
        try:
            lookback = int(lookback_str)
            lookback_periods.append(lookback)
        except ValueError:
            continue
    
    return sorted(lookback_periods)

def generate_momentum_signals(price_data: pd.DataFrame, lookback_days: int = 30, 
                             top_n: int = 5, rebalance_freq: str = 'weekly') -> pd.DataFrame:
    """
    Generate momentum trading signals dynamically.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data with columns for each currency pair (OHLC format)
    lookback_days : int
        Number of days to look back for momentum calculation
    top_n : int
        Number of top/bottom momentum pairs to select
    rebalance_freq : str
        Rebalancing frequency ('weekly', 'daily')
        
    Returns:
    --------
    pd.DataFrame
        Trading signals with same index as price_data
    """
    # Get currency pairs - extract only Close prices
    all_columns = price_data.columns
    close_columns = [col for col in all_columns if col.endswith('_Close')]
    
    # Extract pair names (remove _Close suffix)
    currency_pairs = [col.replace('_Close', '') for col in close_columns]
    
    # Create column names with _weight suffix for compatibility with BacktestEngine
    signal_columns = [f"{pair}_weight" for pair in currency_pairs]
    
    # Initialize signals dataframe with _weight column names
    signals = pd.DataFrame(0.0, index=price_data.index, columns=signal_columns)
    
    # Calculate momentum using only Close prices
    close_prices = price_data[close_columns]
    close_prices.columns = currency_pairs  # Rename columns to pair names
    
    # Calculate momentum (simple return over lookback period)
    momentum_data = close_prices.pct_change(periods=lookback_days).fillna(0)
    
    # Generate signals based on rebalancing frequency
    if rebalance_freq == 'weekly':
        # Rebalance every Friday (weekday 4)
        rebalance_dates = price_data.index[price_data.index.weekday == 4]
    else:  # daily
        rebalance_dates = price_data.index
    
    for date in rebalance_dates:
        if date in momentum_data.index:
            # Get momentum values for this date
            momentum_values = momentum_data.loc[date]
            
            # Skip if we don't have enough data (all NaN or mostly NaN)
            if momentum_values.isna().sum() > len(momentum_values) * 0.5:
                continue
                
            # Remove NaN values before sorting
            momentum_values = momentum_values.dropna()
            
            # Skip if we don't have enough valid pairs
            if len(momentum_values) < top_n * 2:
                continue
            
            # Sort by momentum
            sorted_momentum = momentum_values.sort_values(ascending=False)
            
            # Select top N for long positions (20% each)
            top_long = sorted_momentum.head(top_n).index
            
            # Select bottom N for short positions (20% each)  
            top_short = sorted_momentum.tail(top_n).index
            
            # Set signals for this rebalancing period
            # Find next rebalancing date or end of data
            current_idx = price_data.index.get_loc(date)
            if rebalance_freq == 'weekly':
                # Find next Friday or end of data
                next_rebalance_dates = rebalance_dates[rebalance_dates > date]
                if len(next_rebalance_dates) > 0:
                    next_date = next_rebalance_dates[0]
                    next_idx = price_data.index.get_loc(next_date)
                else:
                    next_idx = len(price_data)
            else:
                next_idx = current_idx + 1
            
            # Apply signals for the period STARTING FROM NEXT TRADING DAY (avoid lookahead bias)
            # Signals calculated on Friday should be applied starting from Monday
            # For weekly rebalancing, ensure at least 1 trading day gap
            if rebalance_freq == 'weekly':
                # For weekly: signals calculated on Friday (weekday 4) should apply from next Monday
                # Find next Monday or first trading day after weekend
                start_application_idx = current_idx + 1
                while (start_application_idx < len(price_data) and 
                       price_data.index[start_application_idx].weekday() > 4):  # Skip weekends
                    start_application_idx += 1
                
                # If Friday signal, make sure to skip to at least Monday
                if price_data.index[current_idx].weekday() == 4:  # Friday
                    # Find next Monday (weekday 0)
                    while (start_application_idx < len(price_data) and 
                           price_data.index[start_application_idx].weekday() != 0):
                        start_application_idx += 1
                
                period_range = price_data.index[start_application_idx:next_idx]
            else:
                # For daily rebalancing, still apply from next day
                period_range = price_data.index[current_idx+1:next_idx]
            
            for period_date in period_range:
                if period_date in signals.index:
                    # Convert pair names to _weight column names
                    top_long_cols = [f"{pair}_weight" for pair in top_long]
                    top_short_cols = [f"{pair}_weight" for pair in top_short]
                    
                    # Long positions (20% each = 0.2)
                    signals.loc[period_date, top_long_cols] = 0.2
                    # Short positions (20% each = -0.2)
                    signals.loc[period_date, top_short_cols] = -0.2
    
    print(f"✓ Generated momentum signals: {lookback_days}d lookback, {top_n} long + {top_n} short")
    print(f"✓ Rebalancing: {rebalance_freq}")
    print(f"✓ Signal range: {signals.index.min().date()} to {signals.index.max().date()}")
    
    return signals

def generate_contrarian_risk_parity_signals(price_data: pd.DataFrame, lookback_days: int = 30,
                                          volatility_window: int = 30, top_n: int = 5, 
                                          rebalance_freq: str = 'weekly') -> pd.DataFrame:
    """
    Generate contrarian risk parity trading signals.
    
    Contrarian strategy: Long the biggest losers, short the biggest winners.
    Risk parity: Weight inversely proportional to volatility within each side.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data with columns for each currency pair (OHLC format)
    lookback_days : int
        Number of days to look back for momentum calculation
    volatility_window : int
        Number of days for rolling volatility calculation
    top_n : int
        Number of top/bottom momentum pairs to select for each side
    rebalance_freq : str
        Rebalancing frequency ('weekly', 'daily')
        
    Returns:
    --------
    pd.DataFrame
        Trading signals with risk parity weights (negative for shorts)
    """
    # Get currency pairs - extract only Close prices
    all_columns = price_data.columns
    close_columns = [col for col in all_columns if col.endswith('_Close')]
    
    # Extract pair names (remove _Close suffix)
    currency_pairs = [col.replace('_Close', '') for col in close_columns]
    
    # Create column names with _weight suffix for compatibility with BacktestEngine
    signal_columns = [f"{pair}_weight" for pair in currency_pairs]
    
    # Initialize signals dataframe with _weight column names
    signals = pd.DataFrame(0.0, index=price_data.index, columns=signal_columns)
    
    # Calculate momentum and volatility using only Close prices
    close_prices = price_data[close_columns]
    close_prices.columns = currency_pairs  # Rename columns to pair names
    
    # Calculate momentum (simple return over lookback period)
    momentum_data = close_prices.pct_change(periods=lookback_days).fillna(0)
    
    # Calculate rolling volatility (annualized) - FIXED: use full window for reliable estimates
    returns = close_prices.pct_change().fillna(0)
    volatility_data = returns.rolling(window=volatility_window, min_periods=volatility_window).std() * np.sqrt(252)
    
    # Generate signals based on rebalancing frequency
    if rebalance_freq == 'weekly':
        # Rebalance every Friday (weekday 4)
        rebalance_dates = price_data.index[price_data.index.weekday == 4]
    else:  # daily
        rebalance_dates = price_data.index
    
    for date in rebalance_dates:
        if date in momentum_data.index and date in volatility_data.index:
            # Get momentum and volatility values for this date
            momentum_values = momentum_data.loc[date]
            volatility_values = volatility_data.loc[date]
            
            # Remove pairs with NaN momentum or volatility
            valid_pairs = ~(momentum_values.isna() | volatility_values.isna())
            momentum_values = momentum_values[valid_pairs]
            volatility_values = volatility_values[valid_pairs]
            
            # Skip if we don't have enough valid pairs
            if len(momentum_values) < top_n * 2:
                continue
            
            # Skip if volatility data is insufficient (>30% missing)
            if len(volatility_values) < len(momentum_values) * 0.7:
                continue
                
            # Skip if volatilities are too similar (no meaningful differences)
            if volatility_values.std() < 0.03:  # Require at least 3% std dev in volatilities
                continue
            
            # Handle edge cases: zero or very low volatility - FIXED: higher minimum threshold
            min_vol = 0.05  # Minimum volatility threshold (5% annualized) - more realistic
            volatility_values = np.maximum(volatility_values, min_vol)
            
            # Sort by momentum for contrarian strategy
            sorted_momentum = momentum_values.sort_values(ascending=True)  # Ascending for contrarian
            
            # Select top N losers for LONG positions (lowest momentum = biggest losers)
            top_losers = sorted_momentum.head(top_n).index
            loser_volatilities = volatility_values[top_losers]
            
            # Select top N winners for SHORT positions (highest momentum = biggest winners)
            top_winners = sorted_momentum.tail(top_n).index
            winner_volatilities = volatility_values[top_winners]
            
            # Calculate risk parity weights
            # For LONG positions (losers): weight = (1/volatility) / sum(1/volatility_losers)
            if len(top_losers) > 0 and not loser_volatilities.isna().all():
                inv_vol_losers = 1.0 / loser_volatilities
                # Normalize to sum to 1.0 (will be scaled later)
                long_weights = inv_vol_losers / inv_vol_losers.sum()
            else:
                long_weights = pd.Series(dtype=float)
            
            # For SHORT positions (winners): weight = -(1/volatility) / sum(1/volatility_winners)
            if len(top_winners) > 0 and not winner_volatilities.isna().all():
                inv_vol_winners = 1.0 / winner_volatilities
                # Normalize to sum to 1.0, then make negative (will be scaled later)
                short_weights = -(inv_vol_winners / inv_vol_winners.sum())
            else:
                short_weights = pd.Series(dtype=float)
            
            # Scale to maintain ~200% total exposure (100% long + 100% short)
            # Each side should sum to 1.0 in absolute terms
            if len(long_weights) > 0:
                long_weights = long_weights * 1.0  # 100% long exposure
            if len(short_weights) > 0:
                short_weights = short_weights * 1.0  # 100% short exposure (already negative)
            
            # Set signals for this rebalancing period
            # Find next rebalancing date or end of data
            current_idx = price_data.index.get_loc(date)
            if rebalance_freq == 'weekly':
                # Find next Friday or end of data
                next_rebalance_dates = rebalance_dates[rebalance_dates > date]
                if len(next_rebalance_dates) > 0:
                    next_date = next_rebalance_dates[0]
                    next_idx = price_data.index.get_loc(next_date)
                else:
                    next_idx = len(price_data)
            else:
                next_idx = current_idx + 1
            
            # Apply signals for the period STARTING FROM NEXT DAY (avoid lookahead bias)
            period_range = price_data.index[current_idx+1:next_idx]
            
            for period_date in period_range:
                if period_date in signals.index:
                    # Apply long weights
                    for pair in long_weights.index:
                        weight_col = f"{pair}_weight"
                        if weight_col in signals.columns:
                            signals.loc[period_date, weight_col] = long_weights[pair]
                    
                    # Apply short weights (already negative)
                    for pair in short_weights.index:
                        weight_col = f"{pair}_weight"
                        if weight_col in signals.columns:
                            signals.loc[period_date, weight_col] = short_weights[pair]
    
    # Validation: check total exposure and weight diversity
    long_exposure = signals[signals > 0].sum(axis=1).mean()
    short_exposure = abs(signals[signals < 0].sum(axis=1)).mean()
    total_exposure = long_exposure + short_exposure
    
    # Check weight diversity
    non_zero_signals = signals[signals != 0]
    weight_std = non_zero_signals.std().mean() if len(non_zero_signals) > 0 else 0
    weight_range = (non_zero_signals.max().max() - non_zero_signals.min().min()) if len(non_zero_signals) > 0 else 0
    
    print(f"✓ Generated contrarian risk parity signals: {lookback_days}d momentum, {volatility_window}d volatility")
    print(f"✓ Rebalancing: {rebalance_freq}")
    print(f"✓ Signal range: {signals.index.min().date()} to {signals.index.max().date()}")
    print(f"✓ Average exposure - Long: {long_exposure:.2f}, Short: {short_exposure:.2f}, Total: {total_exposure:.2f}")
    print(f"✓ Weight diversity - Std: {weight_std:.3f}, Range: {weight_range:.3f}")
    print(f"✓ Risk parity: Lower volatility pairs get higher weights")
    
    return signals

def generate_contrarian_signals_with_filter(price_data: pd.DataFrame, lookback_days: int = 30,
                                          top_n: int = 5, rebalance_freq: str = 'weekly') -> pd.DataFrame:
    """
    Generate contrarian trading signals with weekly performance filter.
    
    Simple filter logic: If portfolio loses money in a week, set all weights to zero for next week.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data with columns for each currency pair (OHLC format)
    lookback_days : int
        Number of days to look back for momentum calculation
    top_n : int
        Number of top/bottom momentum pairs to select
    rebalance_freq : str
        Rebalancing frequency ('weekly' recommended for filter)
        
    Returns:
    --------
    pd.DataFrame
        Trading signals with performance filter applied
    """
    # Step 1: Generate base contrarian signals (inverted momentum)
    print(f"🔄 Generating base contrarian signals ({lookback_days}d lookback)...")
    
    # Get currency pairs - extract only Close prices
    all_columns = price_data.columns
    close_columns = [col for col in all_columns if col.endswith('_Close')]
    currency_pairs = [col.replace('_Close', '') for col in close_columns]
    signal_columns = [f"{pair}_weight" for pair in currency_pairs]
    
    # Initialize signals dataframe
    signals = pd.DataFrame(0.0, index=price_data.index, columns=signal_columns)
    
    # Calculate momentum using only Close prices
    close_prices = price_data[close_columns]
    close_prices.columns = currency_pairs
    momentum_data = close_prices.pct_change(periods=lookback_days).fillna(0)
    
    # Get rebalancing dates (Fridays for weekly)
    if rebalance_freq == 'weekly':
        rebalance_dates = price_data.index[price_data.index.weekday == 4]
    else:
        rebalance_dates = price_data.index
    
    # Step 2: Generate base contrarian signals
    for date in rebalance_dates:
        if date in momentum_data.index:
            momentum_values = momentum_data.loc[date]
            
            # Skip if insufficient data
            if momentum_values.isna().sum() > len(momentum_values) * 0.5:
                continue
                
            momentum_values = momentum_values.dropna()
            if len(momentum_values) < top_n * 2:
                continue
            
            # Sort by momentum for CONTRARIAN strategy (ascending = long losers)
            sorted_momentum = momentum_values.sort_values(ascending=True)
            
            # CONTRARIAN: Long the biggest LOSERS, Short the biggest WINNERS
            top_losers = sorted_momentum.head(top_n).index  # Bottom momentum = Long
            top_winners = sorted_momentum.tail(top_n).index  # Top momentum = Short
            
            # Set period for signals
            current_idx = price_data.index.get_loc(date)
            if rebalance_freq == 'weekly':
                next_rebalance_dates = rebalance_dates[rebalance_dates > date]
                if len(next_rebalance_dates) > 0:
                    next_date = next_rebalance_dates[0]
                    next_idx = price_data.index.get_loc(next_date)
                else:
                    next_idx = len(price_data)
            else:
                next_idx = current_idx + 1
            
            period_range = price_data.index[current_idx+1:next_idx]
            
            for period_date in period_range:
                if period_date in signals.index:
                    # CONTRARIAN positions
                    top_loser_cols = [f"{pair}_weight" for pair in top_losers]
                    top_winner_cols = [f"{pair}_weight" for pair in top_winners]
                    
                    signals.loc[period_date, top_loser_cols] = 0.2   # Long losers
                    signals.loc[period_date, top_winner_cols] = -0.2  # Short winners
    
    print(f"✓ Base contrarian signals generated")
    
    # Step 3: Apply Weekly Performance Filter
    print(f"🔍 Applying weekly performance filter...")
    
    filtered_signals = signals.copy()
    filter_activations = 0
    
    # Calculate weekly portfolio returns using close prices
    portfolio_returns = pd.Series(0.0, index=price_data.index)
    
    for i in range(1, len(price_data)):
        date = price_data.index[i]
        prev_date = price_data.index[i-1]
        
        if date in signals.index and prev_date in signals.index:
            # Get previous day's weights
            prev_weights = signals.loc[prev_date]
            
            # Calculate daily returns for each pair
            daily_returns = {}
            for pair in currency_pairs:
                close_col = f"{pair}_Close"
                if close_col in price_data.columns:
                    price_today = price_data.loc[date, close_col]
                    price_yesterday = price_data.loc[prev_date, close_col]
                    if price_yesterday != 0 and not pd.isna(price_today) and not pd.isna(price_yesterday):
                        daily_returns[pair] = (price_today / price_yesterday) - 1
                    else:
                        daily_returns[pair] = 0.0
                else:
                    daily_returns[pair] = 0.0
            
            # Calculate portfolio return for this day
            portfolio_return = 0.0
            for pair in currency_pairs:
                weight_col = f"{pair}_weight"
                if weight_col in prev_weights.index:
                    weight = prev_weights[weight_col]
                    pair_return = daily_returns.get(pair, 0.0)
                    portfolio_return += weight * pair_return
            
            portfolio_returns.loc[date] = portfolio_return
    
    # Apply filter: check performance every Friday and adjust next week's signals
    for i, date in enumerate(rebalance_dates):
        if i == 0:  # Skip first rebalancing date (no previous week to check)
            continue
            
        # Get previous rebalancing date
        prev_rebalance_date = rebalance_dates[i-1]
        
        # Calculate week return from prev rebalance to current rebalance
        week_dates = price_data.index[(price_data.index > prev_rebalance_date) & 
                                     (price_data.index <= date)]
        
        if len(week_dates) > 0:
            week_return = portfolio_returns.loc[week_dates].sum()  # Sum daily returns for week
            
            # If week was negative, zero out next week's signals
            if week_return < 0:
                filter_activations += 1
                
                # Find next rebalancing period
                current_idx = price_data.index.get_loc(date)
                next_rebalance_dates = rebalance_dates[rebalance_dates > date]
                if len(next_rebalance_dates) > 0:
                    next_date = next_rebalance_dates[0]
                    next_idx = price_data.index.get_loc(next_date)
                else:
                    next_idx = len(price_data)
                
                # Zero out signals for next week
                next_week_range = price_data.index[current_idx:next_idx]
                filtered_signals.loc[next_week_range, :] = 0.0
    
    print(f"✓ Performance filter applied")
    print(f"📊 Filter activations: {filter_activations} weeks out of {len(rebalance_dates)-1} total weeks")
    print(f"📈 Filter rate: {filter_activations/(len(rebalance_dates)-1)*100:.1f}% of weeks filtered")
    
    return filtered_signals

def validate_signals(signals: pd.DataFrame) -> dict:
    """
    Perform comprehensive signal validation.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        Trading signals
        
    Returns:
    --------
    dict
        Signal validation report
    """
    # Count positions
    long_positions = (signals == 0.2).sum().sum()
    short_positions = (signals == -0.2).sum().sum()
    neutral_positions = (signals == 0.0).sum().sum()
    
    report = {
        'total_signals': signals.size,
        'long_positions': long_positions,
        'short_positions': short_positions,
        'neutral_positions': neutral_positions,
        'date_range': {
            'start': signals.index.min(),
            'end': signals.index.max()
        },
        'currency_pairs': signals.shape[1]
    }
    
    return report