"""
Data Loading Module for Forex Momentum Trading System

This module provides functions to load and validate forex price data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def load_forex_data(data_path: str, validate: bool = True) -> pd.DataFrame:
    """
    Load forex price data from parquet file.
    
    Parameters:
    -----------
    data_path : str
        Path to the forex data parquet file
    validate : bool
        Whether to perform data validation
        
    Returns:
    --------
    pd.DataFrame
        Forex price data with columns for each currency pair OHLC
    """
    # Load data
    data = pd.read_parquet(data_path)
    
    if validate:
        # Basic validation
        if data.empty:
            raise ValueError("Data file is empty")
            
        if data.index.duplicated().any():
            raise ValueError("Duplicate dates found in data")
            
        # Check for missing values
        if data.isnull().any().any():
            print(f"Warning: {data.isnull().sum().sum()} missing values found")
    
    print(f"✓ Loaded forex data: {data.shape[0]} days, {data.shape[1]} columns")
    print(f"✓ Date range: {data.index.min().date()} to {data.index.max().date()}")
    
    return data

def get_currency_pairs(data: pd.DataFrame) -> List[str]:
    """
    Extract currency pair names from column names.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Forex price data
        
    Returns:
    --------
    List[str]
        List of currency pair names
    """
    pairs = []
    for col in data.columns:
        if '_Close' in col:
            pair = col.replace('_Close', '')
            pairs.append(pair)
    
    return sorted(pairs)

def generate_commodity_data(start_date: str, end_date: str, 
                           commodities: List[str] = ['GOLD', 'SILVER', 'NATGAS', 'CRUDE']) -> pd.DataFrame:
    """
    Generate realistic commodity price data for backtesting.
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    commodities : List[str]
        List of commodity symbols
        
    Returns:
    --------
    pd.DataFrame
        Commodity price data with OHLC columns
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initial prices (realistic starting points)
    initial_prices = {
        'GOLD': 1200.0,     # Gold $/oz
        'SILVER': 20.0,     # Silver $/oz  
        'NATGAS': 3.0,      # Natural Gas $/MMBtu
        'CRUDE': 80.0       # Crude Oil $/barrel
    }
    
    # Volatility parameters (annualized)
    volatilities = {
        'GOLD': 0.18,       # 18% annual volatility
        'SILVER': 0.25,     # 25% annual volatility
        'NATGAS': 0.40,     # 40% annual volatility
        'CRUDE': 0.30       # 30% annual volatility
    }
    
    # Drift parameters (annual trend)
    drifts = {
        'GOLD': 0.03,       # 3% annual growth
        'SILVER': 0.02,     # 2% annual growth
        'NATGAS': 0.01,     # 1% annual growth
        'CRUDE': 0.015      # 1.5% annual growth
    }
    
    np.random.seed(42)  # For reproducible results
    
    commodity_data = pd.DataFrame(index=dates)
    
    for commodity in commodities:
        if commodity not in initial_prices:
            continue
            
        # Generate price series using geometric Brownian motion
        n_days = len(dates)
        dt = 1/365  # Daily time step
        
        # Parameters
        S0 = initial_prices[commodity]
        mu = drifts[commodity]
        sigma = volatilities[commodity]
        
        # Generate random walks
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
        
        # Add some momentum/mean reversion effects for commodities
        # Commodities tend to have more momentum than forex
        for i in range(1, len(returns)):
            # Add momentum effect (20% weight on previous return)
            momentum_effect = 0.2 * returns[i-1]
            returns[i] += momentum_effect
            
        # Calculate prices
        log_prices = np.log(S0) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Generate OHLC from close prices
        daily_vol = sigma * np.sqrt(dt)
        
        opens = prices * (1 + np.random.normal(0, daily_vol * 0.3, n_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, daily_vol * 0.4, n_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, daily_vol * 0.4, n_days)))
        
        # Ensure price relationships: Low <= Open,Close <= High
        lows = np.minimum(lows, np.minimum(opens, prices))
        highs = np.maximum(highs, np.maximum(opens, prices))
        
        # Add to dataframe
        commodity_data[f'{commodity}_Open'] = opens
        commodity_data[f'{commodity}_High'] = highs
        commodity_data[f'{commodity}_Low'] = lows
        commodity_data[f'{commodity}_Close'] = prices
    
    print(f"✓ Generated commodity data: {len(dates)} days, {len(commodities)} commodities")
    print(f"✓ Commodities: {', '.join(commodities)}")
    
    return commodity_data

def load_forex_and_commodities_data(forex_path: str, include_commodities: bool = True) -> pd.DataFrame:
    """
    Load forex data and optionally add commodity data.
    
    Parameters:
    -----------
    forex_path : str
        Path to forex data
    include_commodities : bool
        Whether to include commodity data
        
    Returns:
    --------
    pd.DataFrame
        Combined forex and commodity data
    """
    # Load forex data
    forex_data = load_forex_data(forex_path)
    
    if not include_commodities:
        return forex_data
    
    # Generate commodity data for the same date range
    start_date = forex_data.index.min().strftime('%Y-%m-%d')
    end_date = forex_data.index.max().strftime('%Y-%m-%d')
    
    commodity_data = generate_commodity_data(start_date, end_date)
    
    # Align dates (keep only business days that exist in forex data)
    commodity_data = commodity_data.reindex(forex_data.index, method='ffill')
    
    # Combine data
    combined_data = pd.concat([forex_data, commodity_data], axis=1)
    
    print(f"✓ Combined data: {combined_data.shape[0]} days, {combined_data.shape[1]} columns")
    print(f"✓ Total instruments: {len(get_currency_pairs(forex_data))} forex + 4 commodities")
    
    return combined_data

def get_asset_names(data: pd.DataFrame) -> dict:
    """
    Extract asset names from column names, separating forex and commodities.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Combined price data
        
    Returns:
    --------
    dict
        Dictionary with 'forex' and 'commodities' lists
    """
    forex_pairs = []
    commodities = []
    
    for col in data.columns:
        if '_Close' in col:
            asset = col.replace('_Close', '')
            # Distinguish forex pairs from commodities
            if len(asset) == 6 and any(c in asset for c in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
                forex_pairs.append(asset)
            else:
                commodities.append(asset)
    
    return {
        'forex': sorted(forex_pairs),
        'commodities': sorted(commodities),
        'all': sorted(forex_pairs + commodities)
    }

def download_extended_forex_data(start_date: str = "2000-01-01", 
                               end_date: str = "2025-12-31",
                               save_path: str = "data/forex_extended_data.parquet") -> pd.DataFrame:
    """
    Download extended forex data from Yahoo Finance from 2000 to present.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str  
        End date in 'YYYY-MM-DD' format
    save_path : str
        Path to save the downloaded data
        
    Returns:
    --------
    pd.DataFrame
        Extended forex price data with OHLC columns
    """
    # Major forex pairs with USD base (Yahoo Finance format)
    forex_symbols = [
        'EURUSD=X',   # EUR/USD
        'GBPUSD=X',   # GBP/USD  
        'USDJPY=X',   # USD/JPY
        'USDCHF=X',   # USD/CHF
        'AUDUSD=X',   # AUD/USD
        'USDCAD=X',   # USD/CAD
        'NZDUSD=X',   # NZD/USD
        'EURGBP=X',   # EUR/GBP
        'EURJPY=X',   # EUR/JPY
        'GBPJPY=X',   # GBP/JPY
        'CHFJPY=X',   # CHF/JPY
        'EURCHF=X',   # EUR/CHF
        'AUDJPY=X',   # AUD/JPY
        'CADJPY=X',   # CAD/JPY
        'NZDJPY=X',   # NZD/JPY
        'GBPCHF=X',   # GBP/CHF
        'AUDCAD=X',   # AUD/CAD
        'AUDCHF=X',   # AUD/CHF
        'CADCHF=X',   # CAD/CHF
        'EURCAD=X',   # EUR/CAD
        'EURAUD=X',   # EUR/AUD
        'GBPCAD=X',   # GBP/CAD
    ]
    
    print(f"📊 Downloading forex data from {start_date} to {end_date}...")
    print(f"🔄 Downloading {len(forex_symbols)} forex pairs...")
    
    all_data = {}
    successful_downloads = 0
    
    for i, symbol in enumerate(forex_symbols):
        try:
            # Clean symbol name for column naming
            pair_name = symbol.replace('=X', '').replace('USD', 'USD')
            if not pair_name.endswith('USD') and not pair_name.startswith('USD'):
                # For cross pairs like EURGBP, keep as is
                clean_pair = pair_name
            elif pair_name.startswith('USD'):
                # For USD/XXX pairs, keep as USDXXX
                clean_pair = pair_name
            else:
                # For XXX/USD pairs, keep as XXXUSD
                clean_pair = pair_name
                
            print(f"  📈 Downloading {symbol} as {clean_pair}... ({i+1}/{len(forex_symbols)})")
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if len(data) > 0:
                # Rename columns to match our format
                data.columns = [f"{clean_pair}_{col}" for col in data.columns]
                all_data[clean_pair] = data
                successful_downloads += 1
                print(f"    ✓ Downloaded {len(data)} days of data")
            else:
                print(f"    ❌ No data available for {symbol}")
                
        except Exception as e:
            print(f"    ❌ Failed to download {symbol}: {str(e)}")
            continue
    
    if successful_downloads == 0:
        raise ValueError("No forex data could be downloaded!")
    
    print(f"\n✅ Successfully downloaded {successful_downloads}/{len(forex_symbols)} forex pairs")
    
    # Combine all data
    print("🔄 Combining and synchronizing data...")
    combined_data = pd.concat(all_data.values(), axis=1)
    
    # Remove any completely empty rows
    combined_data = combined_data.dropna(how='all')
    
    # Forward fill missing values (common in forex weekend gaps)
    combined_data = combined_data.fillna(method='ffill')
    
    # Remove any remaining NaN rows at the beginning
    combined_data = combined_data.dropna(how='all')
    
    # Ensure index is datetime
    combined_data.index = pd.to_datetime(combined_data.index)
    
    # Save to parquet
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    combined_data.to_parquet(save_path)
    
    print(f"✅ Extended forex data saved to: {save_path}")
    print(f"📊 Final dataset: {len(combined_data):,} days, {len(combined_data.columns)} columns")
    print(f"📅 Date range: {combined_data.index.min().date()} to {combined_data.index.max().date()}")
    
    return combined_data

def validate_data_quality(data: pd.DataFrame) -> dict:
    """
    Perform comprehensive data quality checks.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data (forex and/or commodities)
        
    Returns:
    --------
    dict
        Data quality report
    """
    assets = get_asset_names(data)
    
    report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'date_range': {
            'start': data.index.min(),
            'end': data.index.max()
        },
        'forex_pairs': len(assets['forex']),
        'commodities': len(assets['commodities']),
        'total_assets': len(assets['all'])
    }
    
    return report