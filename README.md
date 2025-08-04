# 🚀 Forex Momentum Trading System

A high-performance quantitative trading framework for backtesting momentum-based strategies on forex currency pairs and commodities. Built with numba-optimized vectorized calculations for maximum computational efficiency.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Numba](https://img.shields.io/badge/numba-optimized-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Overview

This system generates trading signals based on momentum indicators and simulates portfolio performance with comprehensive risk metrics. It processes 22 forex pairs + 4 commodities with weekly rebalancing and multiple strategy configurations.

### Key Features

- **⚡ High Performance**: Numba-compiled execution loops processing 1000+ days/second
- **📊 Multi-Strategy**: Momentum, contrarian, and risk parity strategies
- **💹 Comprehensive Coverage**: 22 forex pairs + 4 commodities (Gold, Silver, Natural Gas, Crude Oil)
- **🔄 Advanced Rebalancing**: Weekly rebalancing with configurable leverage
- **📈 Rich Analytics**: Sharpe, Sortino, Calmar ratios, VaR, drawdown analysis
- **🎨 Visualization**: Equity curves, drawdown charts, monthly performance heatmaps

## 🏗️ Architecture

```
fx_mom/
├── core/                          # Core trading engine modules
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and validation
│   ├── signal_generator.py       # Trading signal generation
│   ├── backtest_engine.py        # Numba-optimized backtesting
│   └── commodity_momentum.py     # Commodity momentum strategies
├── notebooks/                    # Analysis and strategy notebooks
│   ├── main.ipynb                # Main analysis notebook
│   ├── momentum_long_only.ipynb  # Momentum-only strategy
│   ├── contrarian_strategy.ipynb # Contrarian strategy analysis
│   ├── commodity_momentum_clean.ipynb    # Commodity momentum analysis
│   ├── contrarian_momentum_clean.ipynb   # Contrarian momentum on commodities
│   ├── contrarian_filtered_strategy.ipynb # Filtered contrarian strategy
│   └── contrarian_vol_scaling.ipynb      # Volatility-scaled contrarian
├── scripts/                      # Data fetching and utilities
│   ├── download_extended_data.py     # Forex data fetching
│   └── download_extended_commodities.py # Commodity data fetching
├── data/                         # Market data storage
│   ├── forex_synchronized_data.parquet
│   ├── forex_extended_data.parquet
│   └── commodities_extended_data.parquet
└── strategies/                   # Strategy implementations (future)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd fx_mom

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Navigate to the project directory and import core modules
import sys
sys.path.append('core')

from data_loader import load_forex_data, load_forex_and_commodities_data
from signal_generator import generate_momentum_signals
from backtest_engine import BacktestEngine

# Load synchronized forex data
df = load_forex_data('data/forex_synchronized_data.parquet')

# Generate momentum signals (30-day lookback)
signals = generate_momentum_signals(df, lookback_days=30)

# Run backtest
engine = BacktestEngine()
results = engine.run_backtest(df, signals)

# View performance metrics
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Working with Notebooks

All analysis notebooks are located in the `notebooks/` directory:

- **Main Analysis**: `notebooks/main.ipynb` - Complete forex + commodities momentum strategy
- **Forex Only**: `notebooks/momentum_long_only.ipynb` - Long-only momentum on forex pairs
- **Contrarian Strategies**: `notebooks/contrarian_*.ipynb` - Various contrarian approaches
- **Commodity Focus**: `notebooks/commodity_momentum_clean.ipynb` - Commodities-specific momentum

### Data Download

Use the scripts in the `scripts/` directory to download fresh market data:

```bash
# Download forex data
python scripts/download_extended_data.py

# Download commodity data  
python scripts/download_extended_commodities.py
```

### Jupyter Notebooks

```bash
# Run main analysis
jupyter notebook notebooks/main.ipynb

# Momentum-only strategy
jupyter notebook notebooks/momentum_long_only.ipynb

# Strategy comparison
jupyter notebook notebooks/contrarian_strategy.ipynb

# Commodity momentum
jupyter notebook notebooks/commodity_momentum_clean.ipynb
```

## 📊 Strategy Details

### Momentum Strategy
- **Position Sizing**: Top 5 long (20% each) + Top 5 short (20% each) = 200% total exposure
- **Lookback Periods**: 5, 14, 30, 60, 90, 120, 180 days (configurable)
- **Rebalancing**: Weekly on Fridays
- **Selection**: Based on cumulative returns ranking

### Contrarian Strategy
- **Risk Parity Weighting**: Inverse volatility weighting within long/short buckets
- **Volatility Window**: 30-day rolling volatility for risk adjustment
- **Selection**: Bottom 5 momentum (long losers) + Top 5 momentum (short winners)

### Risk Management
- **Drawdown Analysis**: Maximum and average drawdown tracking
- **Value at Risk (VaR)**: 95% and 99% confidence intervals
- **Correlation-Based Risk Parity**: Dynamic position sizing based on volatility

## 📈 Performance Analytics

The system generates comprehensive performance metrics:

- **Return Metrics**: Total return, annualized return, excess return
- **Risk Metrics**: Volatility, Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Analysis**: Maximum drawdown, average drawdown, recovery time
- **Statistical Tests**: Normality tests, autocorrelation analysis

### Sample Output

```
=== PERFORMANCE METRICS (30-day lookback) ===
Total Return: 127.45%
Annualized Return: 12.8%
Volatility: 15.2%
Sharpe Ratio: 0.84
Max Drawdown: -8.3%
Calmar Ratio: 1.54
```

## 🛠️ Advanced Configuration

### Custom Lookback Periods

```python
# Test multiple lookback periods
lookback_periods = [5, 14, 30, 60, 90, 120, 180]

for lookback in lookback_periods:
    signals = generate_momentum_signals(df, lookback_days=lookback)
    results = engine.run_backtest(df, signals)
    print(f"Lookback {lookback}d: Sharpe={results['sharpe_ratio']:.2f}")
```

### Adding New Asset Classes

```python
# Extend commodity data generation
def generate_custom_commodity_data(df, commodity_name, volatility=0.02):
    # Custom implementation for new commodities
    pass
```

## 📊 Data Sources

- **Forex Data**: 22 major currency pairs (EUR/USD, GBP/USD, USD/JPY, etc.)
- **Commodities**: Gold (XAU), Silver (XAG), Natural Gas (NG), Crude Oil (CL)
- **Format**: Parquet files with OHLC data, synchronized timestamps
- **Frequency**: Daily data with weekly rebalancing

## 🔧 Performance Optimization

### Numba Acceleration
- Critical simulation loops are JIT-compiled with `@jit(nopython=True)`
- Vectorized operations for returns and position calculations
- Memory-efficient numpy array processing

### Data Efficiency
- Parquet format for compressed storage
- Date alignment and missing value handling  
- Optimized data structures for time series operations

## 📚 Dependencies

- **pandas**: Data manipulation and time series handling
- **numpy**: Numerical computations and array operations
- **numba**: JIT compilation for performance-critical code
- **matplotlib/seaborn**: Visualization and plotting
- **scipy**: Statistical functions and analysis
- **yfinance**: Market data fetching
- **pyarrow**: Parquet file I/O optimization

## 📖 Usage Examples

### Backtesting Different Strategies

```python
# Compare momentum vs contrarian strategies
momentum_results = backtest_momentum_strategy(df, lookback=30)
contrarian_results = backtest_contrarian_strategy(df, lookback=30)

print(f"Momentum Sharpe: {momentum_results['sharpe']:.2f}")
print(f"Contrarian Sharpe: {contrarian_results['sharpe']:.2f}")
```

### Custom Signal Generation

```python
# Create custom momentum signals
def custom_momentum_signals(data, fast_period=10, slow_period=30):
    signals = pd.DataFrame(index=data.index)
    
    for pair in get_currency_pairs(data):
        fast_ma = data[f'{pair}_close'].rolling(fast_period).mean()
        slow_ma = data[f'{pair}_close'].rolling(slow_period).mean()
        
        # Generate signals based on moving average crossover
        signals[f'{pair}_weight'] = np.where(fast_ma > slow_ma, 0.2, -0.2)
    
    return signals
```

## 🤝 Contributing

Contributions are welcome! Please ensure that:

1. Code follows existing style conventions
2. New features include appropriate tests
3. Performance-critical code uses numba optimization
4. Documentation is updated for new functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Disclaimer**: This software is for educational and research purposes only. Trading involves risk and past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.