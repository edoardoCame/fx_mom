# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Forex Momentum Trading System** - a quantitative trading framework that backtests momentum-based strategies on forex currency pairs and commodities. The system generates trading signals based on momentum indicators and simulates portfolio performance with comprehensive risk metrics.

## Key Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running the System
```bash
# Run main analysis (Jupyter notebook)
jupyter notebook main.ipynb

# Alternative: momentum-only strategy 
jupyter notebook momentum_long_only.ipynb

# Strategy comparison
jupyter notebook contrarian_strategy.ipynb
```

### Python Execution
```bash
# Direct module execution
python -m core.backtest_engine

# Import and use in Python
python -c "from core import BacktestEngine, load_forex_data, generate_momentum_signals"
```

## Architecture Overview

### Core Module Structure (`core/`)
- **`data_loader.py`** - Forex/commodity data loading and validation with synthetic commodity generation
- **`signal_generator.py`** - Trading signal generation (momentum, contrarian, risk parity strategies)  
- **`backtest_engine.py`** - High-performance numba-optimized backtesting engine with vectorized calculations

### Data Flow
1. **Data Loading**: Load synchronized forex data from parquet + generate commodity data
2. **Signal Generation**: Calculate momentum indicators and generate position weights
3. **Backtesting**: Run numba-optimized portfolio simulation with weekly rebalancing
4. **Analysis**: Calculate performance metrics, drawdowns, and risk-adjusted returns

### Key Features
- **Vectorized Processing**: Pandas preprocessing + numba-compiled execution loops
- **Multiple Strategies**: Momentum, contrarian, risk parity with dynamic signal generation
- **Asset Classes**: 22 forex pairs + 4 commodities (Gold, Silver, Natural Gas, Crude Oil)
- **Performance Analytics**: Comprehensive metrics including Sharpe, Sortino, Calmar ratios
- **Risk Management**: Drawdown analysis, VaR calculations, correlation-based risk parity

## Data Structure

### Input Data Format
- **Forex Data**: `data/forex_synchronized_data.parquet` with OHLC columns per currency pair
- **Generated Signals**: DataFrames with `{pair}_weight` columns containing position weights (-0.2, 0, +0.2)

### Output Files
- **Portfolio Results**: `portfolio_results_mixed_{lookback}d.csv`
- **Performance Metrics**: `performance_metrics_mixed_{lookback}d.txt` 
- **Visualizations**: Equity curves, drawdown charts, monthly heatmaps

## Strategy Configuration

### Momentum Strategy Parameters
- **Lookback Periods**: 5, 14, 30, 60, 90, 120, 180 days (configurable in notebooks)
- **Position Sizing**: Top 5 long (20% each) + Top 5 short (20% each) = 200% total exposure
- **Rebalancing**: Weekly on Fridays
- **Universe**: 22 forex pairs + 4 commodities = 26 total instruments

### Contrarian Strategy Parameters  
- **Risk Parity Weighting**: Inverse volatility weighting within long/short buckets
- **Volatility Window**: 30-day rolling volatility for risk adjustment
- **Selection**: Bottom 5 momentum (long losers) + Top 5 momentum (short winners)

## Performance Optimization

### Numba Acceleration
- Critical simulation loops are JIT-compiled with `@jit(nopython=True)`
- Portfolio simulation processes ~1000+ days/second
- Vectorized operations for returns and position calculations

### Memory Efficiency
- Parquet format for compressed data storage
- Numpy arrays for numba processing
- Date alignment and missing value handling

## Common Workflows

### Backtesting Different Lookback Periods
1. Modify `lookback_days` variable in notebook cells
2. Re-run signal generation and backtesting
3. Compare performance metrics across periods

### Adding New Asset Classes
1. Extend `generate_commodity_data()` in `data_loader.py`
2. Update currency pair detection logic
3. Ensure signal generation handles new instruments

### Custom Strategy Development
1. Implement new signal generator in `signal_generator.py`
2. Follow naming convention: `{pair}_weight` columns
3. Use `BacktestEngine.run_backtest()` for simulation

## Testing and Validation

- **Data Quality Checks**: `validate_data_quality()` and `validate_signals()`
- **Signal Validation**: Ensures proper weight ranges and exposure limits
- **Performance Verification**: Cross-reference with benchmark strategies

## Key Dependencies

- **pandas**: Data manipulation and time series handling
- **numpy**: Numerical computations and array operations  
- **numba**: JIT compilation for performance-critical code
- **matplotlib/seaborn**: Visualization and plotting
- **scipy**: Statistical functions and analysis
- **yfinance**: Market data fetching (if needed)
- **pyarrow**: Parquet file I/O optimization