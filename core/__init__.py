"""
Forex Momentum Trading System - Core Module

This package contains the core functionality for the forex momentum trading system:
- Data loading and processing
- Signal generation and management  
- Backtesting engine
"""

__version__ = "1.0.0"
__author__ = "Quantitative Trading System"

from .data_loader import load_forex_data
from .signal_generator import load_signals
from .backtest_engine import BacktestEngine

__all__ = ['load_forex_data', 'load_signals', 'BacktestEngine']