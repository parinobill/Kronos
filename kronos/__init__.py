"""Kronos - Time Series Prediction Library for Stock Market Analysis

A fork of shiyu-coder/Kronos with extended support for Chinese and global
stock markets, integrating akshare, baostock, and eastmoney data sources.

Personal fork notes:
- Primary use case: A-share market analysis (SSE/SZSE)
- Default data source set to akshare for convenience
- See examples/ directory for usage with Chinese stock codes
"""

__version__ = "0.2.0"
__author__ = "Kronos Contributors"
__license__ = "MIT"

# Personal default: prefer akshare as the data source for A-share markets
DEFAULT_DATA_SOURCE = "akshare"

from kronos.model import KronosModel
from kronos.predictor import StockPredictor
from kronos.data import DataLoader

__all__ = [
    "KronosModel",
    "StockPredictor",
    "DataLoader",
    "DEFAULT_DATA_SOURCE",
    "__version__",
]
