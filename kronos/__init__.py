"""
Kronos - Time Series Prediction Library for Stock Market Analysis

A fork of shiyu-coder/Kronos with extended support for Chinese and global
stock markets, integrating akshare, baostock, and eastmoney data sources.
"""

__version__ = "0.2.0"
__author__ = "Kronos Contributors"
__license__ = "MIT"

from kronos.model import KronosModel
from kronos.predictor import StockPredictor
from kronos.data import DataLoader

__all__ = [
    "KronosModel",
    "StockPredictor",
    "DataLoader",
    "__version__",
]
