"""核心接口定义"""

from core.interfaces.data import DataLoader, DataValidator
from core.interfaces.factor import Factor, FactorRegistry
from core.interfaces.model import StockPredictor
from core.interfaces.strategy import StockSelector

__all__ = [
    "DataLoader",
    "DataValidator",
    "Factor",
    "FactorRegistry",
    "StockPredictor",
    "StockSelector",
]
