"""模型接口定义"""

from typing import Protocol, Dict, Any, Optional
import numpy as np


class StockPredictor(Protocol):
    """股票预测器接口"""

    @property
    def name(self) -> str:
        """预测器名称"""
        ...

    @property
    def description(self) -> str:
        """预测器描述"""
        ...

    def train(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        """训练模型"""
        ...

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """预测，返回多层结果: daily_prob, trend, target_price"""
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型，返回指标字典"""
        ...
