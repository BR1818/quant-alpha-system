"""集成预测器 — 加权平均多个预测器结果"""

import numpy as np
from typing import Dict, Any, List
import logging


class EnsemblePredictor:
    """集成多个预测器的加权平均结果"""

    name = "ensemble_predictor"
    description = "集成多个预测器加权平均"

    def __init__(self, predictors: List, weights: List[float] = None):
        self.logger = logging.getLogger(__name__)
        self.predictors = predictors
        n = len(predictors)
        self.weights = weights or [1.0 / n] * n

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] = None) -> None:
        """训练所有子预测器"""
        for pred in self.predictors:
            pred.train(X, y, params)

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """集成预测"""
        self.logger.debug(f"集成预测: {len(self.predictors)} 个预测器")
        all_results = [p.predict(X) for p in self.predictors]

        result = {}
        for key in all_results[0]:
            weighted = sum(r[key] * w for r, w in zip(all_results, self.weights))
            result[key] = weighted

        return result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估"""
        preds = self.predict(X)
        return {
            "mse": float(np.mean((preds["target_price"] - y[:, 2]) ** 2)),
            "mae": float(np.mean(np.abs(preds["target_price"] - y[:, 2]))),
            "direction_acc": float(np.mean((preds["daily_prob"] > 0.5) == (y[:, 0] > 0.5))),
        }
