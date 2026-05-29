"""XGBoost 选股器 — 基于监督学习的多因子选股"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

import xgboost as xgb


class XGBoostSelector:
    """XGBoost 选股器 — 训练后输出个股上涨概率作为选股得分"""

    name = "xgboost_selector"
    description = "基于 XGBoost 的多因子机器学习选股器"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self._feature_names: List[str] = []
        self._feature_importance: Dict[str, float] = {}

    def train(self, X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, Any]] = None,
              test_size: float = 0.2) -> None:
        """训练选股模型 — 按时间顺序切分train/test，避免未来函数"""
        self.logger.info(f"训练 XGBoost 选股器: {X.shape[1]} 特征, {X.shape[0]} 样本")
        self._feature_names = list(X.columns)

        if params:
            self.params.update(params)

        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)

        # 时序切分：前80%训练，后20%测试（严格按时间顺序，不shuffle）
        split_idx = int(len(X_clean) * (1 - test_size))
        X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train,
                       eval_set=[(X_test, y_test)],
                       verbose=False)

        # 记录测试集AUC
        from sklearn.metrics import roc_auc_score
        try:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            self.logger.info(f"训练完成. Test AUC: {auc:.4f}")
        except Exception:
            self.logger.info("训练完成. 无法计算AUC(类别不足)")

        self._feature_importance = dict(
            zip(self._feature_names, self.model.feature_importances_)
        )

    def select(self, data: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """选股"""
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        self.logger.info(f"XGBoost 选股: top_{top_n}, 候选: {len(data)}")

        X = data[self._feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        probabilities = self.model.predict_proba(X)[:, 1]

        result = data.copy()
        result["score"] = probabilities
        result = result.nlargest(top_n, "score").copy()

        self.logger.info(f"选股完成: {len(result)} 只, 得分范围 [{result['score'].min():.4f}, {result['score'].max():.4f}]")
        return result

    def get_factor_weights(self) -> Dict[str, float]:
        return self._feature_importance

    def save_model(self, path: Path) -> None:
        """保存模型"""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "features": self._feature_names}, f)
        self.logger.info(f"模型已保存: {path}")

    def load_model(self, path: Path) -> None:
        """加载模型"""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self._feature_names = data["features"]
        self.logger.info(f"模型已加载: {path}")
