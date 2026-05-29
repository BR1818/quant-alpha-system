"""特征标准化与预处理工具 — MAD去极值 + 训练集中位数填充 + RobustScaler"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import RobustScaler
import logging


class FeaturePreprocessor:
    """特征预处理器：MAD去极值 → 训练集中位数填充 → RobustScaler标准化"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = RobustScaler()
        self._fitted = False
        self._feature_cols: List[str] = []
        self._medians: Optional[np.ndarray] = None
        self._mad_bounds: Optional[dict] = None

    def _mad_winsorize(self, X: np.ndarray) -> np.ndarray:
        """MAD去极值：clip到 median ± 3 * 1.4826 * MAD"""
        self._mad_bounds = {}
        for i in range(X.shape[1]):
            col = X[:, i]
            median = np.nanmedian(col)
            mad = np.nanmedian(np.abs(col - median))
            bound = 3.0 * 1.4826 * mad
            lower = median - bound
            upper = median + bound
            self._mad_bounds[i] = (lower, upper)
            X[:, i] = np.clip(col, lower, upper)
        return X

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """完整预处理流水线: MAD去极值 → 中位数填充 → 标准化"""
        self._feature_cols = [c for c in feature_cols if c in df.columns]
        if not self._feature_cols:
            return df

        result = df.copy()
        X = result[self._feature_cols].values.astype(np.float64)

        # 1. inf → nan
        X = np.where(np.isinf(X), np.nan, X)

        # 2. MAD去极值
        X = self._mad_winsorize(X)

        # 3. 保存训练集中位数，用于填充nan
        self._medians = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            fill_val = self._medians[i] if not np.isnan(self._medians[i]) else 0.0
            X[mask, i] = fill_val

        # 4. RobustScaler标准化
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        result[self._feature_cols] = X_scaled
        self._fitted = True
        self.logger.info(f"特征预处理完成(含MAD去极值): {len(self._feature_cols)} 列")
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的参数转换新数据（不偷看新数据统计量）"""
        if not self._fitted:
            raise RuntimeError("Scaler 未拟合，请先调用 fit_transform")
        result = df.copy()
        X = result[self._feature_cols].values.astype(np.float64)
        X = np.where(np.isinf(X), np.nan, X)

        # 使用训练集的MAD bounds去极值
        if self._mad_bounds:
            for i in range(X.shape[1]):
                if i in self._mad_bounds:
                    lower, upper = self._mad_bounds[i]
                    X[:, i] = np.clip(X[:, i], lower, upper)

        # 使用训练集保存的中位数填充（不用新数据的统计量）
        if self._medians is not None:
            for i in range(X.shape[1]):
                mask = np.isnan(X[:, i])
                fill_val = self._medians[i] if not np.isnan(self._medians[i]) else 0.0
                X[mask, i] = fill_val

        result[self._feature_cols] = self.scaler.transform(X)
        return result

    def get_feature_matrix(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """提取干净的特征矩阵 (无 inf / nan)"""
        cols = feature_cols or self._feature_cols
        X = df[cols].values.copy().astype(np.float64)
        X = np.where(np.isinf(X), np.nan, X)
        # 使用保存的中位数填充
        if self._medians is not None and len(self._medians) == X.shape[1]:
            for i in range(X.shape[1]):
                mask = np.isnan(X[:, i])
                fill_val = self._medians[i] if not np.isnan(self._medians[i]) else 0.0
                X[mask, i] = fill_val
        else:
            X = np.nan_to_num(X, nan=0.0)
        return X
