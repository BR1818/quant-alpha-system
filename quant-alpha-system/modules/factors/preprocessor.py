"""特征标准化与预处理工具"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import RobustScaler
import logging


class FeaturePreprocessor:
    """特征预处理器：标准化、去极值、填充缺失值"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = RobustScaler()
        self._fitted = False
        self._feature_cols: List[str] = []

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """对特征列进行 RobustScaler 标准化（中位数+四分位距，对极值更鲁棒）"""
        self._feature_cols = [c for c in feature_cols if c in df.columns]
        if not self._feature_cols:
            return df

        result = df.copy()
        X = result[self._feature_cols].values
        
        # 将 inf 替换为 nan
        X = np.where(np.isinf(X), np.nan, X)
        
        # 逐列用中位数填充 nan
        for i in range(X.shape[1]):
            col = X[:, i]
            median_val = np.nanmedian(col)
            col[np.isnan(col)] = median_val if not np.isnan(median_val) else 0
            X[:, i] = col

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        result[self._feature_cols] = X_scaled
        self._fitted = True
        self.logger.info(f"特征标准化完成: {len(self._feature_cols)} 列")
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的 scaler 转换新数据"""
        if not self._fitted:
            raise RuntimeError("Scaler 未拟合，请先调用 fit_transform")
        result = df.copy()
        X = result[self._feature_cols].values
        X = np.where(np.isinf(X), np.nan, X)
        for i in range(X.shape[1]):
            col = X[:, i]
            median_val = np.nanmedian(col)
            col[np.isnan(col)] = median_val if not np.isnan(median_val) else 0
            X[:, i] = col
        result[self._feature_cols] = self.scaler.transform(X)
        return result

    def get_feature_matrix(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """提取干净的特征矩阵 (无 inf / nan)"""
        cols = feature_cols or self._feature_cols
        X = df[cols].values.copy()
        X = np.where(np.isinf(X), 0, X)
        X = np.nan_to_num(X, nan=0.0)
        return X
