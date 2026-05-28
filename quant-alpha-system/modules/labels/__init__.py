"""标签生成器 — 为监督学习提供训练标签，严格避免未来函数"""

import pandas as pd
import numpy as np
from typing import Optional
import logging


class LabelGenerator:
    """标签生成器：基于未来 N 日收益率生成分类/回归标签"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def forward_return(self, data: pd.DataFrame, horizon: int = 5, price_col: str = "close") -> pd.Series:
        """计算未来 N 日收益率（回归标签）
        
        注意：最后 horizon 行的标签为 NaN，训练时必须 dropna
        """
        return data[price_col].shift(-horizon) / data[price_col] - 1

    def binary_label(self, data: pd.DataFrame, horizon: int = 5, price_col: str = "close") -> pd.Series:
        """二分类标签：未来 N 日上涨=1，下跌=0"""
        fwd = self.forward_return(data, horizon, price_col)
        return (fwd > 0).astype(int)

    def triple_label(self, data: pd.DataFrame, horizon: int = 5, 
                     up_threshold: float = 0.02, down_threshold: float = -0.02,
                     price_col: str = "close") -> pd.Series:
        """三分类标签：涨(2) / 震荡(1) / 跌(0)"""
        fwd = self.forward_return(data, horizon, price_col)
        labels = pd.Series(1, index=data.index)  # 默认震荡
        labels[fwd > up_threshold] = 2   # 涨
        labels[fwd < down_threshold] = 0  # 跌
        return labels

    def create_sequences(self, features: np.ndarray, labels: np.ndarray, 
                         seq_len: int = 30) -> tuple:
        """为 LSTM 创建滑动窗口序列
        
        Args:
            features: (N, n_features) 的特征矩阵
            labels: (N,) 的标签向量
            seq_len: 每个样本的时间步长
            
        Returns:
            X: (N-seq_len, seq_len, n_features)
            y: (N-seq_len,)
        """
        X, y = [], []
        for i in range(seq_len, len(features)):
            if np.isnan(labels[i]):
                continue
            X.append(features[i - seq_len:i])
            y.append(labels[i])
        return np.array(X), np.array(y)
