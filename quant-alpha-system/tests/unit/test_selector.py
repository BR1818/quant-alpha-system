"""选股器测试"""
import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.selectors.composite_selector import CompositeSelector


class TestCompositeSelector:
    """综合打分选股器测试"""

    def test_select_basic(self):
        """测试基础选股功能"""
        selector = CompositeSelector()
        data = pd.DataFrame({
            "ts_code": [f"{i:06d}.SZ" for i in range(100)],
            "technical_score": np.random.rand(100),
            "fundamental_score": np.random.rand(100),
            "moneyflow_score": np.random.rand(100),
            "sentiment_score": np.random.rand(100),
        })
        result = selector.select(data, top_n=20)
        assert len(result) == 20
        assert "score" in result.columns
        assert result["score"].iloc[0] >= result["score"].iloc[-1]

    def test_select_empty_data(self):
        """测试空数据"""
        selector = CompositeSelector()
        data = pd.DataFrame()
        result = selector.select(data, top_n=10)
        assert len(result) == 0

    def test_get_factor_weights(self):
        """测试获取因子权重"""
        weights = {"technical": 0.5, "fundamental": 0.5}
        selector = CompositeSelector(weights=weights)
        assert selector.get_factor_weights() == weights


class TestXGBoostSelector:
    """XGBoost 选股器测试"""

    def test_train_and_select(self):
        """测试训练和选股"""
        from modules.selectors.xgboost_selector import XGBoostSelector

        selector = XGBoostSelector()
        np.random.seed(42)
        X = pd.DataFrame({
            "factor_a": np.random.randn(500),
            "factor_b": np.random.randn(500),
            "factor_c": np.random.randn(500),
        })
        y = pd.Series((np.random.rand(500) > 0.5).astype(int))

        selector.train(X, y)
        data = pd.DataFrame({
            "ts_code": [f"{i:06d}.SZ" for i in range(50)],
            "factor_a": np.random.randn(50),
            "factor_b": np.random.randn(50),
            "factor_c": np.random.randn(50),
        })
        result = selector.select(data, top_n=10)
        assert len(result) == 10
        assert "score" in result.columns
