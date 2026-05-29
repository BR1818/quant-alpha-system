"""回测引擎测试"""
import sys
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.backtest.bt_engine import BTEngine


def _make_test_df(days: int = 200) -> pd.DataFrame:
    """生成回测用的测试DataFrame"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    close = 10.0 + np.cumsum(np.random.randn(days) * 0.1)
    return pd.DataFrame({
        "trade_date": dates,
        "open": close + np.random.randn(days) * 0.05,
        "high": close + abs(np.random.randn(days) * 0.1),
        "low": close - abs(np.random.randn(days) * 0.1),
        "close": close,
        "vol": np.random.randint(1000, 10000, days),
        "amount": close * 5000,
        "score": np.random.rand(days),
    })


class TestBTEngine:
    """回测引擎测试"""

    def test_run_backtest_returns_metrics(self):
        """回测应返回包含关键指标的dict"""
        df = _make_test_df()
        engine = BTEngine({"initial_cash": 100000})
        metrics, _ = engine.run_backtest(df)
        assert "total_return_pct" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "win_rate_pct" in metrics
