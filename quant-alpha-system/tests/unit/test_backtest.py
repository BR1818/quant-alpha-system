"""回测测试"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.metrics import calculate_metrics


class TestMetrics:
    """回测指标测试"""

    def test_calculate_metrics_basic(self):
        """测试基础指标计算"""
        returns = pd.Series(np.random.randn(252) * 0.01)
        metrics = calculate_metrics(returns)
        assert "total_return_pct" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "win_rate_pct" in metrics

    def test_calculate_metrics_with_benchmark(self):
        """测试含基准的指标计算"""
        returns = pd.Series(np.random.randn(252) * 0.01)
        benchmark = pd.Series(np.random.randn(252) * 0.008)
        metrics = calculate_metrics(returns, benchmark)
        assert "excess_return_pct" in metrics
        assert "information_ratio" in metrics

    def test_calculate_metrics_empty(self):
        """测试空数据"""
        returns = pd.Series([], dtype=float)
        metrics = calculate_metrics(returns)
        assert "error" in metrics
