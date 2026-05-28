"""因子系统测试"""
import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.interfaces.factor import FactorRegistry
from modules.factors import init_factor_registry
from modules.factors.engine import FactorEngine


class TestFactorRegistry:
    def test_init_registry(self):
        registry = init_factor_registry()
        factors = registry.list_factors()
        assert len(factors) >= 12
        assert "ma" in factors
        assert "rsi" in factors
        assert "pe_ttm" in factors
        assert "northbound_flow" in factors

    def test_list_by_category(self):
        registry = init_factor_registry()
        tech_factors = registry.list_factors("technical")
        assert len(tech_factors) >= 5
        assert "ma" in tech_factors
        assert "macd" in tech_factors
        fund_factors = registry.list_factors("fundamental")
        assert len(fund_factors) >= 4

    def test_get_factor_info(self):
        registry = init_factor_registry()
        info_list = registry.get_factor_info()
        assert len(info_list) >= 12
        for info in info_list:
            assert "name" in info
            assert "category" in info
            assert "description" in info


class TestFactorEngine:
    def test_compute_single_factor(self):
        registry = init_factor_registry()
        engine = FactorEngine(registry)
        data = pd.DataFrame({
            "close": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 102,
            "low": np.random.randn(100) + 98,
            "open": np.random.randn(100) + 100,
            "volume": np.random.randint(1000, 100000, 100),
        })
        result = engine.compute_factors(data, ["ma"])
        assert "ma" in result.columns

    def test_compute_multiple_factors(self):
        registry = init_factor_registry()
        engine = FactorEngine(registry)
        data = pd.DataFrame({
            "close": np.random.randn(200) + 100,
            "high": np.random.randn(200) + 102,
            "low": np.random.randn(200) + 98,
            "open": np.random.randn(200) + 100,
            "volume": np.random.randint(1000, 100000, 200),
            "pe_ttm": np.abs(np.random.randn(200)) * 20 + 15,
            "pb": np.abs(np.random.randn(200)) * 3 + 1,
            "roe": np.random.randn(200) * 5 + 10,
            "revenue_yoy": np.random.randn(200) * 10 + 5,
            "north_money": np.random.randn(200) * 100,
            "net_mf_amount": np.random.randn(200) * 50,
            "turnover_rate": np.abs(np.random.randn(200)) * 3,
        })
        technical = registry.list_factors("technical")
        result = engine.compute_factors(data, technical)
        for f in technical:
            assert f in result.columns
