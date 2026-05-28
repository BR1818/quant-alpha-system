"""因子模块"""

from core.interfaces.factor import FactorRegistry
from modules.factors.engine import FactorEngine
from modules.factors.technical import register_technical_factors
from modules.factors.fundamental import register_fundamental_factors
from modules.factors.moneyflow import register_moneyflow_factors
from modules.factors.sentiment import register_sentiment_factors
from modules.factors.extended import register_extended_factors


def init_factor_registry() -> FactorRegistry:
    """初始化因子注册器，注册所有内置因子"""
    registry = FactorRegistry()
    register_technical_factors(registry)
    register_fundamental_factors(registry)
    register_moneyflow_factors(registry)
    register_sentiment_factors(registry)
    register_extended_factors(registry)
    return registry


__all__ = [
    "FactorRegistry",
    "FactorEngine",
    "init_factor_registry",
]
