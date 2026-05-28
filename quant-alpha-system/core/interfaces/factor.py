"""因子接口定义"""

from typing import Protocol, Dict, Any, List, Optional
import pandas as pd


class Factor(Protocol):
    """因子计算接口"""

    @property
    def name(self) -> str:
        """因子名称"""
        ...

    @property
    def category(self) -> str:
        """因子类别: technical / fundamental / moneyflow / sentiment"""
        ...

    @property
    def description(self) -> str:
        """因子描述"""
        ...

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """计算因子值"""
        ...

    def get_required_columns(self) -> List[str]:
        """获取计算所需的输入列"""
        ...

    def validate_input(self, data: pd.DataFrame) -> bool:
        """验证输入数据是否包含所需列"""
        ...


class FactorRegistry:
    """因子注册器 — 管理所有已注册因子"""

    def __init__(self):
        self._factors: Dict[str, Factor] = {}

    def register(self, factor: Factor) -> None:
        """注册因子"""
        self._factors[factor.name] = factor

    def get(self, name: str) -> Optional[Factor]:
        """按名称获取因子"""
        return self._factors.get(name)

    def list_factors(self, category: Optional[str] = None) -> List[str]:
        """列出因子名称，可按类别筛选"""
        if category:
            return [name for name, f in self._factors.items() if f.category == category]
        return list(self._factors.keys())

    def get_factor_info(self) -> List[Dict[str, Any]]:
        """获取所有因子信息"""
        return [
            {
                "name": f.name,
                "category": f.category,
                "description": f.description,
                "required_columns": f.get_required_columns(),
            }
            for f in self._factors.values()
        ]

    @property
    def factor_count(self) -> int:
        """已注册因子数量"""
        return len(self._factors)
