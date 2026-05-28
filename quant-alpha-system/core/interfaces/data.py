"""数据接口定义"""

from typing import Protocol, Dict, Any, Optional, List
import pandas as pd


class DataLoader(Protocol):
    """数据加载器接口"""

    def load_stock_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载个股日线数据"""
        ...

    def load_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载指数日线数据"""
        ...

    def load_sector_stocks(self, sector_code: str) -> pd.DataFrame:
        """加载板块成分股"""
        ...

    def load_factors(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载技术因子数据（stk_factor_pro 表）"""
        ...

    def get_available_stocks(self) -> List[str]:
        """获取可用股票列表"""
        ...


class DataValidator(Protocol):
    """数据验证器接口"""

    def validate(self, df: pd.DataFrame, data_type: str) -> bool:
        """验证数据完整性，返回 True/False"""
        ...

    def get_validation_report(self) -> Dict[str, Any]:
        """获取验证报告"""
        ...
