"""策略接口定义"""

from typing import Protocol, Dict, Any, Optional
import pandas as pd


class StockSelector(Protocol):
    """选股器接口"""

    @property
    def name(self) -> str:
        """选股器名称"""
        ...

    @property
    def description(self) -> str:
        """选股器描述"""
        ...

    def select(self, data: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """选股，返回排序后的股票 DataFrame，含 score 列"""
        ...

    def get_factor_weights(self) -> Dict[str, float]:
        """获取因子权重"""
        ...
