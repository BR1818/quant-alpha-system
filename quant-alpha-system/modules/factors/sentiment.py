"""情绪因子库"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class LimitUpCountFactor:
    name = "limit_up_count"
    category = "sentiment"
    description = "连板次数 (limit_times)"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data.get("limit_times", pd.Series(0, index=data.index))

    def get_required_columns(self) -> List[str]:
        return ["limit_times"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "limit_times" in data.columns


class TopListNetBuyFactor:
    name = "top_list_net_buy"
    category = "sentiment"
    description = "龙虎榜净买入金额"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data.get("top_list_net_amount", pd.Series(0, index=data.index))

    def get_required_columns(self) -> List[str]:
        return ["top_list_net_amount"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "top_list_net_amount" in data.columns


class MarginBalanceFactor:
    name = "margin_balance"
    category = "sentiment"
    description = "融资余额变化率"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        if "rzye" in data.columns:
            return data["rzye"].pct_change(5)
        return pd.Series(0, index=data.index)

    def get_required_columns(self) -> List[str]:
        return ["rzye"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "rzye" in data.columns


def register_sentiment_factors(registry) -> None:
    registry.register(LimitUpCountFactor())
    registry.register(TopListNetBuyFactor())
    registry.register(MarginBalanceFactor())
