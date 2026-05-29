"""资金流因子库"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class NorthboundFlowFactor:
    name = "northbound_flow"
    category = "moneyflow"
    description = "北向资金净流入"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["north_money"]

    def get_required_columns(self) -> List[str]:
        return ["north_money"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "north_money" in data.columns


class MainForceFlowFactor:
    name = "main_force_flow"
    category = "moneyflow"
    description = "主力资金净流入"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["net_mf_amount"]

    def get_required_columns(self) -> List[str]:
        return ["net_mf_amount"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "net_mf_amount" in data.columns


class VolumeRatioFactor:
    name = "volume_ratio"
    category = "moneyflow"
    description = "当日成交量 / 5日均量"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        if "volume_ratio" in data.columns:
            return data["volume_ratio"]
        # 量比 = 当日量 / 过去5日均量（分母不含当日）
        avg_vol = data["volume"].shift(1).rolling(window=5).mean()
        return data["volume"] / avg_vol.replace(0, np.nan)

    def get_required_columns(self) -> List[str]:
        return ["volume"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "volume" in data.columns or "volume_ratio" in data.columns


class TurnoverRateFactor:
    name = "turnover_rate"
    category = "moneyflow"
    description = "换手率"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["turnover_rate"]

    def get_required_columns(self) -> List[str]:
        return ["turnover_rate"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "turnover_rate" in data.columns


def register_moneyflow_factors(registry) -> None:
    registry.register(NorthboundFlowFactor())
    registry.register(MainForceFlowFactor())
    registry.register(VolumeRatioFactor())
    registry.register(TurnoverRateFactor())
