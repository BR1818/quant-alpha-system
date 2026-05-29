"""基本面因子库"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class PE_TTM_Factor:
    name = "pe_ttm"
    category = "fundamental"
    description = "滚动市盈率倒数（盈利收益率）"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        pe = data["pe_ttm"].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
        ep = 1.0 / pe
        # 亏损公司PE为负，取倒数后语义反转，设为NaN
        ep[pe < 0] = np.nan
        return ep

    def get_required_columns(self) -> List[str]:
        return ["pe_ttm"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "pe_ttm" in data.columns


class PB_Factor:
    name = "pb"
    category = "fundamental"
    description = "市净率倒数"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        pb = data["pb"].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
        bp = 1.0 / pb
        # 资不抵债公司PB为负，取倒数后语义反转，设为NaN
        bp[pb < 0] = np.nan
        return bp

    def get_required_columns(self) -> List[str]:
        return ["pb"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "pb" in data.columns


class ROE_Factor:
    name = "roe"
    category = "fundamental"
    description = "净资产收益率"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["roe"]

    def get_required_columns(self) -> List[str]:
        return ["roe"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "roe" in data.columns


class RevenueGrowthFactor:
    name = "revenue_growth"
    category = "fundamental"
    description = "营收同比增长率"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["revenue_yoy"]

    def get_required_columns(self) -> List[str]:
        return ["revenue_yoy"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "revenue_yoy" in data.columns


class DividendYieldFactor:
    name = "dividend_yield"
    category = "fundamental"
    description = "股息率 (dv_ratio)"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["dv_ratio"]

    def get_required_columns(self) -> List[str]:
        return ["dv_ratio"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "dv_ratio" in data.columns


class PS_TTM_Factor:
    name = "ps_ttm"
    category = "fundamental"
    description = "市销率倒数"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        ps = data["ps_ttm"].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
        return 1.0 / ps

    def get_required_columns(self) -> List[str]:
        return ["ps_ttm"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "ps_ttm" in data.columns


class NetProfitYoyFactor:
    name = "net_profit_yoy"
    category = "fundamental"
    description = "净利润同比增长率"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["net_profit_yoy"]

    def get_required_columns(self) -> List[str]:
        return ["net_profit_yoy"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "net_profit_yoy" in data.columns


def register_fundamental_factors(registry) -> None:
    registry.register(PE_TTM_Factor())
    registry.register(PB_Factor())
    registry.register(ROE_Factor())
    registry.register(RevenueGrowthFactor())
    registry.register(DividendYieldFactor())
    registry.register(PS_TTM_Factor())
    registry.register(NetProfitYoyFactor())
