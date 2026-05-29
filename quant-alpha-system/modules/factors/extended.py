"""动量因子库 — A股最核心的Alpha来源"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class Return5dFactor:
    name = "return_5d"
    category = "technical"
    description = "过去5日收益率"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["close"].pct_change(5)

    def get_required_columns(self) -> List[str]:
        return ["close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "close" in data.columns


class Return20dFactor:
    name = "return_20d"
    category = "technical"
    description = "过去20日收益率"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["close"].pct_change(20)

    def get_required_columns(self) -> List[str]:
        return ["close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "close" in data.columns


class Return60dFactor:
    name = "return_60d"
    category = "technical"
    description = "过去60日收益率"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["close"].pct_change(60)

    def get_required_columns(self) -> List[str]:
        return ["close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "close" in data.columns


class RealizedVol20dFactor:
    name = "realized_vol_20d"
    category = "technical"
    description = "20日已实现波动率"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        log_ret = np.log(data["close"] / data["close"].shift(1))
        return log_ret.rolling(20).std() * np.sqrt(252)

    def get_required_columns(self) -> List[str]:
        return ["close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "close" in data.columns


class IntradayRangeFactor:
    name = "intraday_range"
    category = "technical"
    description = "日内振幅 (high-low)/close"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return (data["high"] - data["low"]) / data["close"].replace(0, np.nan)

    def get_required_columns(self) -> List[str]:
        return ["high", "low", "close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return all(c in data.columns for c in self.get_required_columns())


class OBVFactor:
    name = "obv"
    category = "moneyflow"
    description = "能量潮指标 (On-Balance Volume)"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        direction = np.sign(data["close"].diff())
        obv = (direction * data["volume"]).cumsum()
        # 标准化为百分比变化
        return obv.pct_change(5)

    def get_required_columns(self) -> List[str]:
        return ["close", "volume"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return all(c in data.columns for c in self.get_required_columns())


class VolumeMA20RatioFactor:
    name = "volume_ma20_ratio"
    category = "moneyflow"
    description = "当日量/20日均量"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        # 当日量 / 过去20日均量（分母不含当日）
        avg_vol = data["volume"].shift(1).rolling(window=20).mean()
        return data["volume"] / avg_vol.replace(0, np.nan)

    def get_required_columns(self) -> List[str]:
        return ["volume"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "volume" in data.columns


class PriceVolumeCorrFactor:
    name = "price_volume_corr"
    category = "moneyflow"
    description = "近20日量价相关系数"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return data["close"].rolling(20).corr(data["volume"])

    def get_required_columns(self) -> List[str]:
        return ["close", "volume"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return all(c in data.columns for c in self.get_required_columns())


class AmihudIlliquidityFactor:
    name = "amihud_illiquidity"
    category = "moneyflow"
    description = "Amihud非流动性因子"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        abs_ret = data["close"].pct_change().abs()
        amount = data["amount"].replace(0, np.nan)
        daily_illiq = abs_ret / amount
        return daily_illiq.rolling(20).mean()

    def get_required_columns(self) -> List[str]:
        return ["close", "amount"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return all(c in data.columns for c in self.get_required_columns())


class KDJFactor:
    name = "kdj_k"
    category = "technical"
    description = "KDJ随机指标-K值"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        period = (params or {}).get("period", 9)
        low_min = data["low"].rolling(period).min()
        high_max = data["high"].rolling(period).max()
        rsv = (data["close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        return k

    def get_required_columns(self) -> List[str]:
        return ["high", "low", "close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return all(c in data.columns for c in self.get_required_columns())


class CCIFactor:
    name = "cci"
    category = "technical"
    description = "顺势指标 CCI"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        period = (params or {}).get("period", 20)
        tp = (data["high"] + data["low"] + data["close"]) / 3
        ma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return (tp - ma_tp) / (0.015 * mad).replace(0, np.nan)

    def get_required_columns(self) -> List[str]:
        return ["high", "low", "close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return all(c in data.columns for c in self.get_required_columns())


class WilliamsRFactor:
    name = "williams_r"
    category = "technical"
    description = "威廉指标 %R"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        period = (params or {}).get("period", 14)
        high_max = data["high"].rolling(period).max()
        low_min = data["low"].rolling(period).min()
        return (high_max - data["close"]) / (high_max - low_min).replace(0, np.nan) * -100

    def get_required_columns(self) -> List[str]:
        return ["high", "low", "close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return all(c in data.columns for c in self.get_required_columns())


class TotalMVFactor:
    name = "total_mv"
    category = "fundamental"
    description = "总市值(对数)"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return np.log(data["total_mv"].replace(0, np.nan))

    def get_required_columns(self) -> List[str]:
        return ["total_mv"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "total_mv" in data.columns


class CircMVFactor:
    name = "circ_mv"
    category = "fundamental"
    description = "流通市值(对数)"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        return np.log(data["circ_mv"].replace(0, np.nan))

    def get_required_columns(self) -> List[str]:
        return ["circ_mv"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "circ_mv" in data.columns


def register_extended_factors(registry) -> None:
    """注册所有扩展因子"""
    # 动量 & 波动
    registry.register(Return5dFactor())
    registry.register(Return20dFactor())
    registry.register(Return60dFactor())
    registry.register(RealizedVol20dFactor())
    registry.register(IntradayRangeFactor())
    # 量价
    registry.register(OBVFactor())
    registry.register(VolumeMA20RatioFactor())
    registry.register(PriceVolumeCorrFactor())
    registry.register(AmihudIlliquidityFactor())
    # 技术扩展
    registry.register(KDJFactor())
    registry.register(CCIFactor())
    registry.register(WilliamsRFactor())
    # 市值
    registry.register(TotalMVFactor())
    registry.register(CircMVFactor())
