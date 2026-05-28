"""技术因子库"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class MAFactor:
    name = "ma"
    category = "technical"
    description = "N日移动平均线偏离度"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        period = (params or {}).get("period", 20)
        ma = data["close"].rolling(window=period).mean()
        return (data["close"] - ma) / ma

    def get_required_columns(self) -> List[str]:
        return ["close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "close" in data.columns


class MACDFactor:
    name = "macd"
    category = "technical"
    description = "MACD 柱状图值"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        p = params or {}
        fast = p.get("fast", 12)
        slow = p.get("slow", 26)
        signal = p.get("signal", 9)

        if HAS_TALIB:
            import talib
            _, _, hist = talib.MACD(data["close"], fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return pd.Series(hist, index=data.index)
        else:
            ema_fast = data["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = data["close"].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            return macd_line - signal_line

    def get_required_columns(self) -> List[str]:
        return ["close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "close" in data.columns


class RSIFactor:
    name = "rsi"
    category = "technical"
    description = "相对强弱指数"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        period = (params or {}).get("period", 14)
        if HAS_TALIB:
            import talib
            rsi = talib.RSI(data["close"], timeperiod=period)
            return pd.Series(rsi, index=data.index)
        else:
            delta = data["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            return 100.0 - (100.0 / (1.0 + rs))

    def get_required_columns(self) -> List[str]:
        return ["close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "close" in data.columns


class BollingerFactor:
    name = "bollinger"
    category = "technical"
    description = "布林带宽度百分比"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        p = params or {}
        period = p.get("period", 20)
        std_dev = p.get("std_dev", 2)
        middle = data["close"].rolling(window=period).mean()
        std = data["close"].rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return (upper - lower) / middle.replace(0, np.nan)

    def get_required_columns(self) -> List[str]:
        return ["close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return "close" in data.columns


class ATRFactor:
    name = "atr"
    category = "technical"
    description = "平均真实波幅"

    def compute(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        period = (params or {}).get("period", 14)
        if HAS_TALIB:
            import talib
            atr = talib.ATR(data["high"], data["low"], data["close"], timeperiod=period)
            return pd.Series(atr, index=data.index)
        else:
            high, low, close = data["high"], data["low"], data["close"]
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=period).mean()

    def get_required_columns(self) -> List[str]:
        return ["high", "low", "close"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        return all(c in data.columns for c in self.get_required_columns())


def register_technical_factors(registry) -> None:
    registry.register(MAFactor())
    registry.register(MACDFactor())
    registry.register(RSIFactor())
    registry.register(BollingerFactor())
    registry.register(ATRFactor())
