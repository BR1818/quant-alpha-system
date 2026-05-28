"""数据验证器 — 验证加载数据的完整性和正确性"""

import pandas as pd
from typing import Dict, Any, List
import logging
from datetime import datetime


class DataValidator:
    """数据验证器，支持多种数据类型的验证规则"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._results: List[Dict[str, Any]] = []
        self._rules = {
            "stock_daily": self._validate_stock_daily,
            "index_daily": self._validate_index_daily,
            "factor": self._validate_factor,
            "sector_member": self._validate_sector_member,
        }

    def validate(self, df: pd.DataFrame, data_type: str) -> bool:
        """验证数据完整性"""
        self.logger.info(f"验证 {data_type} 数据, 行数: {len(df)}")

        result = {
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "row_count": len(df),
            "checks": [],
        }

        if df.empty:
            result["checks"].append({"check": "non_empty", "passed": False, "message": "DataFrame is empty"})
            self._results.append(result)
            return False

        validator_fn = self._rules.get(data_type)
        if validator_fn:
            checks = validator_fn(df)
            result["checks"].extend(checks)
        else:
            result["checks"].append({"check": "unknown_type", "passed": False, "message": f"未知数据类型: {data_type}"})

        self._results.append(result)
        passed = all(c["passed"] for c in result["checks"])
        if not passed:
            self.logger.warning(f"数据验证失败: {data_type}")
        return passed

    def _validate_stock_daily(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """验证个股日线数据"""
        checks = []
        required_cols = ["trade_date", "open", "high", "low", "close", "volume", "amount"]
        missing = [c for c in required_cols if c not in df.columns]
        checks.append({
            "check": "required_columns",
            "passed": len(missing) == 0,
            "message": f"缺失列: {missing}" if missing else "所有必需列存在",
        })

        if all(c in df.columns for c in ["high", "low", "open", "close"]):
            valid_price = (
                (df["high"] >= df["low"])
                & (df["high"] >= df["open"])
                & (df["high"] >= df["close"])
                & (df["low"] <= df["open"])
                & (df["low"] <= df["close"])
            )
            checks.append({
                "check": "price_logic",
                "passed": valid_price.all(),
                "message": f"价格逻辑异常: {(~valid_price).sum()} 行" if not valid_price.all() else "价格逻辑正常",
            })

        if "volume" in df.columns:
            checks.append({
                "check": "volume_non_negative",
                "passed": (df["volume"] >= 0).all(),
                "message": "成交量正常",
            })

        return checks

    def _validate_index_daily(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """验证指数日线数据"""
        return self._validate_stock_daily(df)

    def _validate_factor(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """验证因子数据"""
        checks = []
        if "ts_code" not in df.columns:
            checks.append({"check": "has_ts_code", "passed": False, "message": "缺少 ts_code 列"})
        else:
            checks.append({"check": "has_ts_code", "passed": True, "message": "OK"})

        if "trade_date" not in df.columns:
            checks.append({"check": "has_trade_date", "passed": False, "message": "缺少 trade_date 列"})
        else:
            checks.append({"check": "has_trade_date", "passed": True, "message": "OK"})

        return checks

    def _validate_sector_member(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """验证板块成分数据"""
        checks = []
        if "con_code" not in df.columns:
            checks.append({"check": "has_con_code", "passed": False, "message": "缺少 con_code 列"})
        else:
            checks.append({"check": "has_con_code", "passed": True, "message": "OK"})
        return checks

    def get_validation_report(self) -> Dict[str, Any]:
        """获取验证报告"""
        total = len(self._results)
        passed = sum(1 for r in self._results if all(c["passed"] for c in r.get("checks", [])))
        return {
            "total_validations": total,
            "passed": passed,
            "failed": total - passed,
            "details": self._results,
        }
