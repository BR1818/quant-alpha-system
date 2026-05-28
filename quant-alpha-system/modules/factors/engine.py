"""因子计算引擎"""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from core.interfaces.factor import FactorRegistry


class FactorEngine:
    """因子计算引擎 — 批量计算因子并记录过程"""

    def __init__(self, registry: FactorRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)

    def compute_factors(
        self,
        data: pd.DataFrame,
        factor_names: List[str],
        params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """计算指定因子列表"""
        self.logger.info(f"开始计算 {len(factor_names)} 个因子")
        result = data.copy()
        computed = []
        failed = []

        for factor_name in factor_names:
            factor = self.registry.get(factor_name)
            if factor is None:
                self.logger.warning(f"因子不存在: {factor_name}")
                failed.append(factor_name)
                continue

            try:
                if not factor.validate_input(data):
                    raise ValueError(f"输入验证失败: {factor_name}")

                factor_params = (params or {}).get(factor_name, {})
                t0 = datetime.now()

                factor_values = factor.compute(data, factor_params)
                result[factor_name] = factor_values

                elapsed = (datetime.now() - t0).total_seconds()
                self.logger.debug(f"因子 {factor_name} 计算完成 ({elapsed:.3f}s)")
                computed.append(factor_name)

            except Exception as e:
                self.logger.error(f"因子 {factor_name} 计算失败: {e}")
                failed.append(factor_name)

        self.logger.info(f"因子计算完成: {len(computed)} 成功, {len(failed)} 失败")
        return result

    def compute_all_factors(self, data: pd.DataFrame, category: Optional[str] = None) -> pd.DataFrame:
        """计算所有已注册因子（可按类别筛选）"""
        factor_names = self.registry.list_factors(category)
        return self.compute_factors(data, factor_names)
