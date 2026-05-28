"""综合打分选股器 — 按因子类别加权打分"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging


class CompositeSelector:
    """基于多维度综合加权的选股器，无需训练，即开即用"""

    name = "composite_selector"
    description = "综合多维度因子加权打分选股器"

    def __init__(self, weights: Optional[Dict[str, float]] = None, registry=None):
        self.weights = weights or {
            "technical": 0.30,
            "fundamental": 0.25,
            "moneyflow": 0.25,
            "sentiment": 0.20,
        }
        self.registry = registry
        self.logger = logging.getLogger(__name__)

    def select(self, data: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """综合打分选股"""
        self.logger.info(f"综合打分选股: top_{top_n}, 候选数: {len(data)}")
        result = data.copy()

        score_parts = []
        for category, weight in self.weights.items():
            if self.registry:
                category_cols = [name for name in self.registry.list_factors(category) if name in data.columns]
            else:
                category_cols = [c for c in data.columns if c.startswith(category)]
                
            if not category_cols:
                self.logger.debug(f"类别 {category} 无可用列，跳过")
                continue

            subset = data[category_cols].copy()
            for col in category_cols:
                # 截面标准化：使用秩排序 (Rank) 来防止极值影响 (去极值+标准化)
                # 缺失值填充为中位数(0.5)
                subset[col] = subset[col].rank(pct=True, na_option='keep')
                subset[col] = subset[col].fillna(0.5)

            cat_score = subset.mean(axis=1) * weight
            score_parts.append(cat_score)

        if not score_parts:
            result["score"] = 0.0
        else:
            result["score"] = pd.concat(score_parts, axis=1).sum(axis=1)

        result = result.nlargest(top_n, "score").copy()
        self.logger.info(f"选股完成: {len(result)} 只, 得分范围 [{result['score'].min():.4f}, {result['score'].max():.4f}]")
        return result

    def get_factor_weights(self) -> Dict[str, float]:
        return self.weights
