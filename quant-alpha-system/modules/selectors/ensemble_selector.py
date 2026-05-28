"""集成选股器 — 融合 XGBoost、LSTM 和传统 Composite 打分"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from modules.selectors.composite_selector import CompositeSelector
from modules.selectors.xgboost_selector import XGBoostSelector
from modules.predictors.lstm_predictor import LSTMPredictor


class EnsembleSelector:
    """融合模型：0.5 * XGBoost + 0.3 * LSTM + 0.2 * Composite"""

    name = "ensemble_selector"
    description = "多轨并行集成选股器 (XGBoost + LSTM + 规则综合打分)"

    def __init__(self, registry=None, xgb_model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.registry = registry
        
        # 初始化基础模型
        self.composite = CompositeSelector(registry=registry)
        
        self.xgb = XGBoostSelector()
        self.xgb_loaded = False
        if xgb_model_path and Path(xgb_model_path).exists():
            try:
                self.xgb.load_model(Path(xgb_model_path))
                self.xgb_loaded = True
            except Exception as e:
                self.logger.error(f"无法加载 XGBoost 模型: {e}")

        # LSTM 选股通常针对单只股票，在截面选股中若需全量运行会非常慢
        # 为了生产可用，截面选股这里LSTM以快速打分为主（如果有提前计算好的结果）
        # 这里仅作接口预留，若有传入序列化模型则可支持
        self.lstm_cache = {}

    def select(self, data: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """执行集成选股"""
        if data.empty:
            return data

        self.logger.info(f"开始集成选股: 候选池 {len(data)}")
        result = data.copy()
        
        # 1. 传统多因子打分 (Composite)
        comp_df = self.composite.select(data, top_n=len(data))
        comp_scores = comp_df["score"]
        
        # 2. XGBoost 打分
        xgb_scores = pd.Series(0.0, index=data.index)
        if self.xgb_loaded:
            try:
                # 提取需要的特征，缺失补0
                X = data[self.xgb._feature_names].fillna(0).replace([np.inf, -np.inf], 0)
                probs = self.xgb.model.predict_proba(X)[:, 1]
                xgb_scores = pd.Series(probs, index=data.index)
            except Exception as e:
                self.logger.warning(f"XGBoost 预测失败，降级为 0: {e}")
        else:
            self.logger.debug("XGBoost 未加载，权重置0")

        # 3. LSTM 打分 (目前由于LSTM需要时间序列，截面上暂用0占位或从缓存读取)
        lstm_scores = pd.Series(0.0, index=data.index)
        
        # 模型融合权重
        if self.xgb_loaded:
            # Phase 3 规划: 0.5 * XGB + 0.3 * LSTM + 0.2 * Composite
            # 在没有 LSTM 截面数据时：0.7 * XGB + 0.3 * Composite
            final_scores = 0.7 * xgb_scores + 0.3 * comp_scores
        else:
            final_scores = comp_scores

        result["score"] = final_scores
        result["xgb_score"] = xgb_scores
        result["comp_score"] = comp_scores
        
        result = result.nlargest(top_n, "score").copy()
        self.logger.info(f"集成选股完成: 选出 {len(result)} 只股票, 综合得分范围 [{result['score'].min():.4f}, {result['score'].max():.4f}]")
        return result
