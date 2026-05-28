#!/usr/bin/env python3
"""训练 XGBoost 截面选股模型"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline.context import Context
from core.pipeline.engine import PipelineEngine
from modules.data.quantdb_loader import QuantDBLoader
from modules.factors import init_factor_registry, FactorEngine
from modules.selectors.xgboost_selector import XGBoostSelector
from modules.labels import LabelGenerator


def main():
    parser = argparse.ArgumentParser(description="训练 XGBoost 选股模型")
    parser.add_argument("--sector-code", default="000300.SH", help="用来训练的股票池 (默认沪深300)")
    parser.add_argument("--start-date", default="20200101")
    parser.add_argument("--end-date", default="20251231")
    parser.add_argument("--model-path", default="output/models/xgboost_selector.pkl")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    import yaml
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ctx = Context(run_id=f"train_xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}", config=config)
    
    loader = QuantDBLoader(Path(config["data_dir"]))
    registry = init_factor_registry()
    engine = FactorEngine(registry)
    label_gen = LabelGenerator()

    # 1. 加载股票池
    print(f"正在加载训练数据池: {args.sector_code} ...")
    sector_stocks = loader.load_sector_stocks(args.sector_code)
    if sector_stocks is None or "con_code" not in sector_stocks.columns:
        stock_codes = ["600584.SH", "600519.SH", "000001.SZ", "000858.SZ", "601318.SH"] # Fallback
    else:
        stock_codes = sector_stocks["con_code"].tolist()[:50] # 训练前50只作演示

    # 2. 计算特征和标签
    all_data = []
    print(f"正在计算特征和标签 (总计 {len(stock_codes)} 只股票) ...")
    
    for idx, ts_code in enumerate(stock_codes):
        if idx % 10 == 0:
            print(f"  处理进度: {idx}/{len(stock_codes)}")
        try:
            stock_data = loader.load_stock_daily_enriched(ts_code, args.start_date, args.end_date)
            if len(stock_data) < 100:
                continue
                
            # 计算因子
            factor_data = engine.compute_all_factors(stock_data)
            
            # 生成标签 (未来 5 日是否上涨)
            labels = label_gen.binary_label(stock_data, horizon=5)
            
            # 合并数据
            df = factor_data.copy()
            df["label"] = labels
            df["ts_code"] = ts_code
            df["trade_date"] = stock_data["trade_date"]
            
            # 丢弃最后几天的 NaN 标签
            df = df.dropna(subset=["label"])
            all_data.append(df)
            
        except Exception as e:
            print(f"  跳过 {ts_code}: {e}")
            
    if not all_data:
        print("没有成功加载的数据！")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.sort_values("trade_date")
    
    # 提取特征列
    exclude_cols = {"trade_date", "ts_code", "pre_close", "change", "pct_chg", 
                    "open", "high", "low", "close", "volume", "vol", "amount",
                    "ann_date", "end_date", "q_sales_yoy", "name", "label"}
    feature_cols = [c for c in full_df.columns if c not in exclude_cols and full_df[c].dtype in ['float64', 'float32', 'int64']]

    print(f"\n数据集准备完毕. 总样本数: {len(full_df)}, 特征维度: {len(feature_cols)}")

    # 3. 按时间拆分 Train / Test (Walk-Forward 思想: 前 80% 时间训练，后 20% 测试)
    train_size = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:train_size]
    test_df = full_df.iloc[train_size:]

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    print(f"训练集区间: {train_df['trade_date'].min()} -> {train_df['trade_date'].max()} ({len(train_df)} 样本)")
    print(f"测试集区间: {test_df['trade_date'].min()} -> {test_df['trade_date'].max()} ({len(test_df)} 样本)")

    # 4. 训练模型
    print("\n开始训练 XGBoost 模型 ...")
    selector = XGBoostSelector()
    selector.train(X_train, y_train)

    # 5. 评估模型
    print("\n评估模型表现:")
    y_pred_proba = selector.model.predict_proba(X_test.fillna(0).replace([np.inf, -np.inf], 0))[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test AUC: {auc:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 6. 保存模型
    out_path = Path(args.model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selector.save_model(out_path)
    
if __name__ == "__main__":
    main()
