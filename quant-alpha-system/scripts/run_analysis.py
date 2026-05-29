#!/usr/bin/env python3
"""主分析入口 — 个股全面分析（真实 LSTM 预测 + 规则辅助诊断）"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline.context import Context
from core.pipeline.engine import PipelineEngine
from modules.data.quantdb_loader import QuantDBLoader
from modules.factors import init_factor_registry, FactorEngine
from modules.factors.preprocessor import FeaturePreprocessor
from modules.selectors.composite_selector import CompositeSelector
from modules.predictors.lstm_predictor import LSTMPredictor
from modules.labels import LabelGenerator
from modules.reporters.html_reporter import HTMLReporter


def main():
    parser = argparse.ArgumentParser(description="Quant Alpha 个股分析")
    parser.add_argument("--stock-code", required=True, help="股票代码, 如 600584.SH")
    parser.add_argument("--stock-name", default="", help="股票名称")
    parser.add_argument("--start-date", default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--output-dir", default="output/reports")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    import yaml
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_id = f"analysis_{args.stock_code.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ctx = Context(run_id=run_id, config=config)

    try:
        loader = QuantDBLoader(Path(config["data_dir"]))
        registry = init_factor_registry()
        factor_engine = FactorEngine(registry)
        reporter = HTMLReporter(ctx)

        pipeline = PipelineEngine(ctx)

        def load_data_step(ctx, loader, stock_code, start_date, end_date):
            stock_data = loader.load_stock_daily_enriched(stock_code, start_date, end_date)
            ctx.set_intermediate_result("stock_data", stock_data)
            return {"rows": len(stock_data)}

        def compute_factors_step(ctx, engine, loader):
            stock_data = ctx.intermediate_results["stock_data"]
            result = engine.compute_all_factors(stock_data)
            ctx.set_intermediate_result("factor_data", result)
            return {"factor_count": engine.registry.factor_count}

        def predict_step(ctx):
            """真实 LSTM 预测 + 规则辅助诊断"""
            stock_data = ctx.intermediate_results["stock_data"]
            factor_data = ctx.intermediate_results.get("factor_data", stock_data)

            if len(stock_data) < 60:
                pred = {
                    "daily_prob": [], "target_price": [], "trend": 0,
                    "win_rate": None, "action": "观望",
                    "reason": ["数据不足 (<60日)，无法生成有效预测。"],
                    "current_price": float(stock_data["close"].iloc[-1])
                }
                ctx.set_intermediate_result("predictions", pred)
                return {"status": "insufficient_data"}

            close = stock_data["close"].values
            current_price = float(close[-1])

            # ========== 真实 LSTM 预测 ==========
            exclude_cols = {"trade_date", "ts_code", "pre_close", "change", "pct_chg",
                           "open", "high", "low", "close", "volume", "vol", "amount",
                           "ann_date", "end_date", "q_sales_yoy", "name",
                           "is_tradable", "is_st", "up_limit", "down_limit"}
            factor_cols = [c for c in factor_data.columns if c not in exclude_cols
                          and factor_data[c].dtype in ['float64', 'float32', 'int64']]

            preprocessor = FeaturePreprocessor()
            scaled_data = preprocessor.fit_transform(factor_data, factor_cols)
            feature_matrix = preprocessor.get_feature_matrix(scaled_data, factor_cols)

            label_gen = LabelGenerator()
            fwd_returns = label_gen.forward_return(stock_data, horizon=5).values

            seq_len = 30
            X_seq, y_seq = label_gen.create_sequences(feature_matrix, fwd_returns, seq_len=seq_len)

            lstm_probs = []
            lstm_targets = []
            pred_return = 0.0
            pred_trend = 0.0

            if len(X_seq) > 50:
                split_idx = int(len(X_seq) * 0.8)
                X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]

                n_features = X_train.shape[2]
                lstm_params = config.get("lstm", {})
                lstm = LSTMPredictor(input_dim=n_features, params={
                    "hidden_dim": lstm_params.get("hidden_dim", 64),
                    "num_layers": lstm_params.get("num_layers", 2),
                    "epochs": lstm_params.get("epochs", 50),
                    "batch_size": lstm_params.get("batch_size", 32),
                    "patience": lstm_params.get("patience", 15),
                })

                y_train_3d = np.column_stack([
                    y_train,
                    np.sign(y_train),
                    1 + y_train
                ])

                try:
                    lstm.train(X_train, y_train_3d)

                    latest_features = feature_matrix[-seq_len:]
                    raw_pred = lstm.predict(latest_features)

                    pred_prob = float(raw_pred["daily_prob"][0])  # 已经是[0,1]概率
                    pred_trend = float(raw_pred["trend"][0])
                    pred_multiplier = float(raw_pred["target_price"][0])

                    pred_return = pred_prob  # 这是真实的上涨概率

                    # 7天递推预测：基于概率逐步衰减置信度
                    for i in range(1, 8):
                        # 距离预测越远，向50%回归越多
                        decay = 1.0 / (1.0 + i * 0.3)
                        day_prob = 0.5 + (pred_prob - 0.5) * decay
                        lstm_probs.append(round(day_prob * 100, 1))
                        day_target = current_price * (1 + (pred_multiplier - 1) * decay * i)
                        lstm_targets.append(round(day_target, 2))

                    ctx.add_marker("lstm_trained", True)
                    ctx.add_marker("lstm_pred_prob", pred_prob)
                except Exception as e:
                    ctx.add_error("lstm_predict", e)
                    ctx.add_marker("lstm_trained", False)
                    lstm_probs = [None] * 7
                    lstm_targets = [None] * 7
            else:
                ctx.add_marker("lstm_trained", False)
                lstm_probs = [None] * 7
                lstm_targets = [None] * 7

            # ========== 规则辅助诊断（补充解释性，不生成概率） ==========
            ma5 = close[-5:].mean()
            ma20 = close[-20:].mean()
            momentum = (close[-1] / close[-20]) - 1

            reasons = []
            action = "观望"

            if ma5 > ma20 and momentum > 0:
                reasons.append("短期均线(5日)上穿长期均线(20日)，动量强劲，多头趋势显著。")
                if lstm_probs[0] is not None and lstm_probs[0] > 55:
                    action = "买入"
                elif lstm_probs[0] is not None and lstm_probs[0] > 50:
                    action = "持有"
                else:
                    action = "观望"
            elif ma5 < ma20 and momentum < 0:
                reasons.append("短期均线跌破长期均线，处于空头弱势区间，有进一步下探风险。")
                action = "卖出"
            else:
                reasons.append("当前处于震荡洗盘阶段，趋势暂不明朗。")
                action = "观望"

            # 基本面辅助
            latest_stock = stock_data.iloc[-1]
            raw_pe = latest_stock.get("pe_ttm", 100)
            if pd.notna(raw_pe) and 0 < raw_pe < 20:
                reasons.append(f"估值较低（当前市盈率 {raw_pe:.1f} 倍），具备安全垫。")
            elif pd.notna(raw_pe) and raw_pe > 80:
                reasons.append(f"估值偏高（当前市盈率 {raw_pe:.1f} 倍），追高需警惕。")

            raw_profit_yoy = latest_stock.get("net_profit_yoy", 0)
            if pd.notna(raw_profit_yoy) and raw_profit_yoy > 20:
                reasons.append(f"净利润同比高增（{raw_profit_yoy:.1f}%），盈利驱动力强劲。")

            limit_t = latest_stock.get("limit_times", 0)
            if pd.notna(limit_t) and limit_t > 0:
                reasons.append(f"近期有 {int(limit_t)} 次涨停异动，股性活跃。")

            if ctx.markers.get("lstm_trained"):
                reasons.append(f"LSTM 深度学习模型已训练完成，预测 T+1 上涨概率: {lstm_probs[0]:.1f}%")
            else:
                reasons.append("LSTM 模型因数据量不足未能训练，暂无概率预测。")

            pred = {
                "daily_prob": lstm_probs,
                "target_price": lstm_targets,
                "trend": 1 if action == "买入" else (-1 if action == "卖出" else 0),
                "win_rate": None,  # 不再伪造胜率，需历史回测才能计算
                "action": action,
                "reason": reasons,
                "current_price": current_price
            }

            ctx.set_intermediate_result("predictions", pred)
            return {"status": "success"}

        def report_step(ctx, reporter, stock_code, stock_name, output_dir):
            stock_data = ctx.intermediate_results.get("stock_data")
            predictions = ctx.intermediate_results.get("predictions", {})
            if "factor_scores" not in predictions:
                predictions["factor_scores"] = {}
                
            if "factor_data" in ctx.intermediate_results:
                latest = ctx.intermediate_results["factor_data"].iloc[-1].to_dict()
                predictions["factor_scores"] = {k: float(v) if hasattr(v, 'item') else v for k, v in latest.items() if isinstance(v, (int, float))}

            report_data = {
                "stock_name": stock_name,
                "stock_code": stock_code,
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "stock_data": stock_data.tail(90).to_dict("records") if stock_data is not None else [],
                "predictions": predictions,
            }
            output_path = Path(output_dir) / f"{stock_name}_{stock_code}_分析报告_{datetime.now().strftime('%Y%m%d')}.html"
            report_path = str(reporter.generate_analysis_report(report_data, output_path))
            ctx.set_intermediate_result("report_path", report_path)
            return {"report_path": report_path}

        def compute_factors_wrapper(ctx):
            return compute_factors_step(ctx, factor_engine, loader)

        pipeline.add_step("load_data", load_data_step, loader=loader, stock_code=args.stock_code, start_date=args.start_date, end_date=args.end_date)
        pipeline.add_step("compute_factors", compute_factors_wrapper)
        pipeline.add_step("predict", predict_step)
        pipeline.add_step("report", report_step, reporter=reporter, stock_code=args.stock_code, stock_name=args.stock_name, output_dir=args.output_dir)

        pipeline.execute()
        report_path = ctx.intermediate_results.get("report_path")
        print(f"分析完成! 报告: {report_path}")

    finally:
        ctx.save_execution_report(Path(config.get("log_dir", "output/logs")))


if __name__ == "__main__":
    main()
