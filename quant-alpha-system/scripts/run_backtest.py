#!/usr/bin/env python3
"""回测入口 — Backtrader 严谨回测（支持任意股票 + 大盘过滤 + ATR动态止损）"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline.context import Context
from modules.reporters.html_reporter import HTMLReporter


def main():
    parser = argparse.ArgumentParser(description="Quant Alpha 策略回测")
    parser.add_argument("--strategy", default="composite_selector", help="策略名称")
    parser.add_argument("--stock-code", default="600584.SH", help="回测标的股票代码")
    parser.add_argument("--start-date", default="20200101")
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

    run_id = f"backtest_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ctx = Context(run_id=run_id, config=config)

    import pandas as pd
    from modules.data.quantdb_loader import QuantDBLoader
    from modules.factors import init_factor_registry, FactorEngine
    from modules.selectors.composite_selector import CompositeSelector
    from modules.backtest.bt_engine import BTEngine

    try:
        reporter = HTMLReporter(ctx)

        stock_code = args.stock_code
        print(f"开始执行 Backtrader 严谨回测 (标的: {stock_code}, 期间: {args.start_date} -> {args.end_date})")

        loader = QuantDBLoader(Path(config["data_dir"]))
        stock_data = loader.load_stock_daily_enriched(stock_code, args.start_date, args.end_date)

        if len(stock_data) < 60:
            print("数据量不足，无法回测")
            return

        # 计算所有因子并打分
        registry = init_factor_registry()
        engine = FactorEngine(registry)
        factor_data = engine.compute_all_factors(stock_data)

        selector = CompositeSelector(registry=registry)
        scored_data = selector.select(factor_data, top_n=len(factor_data))
        stock_data = stock_data.merge(scored_data[["score"]], left_index=True, right_index=True, how="left")
        stock_data["score"] = stock_data["score"].fillna(0)

        # 加载基准数据 (沪深300)，可选
        benchmark_data = None
        try:
            benchmark_data = loader.load_index_daily("000300.SH", args.start_date, args.end_date)
        except FileNotFoundError:
            print("未找到沪深300基准数据，将跳过基准对比")

        # 运行 Backtrader 引擎
        bt_engine = BTEngine(config)
        backtest_results = bt_engine.run_backtest(stock_data, benchmark_df=benchmark_data)

        output_path = Path(args.output_dir) / f"回测报告_BT_{stock_code}_{datetime.now().strftime('%Y%m%d')}.html"
        reporter.generate_backtest_report(backtest_results, output_path)

        m = backtest_results["metrics"]
        print(f"\n======================================")
        print(f"回测标的: {stock_code}")
        print(f"交易次数: {m['trade_count']}")
        print(f"策略胜率: {m['win_rate_pct']:.2f}%")
        print(f"区间累计收益: {m['total_return_pct']:.2f}%")
        print(f"年化收益率(CAGR): {m['annual_return_pct']:.2f}%")
        print(f"最大回撤: {m['max_drawdown_pct']:.2f}%")
        print(f"夏普比率: {m['sharpe_ratio']:.2f}")
        if m.get('benchmark_annual_return_pct') is not None:
            print(f"基准年化收益: {m['benchmark_annual_return_pct']:.2f}%")
            print(f"超额收益: {m['excess_return_pct']:.2f}%")
            print(f"Alpha: {m['alpha']:.2f}  Beta: {m['beta']:.2f}")
        print(f"======================================")
        print(f"严谨回测报告已生成: {output_path}")

    finally:
        ctx.save_execution_report(Path(config.get("log_dir", "output/logs")))


if __name__ == "__main__":
    main()
