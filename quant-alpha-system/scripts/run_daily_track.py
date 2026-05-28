#!/usr/bin/env python3
"""每日跟踪入口"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline.context import Context
from modules.data.quantdb_loader import QuantDBLoader
from modules.factors import init_factor_registry, FactorEngine
from modules.reporters.html_reporter import HTMLReporter


def main():
    parser = argparse.ArgumentParser(description="Quant Alpha 每日跟踪")
    parser.add_argument("--stock-code", required=True, help="股票代码")
    parser.add_argument("--stock-name", default="", help="股票名称")
    parser.add_argument("--lookback-days", type=int, default=60, help="回顾天数")
    parser.add_argument("--output-dir", default="output/reports")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    import yaml
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    today = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=args.lookback_days)).strftime("%Y%m%d")
    run_id = f"track_{args.stock_code.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ctx = Context(run_id=run_id, config=config)

    try:
        loader = QuantDBLoader(Path(config["data_dir"]))
        registry = init_factor_registry()
        factor_engine = FactorEngine(registry)
        reporter = HTMLReporter(ctx)

        stock_data = loader.load_stock_daily_enriched(args.stock_code, start, today)
        factor_data = factor_engine.compute_all_factors(stock_data)

        latest = factor_data.iloc[-1].to_dict() if len(factor_data) > 0 else {}
        print(f"[{args.stock_code} {args.stock_name}] 最新交易日: {stock_data['trade_date'].iloc[-1] if len(stock_data) > 0 else 'N/A'}")

        output_path = Path(args.output_dir) / f"{args.stock_name}_{args.stock_code}_每日跟踪_{today}.html"
        report_data = {
            "stock_name": args.stock_name,
            "stock_code": args.stock_code,
            "analysis_date": today,
            "stock_data": stock_data.to_dict("records"),
            "predictions": {"factor_scores": {k: v for k, v in latest.items() if isinstance(v, float)}},
        }
        reporter.generate_analysis_report(report_data, output_path)
        print(f"跟踪报告已生成: {output_path}")

    finally:
        ctx.save_execution_report(Path(config.get("log_dir", "output/logs")))


if __name__ == "__main__":
    main()
