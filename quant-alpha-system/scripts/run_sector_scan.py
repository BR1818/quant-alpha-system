#!/usr/bin/env python3
"""板块选股入口"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline.context import Context
from core.pipeline.engine import PipelineEngine
from modules.data.quantdb_loader import QuantDBLoader
from modules.factors import init_factor_registry, FactorEngine
from modules.selectors.composite_selector import CompositeSelector
from modules.selectors.ensemble_selector import EnsembleSelector
from modules.reporters.html_reporter import HTMLReporter


def main():
    parser = argparse.ArgumentParser(description="Quant Alpha 板块选股")
    parser.add_argument("--sector-code", required=True, help="板块代码")
    parser.add_argument("--sector-name", default="", help="板块名称")
    parser.add_argument("--top-n", type=int, default=20, help="选股数量")
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

    run_id = f"sector_{args.sector_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ctx = Context(run_id=run_id, config=config)

    try:
        loader = QuantDBLoader(Path(config["data_dir"]))
        factor_engine = FactorEngine(registry)
        model_path = str(Path(config.get("model_dir", "output/models")) / "xgboost_selector.pkl")
        selector = EnsembleSelector(registry=registry, xgb_model_path=model_path)
        reporter = HTMLReporter(ctx)

        pipeline = PipelineEngine(ctx)

        def load_sector_step(ctx, loader, sector_code):
            sector_data = loader.load_sector_stocks(sector_code)
            ctx.set_intermediate_result("sector_stocks", sector_data)
            ctx.add_marker("sector_stock_count", len(sector_data))
            return {"stocks": len(sector_data)}

        def scan_step(ctx, loader, engine, selector, top_n, start_date, end_date):
            sector_stocks = ctx.intermediate_results.get("sector_stocks")
            if sector_stocks is None or "con_code" not in sector_stocks.columns:
                return {"error": "no_sector_data"}

            stock_codes = sector_stocks["con_code"].tolist()[:100]
            all_scores = []
            for ts_code in stock_codes:
                try:
                    # 使用 _enriched 确保带有基本面和资金流数据，否则只能计算技术因子
                    stock_data = loader.load_stock_daily_enriched(ts_code, start_date, end_date)
                    if len(stock_data) < 60:
                        continue
                    factor_data = engine.compute_all_factors(stock_data)
                    latest = factor_data.iloc[-1:].copy()
                    latest["ts_code"] = ts_code
                    all_scores.append(latest)
                except Exception as e:
                    ctx.add_error("scan_stock", e, {"ts_code": ts_code})
                    continue

            if not all_scores:
                return {"error": "no_valid_stocks"}

            import pandas as pd
            combined = pd.concat(all_scores, ignore_index=True)
            selected = selector.select(combined, top_n=top_n)
            ctx.set_intermediate_result("selected_stocks", selected)
            ctx.add_marker("top_picks", selected["ts_code"].tolist()[:5])
            return {"selected": len(selected)}

        pipeline.add_step("load_sector", load_sector_step, loader=loader, sector_code=args.sector_code)
        pipeline.add_step("scan", scan_step, loader=loader, engine=factor_engine, selector=selector, top_n=args.top_n, start_date=args.start_date, end_date=args.end_date)

        results = pipeline.execute()
        selected = ctx.intermediate_results.get("selected_stocks")
        sector_stocks = ctx.intermediate_results.get("sector_stocks")
        name_map = dict(zip(sector_stocks["con_code"], sector_stocks["name"])) if sector_stocks is not None else {}
        
        if selected is not None and len(selected) > 0:
            print(f"\n板块 {args.sector_name} 选股结果 Top {args.top_n}:")
            for i, (_, row) in enumerate(selected.head(args.top_n).iterrows()):
                ts_code = row['ts_code']
                stock_name = name_map.get(ts_code, "未知")
                score = row.get('score', 0)
                print(f"  {i+1}. {ts_code} | {stock_name:<6} | 综合打分: {score:.4f}")
        else:
            print("未选出符合条件的股票")

    finally:
        ctx.save_execution_report(Path(config.get("log_dir", "output/logs")))


if __name__ == "__main__":
    main()
