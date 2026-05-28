"""管道集成测试"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
import pytest

from core.pipeline.context import Context
from core.pipeline.engine import PipelineEngine
from modules.factors import init_factor_registry, FactorEngine
from modules.selectors.composite_selector import CompositeSelector


class TestPipelineIntegration:
    """管道集成测试"""

    def test_full_pipeline_flow(self):
        """测试完整管道流程: 加载 → 因子 → 选股"""
        tmpdir = tempfile.mkdtemp()
        config = {"log_dir": tmpdir, "cache_dir": tmpdir}

        run_id = f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ctx = Context(run_id=run_id, config=config)

        np.random.seed(42)
        data = pd.DataFrame({
            "ts_code": [f"{i:06d}.SZ" for i in range(50)],
            "close": np.random.randn(50) + 100,
            "high": np.random.randn(50) + 102,
            "low": np.random.randn(50) + 98,
            "open": np.random.randn(50) + 100,
            "volume": np.random.randint(1000, 100000, 50),
            "pe_ttm": np.abs(np.random.randn(50)) * 20 + 15,
            "pb": np.abs(np.random.randn(50)) * 3 + 1,
            "roe": np.random.randn(50) * 5 + 10,
            "revenue_yoy": np.random.randn(50) * 10 + 5,
            "north_money": np.random.randn(50) * 100,
            "net_mf_amount": np.random.randn(50) * 50,
            "turnover_rate": np.abs(np.random.randn(50)) * 3,
        })

        registry = init_factor_registry()
        factor_engine = FactorEngine(registry)
        selector = CompositeSelector()

        pipeline = PipelineEngine(ctx)

        def fake_load(ctx):
            ctx.set_intermediate_result("stock_data", data)
            return {"rows": len(data)}

        def compute_factors(ctx, engine):
            result = engine.compute_all_factors(data)
            ctx.set_intermediate_result("factor_data", result)
            return {"factor_count": engine.registry.factor_count}

        def select(ctx, selector):
            factors = ctx.intermediate_results["factor_data"]
            selected = selector.select(factors, top_n=10)
            ctx.set_intermediate_result("selected_stocks", selected)
            ctx.add_marker("top_stock", selected["ts_code"].iloc[0] if len(selected) > 0 else "none")
            return {"selected_count": len(selected)}

        pipeline.add_step("load", fake_load)
        pipeline.add_step("factors", compute_factors, engine=factor_engine)
        pipeline.add_step("select", select, selector=selector)

        results = pipeline.execute()

        assert "load" in results
        assert "factors" in results
        assert "select" in results
        assert results["select"]["selected_count"] == 10
        assert len(ctx.traces) >= 6
        assert "top_stock" in ctx.markers

        report_path = ctx.save_execution_report(Path(tmpdir))
        assert report_path.exists()

    def test_pipeline_error_handling(self):
        """测试管道错误处理"""
        tmpdir = tempfile.mkdtemp()
        config = {"log_dir": tmpdir}
        ctx = Context(run_id="error_test", config=config)

        pipeline = PipelineEngine(ctx)

        def failing_step(ctx):
            raise ValueError("模拟的错误")

        pipeline.add_step("will_fail", failing_step)

        with pytest.raises(Exception):
            pipeline.execute()

        assert len(ctx.errors) == 1
        assert ctx.errors[0]["error_type"] == "ValueError"
