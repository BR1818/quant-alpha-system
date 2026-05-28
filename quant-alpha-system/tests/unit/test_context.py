"""Context 执行上下文测试"""
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.pipeline.context import Context


class TestContext:
    """Context 单元测试"""

    def test_context_initialization(self):
        """测试 Context 初始化"""
        config = {"log_dir": tempfile.mkdtemp()}
        ctx = Context(run_id="test_run", config=config)
        assert ctx.run_id == "test_run"
        assert ctx.config == config
        assert len(ctx.traces) == 0
        assert len(ctx.errors) == 0
        assert len(ctx.intermediate_results) == 0

    def test_add_trace(self):
        """测试添加执行追踪"""
        config = {"log_dir": tempfile.mkdtemp()}
        ctx = Context(run_id="test_trace", config=config)
        ctx.add_trace("load_data", {"ts_code": "000001.SZ", "rows": 100})
        assert len(ctx.traces) == 1
        assert ctx.traces[0]["step"] == "load_data"
        assert ctx.traces[0]["data"]["rows"] == 100
        assert "timestamp" in ctx.traces[0]

    def test_add_marker(self):
        """测试添加标记"""
        config = {"log_dir": tempfile.mkdtemp()}
        ctx = Context(run_id="test_marker", config=config)
        ctx.add_marker("feature_count", 42)
        assert ctx.markers["feature_count"] == 42

    def test_set_intermediate_result(self):
        """测试保存中间结果"""
        config = {"log_dir": tempfile.mkdtemp()}
        ctx = Context(run_id="test_result", config=config)
        ctx.set_intermediate_result("factor_data", {"ma": [1, 2, 3]})
        assert "factor_data" in ctx.intermediate_results
        assert ctx.intermediate_results["factor_data"] == {"ma": [1, 2, 3]}

    def test_add_error(self):
        """测试记录错误"""
        config = {"log_dir": tempfile.mkdtemp()}
        ctx = Context(run_id="test_error", config=config)
        try:
            raise ValueError("test error message")
        except ValueError as e:
            ctx.add_error("test_step", e, {"extra": "context"})
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "test_step"
        assert ctx.errors[0]["error_type"] == "ValueError"
        assert ctx.errors[0]["error_message"] == "test error message"

    def test_save_execution_report(self):
        """测试保存执行报告"""
        tmpdir = tempfile.mkdtemp()
        config = {"log_dir": tmpdir}
        ctx = Context(run_id="test_report", config=config)
        ctx.add_trace("step1", {"data": "ok"})
        ctx.add_marker("score", 0.95)

        ctx.save_execution_report(Path(tmpdir))
        report_files = list(Path(tmpdir).glob("*_execution_report.json"))
        assert len(report_files) == 1
