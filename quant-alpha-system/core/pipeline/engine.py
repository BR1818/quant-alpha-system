"""管道执行引擎"""

from typing import List, Dict, Any, Callable
import logging
from datetime import datetime

from core.pipeline.context import Context
from core.exceptions import PipelineError


class PipelineEngine:
    """管道执行器 — 按顺序执行步骤，记录全过程"""

    def __init__(self, context: Context):
        self.context = context
        self.logger = context.get_logger(__name__)
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, name: str, func: Callable, **kwargs) -> None:
        """添加管道步骤"""
        self.steps.append({"name": name, "func": func, "kwargs": kwargs})
        self.logger.debug(f"添加管道步骤: {name}")

    def execute(self) -> Dict[str, Any]:
        """执行全部步骤"""
        self.logger.info(f"开始执行管道: {len(self.steps)} 个步骤")
        self.context.add_trace("pipeline_start", {
            "step_count": len(self.steps),
            "steps": [s["name"] for s in self.steps],
        })

        results = {}
        for i, step in enumerate(self.steps):
            name = step["name"]
            self.logger.info(f"[{i+1}/{len(self.steps)}] 执行: {name}")
            self.context.add_trace("step_start", {"step": name, "index": i})

            try:
                t0 = datetime.now()
                result = step["func"](self.context, **step["kwargs"])
                elapsed = (datetime.now() - t0).total_seconds()
                results[name] = result
                self.context.add_trace("step_complete", {
                    "step": name,
                    "duration_seconds": round(elapsed, 3),
                })
                self.logger.info(f"[{i+1}/{len(self.steps)}] 完成: {name} ({elapsed:.1f}s)")

            except Exception as e:
                self.logger.error(f"步骤 {name} 执行失败: {e}")
                self.context.add_error(name, e)
                raise PipelineError(f"步骤 '{name}' 执行失败") from e

        self.context.add_trace("pipeline_complete", {"result_keys": list(results.keys())})
        self.logger.info("管道执行完毕")
        return results
