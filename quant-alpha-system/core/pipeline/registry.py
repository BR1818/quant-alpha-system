"""管道注册器"""

from typing import Dict, Callable


class PipelineRegistry:
    """管道注册器 — 管理多个预定义管道"""

    def __init__(self):
        self._pipelines: Dict[str, Callable] = {}

    def register(self, name: str, pipeline_func: Callable) -> None:
        """注册管道"""
        self._pipelines[name] = pipeline_func

    def get(self, name: str) -> Callable:
        """获取管道"""
        if name not in self._pipelines:
            raise KeyError(f"管道不存在: {name}. 可用: {list(self._pipelines.keys())}")
        return self._pipelines[name]

    def list_pipelines(self) -> list:
        """列出所有管道"""
        return list(self._pipelines.keys())
