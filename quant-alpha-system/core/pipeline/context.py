"""管道执行上下文，提供日志、追踪、标记、中间结果存储"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


class Context:
    """管道执行上下文 — 贯穿整个管道生命周期"""

    def __init__(self, run_id: str, config: Dict[str, Any]):
        self.run_id = run_id
        self.config = config
        self.start_time = datetime.now()

        # 日志系统
        self.logger = self._setup_logger()

        # 执行追踪
        self.traces: List[Dict[str, Any]] = []
        self.markers: Dict[str, Any] = {}

        # 中间结果
        self.intermediate_results: Dict[str, Any] = {}

        # 错误记录
        self.errors: List[Dict[str, Any]] = []

    def _setup_logger(self) -> logging.Logger:
        """设置日志系统: 控制台 INFO + 文件 DEBUG"""
        logger = logging.getLogger(f"quant_alpha_{self.run_id}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

        # 文件 handler
        log_dir = Path(self.config.get("log_dir", "output/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{self.run_id}.log", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """获取子 logger"""
        return logging.getLogger(f"quant_alpha_{self.run_id}.{name}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔的嵌套键"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def add_trace(self, step: str, data: Dict[str, Any]) -> None:
        """添加执行追踪记录"""
        trace = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        self.traces.append(trace)
        self.logger.debug(f"[TRACE] {step}: {json.dumps(data, ensure_ascii=False, default=str)}")

    def add_marker(self, key: str, value: Any) -> None:
        """添加标记点（如模型指标、数据统计）"""
        self.markers[key] = value
        self.logger.debug(f"[MARKER] {key} = {value}")

    def set_intermediate_result(self, key: str, result: Any) -> None:
        """保存中间结果，供后续步骤使用"""
        self.intermediate_results[key] = result
        self.logger.debug(f"[RESULT] saved: {key}")

    def add_error(self, step: str, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """记录错误"""
        error_record = {
            "step": step,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
        }
        self.errors.append(error_record)
        self.logger.error(f"[ERROR] {step}: {error}", exc_info=True)

    def save_execution_report(self, output_dir: Path) -> Path:
        """保存完整执行报告为 JSON 文件"""
        report = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "traces": self.traces,
            "markers": self.markers,
            "errors": self.errors,
            "intermediate_result_keys": list(self.intermediate_results.keys()),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.run_id}_execution_report.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info(f"执行报告已保存: {output_file}")
        return output_file
