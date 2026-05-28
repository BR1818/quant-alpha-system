"""自定义异常类"""

class QuantAlphaError(Exception):
    """系统基础异常"""
    pass

class DataLoadError(QuantAlphaError):
    """数据加载异常"""
    pass

class DataValidationError(QuantAlphaError):
    """数据验证异常"""
    pass

class FactorComputeError(QuantAlphaError):
    """因子计算异常"""
    pass

class ModelError(QuantAlphaError):
    """模型异常"""
    pass

class BacktestError(QuantAlphaError):
    """回测异常"""
    pass

class PipelineError(QuantAlphaError):
    """管道执行异常"""
    pass

class ConfigError(QuantAlphaError):
    """配置异常"""
    pass
