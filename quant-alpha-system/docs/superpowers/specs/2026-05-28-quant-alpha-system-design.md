# Quant Alpha System 设计文档

> 日期：2026-05-28
> 作者：AI 量化研究助手
> 版本：1.0

---

## 1. 系统概述

### 1.1 项目目标

构建一个专业的 A 股量化选股与预测系统，支持：
- 定期深度分析 + 手动触发研究
- 综合打分选股（技术面、资金面、基本面、情绪面）
- 多层次预测（1-7 天涨跌概率 + 1-4 周趋势判断 + 具体价格预测）
- 全面回测验证（策略表现 + 模型准确率 + 基准对比 + 分年度/分市场环境）
- HTML 交互报告输出

### 1.2 技术选型

| 组件 | 技术方案 | 说明 |
|------|---------|------|
| 选股模型 | XGBoost | 多因子打分，解释性强 |
| 预测模型 | LSTM | 时序模式捕捉 |
| 数据源 | QuantDB Parquet | 本地数据底座 |
| 报告格式 | HTML + Plotly | 交互式图表 |
| 架构模式 | 管道式 + 清晰接口 | 灵活可测试 |

---

## 2. 系统架构

### 2.1 目录结构

```
quant-alpha-system/
├── config/                    # 配置文件
│   ├── settings.yaml          # 全局配置
│   ├── strategies/            # 策略配置
│   └── logging.yaml           # 日志配置
├── core/                      # 核心引擎
│   ├── pipeline/              # 管道引擎
│   │   ├── engine.py          # 管道执行器
│   │   ├── context.py         # 执行上下文（日志、标记）
│   │   └── registry.py        # 模块注册器
│   ├── interfaces/            # 接口定义（Protocol）
│   │   ├── data.py            # 数据接口
│   │   ├── factor.py          # 因子接口
│   │   ├── model.py           # 模型接口
│   │   └── strategy.py        # 策略接口
│   └── exceptions.py          # 自定义异常
├── modules/                   # 功能模块
│   ├── data/                  # 数据加载模块
│   │   ├── quantdb_loader.py  # QuantDB 数据加载
│   │   ├── validator.py       # 数据验证器
│   │   └── cache.py           # 数据缓存
│   ├── factors/               # 因子计算模块
│   │   ├── engine.py          # 因子计算引擎
│   │   ├── technical.py       # 技术因子
│   │   ├── fundamental.py     # 基本面因子
│   │   ├── moneyflow.py       # 资金流因子
│   │   └── sentiment.py       # 情绪因子
│   ├── selectors/             # 选股模块
│   │   ├── xgboost_selector.py
│   │   └── composite_selector.py
│   ├── predictors/            # 预测模块
│   │   ├── lstm_predictor.py
│   │   └── ensemble_predictor.py
│   └── reporters/             # 报告生成模块
│       ├── html_reporter.py
│       └── templates/
├── backtest/                  # 回测引擎
│   ├── engine.py              # 回测执行器
│   ├── metrics.py             # 指标计算
│   └── validator.py           # 模型验证
├── tests/                     # 测试套件
│   ├── unit/                  # 单元测试
│   ├── integration/           # 集成测试
│   └── backtest/              # 回测验证测试
├── scripts/                   # 可执行脚本
│   ├── run_analysis.py        # 主分析入口
│   ├── run_backtest.py        # 回测入口
│   └── run_daily_track.py     # 每日跟踪入口
└── output/                    # 输出目录
    ├── reports/               # HTML 报告
    ├── models/                # 训练好的模型
    └── logs/                  # 运行日志
```

### 2.2 核心设计原则

1. **接口驱动**：所有模块实现统一接口，可独立测试和替换
2. **上下文传递**：管道执行时传递 Context 对象，包含日志、标记、中间结果
3. **模块注册**：新模块只需实现接口并注册，无需修改核心引擎
4. **全面日志**：每个环节记录输入、输出、耗时、异常，便于 AI 诊断问题

---

## 3. 数据层设计

### 3.1 数据接口

```python
class DataLoader(Protocol):
    """数据加载器接口"""
    
    def load_stock_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载个股日线数据"""
        ...
    
    def load_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载指数日线数据"""
        ...
    
    def load_sector_stocks(self, sector_code: str) -> pd.DataFrame:
        """加载板块成分股"""
        ...
    
    def load_factors(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载技术因子数据"""
        ...

class DataValidator(Protocol):
    """数据验证器接口"""
    
    def validate(self, df: pd.DataFrame, data_type: str) -> bool:
        """验证数据完整性"""
        ...
    
    def get_validation_report(self) -> Dict[str, Any]:
        """获取验证报告"""
        ...
```

### 3.2 QuantDB 数据加载器

- 支持加载 daily、daily_basic、index_daily、stk_factor_pro、ths_member、dc_member 等表
- 内置数据缓存机制（内存 + 磁盘）
- 自动验证数据完整性（列检查、日期连续性、价格合理性）

### 3.3 上下文对象

```python
class Context:
    """管道执行上下文"""
    
    def __init__(self, run_id: str, config: Dict[str, Any]):
        self.run_id = run_id
        self.config = config
        self.logger = self._setup_logger()
        self.traces: List[Dict[str, Any]] = []
        self.markers: Dict[str, Any] = {}
        self.intermediate_results: Dict[str, Any] = {}
        self.errors: List[Dict[str, Any]] = []
    
    def add_trace(self, step: str, data: Dict[str, Any]) -> None:
        """添加执行追踪"""
        ...
    
    def add_marker(self, key: str, value: Any) -> None:
        """添加标记"""
        ...
    
    def set_intermediate_result(self, key: str, result: Any) -> None:
        """保存中间结果"""
        ...
    
    def add_error(self, step: str, error: Exception, context: Dict[str, Any] = None) -> None:
        """记录错误"""
        ...
```

---

## 4. 因子层设计

### 4.1 因子接口

```python
class Factor(Protocol):
    """因子计算接口"""
    
    @property
    def name(self) -> str:
        """因子名称"""
        ...
    
    @property
    def category(self) -> str:
        """因子类别：technical/fundamental/moneyflow/sentiment"""
        ...
    
    @property
    def description(self) -> str:
        """因子描述"""
        ...
    
    def compute(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
        """计算因子值"""
        ...
    
    def get_required_columns(self) -> List[str]:
        """获取所需数据列"""
        ...
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        ...
```

### 4.2 因子库

#### 技术因子（technical）
- MA（移动平均线）
- MACD（指数平滑异同移动平均线）
- RSI（相对强弱指数）
- Bollinger Bands（布林带）
- ATR（平均真实波幅）

#### 基本面因子（fundamental）
- PE_TTM（滚动市盈率）
- PB（市净率）
- ROE（净资产收益率）
- Revenue Growth（营收增长率）

#### 资金流因子（moneyflow）
- Northbound Flow（北向资金净流入）
- Main Force Flow（主力资金净流入）
- Volume Ratio（量比）
- Turnover Rate（换手率）

#### 情绪因子（sentiment）
- Limit Up Count（涨停板数量）
- Top List Net Buy（龙虎榜净买入）
- Margin Balance（融资余额）

### 4.3 因子注册器

```python
class FactorRegistry:
    """因子注册器"""
    
    def register(self, factor: Factor) -> None:
        """注册因子"""
        ...
    
    def get(self, name: str) -> Optional[Factor]:
        """获取因子"""
        ...
    
    def list_factors(self, category: str = None) -> List[str]:
        """列出所有因子"""
        ...
```

---

## 5. 选股层设计

### 5.1 选股接口

```python
class StockSelector(Protocol):
    """选股器接口"""
    
    @property
    def name(self) -> str:
        """选股器名称"""
        ...
    
    def select(self, data: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """选股，返回排序后的股票列表"""
        ...
    
    def get_factor_weights(self) -> Dict[str, float]:
        """获取因子权重"""
        ...
```

### 5.2 XGBoost 选股器

- 训练 XGBClassifier 模型
- 输出选股概率得分
- 记录特征重要性
- 支持模型保存/加载

### 5.3 综合打分选股器

- 按类别计算因子得分
- 支持自定义权重配置
- 归一化处理
- 加权综合打分

---

## 6. 预测层设计

### 6.1 预测接口

```python
class StockPredictor(Protocol):
    """股票预测器接口"""
    
    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] = None) -> None:
        """训练模型"""
        ...
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """预测，返回多层次结果"""
        ...
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        ...
```

### 6.2 LSTM 预测器

- 2 层 LSTM + 全连接层
- 输出：涨跌概率、趋势判断、目标价位
- 支持 GPU 加速
- 模型保存/加载

### 6.3 集成预测器

- 加权平均多个预测器结果
- 支持动态权重调整
- 降低单一模型风险

---

## 7. 回测层设计

### 7.1 回测引擎

```python
class BacktestEngine:
    """回测引擎"""
    
    def run(self, strategy: StockSelector, data: pd.DataFrame, 
            start_date: str, end_date: str) -> Dict[str, Any]:
        """运行回测"""
        ...
    
    def calculate_metrics(self, returns: pd.Series, benchmark: pd.Series = None) -> Dict[str, float]:
        """计算回测指标"""
        ...
```

### 7.2 回测指标

- 总收益率、年化收益率
- 夏普比率
- 最大回撤
- 胜率
- 基准对比（超额收益、信息比率）

### 7.3 模型验证器

- 时间序列交叉验证
- 向前验证（Walk-Forward）
- 分年度/分市场环境分析

---

## 8. 报告层设计

### 8.1 HTML 报告生成器

- 使用 Jinja2 模板引擎
- 集成 Plotly 交互式图表
- 支持分析报告和回测报告

### 8.2 报告内容

#### 分析报告
- 股价走势图（K 线）
- 因子得分图
- 预测结果图
- 预测摘要（涨跌概率、趋势、目标价位）
- 执行日志

#### 回测报告
- 净值曲线
- 回撤曲线
- 回测指标（总收益、年化、夏普、最大回撤、胜率）
- 执行日志

---

## 9. 管道引擎设计

### 9.1 管道执行器

```python
class PipelineEngine:
    """管道执行引擎"""
    
    def add_step(self, name: str, func: callable, **kwargs) -> None:
        """添加执行步骤"""
        ...
    
    def execute(self) -> Dict[str, Any]:
        """执行管道"""
        ...
```

### 9.2 执行流程

1. 加载数据
2. 计算因子
3. 选股打分
4. 预测分析
5. 生成报告

### 9.3 错误处理

- 每个步骤独立 try-catch
- 错误记录到上下文
- 支持断点重试

---

## 10. 使用方式

### 10.1 分析单只股票

```bash
python scripts/run_analysis.py \
  --stock-code 600584.SH \
  --stock-name 长电科技 \
  --start-date 20200101 \
  --end-date 20260528
```

### 10.2 板块选股

```bash
python scripts/run_sector_scan.py \
  --sector-code THS_ROBOT \
  --sector-name 机器人概念 \
  --top-n 20
```

### 10.3 每日跟踪

```bash
python scripts/run_daily_track.py \
  --stock-code 600584.SH \
  --stock-name 长电科技
```

### 10.4 回测验证

```bash
python scripts/run_backtest.py \
  --strategy xgboost_selector \
  --start-date 20200101 \
  --end-date 20260528
```

---

## 11. 扩展性设计

### 11.1 添加新因子

1. 在 `modules/factors/` 下创建新文件
2. 实现 `Factor` 接口
3. 在 `init.py` 中注册

### 11.2 添加新选股策略

1. 在 `modules/selectors/` 下创建新文件
2. 实现 `StockSelector` 接口
3. 在配置中指定使用

### 11.3 添加新预测模型

1. 在 `modules/predictors/` 下创建新文件
2. 实现 `StockPredictor` 接口
3. 在集成预测器中配置权重

### 11.4 添加新报告类型

1. 在 `modules/reporters/` 下创建新文件
2. 实现报告生成方法
3. 添加 HTML 模板

---

## 12. 测试策略

### 12.1 单元测试

- 每个模块独立测试
- Mock 依赖组件
- 覆盖正常/异常场景

### 12.2 集成测试

- 测试管道完整流程
- 验证数据流转
- 检查日志输出

### 12.3 回测验证测试

- 使用历史数据验证策略
- 对比基准收益
- 分年度/分市场环境分析

---

## 13. 配置管理

### 13.1 全局配置

```yaml
# config/settings.yaml
data_dir: "/Users/zhengborui/Documents/Claude-workspace/quant-DB/quantdb/data/curated"
cache_dir: "output/cache"
log_dir: "output/logs"
model_dir: "output/models"
report_dir: "output/reports"

# 因子权重
factor_weights:
  technical: 0.3
  fundamental: 0.25
  moneyflow: 0.25
  sentiment: 0.2

# 模型参数
lstm_params:
  hidden_dim: 64
  num_layers: 2
  epochs: 100
  batch_size: 32

xgboost_params:
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 100
```

### 13.2 策略配置

```yaml
# config/strategies/default.yaml
name: "default"
description: "默认综合策略"
selector: "xgboost_selector"
predictor: "ensemble_predictor"
top_n: 20
rebalance_freq: "W"
```

---

## 14. 日志与监控

### 14.1 日志级别

- DEBUG：详细调试信息
- INFO：关键步骤信息
- WARNING：警告信息
- ERROR：错误信息

### 14.2 执行追踪

每个步骤记录：
- 步骤名称
- 开始/结束时间
- 输入参数
- 输出结果
- 耗时统计

### 14.3 错误追踪

- 错误类型
- 错误消息
- 错误上下文
- 堆栈信息

### 14.4 执行报告

每次运行生成 JSON 格式的执行报告：
- 运行 ID
- 开始/结束时间
- 所有追踪记录
- 所有标记
- 错误记录

---

## 15. 未来规划

### 15.1 短期（1-2 周）

- [ ] 实现核心框架
- [ ] 实现数据层
- [ ] 实现因子层
- [ ] 实现基础选股器

### 15.2 中期（3-4 周）

- [ ] 实现预测层
- [ ] 实现回测层
- [ ] 实现报告层
- [ ] 集成测试

### 15.3 长期（持续）

- [ ] 添加更多因子
- [ ] 优化模型参数
- [ ] 支持更多板块
- [ ] 实盘信号输出

---

## 附录 A：QuantDB 数据表参考

| 表名 | 关键字段 | 用途 |
|------|---------|------|
| daily | ts_code, trade_date, open, high, low, close, volume | 日线行情 |
| daily_basic | ts_code, trade_date, pe_ttm, pb, turnover_rate, total_mv | 每日指标 |
| stk_factor_pro | ts_code, trade_date + 200+ 技术指标 | 技术因子 |
| moneyflow | ts_code, trade_date, net_mf_amount | 个股资金流向 |
| moneyflow_hsgt | trade_date, north_money | 北向资金 |
| fina_indicator | ts_code, ann_date, roe, revenue_yoy | 财务指标 |
| ths_member | ts_code, con_code | 同花顺概念成分 |
| dc_member | ts_code, con_code | 东财概念成分 |
| limit_list_d | trade_date, ts_code, limit | 涨跌停统计 |
| top_list | trade_date, ts_code, net_amount | 龙虎榜 |
| margin | trade_date, rzye | 融资融券 |

---

## 附录 B：接口定义汇总

### 数据接口
- DataLoader：load_stock_daily, load_index_daily, load_sector_stocks, load_factors
- DataValidator：validate, get_validation_report

### 因子接口
- Factor：name, category, description, compute, get_required_columns, validate_input
- FactorRegistry：register, get, list_factors, get_factor_info

### 选股接口
- StockSelector：name, description, select, get_factor_weights

### 预测接口
- StockPredictor：name, description, train, predict, evaluate

### 回测接口
- BacktestEngine：run, calculate_metrics, generate_report
- ModelValidator：cross_validate, walk_forward_validation

---

**文档完成时间：2026-05-28**
