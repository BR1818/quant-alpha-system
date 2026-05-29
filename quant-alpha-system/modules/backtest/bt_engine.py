"""Backtrader 回测引擎核心 — ATR动态止损 + T+1 + A股佣金 + 涨跌停 + 停牌"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import math
import logging

from modules.backtest.ac_stock_commission import ACStockCommission


class PortfolioValueRecorder(bt.Analyzer):
    """记录每个bar的组合价值，用于净值曲线和回撤计算"""
    params = (
        ('fund', None),
    )

    def start(self):
        self._values = []

    def next(self):
        dt = self.data.datetime.date(0)
        val = self.strategy.broker.getvalue()
        self._values.append((dt, val))

    def get_analysis(self):
        return self._values


class AlphaPandasData(bt.feeds.PandasData):
    """扩展 PandasData，支持动态传入自定义列"""
    lines = ('score',)
    params = (
        ('score', -1),
    )


class QuantAlphaStrategy(bt.Strategy):
    """多因子打分量化交易策略 — ATR动态止损 + T+1 + 涨跌停 + 停牌"""
    params = (
        ("buy_threshold", 0.5),
        ("take_profit_pct", 0.10),
        ("stop_loss_atr_mult", 2.0),
        ("max_hold_days", 10),
        ("position_pct", 0.30),
        ("atr_period", 14),
        ("trailing_stop", True),
    )

    def __init__(self):
        self.score = self.datas[0].score
        self.order = None
        self.buy_price = 0.0
        self.buy_bar = None
        self.hold_days = 0
        self.stop_price = 0.0
        self.highest_since_buy = 0.0
        self.entry_date = None
        self.entry_price = None

        # ATR 指标
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)

        # 交易日志
        self.trade_log = []

    def notify_order(self, order):
        """订单状态回调 — 记录实际成交价"""
        if order.status == order.Completed:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_bar = len(self)
                self.entry_date = self.data.datetime.date(0)
                self.entry_price = order.executed.price
                self.hold_days = 0
                self.highest_since_buy = order.executed.price
                atr_val = self.atr[0] if self.atr[0] > 0 else order.executed.price * 0.03
                self.stop_price = order.executed.price - self.p.stop_loss_atr_mult * atr_val
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_log.append({
                "entry_date": self.entry_date,
                "exit_date": self.data.datetime.date(0),
                "entry_price": self.entry_price,
                "exit_price": trade.pnl / trade.size + trade.price if trade.size else 0,
                "size": trade.size,
                "commission": trade.commission,
                "pnl": trade.pnl,
                "pnlcomm": trade.pnlcomm,
                "barlen": trade.barlen,
            })

    def next(self):
        if self.order:
            return

        # 停牌日过滤：成交量为0则不可交易
        if self.data.volume[0] == 0:
            return

        if not self.position:
            # 买入逻辑
            if self.score[0] > self.p.buy_threshold:
                # 涨停检查：收盘价达到涨停价时不可买入
                if hasattr(self.data, 'up_limit') and self.data.up_limit[0] > 0:
                    if self.data.close[0] >= self.data.up_limit[0]:
                        return

                target_value = self.broker.get_cash() * self.p.position_pct
                size = math.floor(target_value / self.data.close[0] / 100) * 100
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # T+1检查：买入当日不可卖出
            if self.buy_bar is not None and len(self) - self.buy_bar < 2:
                self.hold_days += 1
                return

            self.hold_days += 1
            current_close = self.data.close[0]

            # 更新最高价
            if current_close > self.highest_since_buy:
                self.highest_since_buy = current_close

            # 移动止损
            if self.p.trailing_stop and self.highest_since_buy > self.buy_price * 1.03:
                atr_val = self.atr[0] if self.atr[0] > 0 else current_close * 0.02
                new_stop = self.highest_since_buy - self.p.stop_loss_atr_mult * atr_val
                self.stop_price = max(self.stop_price, new_stop, self.buy_price)

            # 跌停检查：收盘价达到跌停价时不可卖出
            can_sell = True
            if hasattr(self.data, 'down_limit') and self.data.down_limit[0] > 0:
                if self.data.close[0] <= self.data.down_limit[0]:
                    can_sell = False

            if not can_sell:
                return

            pnl_pct = (current_close - self.position.price) / self.position.price

            # 止盈
            if pnl_pct >= self.p.take_profit_pct:
                self.order = self.close()
            # 止损：用盘中低点判断
            elif self.data.low[0] <= self.stop_price:
                self.order = self.close()
            # 时间止损
            elif self.hold_days >= self.p.max_hold_days:
                self.order = self.close()


def compute_alpha_beta(strat_ret: pd.Series, bench_ret: pd.Series):
    """用 numpy 计算 Alpha 和 Beta (年化)"""
    X = bench_ret.values
    Y = strat_ret.values
    X_mean = X.mean()
    Y_mean = Y.mean()
    ss_xy = np.sum((X - X_mean) * (Y - Y_mean))
    ss_xx = np.sum((X - X_mean) ** 2)
    beta = ss_xy / ss_xx if ss_xx > 0 else 0.0
    alpha = (Y_mean - beta * X_mean) * 252 * 100
    return beta, alpha


def compute_sortino(daily_returns: pd.Series, rf: float = 0.02) -> float:
    """计算年化 Sortino 比率"""
    annual_return = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
    downside = daily_returns[daily_returns < rf / 252]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
    return (annual_return - rf) / downside_std if downside_std > 0 else 0.0


def compute_information_ratio(strat_ret: pd.Series, bench_ret: pd.Series) -> float:
    """计算年化 Information Ratio"""
    excess = strat_ret.values - bench_ret.values
    std = excess.std()
    return excess.mean() / std * np.sqrt(252) if std > 0 else 0.0


class BTEngine:
    """封装 Backtrader 框架，处理数据挂载与分析器"""
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.cerebro = bt.Cerebro()

        self.initial_cash = config.get("initial_cash", 1000000.0)
        self.cerebro.broker.setcash(self.initial_cash)

        # A股真实佣金
        comm = ACStockCommission()
        self.cerebro.broker.addcommissioninfo(comm)

        self.cerebro.broker.set_slippage_perc(perc=0.001)

    def run_backtest(self, df: pd.DataFrame, strategy_params: Dict[str, Any] = None,
                     benchmark_df: pd.DataFrame = None) -> Dict[str, Any]:
        """执行回测并返回完整结果字典"""
        df = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if "trade_date" in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
                    df["datetime"] = df["trade_date"]
                else:
                    df["datetime"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                df.set_index("datetime", inplace=True)

        df = df.sort_index()

        data = AlphaPandasData(dataname=df,
                               open='open', high='high', low='low', close='close', volume='vol',
                               score='score',
                               openinterest=-1)
        self.cerebro.adddata(data)

        p = strategy_params or {}
        self.cerebro.addstrategy(QuantAlphaStrategy, **p)

        # 分析器
        self.cerebro.addanalyzer(PortfolioValueRecorder, _name="portfolio_value")
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="daily_returns",
                                 timeframe=bt.TimeFrame.Days)
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="sharpe_annual", riskfreerate=0.02)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        self.logger.info(f"开启回测: 初始资金={self.initial_cash}, A股佣金(万2.5+千1印花税), T+1限制")

        results = self.cerebro.run()
        strat = results[0]

        # 提取组合价值时间序列
        pv_data = strat.analyzers.portfolio_value.get_analysis()
        portfolio_df = pd.DataFrame(pv_data, columns=["date", "value"])

        # 提取日收益率时间序列
        daily_returns_dict = strat.analyzers.daily_returns.get_analysis()
        daily_returns = pd.Series(daily_returns_dict, name="daily_return")
        daily_returns.index = pd.to_datetime(daily_returns.index)

        # 提取交易日志
        trade_log = strat.trade_log

        # 基础指标
        final_value = self.cerebro.broker.getvalue()
        total_return_pct = ((final_value / self.initial_cash) - 1) * 100

        years = (df.index[-1] - df.index[0]).days / 365.25
        if years > 0 and total_return_pct > -100:
            annual_return_pct = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
        else:
            annual_return_pct = 0

        trade_analyzer = strat.analyzers.trades.get_analysis()
        sharpe_analyzer = strat.analyzers.sharpe_annual.get_analysis()
        drawdown_analyzer = strat.analyzers.drawdown.get_analysis()

        total_trades = trade_analyzer.get('total', {}).get('closed', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        lost_trades = trade_analyzer.get('lost', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0

        avg_won = trade_analyzer.get('won', {}).get('pnl', {}).get('average', 0) or 0
        avg_lost = abs(trade_analyzer.get('lost', {}).get('pnl', {}).get('average', 1) or 1)
        profit_loss_ratio = avg_won / avg_lost if avg_lost > 0 else 0

        # 从日收益率计算扩展指标
        volatility_pct = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0.0
        downside = daily_returns[daily_returns < 0]
        downside_risk_pct = downside.std() * np.sqrt(252) * 100 if len(downside) > 1 else 0.0
        sortino_ratio = compute_sortino(daily_returns) if len(daily_returns) > 1 else 0.0
        max_dd_pct = drawdown_analyzer.get('max', {}).get('drawdown', 0.0)
        calmar_ratio = annual_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0
        avg_hold_days = np.mean([t["barlen"] for t in trade_log]) if trade_log else 0.0

        metrics = {
            "total_return_pct": total_return_pct,
            "annual_return_pct": annual_return_pct,
            "sharpe_ratio": sharpe_analyzer.get('sharperatio', 0.0) or 0.0,
            "max_drawdown_pct": max_dd_pct,
            "win_rate_pct": win_rate,
            "trade_count": total_trades,
            "won_trades": won_trades,
            "lost_trades": lost_trades,
            "profit_loss_ratio": profit_loss_ratio,
            "final_value": final_value,
            "volatility_pct": volatility_pct,
            "downside_risk_pct": downside_risk_pct,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "avg_hold_days": avg_hold_days,
            # 基准相关，稍后填充
            "benchmark_annual_return_pct": None,
            "excess_return_pct": None,
            "alpha": None,
            "beta": None,
            "information_ratio": None,
        }

        # 基准处理：回测后用 pandas 对齐日收益率
        benchmark_daily_returns = None
        if benchmark_df is not None and len(benchmark_df) > 0:
            bench = benchmark_df.copy()
            if "trade_date" in bench.columns:
                bench["date"] = pd.to_datetime(bench["trade_date"])
                bench = bench.set_index("date").sort_index()
            bench_daily = bench["close"].pct_change().dropna()
            bench_daily.name = "benchmark_return"

            # 对齐日期
            aligned = pd.DataFrame({"strategy": daily_returns, "benchmark": bench_daily}).dropna()
            if len(aligned) > 30:
                benchmark_daily_returns = aligned["benchmark"]
                strat_aligned = aligned["strategy"]
                bench_aligned = aligned["benchmark"]

                beta, alpha = compute_alpha_beta(strat_aligned, bench_aligned)
                metrics["alpha"] = alpha
                metrics["beta"] = beta
                metrics["information_ratio"] = compute_information_ratio(strat_aligned, bench_aligned)

                bench_cum = (1 + bench_aligned).prod()
                bench_annual = (bench_cum ** (252 / len(bench_aligned)) - 1) * 100
                metrics["benchmark_annual_return_pct"] = bench_annual
                metrics["excess_return_pct"] = annual_return_pct - bench_annual

        start_date = str(df.index[0].date()) if len(df) > 0 else ""
        end_date = str(df.index[-1].date()) if len(df) > 0 else ""

        return {
            "metrics": metrics,
            "portfolio_values": portfolio_df,
            "daily_returns": daily_returns,
            "benchmark_daily_returns": benchmark_daily_returns,
            "trade_log": trade_log,
            "strategy_name": "QuantAlphaStrategy",
            "initial_cash": self.initial_cash,
            "start_date": start_date,
            "end_date": end_date,
        }
