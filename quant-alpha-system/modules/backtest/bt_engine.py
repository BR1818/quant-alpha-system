"""Backtrader 回测引擎核心 — ATR动态止损 + T+1 + A股佣金 + 涨跌停 + 停牌"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import math
import logging

from modules.backtest.ac_stock_commission import ACStockCommission


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
                self.hold_days = 0
                self.highest_since_buy = order.executed.price
                atr_val = self.atr[0] if self.atr[0] > 0 else order.executed.price * 0.03
                self.stop_price = order.executed.price - self.p.stop_loss_atr_mult * atr_val
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_log.append({
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

    def run_backtest(self, df: pd.DataFrame, strategy_params: Dict[str, Any] = None) -> Tuple[Dict[str, Any], bt.Cerebro]:
        """执行回测并提取严谨的数据指标"""
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

        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        self.logger.info(f"开启回测: 初始资金={self.initial_cash}, A股佣金(万2.5+千1印花税), T+1限制")

        results = self.cerebro.run()
        strat = results[0]

        final_value = self.cerebro.broker.getvalue()
        total_return_pct = ((final_value / self.initial_cash) - 1) * 100

        years = (df.index[-1] - df.index[0]).days / 365.25
        if years > 0 and total_return_pct > -100:
            annual_return_pct = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
        else:
            annual_return_pct = 0

        trade_analyzer = strat.analyzers.trades.get_analysis()
        sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
        drawdown_analyzer = strat.analyzers.drawdown.get_analysis()

        total_trades = trade_analyzer.get('total', {}).get('closed', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        lost_trades = trade_analyzer.get('lost', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0

        avg_won = trade_analyzer.get('won', {}).get('pnl', {}).get('average', 0) or 0
        avg_lost = abs(trade_analyzer.get('lost', {}).get('pnl', {}).get('average', 1) or 1)
        profit_loss_ratio = avg_won / avg_lost if avg_lost > 0 else 0

        metrics = {
            "total_return_pct": total_return_pct,
            "annual_return_pct": annual_return_pct,
            "sharpe_ratio": sharpe_analyzer.get('sharperatio', 0.0) or 0.0,
            "max_drawdown_pct": drawdown_analyzer.get('max', {}).get('drawdown', 0.0),
            "win_rate_pct": win_rate,
            "trade_count": total_trades,
            "won_trades": won_trades,
            "lost_trades": lost_trades,
            "profit_loss_ratio": profit_loss_ratio,
            "final_value": final_value,
        }

        return metrics, self.cerebro
