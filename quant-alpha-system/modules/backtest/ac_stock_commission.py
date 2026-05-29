"""A股交易佣金类 — 佣金+印花税+过户费"""

import backtrader as bt


class ACStockCommission(bt.CommInfoBase):
    """A股真实交易成本：
    - 佣金: 万2.5双向, 最低5元
    - 印花税: 千1卖出单向
    - 过户费: 十万分之一双向(沪市)
    """

    params = (
        ('commission', 0.00025),     # 佣金率 万2.5
        ('stamp_duty', 0.001),       # 印花税 千1 卖出
        ('transfer_fee', 0.00001),   # 过户费 十万分之一 双向
        ('min_commission', 5.0),     # 最低佣金 5元
    )

    def _getcommission(self, size, price, pseudoexec):
        """计算单笔交易手续费"""
        abs_size = abs(size)
        trade_value = abs_size * price

        # 佣金(双向)
        comm = max(trade_value * self.p.commission, self.p.min_commission)

        # 过户费(双向，仅沪市，这里简化为都收)
        comm += trade_value * self.p.transfer_fee

        # 印花税(仅卖出)
        if size < 0:
            comm += trade_value * self.p.stamp_duty

        return comm
