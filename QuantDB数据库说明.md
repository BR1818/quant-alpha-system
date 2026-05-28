# QuantDB 数据库表说明

> 数据库路径：`/Users/zhengborui/Documents/Claude-workspace/quant-DB/quantdb/data/curated/`
> 格式：Parquet（每只股票/指数一个文件）
> 虚拟环境：`/Users/zhengborui/Documents/Claude-workspace/quant-DB/.venv/bin/python3`

---

## 读取方式

```python
import pyarrow.parquet as pq
from pathlib import Path

CURATED_DIR = Path("/Users/zhengborui/Documents/Claude-workspace/quant-DB/quantdb/data/curated")
df = pq.read_table(CURATED_DIR / "daily" / "000001.SZ.parquet").to_pandas()
```

---

## 一、行情数据（日线）

| 表名 | 文件数 | 关键字段 | 说明 |
|------|--------|---------|------|
| `daily` | 5,845 | ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, volume, amount | A股日线行情（不复权） |
| `daily_full` | 1 | 同 daily | 全量合并日线（单文件，适合快速全市场扫描） |
| `daily_basic` | 5,846 | ts_code, trade_date, close, turnover_rate, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv | 每日指标：估值、换手率、股本市值 |
| `adj_factor` | 5,848 | ts_code, trade_date, adj_factor | 复权因子（前复权/后复权换算） |
| `stk_factor_pro` | 5,776 | ts_code, trade_date + 200+ 技术指标字段 | 综合技术因子：MA/EMA/MACD/RSI/KDJ/BOLL/ATR/OBV/CCI/DMI/WR/TRIX/MFI/ROC/PSY 等，含不复权(bfq)/后复权(hfq)/前复权(qfq)三套 |
| `stk_limit` | 5,796 | trade_date, ts_code, up_limit, down_limit | 每日涨跌停价格 |
| `suspend_d` | 4,449 | ts_code, trade_date, suspend_timing, suspend_type | 停牌数据 |

## 二、资金流向

| 表名 | 文件数 | 关键字段 | 说明 |
|------|--------|---------|------|
| `moneyflow` | 5,656 | ts_code, trade_date, buy_sm/md/lg/elg_vol/amount, sell_sm/md/lg/elg_vol/amount, net_mf_vol/amount | 个股资金流向（小单/中单/大单/特大单） |
| `moneyflow_hsgt` | 1 | trade_date, ggt_ss, ggt_sz, hgt, sgt, north_money, south_money | 沪深港通资金流向（北向/南向） |
| `margin` | 1 | trade_date, rzye, rzmre, rzche, rqye, rqmcl, rzrqye | 融资融券数据 |
| `stock_hsgt` | 1 | ts_code, trade_date, type, name | 个股沪深通持股明细 |

## 三、财务数据

| 表名 | 文件数 | 关键字段 | 说明 |
|------|--------|---------|------|
| `fina_indicator` | 5,814 | ts_code, ann_date, end_date + 100+ 财务指标 | 核心财务指标：EPS/ROE/ROA/毛利率/净利率/资产负债率/流动比率/周转率/YoY增速等 |
| `income` | 5,845 | ts_code, ann_date, end_date, revenue, net_profit, operate_profit, total_profit | 利润表 |
| `balancesheet` | 5,811 | ts_code, ann_date, end_date, total_assets, total_liab, total_equity + 100+ 字段 | 资产负债表 |
| `cashflow` | 5,804 | ts_code, ann_date, end_date, net_profit, n_cashflow_act, free_cashflow + 80+ 字段 | 现金流量表 |
| `fina_audit` | 5,811 | ts_code, ann_date, end_date, audit_result, audit_fees, audit_agency | 审计意见 |
| `express` | 4,351 | ts_code, ann_date, end_date, revenue, net_profit, eps, roe, yoy_net_profit | 业绩快报 |
| `forecast` | 5,695 | ts_code, ann_date, end_date, type, p_change_min, p_change_max, net_profit_min/max | 业绩预告 |
| `dividend` | 5,819 | ts_code, end_date, ann_date, div_cash, div_stock, record_date, ex_date | 分红送股 |
| `disclosure_date` | 5,839 | ts_code, ann_date, end_date, pre_date, actual_date | 财报披露日期 |

## 四、指数数据

| 表名 | 文件数 | 关键字段 | 说明 |
|------|--------|---------|------|
| `index_daily` | 610 | ts_code, trade_date, close, open, high, low, pct_chg, vol, amount | 指数日线行情 |
| `index_dailybasic` | 1 | ts_code, trade_date, total_mv, pe, pb, turnover_rate | 指数每日指标 |
| `index_basic` | 1 | ts_code, name, market, publisher, category, base_date, base_point | 指数基础信息 |
| `index_classify` | 1 | index_code, industry_name, level, industry_code | 指数行业分类 |
| `index_weight` | 21 | index_code, con_code, trade_date, weight | 指数成分股及权重 |

## 五、板块概念（同花顺/东财）

| 表名 | 文件数 | 关键字段 | 说明 |
|------|--------|---------|------|
| `ths_index` | 1 | ts_code, name, count, exchange, type | 同花顺概念和行业指数列表 |
| `ths_member` | 1 | ts_code, con_code, con_name | 同花顺概念板块成分 |
| `ths_daily` | 1,571 | ts_code, trade_date, open, high, low, close, pct_change, vol, turnover_rate | 同花顺板块日线行情 |
| `dc_index` | 1 | ts_code, trade_date, name, leading, pct_change, total_mv, up_num, down_num | 东财概念板块指数 |
| `dc_member` | 1 | trade_date, ts_code, con_code, name | 东财板块成分 |
| `dc_daily` | 1,013 | ts_code, trade_date, close, pct_change, vol, amount, swing, turnover_rate | 东财板块日线 |

## 六、市场事件与情绪

| 表名 | 文件数 | 关键字段 | 说明 |
|------|--------|---------|------|
| `limit_list_d` | 1 | trade_date, ts_code, close, pct_chg, amount, limit_amount, float_mv, up_stat, limit_times, limit | 每日涨跌停统计（连板、开板次数） |
| `limit_step` | 1 | ts_code, name, trade_date, nums | 连板阶梯数据 |
| `top_list` | 1 | trade_date, ts_code, close, amount, l_sell, l_buy, net_amount, reason | 龙虎榜数据 |
| `block_trade` | 5,369 | ts_code, trade_date, price, vol, amount, buyer, seller | 大宗交易 |
| `share_float` | 5,733 | ts_code, ann_date, float_date, float_share, float_ratio, holder_name | 限售股解禁 |
| `new_share` | 1 | ts_code, name, ipo_date, issue_date, price, pe, amount | 新股上市信息 |
| `repurchase` | 3,666 | ts_code, ann_date, end_date, vol, amount, high_limit, low_limit | 股票回购 |

## 七、股东与持股

| 表名 | 文件数 | 关键字段 | 说明 |
|------|--------|---------|------|
| `stk_holdernumber` | 5,835 | ts_code, ann_date, end_date, holder_num | 股东人数变化 |
| `stk_holdertrade` | 5,239 | ts_code, ann_date, holder_name, holder_type, in_de, change_vol, avg_price | 股东增减持 |

## 八、基础信息

| 表名 | 文件数 | 关键字段 | 说明 |
|------|--------|---------|------|
| `stock_list` | 1 | ts_code, symbol, name, area, industry, market, list_date, act_name | 全市场股票列表 |
| `trade_calendar` | 1 | exchange, cal_date, is_open, pretrade_date | 交易日历 |
| `st` | 1 | ts_code, name, pub_date, imp_date, st_tpye, st_reason | ST/*ST 股票标记 |

---

## 当前模型已使用的表

| 脚本 | 使用的表 |
|------|---------|
| `01_data_preparation.py` | daily, daily_basic, index_daily |
| `01b_global_factors_extraction.py` | moneyflow_hsgt, index_daily |
| `01c_advanced_factors_extraction.py` | daily, daily_basic, moneyflow, stk_limit, limit_list_d |
| `01d_macro_valuation_extraction.py` | daily_basic, fina_indicator |
| `05_stock_data_prep.py` | daily, daily_basic, moneyflow, fina_indicator |

## 升级模型时可考虑引入的表

| 表 | 潜在用途 |
|---|---------|
| `stk_factor_pro` | 替代手动计算技术指标，200+ 预计算因子直接可用 |
| `moneyflow_hsgt` | 北向资金动向，外资定价因子 |
| `top_list` | 龙虎榜数据，游资/机构动向 |
| `block_trade` | 大宗交易折溢价信号 |
| `limit_list_d` | 连板高度、市场情绪温度计 |
| `forecast` / `express` | 业绩预告/快报，事件驱动因子 |
| `dividend` | 股息率、分红事件因子 |
| `share_float` | 限售解禁压力 |
| `stk_holdernumber` | 筹码集中度变化 |
| `stk_holdertrade` | 大股东增减持信号 |
| `ths_member` / `dc_member` | 概念板块归属，板块轮动因子 |
| `margin` | 两融余额，杠杆资金情绪 |
| `repurchase` | 回购信号 |
