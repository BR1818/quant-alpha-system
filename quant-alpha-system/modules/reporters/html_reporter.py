"""HTML 报告生成器 — 聚宽风格回测报告"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots


DARK_LAYOUT = dict(
    paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e',
    font=dict(color='#e0e0e0', family="'SF Pro Display', -apple-system, sans-serif"),
    margin=dict(l=60, r=30, t=50, b=40),
)
DARK_AXIS = dict(showgrid=True, gridwidth=1, gridcolor='#2d2d44', zerolinecolor='#2d2d44')
COLORS = dict(strategy='#00ff88', benchmark='#4a90e2', drawdown='#ff4d4f',
              excess='#ffd700', positive='#00ff88', negative='#ff4d4f',
              sharpe='#4a90e2', beta='#ffd700', vol='#ff8c00')


class HTMLReporter:
    """HTML 报告生成器 — 使用 Plotly 生成交互式图表"""

    def __init__(self, context):
        self.context = context
        self.logger = context.get_logger(__name__)

    def generate_analysis_report(self, results: Dict[str, Any], output_path: Path) -> Path:
        """生成个股分析报告"""
        self.logger.info(f"生成分析报告: {output_path}")
        charts = self._create_analysis_charts(results)
        html = self._render_analysis_template(results, charts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        self.logger.info(f"报告已保存: {output_path}")
        return output_path

    def generate_backtest_report(self, results: Dict[str, Any], output_path: Path) -> Path:
        """生成回测报告"""
        self.logger.info(f"生成回测报告: {output_path}")
        charts = self._create_backtest_charts(results)
        html = self._render_backtest_template(results, charts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        self.logger.info(f"报告已保存: {output_path}")
        return output_path

    # ============================================================
    # 分析报告 (保留不变)
    # ============================================================

    def _create_analysis_charts(self, results: Dict[str, Any]) -> str:
        figs = []

        if "stock_data" in results and len(results["stock_data"]) > 0:
            sd = results["stock_data"]
            if isinstance(sd, list):
                import pandas as pd
                sd = pd.DataFrame(sd)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, subplot_titles=('K线及趋势', '成交量'),
                                row_width=[0.2, 0.7])

            fig.add_trace(go.Candlestick(
                x=sd.get("trade_date", sd.index),
                open=sd["open"], high=sd["high"],
                low=sd["low"], close=sd["close"],
                name="K线"
            ), row=1, col=1)

            if "volume" in sd.columns:
                colors = ['red' if close > open else 'green' for close, open in zip(sd['close'], sd['open'])]
                fig.add_trace(go.Bar(
                    x=sd.get("trade_date", sd.index),
                    y=sd["volume"],
                    marker_color=colors,
                    name="成交量"
                ), row=2, col=1)

            fig.update_layout(title="个股行情分析", xaxis_rangeslider_visible=False, height=600,
                              paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e', font=dict(color='white'))
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')

            figs.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        if "predictions" in results:
            pred = results["predictions"]

            prob_data = pred.get("daily_prob", [])
            target_data = pred.get("target_price", [])

            fig = make_subplots(rows=2, cols=1, subplot_titles=("未来7个交易日上涨概率 (%)", "未来7个交易日目标价位"))
            if len(prob_data) > 0:
                days = [f"T+{i+1}" for i in range(len(prob_data))]
                colors = ['#ff4d4f' if p > 50 else '#52c41a' for p in prob_data]
                fig.add_trace(go.Bar(x=days, y=list(prob_data), marker_color=colors, name="上涨概率"), row=1, col=1)
                fig.update_yaxes(range=[0, 100], row=1, col=1)
            if len(target_data) > 0:
                fig.add_trace(go.Scatter(x=days, y=list(target_data), mode="lines+markers", line=dict(color="#4a90e2", width=3), marker=dict(size=8), name="目标价"), row=2, col=1)
            fig.update_layout(height=500, paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e', font=dict(color='white'))
            figs.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        return "\n".join(figs)

    def _render_analysis_template(self, results: Dict[str, Any], charts: str) -> str:
        predictions = results.get('predictions', {})
        action = predictions.get("action", "暂无")
        reasons = predictions.get("reason", [])
        action_color = "#ff4d4f" if action == "买入" else ("#52c41a" if action == "卖出" else "#faad14")

        reason_html = "".join([f'<div style="margin-bottom: 10px; font-size: 15px; line-height: 1.6;">{r}</div>' for r in reasons])

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化分析报告 - {results.get('stock_name', '')} {results.get('stock_code', '')}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #2c2c2c; display: flex; justify-content: space-between; align-items: center; }}
.header h1 {{ font-size: 32px; margin-bottom: 10px; font-weight: 600; letter-spacing: 1px; }}
.header p {{ color: #b0c4de; font-size: 16px; }}
.action-badge {{ background: {action_color}; color: white; padding: 10px 25px; border-radius: 30px; font-size: 24px; font-weight: bold; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 24px; }}
.metric-card {{ background: #1e1e1e; padding: 24px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); border: 1px solid #2c2c2c; transition: transform 0.2s; }}
.metric-card:hover {{ transform: translateY(-5px); border-color: #4a90e2; }}
.metric-value {{ font-size: 28px; font-weight: bold; color: #4a90e2; margin-bottom: 8px; }}
.metric-label {{ font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
.section {{ background: #1e1e1e; padding: 24px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); border: 1px solid #2c2c2c; }}
.section h2 {{ font-size: 22px; margin-bottom: 20px; color: #e0e0e0; border-bottom: 2px solid #2c2c2c; padding-bottom: 10px; }}
.chart {{ margin: 20px 0; border-radius: 8px; overflow: hidden; }}
.reason-box {{ background: rgba(74, 144, 226, 0.1); border-left: 4px solid #4a90e2; padding: 20px; border-radius: 4px; }}
.logs {{ background: #000000; color: #00ff00; padding: 20px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 13px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; border: 1px solid #333; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <div>
    <h1>量化分析报告</h1>
    <p>{results.get('stock_name', '')} | {results.get('stock_code', '')} | {results.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))}</p>
  </div>
  <div class="action-badge">AI建议: {action}</div>
</div>

<div class="section">
  <h2>AI 核心诊断逻辑</h2>
  <div class="reason-box">
    {reason_html}
  </div>
</div>

<div class="section"><h2>未来走势预测摘要</h2><div class="metrics">{self._render_metrics(predictions)}</div></div>
<div class="section"><h2>图表分析</h2><div class="chart">{charts}</div></div>
<div class="section"><h2>系统执行日志</h2><div class="logs">{json.dumps(self.context.traces[-20:], indent=2, ensure_ascii=False, default=str)}</div></div>
</div>
</body>
</html>"""

    def _render_metrics(self, predictions: Dict[str, Any]) -> str:
        items = []
        if "win_rate" in predictions:
            wr = predictions["win_rate"]
            color = "#ff4d4f" if wr > 50 else "#52c41a"
            items.append(f'<div class="metric-card"><div class="metric-value" style="color:{color}">{wr}%</div><div class="metric-label">策略综合胜率</div></div>')

        if "daily_prob" in predictions and len(predictions["daily_prob"]) > 0:
            p = predictions["daily_prob"][0]
            items.append(f'<div class="metric-card"><div class="metric-value">{p}%</div><div class="metric-label">T+1 上涨概率</div></div>')

        if "target_price" in predictions and len(predictions["target_price"]) > 0:
            tp = predictions["target_price"][-1]
            curr = predictions.get("current_price", 1)
            space = ((tp / curr) - 1) * 100 if curr > 0 else 0
            items.append(f'<div class="metric-card"><div class="metric-value">{tp}</div><div class="metric-label">T+7 目标价 (空间 {space:.1f}%)</div></div>')

        return "\n".join(items)

    # ============================================================
    # 回测报告 — 聚宽风格
    # ============================================================

    def _create_backtest_charts(self, results: Dict[str, Any]) -> Dict[str, str]:
        """生成7个回测图表，返回 {name: html_script} 字典"""
        charts = {}
        pv = results.get("portfolio_values")
        dr = results.get("daily_returns")
        bdr = results.get("benchmark_daily_returns")

        if pv is None or len(pv) == 0 or dr is None or len(dr) == 0:
            return charts

        charts["equity_curve"] = self._chart_equity_curve(pv, dr, bdr)
        charts["drawdown"] = self._chart_drawdown(pv)
        charts["monthly_heatmap"] = self._chart_monthly_heatmap(dr)
        charts["daily_returns_dist"] = self._chart_daily_returns_dist(dr)
        charts["rolling_sharpe"] = self._chart_rolling_sharpe(dr)
        charts["rolling_volatility"] = self._chart_rolling_volatility(dr)
        if bdr is not None:
            charts["rolling_beta"] = self._chart_rolling_beta(dr, bdr)

        return charts

    def _chart_equity_curve(self, pv, dr, bdr) -> str:
        """净值曲线 + 基准对比 + 超额收益"""
        dates = pv["date"]
        nav = pv["value"] / pv["value"].iloc[0]

        rows = 2 if bdr is not None else 1
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25] if rows == 2 else [1.0],
                            vertical_spacing=0.05,
                            subplot_titles=('净值曲线', '超额收益') if rows == 2 else ('净值曲线',))

        fig.add_trace(go.Scatter(x=dates, y=nav, name='策略净值',
                                 line=dict(color=COLORS['strategy'], width=2)), row=1, col=1)

        if bdr is not None:
            bench_cum = (1 + bdr).cumprod()
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name='基准净值',
                                     line=dict(color=COLORS['benchmark'], width=2, dash='dash')), row=1, col=1)

            strat_cum = (1 + dr).cumprod()
            excess = (strat_cum - bench_cum) * 100
            fig.add_trace(go.Scatter(x=excess.index, y=excess.values, name='超额收益%',
                                     line=dict(color=COLORS['excess'], width=1.5),
                                     fill='tozeroy', fillcolor='rgba(255,215,0,0.1)'),
                          row=2, col=1)

        fig.update_layout(height=500, **DARK_LAYOUT)
        fig.update_xaxes(**DARK_AXIS)
        fig.update_yaxes(**DARK_AXIS)
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart-equity")

    def _chart_drawdown(self, pv) -> str:
        """回撤水下图"""
        cumulative = pv["value"] / pv["value"].iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        dates = pv["date"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=drawdown, fill='tozeroy',
                                 fillcolor='rgba(255,77,79,0.3)',
                                 line=dict(color=COLORS['drawdown'], width=1.5), name='回撤'))

        # 标注最大回撤
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        max_dd_date = dates.iloc[max_dd_idx] if max_dd_idx < len(dates) else dates.iloc[-1]
        fig.add_annotation(x=max_dd_date, y=max_dd_val,
                           text=f'最大回撤: {max_dd_val:.2f}%',
                           showarrow=True, arrowhead=2,
                           font=dict(color='white', size=12),
                           arrowcolor='#ff4d4f', ax=40, ay=-30)

        fig.update_layout(title='回撤分析', height=300, yaxis_title='回撤 (%)', **DARK_LAYOUT)
        fig.update_xaxes(**DARK_AXIS)
        fig.update_yaxes(**DARK_AXIS)
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart-drawdown")

    def _chart_monthly_heatmap(self, dr) -> str:
        """月度收益热力图"""
        monthly = dr.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        pivot = monthly_df.pivot_table(index='year', columns='month', values='return', aggfunc='first')

        month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                        '7月', '8月', '9月', '10月', '11月', '12月']
        # 补齐12列
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot[sorted(pivot.columns)]

        text_vals = [[f'{v:.1f}%' if not np.isnan(v) else '' for v in row] for row in pivot.values]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=month_labels[:len(pivot.columns)],
            y=[str(y) for y in pivot.index],
            colorscale=[[0, '#ff4d4f'], [0.5, '#1a1a2e'], [1, '#00ff88']],
            zmid=0,
            text=text_vals,
            texttemplate='%{text}',
            textfont=dict(size=12),
            colorbar=dict(title=dict(text='收益率%', font=dict(color='#e0e0e0')),
                          tickfont=dict(color='#e0e0e0')),
        ))

        fig.update_layout(title='月度收益率热力图 (%)', height=400, **DARK_LAYOUT)
        fig.update_xaxes(side='bottom', **DARK_AXIS)
        fig.update_yaxes(**DARK_AXIS)
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart-monthly")

    def _chart_daily_returns_dist(self, dr) -> str:
        """日收益率分布直方图"""
        ret_pct = dr.values * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ret_pct, nbinsx=60,
                                   marker_color=COLORS['sharpe'], opacity=0.75,
                                   name='日收益率', histnorm=''))

        # 正态分布叠加
        mu, sigma = ret_pct.mean(), ret_pct.std()
        if sigma > 0:
            x_range = np.linspace(ret_pct.min(), ret_pct.max(), 200)
            pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
            bin_width = (ret_pct.max() - ret_pct.min()) / 60
            pdf_scaled = pdf * len(ret_pct) * bin_width
            fig.add_trace(go.Scatter(x=x_range, y=pdf_scaled, mode='lines',
                                     line=dict(color=COLORS['excess'], width=2),
                                     name='正态分布'))

        # 均值线
        fig.add_vline(x=mu, line_dash='dash', line_color='#00ff88',
                      annotation_text=f'均值: {mu:.2f}%', annotation_position='top left',
                      annotation_font_color='#00ff88')

        fig.update_layout(title='日收益率分布', height=350,
                          xaxis_title='日收益率 (%)', yaxis_title='频次', **DARK_LAYOUT)
        fig.update_xaxes(**DARK_AXIS)
        fig.update_yaxes(**DARK_AXIS)
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart-dist")

    def _chart_rolling_sharpe(self, dr) -> str:
        """滚动夏普比率"""
        window = 120
        rf_daily = 0.02 / 252
        rolling_sharpe = dr.rolling(window).apply(
            lambda x: (x.mean() - rf_daily) / x.std() * np.sqrt(252) if x.std() > 0 else 0,
            raw=True)
        rolling_sharpe = rolling_sharpe.dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                                 line=dict(color=COLORS['sharpe'], width=1.5), name='滚动夏普'))
        fig.add_hline(y=0, line_dash='dash', line_color='#555')
        fig.add_hline(y=1, line_dash='dash', line_color=COLORS['positive'],
                      annotation_text='夏普=1', annotation_position='top right',
                      annotation_font_color=COLORS['positive'])

        fig.update_layout(title=f'滚动夏普比率 ({window}日)', height=300,
                          yaxis_title='夏普比率', **DARK_LAYOUT)
        fig.update_xaxes(**DARK_AXIS)
        fig.update_yaxes(**DARK_AXIS)
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart-sharpe")

    def _chart_rolling_beta(self, dr, bdr) -> str:
        """滚动Beta"""
        import pandas as pd
        window = 120
        aligned = pd.DataFrame({"strategy": dr, "benchmark": bdr}).dropna()
        if len(aligned) < window:
            return ""

        betas = []
        dates_list = []
        for i in range(window, len(aligned)):
            w_s = aligned["strategy"].values[i - window:i]
            w_b = aligned["benchmark"].values[i - window:i]
            cov = np.cov(w_s, w_b)[0, 1]
            var = np.var(w_b)
            betas.append(cov / var if var > 0 else 0)
            dates_list.append(aligned.index[i])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates_list, y=betas,
                                 line=dict(color=COLORS['beta'], width=1.5), name='滚动Beta'))
        fig.add_hline(y=1, line_dash='dash', line_color='#555',
                      annotation_text='Beta=1', annotation_position='top right',
                      annotation_font_color='#888')

        fig.update_layout(title=f'滚动Beta ({window}日)', height=300,
                          yaxis_title='Beta', **DARK_LAYOUT)
        fig.update_xaxes(**DARK_AXIS)
        fig.update_yaxes(**DARK_AXIS)
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart-beta")

    def _chart_rolling_volatility(self, dr) -> str:
        """滚动波动率"""
        window = 120
        rolling_vol = dr.rolling(window).std() * np.sqrt(252) * 100
        rolling_vol = rolling_vol.dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                                 line=dict(color=COLORS['vol'], width=1.5), name='滚动波动率'))

        fig.update_layout(title=f'滚动年化波动率 ({window}日)', height=300,
                          yaxis_title='波动率 (%)', **DARK_LAYOUT)
        fig.update_xaxes(**DARK_AXIS)
        fig.update_yaxes(**DARK_AXIS)
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart-vol")

    # ============================================================
    # 回测报告模板 — 聚宽风格
    # ============================================================

    def _render_backtest_template(self, results: Dict[str, Any], charts: Dict[str, str]) -> str:
        m = results.get("metrics", {})
        trade_log = results.get("trade_log", [])
        has_benchmark = m.get("benchmark_annual_return_pct") is not None
        initial_cash = results.get("initial_cash", 1000000)

        # 指标卡片颜色
        def val_color(v, positive_good=True):
            if v is None:
                return '#555'
            if positive_good:
                return '#00ff88' if v > 0 else '#ff4d4f'
            else:
                return '#ff4d4f' if v > 0 else '#00ff88'

        def fmt(v, suffix='', dash='--'):
            return f'{v:.2f}{suffix}' if v is not None else dash

        # 指标卡片HTML
        def card(label, value, color='#4a90e2'):
            return f'<div class="metric-card"><div class="metric-value" style="color:{color}">{value}</div><div class="metric-label">{label}</div></div>'

        row1 = "".join([
            card('累计收益率', fmt(m.get('total_return_pct'), '%'), val_color(m.get('total_return_pct'))),
            card('年化收益率', fmt(m.get('annual_return_pct'), '%'), val_color(m.get('annual_return_pct'))),
            card('基准年化收益率', fmt(m.get('benchmark_annual_return_pct'), '%', '无基准'),
                 val_color(m.get('benchmark_annual_return_pct')) if has_benchmark else '#555'),
            card('超额收益率', fmt(m.get('excess_return_pct'), '%', '无基准'),
                 val_color(m.get('excess_return_pct')) if has_benchmark else '#555'),
        ])

        row2 = "".join([
            card('最大回撤', fmt(m.get('max_drawdown_pct'), '%'), '#ff4d4f'),
            card('收益波动率', fmt(m.get('volatility_pct'), '%'), '#4a90e2'),
            card('下行风险', fmt(m.get('downside_risk_pct'), '%'), '#4a90e2'),
        ])

        row3 = "".join([
            card('夏普比率', fmt(m.get('sharpe_ratio')), '#4a90e2'),
            card('索提诺比率', fmt(m.get('sortino_ratio')), '#4a90e2'),
            card('卡玛比率', fmt(m.get('calmar_ratio')), '#4a90e2'),
            card('信息比率', fmt(m.get('information_ratio'), dash='无基准'),
                 '#4a90e2' if has_benchmark else '#555'),
            card('Alpha·Beta',
                 f'{fmt(m.get("alpha"))} / {fmt(m.get("beta"))}' if has_benchmark else '无基准',
                 '#ffd700' if has_benchmark else '#555'),
        ])

        row4 = "".join([
            card('胜率', fmt(m.get('win_rate_pct'), '%'), val_color(m.get('win_rate_pct'))),
            card('盈亏比', fmt(m.get('profit_loss_ratio')), '#4a90e2'),
            card('交易次数', str(m.get('trade_count', 0)), '#e0e0e0'),
            card('平均持仓天数', fmt(m.get('avg_hold_days'), '天'), '#e0e0e0'),
        ])

        # 图表HTML
        chart_equity = charts.get("equity_curve", "")
        chart_drawdown = charts.get("drawdown", "")
        chart_monthly = charts.get("monthly_heatmap", "")
        chart_dist = charts.get("daily_returns_dist", "")
        chart_sharpe = charts.get("rolling_sharpe", "")
        chart_beta = charts.get("rolling_beta", "")
        chart_vol = charts.get("rolling_volatility", "")

        # 交易明细表
        trade_rows = ""
        for i, t in enumerate(trade_log):
            ret_pct = (t.get('pnlcomm', 0) / (t.get('entry_price', 1) * abs(t.get('size', 1)))) * 100 if t.get('size') else 0
            ret_color = '#00ff88' if t.get('pnlcomm', 0) > 0 else '#ff4d4f'
            trade_rows += f"""<tr>
                <td>{i+1}</td>
                <td>{t.get('entry_date', '')}</td>
                <td>{t.get('exit_date', '')}</td>
                <td>{t.get('barlen', 0)}</td>
                <td>{t.get('entry_price', 0):.2f}</td>
                <td>{t.get('exit_price', 0):.2f}</td>
                <td style="color:{' #00ff88' if t.get('pnl', 0) > 0 else '#ff4d4f'}">{t.get('pnl', 0):.2f}</td>
                <td>{t.get('commission', 0):.2f}</td>
                <td style="color:{ret_color}">{ret_pct:.2f}%</td>
            </tr>"""

        # 持仓分析
        if trade_log:
            pnls = [t.get('pnlcomm', 0) for t in trade_log]
            best_trade = max(pnls)
            worst_trade = min(pnls)
            avg_days = m.get('avg_hold_days', 0)
        else:
            best_trade = worst_trade = avg_days = 0

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>策略回测报告</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #2c2c2c; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; font-weight: 600; letter-spacing: 1px; }}
.header p {{ color: #b0c4de; font-size: 14px; }}
.header-info {{ display: flex; gap: 30px; margin-top: 12px; }}
.header-info span {{ color: #8ab4f8; font-size: 13px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 20px; }}
.metric-card {{ background: #1e1e2e; padding: 18px 20px; border-radius: 10px; border: 1px solid #2d2d44; text-align: center; transition: transform 0.2s, border-color 0.2s; }}
.metric-card:hover {{ transform: translateY(-3px); border-color: #4a90e2; }}
.metric-value {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
.metric-label {{ font-size: 12px; color: #888; letter-spacing: 0.5px; }}
.section {{ background: #1e1e2e; padding: 24px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); border: 1px solid #2d2d44; }}
.section h2 {{ font-size: 20px; margin-bottom: 16px; color: #e0e0e0; border-bottom: 2px solid #2d2d44; padding-bottom: 10px; }}
.tab-header {{ display: flex; border-bottom: 2px solid #2d2d44; margin-bottom: 20px; }}
.tab-btn {{ padding: 12px 28px; background: transparent; color: #888; border: none; font-size: 15px; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: color 0.2s, border-color 0.2s; }}
.tab-btn:hover {{ color: #ccc; }}
.tab-btn.active {{ color: #4a90e2; border-bottom-color: #4a90e2; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.chart {{ margin: 16px 0; border-radius: 8px; overflow: hidden; }}
.trade-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.trade-table th {{ background: #2d2d44; color: #e0e0e0; padding: 10px 12px; text-align: left; font-weight: 600; }}
.trade-table td {{ padding: 8px 12px; border-bottom: 1px solid #2d2d44; }}
.trade-table tr:hover {{ background: rgba(74, 144, 226, 0.1); }}
.pos-summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
.pos-item {{ background: #16162a; padding: 16px; border-radius: 8px; border: 1px solid #2d2d44; }}
.pos-item .label {{ font-size: 12px; color: #888; margin-bottom: 4px; }}
.pos-item .value {{ font-size: 20px; font-weight: 600; }}
.logs {{ background: #000000; color: #00ff00; padding: 20px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 13px; max-height: 300px; overflow-y: auto; white-space: pre-wrap; border: 1px solid #333; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>策略回测报告</h1>
  <p>策略名称: {results.get('strategy_name', 'QuantAlphaStrategy')}</p>
  <div class="header-info">
    <span>回测区间: {results.get('start_date', '')} ~ {results.get('end_date', '')}</span>
    <span>初始资金: ¥{initial_cash:,.0f}</span>
    <span>基准: {'沪深300' if has_benchmark else '无'}</span>
  </div>
</div>

<div class="section">
  <div class="metrics">{row1}</div>
  <div class="metrics">{row2}</div>
  <div class="metrics">{row3}</div>
  <div class="metrics">{row4}</div>
</div>

<div class="section">
  <div class="tab-header">
    <button class="tab-btn active" onclick="switchTab('overview', this)">收益概况</button>
    <button class="tab-btn" onclick="switchTab('risk', this)">风险分析</button>
    <button class="tab-btn" onclick="switchTab('trades', this)">交易详情</button>
    <button class="tab-btn" onclick="switchTab('positions', this)">持仓分析</button>
  </div>

  <div id="tab-overview" class="tab-content active">
    <div class="chart">{chart_equity}</div>
    <div class="chart">{chart_monthly}</div>
  </div>

  <div id="tab-risk" class="tab-content">
    <div class="chart">{chart_drawdown}</div>
    <div class="chart">{chart_sharpe}</div>
    {'<div class="chart">' + chart_beta + '</div>' if chart_beta else ''}
    <div class="chart">{chart_vol}</div>
    <div class="chart">{chart_dist}</div>
  </div>

  <div id="tab-trades" class="tab-content">
    {'<table class="trade-table"><thead><tr><th>序号</th><th>买入日期</th><th>卖出日期</th><th>持仓天数</th><th>买入价</th><th>卖出价</th><th>净利润</th><th>手续费</th><th>净收益率</th></tr></thead><tbody>' + trade_rows + '</tbody></table>' if trade_log else '<p style="color:#888;text-align:center;padding:40px">无交易记录</p>'}
  </div>

  <div id="tab-positions" class="tab-content">
    <div class="pos-summary">
      <div class="pos-item"><div class="label">盈利交易</div><div class="value" style="color:#00ff88">{m.get('won_trades', 0)} 笔</div></div>
      <div class="pos-item"><div class="label">亏损交易</div><div class="value" style="color:#ff4d4f">{m.get('lost_trades', 0)} 笔</div></div>
      <div class="pos-item"><div class="label">最大单笔盈利</div><div class="value" style="color:#00ff88">¥{best_trade:,.2f}</div></div>
      <div class="pos-item"><div class="label">最大单笔亏损</div><div class="value" style="color:#ff4d4f">¥{worst_trade:,.2f}</div></div>
      <div class="pos-item"><div class="label">平均持仓天数</div><div class="value" style="color:#4a90e2">{avg_days:.1f} 天</div></div>
      <div class="pos-item"><div class="label">期末资产</div><div class="value" style="color:#e0e0e0">¥{m.get('final_value', 0):,.2f}</div></div>
    </div>
  </div>
</div>

<div class="section">
  <h2>执行日志</h2>
  <div class="logs">{json.dumps(self.context.traces[-20:], indent=2, ensure_ascii=False, default=str)}</div>
</div>
</div>

<script>
function switchTab(tabName, btn) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    btn.classList.add('active');
    // 触发Plotly图表重绘
    window.dispatchEvent(new Event('resize'));
}}
</script>
</body>
</html>"""
