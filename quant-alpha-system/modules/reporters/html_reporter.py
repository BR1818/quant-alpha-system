"""HTML 报告生成器"""

import json
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

    def _create_analysis_charts(self, results: Dict[str, Any]) -> str:
        """生成分析图表"""
        figs = []

        if "stock_data" in results and len(results["stock_data"]) > 0:
            sd = results["stock_data"]
            if isinstance(sd, list):
                import pandas as pd
                sd = pd.DataFrame(sd)

            # 使用 subplots 添加成交量
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

    def _create_backtest_charts(self, results: Dict[str, Any]) -> str:
        """生成回测图表"""
        figs = []

        portfolio_values = results.get("portfolio_values")
        if portfolio_values is not None and len(portfolio_values) > 0:
            import pandas as pd
            pv = portfolio_values if isinstance(portfolio_values, pd.DataFrame) else pd.DataFrame(portfolio_values)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pv["date"], y=pv["value"], mode="lines", name="净值"))
            fig.update_layout(title="组合净值曲线", xaxis_title="日期", yaxis_title="净值", height=400)
            figs.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

            cumulative = pv["value"] / pv["value"].iloc[0]
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max * 100

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=pv["date"], y=drawdown, fill="tozeroy", name="回撤%"))
            fig2.update_layout(title="回撤曲线", xaxis_title="日期", yaxis_title="回撤 (%)", height=300)
            figs.append(fig2.to_html(full_html=False, include_plotlyjs="cdn"))

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

    def _render_backtest_template(self, results: Dict[str, Any], charts: str) -> str:
        metrics = results.get("metrics", {})
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>回测报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #2c2c2c; }}
.header h1 {{ font-size: 32px; margin-bottom: 10px; font-weight: 600; letter-spacing: 1px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 24px; }}
.metric-card {{ background: #1e1e1e; padding: 24px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); border: 1px solid #2c2c2c; text-align: center; transition: transform 0.2s; }}
.metric-card:hover {{ transform: translateY(-5px); border-color: #2c5364; }}
.metric-value {{ font-size: 28px; font-weight: bold; color: #00ff88; margin-bottom: 8px; }}
.metric-label {{ font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
.section {{ background: #1e1e1e; padding: 24px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); border: 1px solid #2c2c2c; }}
.section h2 {{ font-size: 22px; margin-bottom: 20px; color: #e0e0e0; border-bottom: 2px solid #2c2c2c; padding-bottom: 10px; }}
.chart {{ margin: 20px 0; border-radius: 8px; overflow: hidden; }}
.logs {{ background: #000000; color: #00ff00; padding: 20px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 13px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; border: 1px solid #333; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>策略回测报告</h1>
  <p>回测区间: {results.get('start_date', '')} ~ {results.get('end_date', '')}</p>
</div>
<div class="section"><h2>回测指标</h2>
<div class="metrics">
  <div class="metric-card"><div class="metric-value">{metrics.get('total_return_pct', 0):.2f}%</div><div class="metric-label">总收益率</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get('annual_return_pct', 0):.2f}%</div><div class="metric-label">年化收益率</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div><div class="metric-label">夏普比率</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get('max_drawdown_pct', 0):.2f}%</div><div class="metric-label">最大回撤</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get('win_rate_pct', 0):.2f}%</div><div class="metric-label">胜率</div></div>
</div>
</div>
<div class="section"><h2>图表分析</h2><div class="chart">{charts}</div></div>
<div class="section"><h2>执行日志</h2><div class="logs">{json.dumps(self.context.traces[-20:], indent=2, ensure_ascii=False, default=str)}</div></div>
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
