"""QuantDB Parquet 数据加载器"""

import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

from modules.data.cache import DataCache
from modules.data.validator import DataValidator


class QuantDBLoader:
    """QuantDB 数据加载器 — 从 Parquet 文件加载量化数据"""

    def __init__(self, curated_dir: Path, cache: Optional[DataCache] = None):
        self.curated_dir = Path(curated_dir)
        self.logger = logging.getLogger(__name__)
        self.cache = cache or DataCache()
        self.validator = DataValidator()

    def _load_parquet(self, table: str, ts_code: str) -> Optional[pd.DataFrame]:
        """加载单个 Parquet 文件"""
        file_path = self.curated_dir / table / f"{ts_code}.parquet"
        if not file_path.exists():
            self.logger.warning(f"文件不存在: {file_path}")
            return None
        table_obj = pq.read_table(file_path)
        return table_obj.to_pandas()

    def load_stock_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载个股日线数据并过滤日期"""
        cache_key = f"daily_{ts_code}_{start_date}_{end_date}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self.logger.info(f"加载个股日线: {ts_code}, {start_date} ~ {end_date}")
        df = self._load_parquet("daily", ts_code)
        if df is None:
            raise FileNotFoundError(f"日线数据文件不存在: {ts_code}")

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]

        self.validator.validate(df, "stock_daily")
        self.cache.set(cache_key, df)
        return df

    def load_stock_daily_enriched(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载个股日线数据，自动合并 daily_basic 和 moneyflow 表"""
        cache_key = f"daily_enriched_{ts_code}_{start_date}_{end_date}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self.logger.info(f"加载个股增强日线: {ts_code}, {start_date} ~ {end_date}")
        df = self.load_stock_daily(ts_code, start_date, end_date)

        # 合并 daily_basic（PE、PB、换手率、量比）
        try:
            basic = self._load_parquet("daily_basic", ts_code)
            if basic is not None:
                basic["trade_date"] = pd.to_datetime(basic["trade_date"])
                merge_cols = ["trade_date", "pe_ttm", "pb", "turnover_rate", "volume_ratio", "dv_ratio", "ps_ttm", "total_mv", "circ_mv"]
                available = [c for c in merge_cols if c in basic.columns]
                df = df.merge(basic[available], on="trade_date", how="left")
                self.logger.info(f"已合并 daily_basic: {[c for c in available if c != 'trade_date']}")
        except Exception as e:
            self.logger.warning(f"合并 daily_basic 失败: {e}")

        # 合并 moneyflow（主力资金净流入）
        try:
            mf = self._load_parquet("moneyflow", ts_code)
            if mf is not None:
                mf["trade_date"] = pd.to_datetime(mf["trade_date"])
                mf_cols = ["trade_date", "net_mf_amount"]
                available = [c for c in mf_cols if c in mf.columns]
                df = df.merge(mf[available], on="trade_date", how="left")
                self.logger.info("已合并 moneyflow: net_mf_amount")
        except Exception as e:
            self.logger.warning(f"合并 moneyflow 失败: {e}")

        # 合并 fina_indicator（ROE、营收增长率 — 季度数据，前向填充）
        try:
            fina = self._load_parquet("fina_indicator", ts_code)
            if fina is not None:
                # 修复未来函数：必须使用真实披露日(ann_date)而不是财报截止日(end_date)进行对齐
                fina["trade_date"] = pd.to_datetime(fina["ann_date"])
                fina_cols = ["trade_date", "roe", "q_sales_yoy", "eps", "net_profit_yoy"]
                available = [c for c in fina_cols if c in fina.columns]
                fina_sub = fina[available].sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last")
                df = df.merge(fina_sub, on="trade_date", how="left")
                # 前向填充季度数据到每日
                for c in [col for col in available if col != "trade_date"]:
                    df[c] = df[c].ffill()
                # 营收增长率映射到 revenue_yoy
                if "q_sales_yoy" in df.columns and "revenue_yoy" not in df.columns:
                    df["revenue_yoy"] = df["q_sales_yoy"]
                self.logger.info(f"已合并 fina_indicator (Point-in-Time): {[c for c in available if c != 'trade_date']}")
        except Exception as e:
            self.logger.warning(f"合并 fina_indicator 失败: {e}")

        # 合并 limit_list_d（涨跌停统计）
        try:
            limit = self._load_parquet("limit_list_d", ts_code)
            if limit is not None:
                limit["trade_date"] = pd.to_datetime(limit["trade_date"])
                limit_cols = ["trade_date", "limit_times"]
                available = [c for c in limit_cols if c in limit.columns]
                df = df.merge(limit[available], on="trade_date", how="left")
                if "limit_times" in df.columns:
                    df["limit_times"] = df["limit_times"].fillna(0)
                self.logger.info(f"已合并 limit_list_d: {[c for c in available if c != 'trade_date']}")
        except Exception as e:
            self.logger.warning(f"合并 limit_list_d 失败: {e}")

        self.validator.validate(df, "stock_daily")
        self.cache.set(cache_key, df)
        return df

    def load_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载指数日线数据"""
        cache_key = f"index_daily_{ts_code}_{start_date}_{end_date}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self.logger.info(f"加载指数日线: {ts_code}")
        df = self._load_parquet("index_daily", ts_code)
        if df is None:
            raise FileNotFoundError(f"指数日线数据文件不存在: {ts_code}")

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]

        self.validator.validate(df, "index_daily")
        self.cache.set(cache_key, df)
        return df

    def load_sector_stocks(self, sector_code: str) -> pd.DataFrame:
        """加载板块成分股 — 自动判断 ths_member 或 dc_member"""
        cache_key = f"sector_{sector_code}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self.logger.info(f"加载板块成分: {sector_code}")

        for source in ["ths_member", "dc_member"]:
            file_path = self.curated_dir / source / f"{source}.parquet"
            if file_path.exists():
                table = pq.read_table(file_path).to_pandas()
                # 过滤出该板块的成分股
                df = table[table["ts_code"] == sector_code]
                if not df.empty:
                    self.validator.validate(df, "sector_member")
                    self.cache.set(cache_key, df)
                    return df

        raise FileNotFoundError(f"板块数据文件不存在: {sector_code}")

    def load_factors(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载技术因子数据（stk_factor_pro 表）"""
        cache_key = f"factors_{ts_code}_{start_date}_{end_date}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self.logger.info(f"加载技术因子: {ts_code}")
        df = self._load_parquet("stk_factor_pro", ts_code)
        if df is None:
            raise FileNotFoundError(f"技术因子数据文件不存在: {ts_code}")

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]

        self.validator.validate(df, "factor")
        self.cache.set(cache_key, df)
        return df

    def get_available_stocks(self) -> List[str]:
        """获取可用股票列表"""
        daily_dir = self.curated_dir / "daily"
        if not daily_dir.exists():
            return []
        return sorted([f.stem for f in daily_dir.glob("*.parquet")])
