"""数据缓存：内存 + 磁盘双层缓存"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional
import pandas as pd
import logging


class DataCache:
    """数据缓存管理器 — 内存优先，磁盘兜底"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self._memory_cache: dict = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        if key in self._memory_cache:
            self.logger.debug(f"内存缓存命中: {key}")
            return self._memory_cache[key]

        if self.cache_dir:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
            if cache_file.exists():
                self.logger.debug(f"磁盘缓存命中: {key}")
                df = pd.read_pickle(cache_file)
                self._memory_cache[key] = df
                return df

        return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        """缓存数据"""
        self._memory_cache[key] = df.copy()
        if self.cache_dir:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
            df.to_pickle(cache_file)
            self.logger.debug(f"已缓存至磁盘: {key}")

    def clear(self) -> None:
        """清空所有缓存"""
        self._memory_cache.clear()
        if self.cache_dir:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
