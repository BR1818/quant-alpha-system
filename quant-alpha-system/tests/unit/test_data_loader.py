"""数据加载器测试"""
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.data.validator import DataValidator
from modules.data.cache import DataCache
from modules.data.quantdb_loader import QuantDBLoader


class TestDataValidator:
    """数据验证器测试"""

    def test_validate_empty_dataframe(self):
        """测试空 DataFrame 验证失败"""
        import pandas as pd
        validator = DataValidator()
        result = validator.validate(pd.DataFrame(), "stock_daily")
        assert result is False

    def test_validate_stock_daily_missing_columns(self):
        """测试缺少必需列验证失败"""
        import pandas as pd
        validator = DataValidator()
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        result = validator.validate(df, "stock_daily")
        assert result is False

    def test_validate_report(self):
        """测试验证报告生成"""
        import pandas as pd
        validator = DataValidator()
        validator.validate(
            pd.DataFrame({
                "trade_date": [], "open": [], "high": [], "low": [],
                "close": [], "volume": [], "amount": [],
            }),
            "stock_daily"
        )
        report = validator.get_validation_report()
        assert "total_validations" in report
        assert "passed" in report
        assert "failed" in report


class TestDataCache:
    """数据缓存测试"""

    def test_cache_set_and_get(self):
        """测试缓存写入和读取"""
        import pandas as pd
        tmpdir = Path(tempfile.mkdtemp())
        cache = DataCache(cache_dir=tmpdir)

        df = pd.DataFrame({"a": [1, 2, 3]})
        cache.set("test_key", df)

        result = cache.get("test_key")
        assert result is not None
        assert len(result) == 3
        assert list(result.columns) == ["a"]


class TestQuantDBLoader:
    """QuantDB 数据加载器测试"""

    def test_loader_initialization(self):
        """测试加载器初始化"""
        loader = QuantDBLoader(curated_dir=Path("/tmp/fake"))
        assert loader.curated_dir == Path("/tmp/fake")

    def test_load_stock_daily_file_not_found(self):
        """测试不存在的股票代码抛出异常"""
        loader = QuantDBLoader(curated_dir=Path("/tmp/fake"))
        with pytest.raises(FileNotFoundError):
            loader.load_stock_daily("INVALID.SZ", "20200101", "20201231")
