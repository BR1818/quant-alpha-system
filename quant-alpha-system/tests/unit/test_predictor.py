"""预测器测试"""
import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestLSTMPredictor:
    """LSTM 预测器测试"""

    def test_model_creation(self):
        """测试模型创建"""
        from modules.predictors.lstm_predictor import LSTMPredictor
        predictor = LSTMPredictor(input_dim=10, params={"epochs": 5, "hidden_dim": 32})
        assert predictor.name == "lstm_predictor"
        assert predictor.input_dim == 10

    def test_train_and_predict(self):
        """测试训练和预测"""
        from modules.predictors.lstm_predictor import LSTMPredictor

        np.random.seed(42)
        predictor = LSTMPredictor(input_dim=5, params={"epochs": 10, "hidden_dim": 16})

        X = np.random.randn(100, 20, 5).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)

        predictor.train(X, y)
        result = predictor.predict(X[:1])

        assert "daily_prob" in result
        assert "trend" in result
        assert "target_price" in result


class TestEnsemblePredictor:
    """集成预测器测试"""

    def test_ensemble_predict(self):
        """测试集成预测"""
        from modules.predictors.lstm_predictor import LSTMPredictor
        from modules.predictors.ensemble_predictor import EnsemblePredictor

        np.random.seed(42)
        p1 = LSTMPredictor(input_dim=5, params={"epochs": 5, "hidden_dim": 16})
        p2 = LSTMPredictor(input_dim=5, params={"epochs": 5, "hidden_dim": 16})

        ensemble = EnsemblePredictor([p1, p2], weights=[0.6, 0.4])

        X = np.random.randn(50, 20, 5).astype(np.float32)
        y = np.random.randn(50, 3).astype(np.float32)

        ensemble.train(X, y)
        result = ensemble.predict(X[:1])

        assert "daily_prob" in result
        assert "trend" in result
        assert "target_price" in result
