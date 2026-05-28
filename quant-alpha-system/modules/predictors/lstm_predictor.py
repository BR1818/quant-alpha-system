"""LSTM 预测器 — 时序深度学习预测"""

import os
import platform

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# 修复 macOS + Python 3.14 环境下 PyTorch LSTM 多线程 segfault
if platform.system() == "Darwin":
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


class LSTMModel(nn.Module):
    """2层 LSTM + 全连接输出"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class LSTMPredictor:
    """LSTM 时序预测器 — 输出涨跌概率、趋势、目标价位"""

    name = "lstm_predictor"
    description = "基于 LSTM 的时序预测器，输出 7 日涨跌概率 + 4 周趋势 + 目标价位"

    def __init__(self, input_dim: int, params: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        p = params or {}
        self.input_dim = input_dim
        self.hidden_dim = p.get("hidden_dim", 64)
        self.num_layers = p.get("num_layers", 2)
        self.epochs = p.get("epochs", 100)
        self.batch_size = p.get("batch_size", 32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        """训练 LSTM 模型"""
        self.logger.info(f"训练 LSTM: {X.shape}, epochs={self.epochs}")
        self.model.train()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")

        self.logger.info(f"训练完成. Final Loss: {loss.item():.6f}")

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """预测"""
        self.logger.debug(f"预测: input shape {X.shape}")
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(0)
            predictions = self.model(X_tensor).cpu().numpy()

        result = {
            "daily_prob": predictions[:, 0],
            "trend": predictions[:, 1],
            "target_price": predictions[:, 2],
        }
        return result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        preds = self.predict(X)
        return {
            "mse": float(np.mean((preds["target_price"] - y[:, 2]) ** 2)),
            "mae": float(np.mean(np.abs(preds["target_price"] - y[:, 2]))),
            "direction_acc": float(np.mean((preds["daily_prob"] > 0.5) == (y[:, 0] > 0.5))),
        }

    def save_model(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
