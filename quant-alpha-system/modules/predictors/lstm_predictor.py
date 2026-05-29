"""LSTM 预测器 — 时序深度学习预测（含标准化+验证集+早停+sigmoid）"""

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
    """2层 LSTM + FC输出，概率输出加sigmoid"""

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


class MultiTaskLoss(nn.Module):
    """多任务损失：概率用BCE + 趋势用BCE + 目标价用Huber"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred[:, 0] = 上涨概率logits, target[:, 0] = 上涨概率(0/1)
        prob_loss = self.bce(pred[:, 0], target[:, 0])
        # pred[:, 1] = 趋势方向logits, target[:, 1] = 趋势(-1/0/1)归一化到0/1
        trend_loss = self.bce(pred[:, 1], (target[:, 1] + 1) / 2)
        # pred[:, 2] = 目标价倍数, target[:, 2] = 1+收益率
        price_loss = self.huber(pred[:, 2], target[:, 2])
        return prob_loss + 0.5 * trend_loss + 0.5 * price_loss


class LSTMPredictor:
    """LSTM 时序预测器 — 输出涨跌概率 + 趋势 + 目标价位"""

    name = "lstm_predictor"
    description = "基于 LSTM 的时序预测器，输出涨跌概率 + 趋势 + 目标价位"

    def __init__(self, input_dim: int, params: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        p = params or {}
        self.input_dim = input_dim
        self.hidden_dim = p.get("hidden_dim", 64)
        self.num_layers = p.get("num_layers", 2)
        self.epochs = p.get("epochs", 100)
        self.batch_size = p.get("batch_size", 32)
        self.patience = p.get("patience", 15)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.criterion = MultiTaskLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

        # 标准化参数(训练时fit)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        """在训练集上计算均值/标准差并标准化"""
        X_2d = X.reshape(-1, X.shape[-1])
        self._mean = np.nanmean(X_2d, axis=0)
        self._std = np.nanstd(X_2d, axis=0)
        self._std[self._std == 0] = 1.0
        return self._standardize_transform(X)

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        """使用已计算的均值/标准差标准化"""
        return (X - self._mean) / self._std

    def train(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        """训练 LSTM 模型 — 含验证集+早停"""
        self.logger.info(f"训练 LSTM: {X.shape}, epochs={self.epochs}, patience={self.patience}")

        # 标准化特征
        X_scaled = self._standardize_fit(X)

        # 时序切分: 前80%训练，后20%验证
        split_idx = int(len(X_scaled) * 0.8)
        X_train = torch.FloatTensor(X_scaled[:split_idx]).to(self.device)
        y_train = torch.FloatTensor(y[:split_idx]).to(self.device)
        X_val = torch.FloatTensor(X_scaled[split_idx:]).to(self.device)
        y_val = torch.FloatTensor(y[split_idx:]).to(self.device)

        # 转换标签: y[:,0]=收益率 → 二分类概率(正收益=1)
        y_train_prob = (y_train[:, 0] > 0).float()
        y_val_prob = (y_val[:, 0] > 0).float()
        y_train_adj = torch.column_stack([y_train_prob, y_train[:, 1], y_train[:, 2]])
        y_val_adj = torch.column_stack([y_val_prob, y_val[:, 1], y_val[:, 2]])

        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train_adj)
            loss.backward()
            self.optimizer.step()

            # 验证集评估
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val_adj)

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

            # 早停
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"早停触发: epoch {epoch+1}, best val loss: {best_val_loss:.6f}")
                    break

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.logger.info(f"训练完成. Best Val Loss: {best_val_loss:.6f}")

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """预测 — 概率输出经sigmoid激活"""
        self.logger.debug(f"预测: input shape {X.shape}")
        self.model.eval()

        X_scaled = self._standardize_transform(X)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(0)
            logits = self.model(X_tensor).cpu().numpy()

        # daily_prob: sigmoid(logits) → [0, 1]概率
        # trend: sigmoid(logits) → [0, 1]，>0.5看涨
        result = {
            "daily_prob": 1.0 / (1.0 + np.exp(-logits[:, 0])),  # sigmoid
            "trend": 1.0 / (1.0 + np.exp(-logits[:, 1])),
            "target_price": logits[:, 2],
        }
        return result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        preds = self.predict(X)
        prob_binary = (preds["daily_prob"] > 0.5).astype(float)
        return {
            "mse": float(np.mean((preds["target_price"] - y[:, 2]) ** 2)),
            "mae": float(np.mean(np.abs(preds["target_price"] - y[:, 2]))),
            "direction_acc": float(np.mean(prob_binary == (y[:, 0] > 0).astype(float))),
        }

    def save_model(self, path: Path) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "mean": self._mean,
            "std": self._std,
        }, path)

    def load_model(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self._mean = checkpoint.get("mean")
        self._std = checkpoint.get("std")
