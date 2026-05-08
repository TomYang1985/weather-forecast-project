"""
训练循环工具
支持 GRU 和 Transformer 模型的训练、验证和早停。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import time
import os
import json


class EarlyStopping:
    """早停机制：当验证损失不再下降时停止训练，防止过拟合。"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    """
    训练一个 epoch。

    参数:
        model: 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备
        clip_grad: 梯度裁剪阈值

    返回:
        平均训练损失
    """
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    验证模型性能。

    返回包含 loss, mae, rmse 的字典。
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)

        # MSE → RMSE
        mse = torch.mean((pred - y) ** 2, dim=1)
        rmse = torch.sqrt(mse)
        total_rmse += rmse.sum().item()

        # MAE
        mae = torch.mean(torch.abs(pred - y), dim=1)
        total_mae += mae.sum().item()

        n_samples += x.size(0)

    return {
        "loss": total_loss / n_samples,
        "mae": total_mae / n_samples,
        "rmse": total_rmse / n_samples,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    device: str = "cuda",
    save_dir: str = "results",
    model_name: str = "weather_model",
    scheduler_type: str = "reduce_on_plateau",
) -> Dict:
    """
    完整的训练流程。

    参数:
        model: 要训练的模型
        train_loader: 训练数据
        val_loader: 验证数据
        epochs: 最大训练轮数
        lr: 学习率
        weight_decay: 权重衰减（L2 正则）
        patience: 早停耐心值
        device: 计算设备
        save_dir: 模型保存目录
        model_name: 模型名称
        scheduler_type: 学习率调度器类型

    返回:
        训练历史记录
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.makedirs(save_dir, exist_ok=True)

    # 损失函数：MSE
    criterion = nn.MSELoss()

    # 优化器：AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # 学习率调度器
    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

    # 早停
    early_stopping = EarlyStopping(patience=patience)

    # 记录
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training {model_name} on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_metrics = validate(model, val_loader, criterion, device)

        # 学习率调整
        if scheduler_type == "reduce_on_plateau":
            scheduler.step(val_metrics["loss"])
        elif scheduler_type == "cosine":
            scheduler.step()

        # 记录
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])

        elapsed = time.time() - start_time

        # 保存最佳模型
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                },
                os.path.join(save_dir, f"{model_name}_best.pth"),
            )

        # 打印进度
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | "
            f"RMSE: {val_metrics['rmse']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # 早停检查
        if early_stopping(val_metrics["loss"]):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"Training complete! Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Model saved to {os.path.join(save_dir, model_name + '_best.pth')}")
    print(f"{'='*60}")

    # 保存训练历史
    with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history
