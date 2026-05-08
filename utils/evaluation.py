"""
评估与可视化工具
包含预测结果可视化、模型性能对比和误差分析。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合服务器

from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import os


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对数据集进行预测，返回所有预测值和真实值。

    返回:
        predictions: (n_samples, pred_len)
        targets: (n_samples, pred_len)
    """
    model.eval()
    all_preds = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device)
        pred = model(x)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_idx: int = 0,
    save_path: str = None,
    title: str = "Weather Forecast",
):
    """
    绘制单个样本的预测 vs 真实值对比图。

    参数:
        y_true: 真实值 (samples, pred_len)
        y_pred: 预测值 (samples, pred_len)
        sample_idx: 可视化第几个样本
        save_path: 图片保存路径
    """
    true = y_true[sample_idx]
    pred = y_pred[sample_idx]
    hours = np.arange(len(true))

    plt.figure(figsize=(10, 5))
    plt.plot(hours, true, "b-o", label="True Temperature", markersize=4)
    plt.plot(hours, pred, "r--s", label="Predicted Temperature", markersize=4)
    plt.fill_between(
        hours,
        true,
        pred,
        alpha=0.15,
        color="gray",
        label=f"Error (MAE: {np.mean(np.abs(true - pred)):.2f})",
    )

    plt.xlabel("Hours Ahead")
    plt.ylabel("Temperature (°C)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()


def plot_error_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    title: str = "Prediction Error by Horizon",
):
    """
    绘制不同预测步数的误差曲线。
    展示模型在短期(1-6h)、中期(7-12h)、长期(13-24h)预测能力。
    """
    errors = np.abs(y_true - y_pred)  # (samples, pred_len)
    mean_mae = np.mean(errors, axis=0)
    std_mae = np.std(errors, axis=0)
    hours = np.arange(1, len(mean_mae) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(hours, mean_mae, "b-", linewidth=2, label="Mean MAE")
    plt.fill_between(
        hours,
        mean_mae - std_mae,
        mean_mae + std_mae,
        alpha=0.2,
        color="blue",
        label="±1 Std",
    )

    # 标注短期/中期/长期区域
    pred_len = len(hours)
    third = pred_len // 3
    for start, end, label, color in [
        (0, third, "Short-term", "green"),
        (third, 2 * third, "Medium-term", "orange"),
        (2 * third, pred_len, "Long-term", "red"),
    ]:
        plt.axvspan(start, end, alpha=0.08, color=color)
        plt.text(
            (start + end) / 2,
            plt.ylim()[1] * 0.95,
            label,
            ha="center",
            fontsize=10,
            color=color,
        )

    plt.xlabel("Forecast Horizon (hours)")
    plt.ylabel("Mean Absolute Error (°C)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()


def plot_training_history(
    history: Dict,
    save_path: str = None,
    title: str = "Training History",
):
    """
    绘制训练过程曲线：损失、MAE、RMSE 随 epoch 的变化。
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(epochs, history["val_mae"], "g-", label="Val MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (°C)")
    axes[1].set_title("Validation MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # RMSE
    axes[2].plot(epochs, history["val_rmse"], "m-", label="Val RMSE")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("RMSE (°C)")
    axes[2].set_title("Validation RMSE")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()


def compare_models(
    model_results: Dict[str, Dict],
    save_path: str = None,
):
    """
    对比多个模型的性能。

    参数:
        model_results: {模型名: {"mae": float, "rmse": float, ...}, ...}
    """
    names = list(model_results.keys())
    maes = [model_results[n]["mae"] for n in names]
    rmses = [model_results[n]["rmse"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, maes, width, label="MAE (°C)", color="steelblue")
    bars2 = ax.bar(x + width / 2, rmses, width, label="RMSE (°C)", color="coral")

    ax.set_xlabel("Model")
    ax.set_ylabel("Error (°C)")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 在柱子上标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()
