#!/usr/bin/env python3
"""
天气预测模型训练入口脚本
支持 GRU 和 Transformer 两种架构。
"""

import argparse
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from utils.data_loader import load_jena_data, prepare_weather_data, create_data_loaders
from utils.training import train_model
from utils.evaluation import (
    predict,
    plot_predictions,
    plot_error_by_horizon,
    plot_training_history,
)
from models.gru_model import WeatherGRU, WeatherGRUWithAttention
from models.transformer_model import WeatherTransformer, PatchTSTStyleTransformer


def main():
    parser = argparse.ArgumentParser(description="Train weather forecasting model")

    # 模型选择
    parser.add_argument(
        "--model",
        type=str,
        default="gru",
        choices=["gru", "gru_attn", "transformer", "patchtst"],
        help="Model architecture",
    )

    # 数据参数
    parser.add_argument("--data_path", type=str, default=None, help="Path to CSV data")
    parser.add_argument("--seq_len", type=int, default=168, help="Input sequence length (e.g., 168 = 7 days)")
    parser.add_argument("--pred_len", type=int, default=24, help="Prediction length (e.g., 24 hours)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # 模型超参数
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GRU/Transformer layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # 其他
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="results", help="Save directory")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 加载数据 ───────────────────────────
    print("\n📂 Loading weather data...")
    df = load_jena_data(args.data_path)

    # 使用 Jena 数据集的 14 个特征
    feature_cols = [
        "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)",
        "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)",
        "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
        "wv (m/s)", "max. wv (m/s)", "wd (deg)",
    ]

    df = prepare_weather_data(
        df,
        feature_cols=feature_cols,
        target_col="T (degC)",
        resample="1h",  # 重采样为每小时
    )

    data = df.values
    target_idx = list(df.columns).index("T (degC)")
    print(f"Data shape: {data.shape}, Target index: {target_idx}")

    # ── 创建数据加载器 ────────────────────
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        target_idx=target_idx,
    )

    # ── 创建模型 ───────────────────────────
    input_dim = data.shape[1]  # 特征数量

    if args.model == "gru":
        model = WeatherGRU(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=args.pred_len,
            dropout=args.dropout,
        )
    elif args.model == "gru_attn":
        model = WeatherGRUWithAttention(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=args.pred_len,
            dropout=args.dropout,
        )
    elif args.model == "transformer":
        model = WeatherTransformer(
            input_dim=input_dim,
            d_model=args.hidden_dim,
            nhead=8,
            num_layers=args.num_layers,
            dim_feedforward=args.hidden_dim * 4,
            output_dim=args.pred_len,
            dropout=args.dropout,
        )
    elif args.model == "patchtst":
        model = PatchTSTStyleTransformer(
            input_dim=input_dim,
            d_model=args.hidden_dim,
            nhead=8,
            num_layers=args.num_layers,
            output_dim=args.pred_len,
            dropout=args.dropout,
        )

    print(f"\n🤖 Model: {args.model}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 训练 ───────────────────────────────
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
        save_dir=args.save_dir,
        model_name=args.model,
    )

    # ── 测试评估 ───────────────────────────
    print("\n🧪 Evaluating on test set...")
    checkpoint = torch.load(
        os.path.join(args.save_dir, f"{args.model}_best.pth"),
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # 预测
    y_pred, y_true = predict(model, test_loader, device)

    # 反标准化
    dummy = np.zeros((y_true.shape[0], scaler.n_features_in_))
    dummy[:, target_idx] = y_true[:, 0]
    y_true_orig = scaler.inverse_transform(dummy)[:, target_idx]

    dummy[:, target_idx] = y_pred[:, 0]
    y_pred_orig = scaler.inverse_transform(dummy)[:, target_idx]

    # 也可以反标准化全部预测
    # 这里简化处理：对每个预测步独立反标准化
    y_true_orig_full = np.zeros_like(y_true)
    y_pred_orig_full = np.zeros_like(y_pred)
    for i in range(y_true.shape[1]):
        d = np.zeros((y_true.shape[0], scaler.n_features_in_))
        d[:, target_idx] = y_true[:, i]
        y_true_orig_full[:, i] = scaler.inverse_transform(d)[:, target_idx]
        d[:, target_idx] = y_pred[:, i]
        y_pred_orig_full[:, i] = scaler.inverse_transform(d)[:, target_idx]

    # 计算指标
    mae = np.mean(np.abs(y_true_orig_full - y_pred_orig_full))
    rmse = np.sqrt(np.mean((y_true_orig_full - y_pred_orig_full) ** 2))
    print(f"\n📊 Test Results:")
    print(f"   MAE:  {mae:.4f} °C")
    print(f"   RMSE: {rmse:.4f} °C")

    # ── 可视化 ─────────────────────────────
    print("\n📈 Generating plots...")

    plot_training_history(
        history,
        save_path=os.path.join(args.save_dir, f"{args.model}_training_history.png"),
        title=f"{args.model.upper()} Training History",
    )

    plot_predictions(
        y_true_orig_full,
        y_pred_orig_full,
        sample_idx=0,
        save_path=os.path.join(args.save_dir, f"{args.model}_sample_prediction.png"),
        title=f"{args.model.upper()} — 24-Hour Temperature Forecast",
    )

    plot_error_by_horizon(
        y_true_orig_full,
        y_pred_orig_full,
        save_path=os.path.join(args.save_dir, f"{args.model}_error_by_horizon.png"),
        title=f"{args.model.upper()} — MAE by Forecast Horizon",
    )

    print(f"\n✅ All results saved to '{args.save_dir}/'")
    print(f"   Checkpoint: {args.save_dir}/{args.model}_best.pth")


if __name__ == "__main__":
    main()
