#!/usr/bin/env python3
"""
仅训练 Transformer（内存优化版）
针对 3.6GB 内存优化：减少 hidden_dim、batch_size
"""

import numpy as np
import pandas as pd
import torch
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.transformer_model import WeatherTransformer
from utils.data_loader import create_data_loaders
from utils.training import train_model
from utils.evaluation import (
    predict,
    plot_predictions,
    plot_error_by_horizon,
    plot_training_history,
    compare_models,
)

# ── 内存优化配置 ──────────────────────────
SEQ_LEN = 168
PRED_LEN = 24
BATCH_SIZE = 32        # 减小 batch
HIDDEN_DIM = 64        # 减小 hidden dim
NUM_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 25
LR = 1e-3
SAVE_DIR = "results"
DATA_PATH = "data/shenzhen_weather_hourly.csv"

device = torch.device("cpu")  # 强制 CPU，避免 CUDA 内存开销
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("📂 Loading data...")
print("=" * 60)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
feature_cols = [
    "temperature", "humidity", "dew_point", "feels_like",
    "pressure_msl", "pressure_surface", "cloud", "wind_speed",
    "wind_dir", "wind_gust", "precip",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]
df = df[feature_cols]
target_idx = 0

data = df.values.astype(np.float32)
split_date = "2026-05-05"
train_val_data = data[df.index < split_date].copy()
today_data = data[df.index >= split_date].copy()

print(f"Train/Val: {len(train_val_data)} rows, Today: {len(today_data)} rows")

train_loader, val_loader, test_loader, scaler = create_data_loaders(
    train_val_data,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    batch_size=BATCH_SIZE,
    target_idx=target_idx,
)

input_dim = data.shape[1]

# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("🔴 Training Transformer (memory-optimized)...")
print("=" * 60)

model_tf = WeatherTransformer(
    input_dim=input_dim,
    d_model=HIDDEN_DIM,
    nhead=4,  # 减少注意力头
    num_layers=NUM_LAYERS,
    dim_feedforward=HIDDEN_DIM * 2,  # 减少 FFN 维度
    output_dim=PRED_LEN,
    dropout=DROPOUT,
)
print(f"Parameters: {sum(p.numel() for p in model_tf.parameters()):,}")

history_tf = train_model(
    model_tf,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LR,
    patience=10,
    device=device,
    save_dir=SAVE_DIR,
    model_name="transformer_shenzhen",
)

# ═══════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════
ckpt = torch.load(os.path.join(SAVE_DIR, "transformer_shenzhen_best.pth"), map_location=device)
model_tf.load_state_dict(ckpt["model_state_dict"])

y_pred_tf, y_true_tf = predict(model_tf, test_loader, device)
n_feat = scaler.n_features_in_
y_pred_orig = np.zeros_like(y_pred_tf)
y_true_orig = np.zeros_like(y_true_tf)
for i in range(PRED_LEN):
    d_p = np.zeros((y_pred_tf.shape[0], n_feat))
    d_t = np.zeros((y_true_tf.shape[0], n_feat))
    d_p[:, target_idx] = y_pred_tf[:, i]
    d_t[:, target_idx] = y_true_tf[:, i]
    y_pred_orig[:, i] = scaler.inverse_transform(d_p)[:, target_idx]
    y_true_orig[:, i] = scaler.inverse_transform(d_t)[:, target_idx]

mae_tf = np.mean(np.abs(y_true_orig - y_pred_orig))
rmse_tf = np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))
print(f"\nTransformer Test: MAE={mae_tf:.3f}°C, RMSE={rmse_tf:.3f}°C")

# ── 评估 GRU ──
from models.gru_model import WeatherGRU
model_gru = WeatherGRU(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=PRED_LEN)
ckpt_gru = torch.load(os.path.join(SAVE_DIR, "gru_shenzhen_best.pth"), map_location=device)
model_gru.load_state_dict(ckpt_gru["model_state_dict"])

y_pred_gru, y_true_gru = predict(model_gru, test_loader, device)
y_pred_gru_orig = np.zeros_like(y_pred_gru)
y_true_gru_orig = np.zeros_like(y_true_gru)
for i in range(PRED_LEN):
    d_p = np.zeros((y_pred_gru.shape[0], n_feat))
    d_t = np.zeros((y_true_gru.shape[0], n_feat))
    d_p[:, target_idx] = y_pred_gru[:, i]
    d_t[:, target_idx] = y_true_gru[:, i]
    y_pred_gru_orig[:, i] = scaler.inverse_transform(d_p)[:, target_idx]
    y_true_gru_orig[:, i] = scaler.inverse_transform(d_t)[:, target_idx]

mae_gru = np.mean(np.abs(y_true_gru_orig - y_pred_gru_orig))
rmse_gru = np.sqrt(np.mean((y_true_gru_orig - y_pred_gru_orig) ** 2))
print(f"GRU Test:         MAE={mae_gru:.3f}°C, RMSE={rmse_gru:.3f}°C")

# ═══════════════════════════════════════════
# 预测今天
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("🎯 Today's Forecast (2026-05-05)")
print("=" * 60)

input_data = scaler.transform(train_val_data[-SEQ_LEN:])
x = torch.from_numpy(input_data).unsqueeze(0)

model_gru.eval()
model_tf.eval()
with torch.no_grad():
    p_gru = model_gru(x).numpy().squeeze()
    p_tf = model_tf(x).numpy().squeeze()

p_gru_c = np.array([scaler.inverse_transform(
    np.eye(1, n_feat, target_idx)[0] * p_gru[i] + np.zeros((1, n_feat))
)[0, target_idx] for i in range(PRED_LEN)])
p_tf_c = np.array([scaler.inverse_transform(
    np.eye(1, n_feat, target_idx)[0] * p_tf[i] + np.zeros((1, n_feat))
)[0, target_idx] for i in range(PRED_LEN)])

# 简单反标准化
p_gru_clean = []
p_tf_clean = []
for i in range(PRED_LEN):
    d_g = np.zeros((1, n_feat))
    d_t = np.zeros((1, n_feat))
    d_g[0, target_idx] = p_gru[i]
    d_t[0, target_idx] = p_tf[i]
    p_gru_clean.append(scaler.inverse_transform(d_g)[0, target_idx])
    p_tf_clean.append(scaler.inverse_transform(d_t)[0, target_idx])
p_gru_c = np.array(p_gru_clean)
p_tf_c = np.array(p_tf_clean)

n_avail = len(today_data)
today_actual = None
if n_avail >= 24:
    today_actual = today_data[:24, target_idx]
    d_a = np.zeros((24, n_feat))
    d_a[:, target_idx] = today_actual
    today_actual = scaler.inverse_transform(d_a)[:, target_idx]

print(f"\n🌡️ {'Hour':<8} {'GRU':<10} {'Transformer':<14}", end="")
if today_actual is not None:
    print(f"{'Actual':<10} {'GRU Err':<10} {'TF Err':<10}")
else:
    print()
print("-" * 70)
for i in range(24):
    h = pd.Timestamp("2026-05-05") + pd.Timedelta(hours=i)
    hs = h.strftime("%H:00")
    avg = (p_gru_c[i] + p_tf_c[i]) / 2
    e = "☀️" if 6 <= h.hour < 18 and avg > 25 else "🌤️" if 6 <= h.hour < 18 else "🌙"
    line = f"  {e} {hs:<4} {p_gru_c[i]:>5.1f}°C   {p_tf_c[i]:>5.1f}°C     "
    if today_actual is not None and i < len(today_actual):
        eg = abs(today_actual[i] - p_gru_c[i])
        et = abs(today_actual[i] - p_tf_c[i])
        line += f"{today_actual[i]:>5.1f}°C    {eg:>5.1f}°C    {et:>5.1f}°C"
    print(line)

print(f"\n📋 Summary:")
print(f"   GRU:         avg {p_gru_c.mean():.1f}°C ({p_gru_c.min():.1f}~{p_gru_c.max():.1f})")
print(f"   Transformer: avg {p_tf_c.mean():.1f}°C ({p_tf_c.min():.1f}~{p_tf_c.max():.1f})")
if today_actual is not None:
    print(f"   Actual:      avg {today_actual.mean():.1f}°C ({today_actual.min():.1f}~{today_actual.max():.1f})")

# ═══════════════════════════════════════════
# 图表
# ═══════════════════════════════════════════
print("\n📈 Generating plots...")

plot_training_history(history_tf, save_path=f"{SAVE_DIR}/shenzhen_transformer_training.png",
                      title="Transformer Training — Shenzhen")
plot_predictions(y_true_orig, y_pred_orig, sample_idx=50,
                 save_path=f"{SAVE_DIR}/shenzhen_transformer_prediction.png",
                 title="Transformer — Shenzhen 24h Forecast")
plot_error_by_horizon(y_true_orig, y_pred_orig,
                      save_path=f"{SAVE_DIR}/shenzhen_transformer_error_horizon.png",
                      title="Transformer — Error by Horizon")

test_results = {
    "GRU": {"mae": round(float(mae_gru), 3), "rmse": round(float(rmse_gru), 3)},
    "Transformer": {"mae": round(float(mae_tf), 3), "rmse": round(float(rmse_tf), 3)},
}
compare_models(test_results, save_path=f"{SAVE_DIR}/shenzhen_model_comparison.png")

# 今日对比
if today_actual is not None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hrs = np.arange(24)
    plt.figure(figsize=(12, 5))
    plt.plot(hrs, today_actual, "k-o", label="Actual", lw=2, ms=5)
    plt.plot(hrs, p_gru_c, "b--s", label="GRU", alpha=0.8, ms=4)
    plt.plot(hrs, p_tf_c, "r--^", label="Transformer", alpha=0.8, ms=4)
    plt.fill_between(hrs, today_actual, p_gru_c, alpha=0.08, color="blue")
    plt.fill_between(hrs, today_actual, p_tf_c, alpha=0.08, color="red")
    plt.xlabel("Hour (2026-05-05)", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.title("Today: GRU vs Transformer vs Actual — Shenzhen", fontsize=14)
    plt.xticks(hrs, [f"{h:02d}:00" for h in hrs], rotation=45)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/shenzhen_today_forecast.png", dpi=150)
    plt.close()

summary = {
    "test_set": test_results,
    "today": {
        "gru_avg": round(float(p_gru_c.mean()), 1),
        "tf_avg": round(float(p_tf_c.mean()), 1),
    }
}
if today_actual is not None:
    summary["today"]["actual_avg"] = round(float(today_actual.mean()), 1)
    summary["today_mae"] = {
        "gru": round(float(np.mean(np.abs(today_actual - p_gru_c))), 2),
        "transformer": round(float(np.mean(np.abs(today_actual - p_tf_c))), 2),
    }

with open(f"{SAVE_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Summary:")
print(json.dumps(summary, indent=2, ensure_ascii=False))
