#!/usr/bin/env python3
"""
深圳天气预测：GRU vs Transformer 完整训练对比
数据源：深圳逐小时天气数据 (Open-Meteo ERA5)
任务：用过去7天(168h)数据预测未来24h温度
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gru_model import WeatherGRU, WeatherGRUWithAttention
from models.transformer_model import WeatherTransformer
from utils.data_loader import create_data_loaders, WeatherDataset
from utils.training import train_model, validate
from utils.evaluation import (
    predict,
    plot_predictions,
    plot_error_by_horizon,
    plot_training_history,
    compare_models,
)

# ── 配置 ──────────────────────────────────
SEQ_LEN = 168       # 输入：过去 168 小时（7天）
PRED_LEN = 24       # 输出：未来 24 小时
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
SAVE_DIR = "results"
DATA_PATH = "data/shenzhen_weather_hourly.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────
print("=" * 60)
print("📂 Loading Shenzhen weather data...")
print("=" * 60)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
print(f"Loaded: {len(df)} rows, {df.shape[1]} columns")
print(f"Range: {df.index.min()} → {df.index.max()}")

# 选择特征列
feature_cols = [
    "temperature",      # 气温
    "humidity",         # 湿度
    "dew_point",        # 露点
    "feels_like",       # 体感温度
    "pressure_msl",     # 海平面气压
    "pressure_surface", # 地表气压
    "cloud",            # 云量
    "wind_speed",       # 风速
    "wind_dir",         # 风向
    "wind_gust",        # 阵风
    "precip",           # 降水
    "hour_sin",         # 时间特征
    "hour_cos",
    "day_sin",
    "day_cos",
]
df = df[feature_cols]
target_idx = 0  # temperature

data = df.values.astype(np.float32)
print(f"Features: {data.shape[1]}, Target: temperature (idx={target_idx})")

# 划分：用最后 24h 作为"今天的真值"用于评估
# 2024-05-01 ~ 2026-05-04 训练，最后 2026-05-05 用来验证预测
split_date = "2026-05-05"  # 今天
train_val_data = data[df.index < split_date].copy()
today_data = data[df.index >= split_date].copy()

print(f"Train/Val data: {len(train_val_data)} rows (2024-05 → 2026-05-04)")
print(f"Today's data:   {len(today_data)} rows (2026-05-05, {len(today_data)}h)")

# ── 创建数据加载器 ─────────────────────────
train_loader, val_loader, test_loader, scaler = create_data_loaders(
    train_val_data,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    batch_size=BATCH_SIZE,
    target_idx=target_idx,
)

# ── 模型训练 ─────────────────────────────
input_dim = data.shape[1]
results = {}

# ═══════════════════════════════════════════
# 训练 GRU
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("🔵 Training GRU Model...")
print("=" * 60)

model_gru = WeatherGRU(
    input_dim=input_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    output_dim=PRED_LEN,
    dropout=DROPOUT,
)
print(f"GRU parameters: {sum(p.numel() for p in model_gru.parameters()):,}")

history_gru = train_model(
    model_gru,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LR,
    patience=10,
    device=device,
    save_dir=SAVE_DIR,
    model_name="gru_shenzhen",
)

# 加载最佳模型
ckpt_gru = torch.load(
    os.path.join(SAVE_DIR, "gru_shenzhen_best.pth"), map_location=device
)
model_gru.load_state_dict(ckpt_gru["model_state_dict"])

# ═══════════════════════════════════════════
# 训练 Transformer
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("🔴 Training Transformer Model...")
print("=" * 60)

model_tf = WeatherTransformer(
    input_dim=input_dim,
    d_model=HIDDEN_DIM,
    nhead=8,
    num_layers=NUM_LAYERS,
    dim_feedforward=HIDDEN_DIM * 4,
    output_dim=PRED_LEN,
    dropout=DROPOUT,
)
print(f"Transformer parameters: {sum(p.numel() for p in model_tf.parameters()):,}")

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

ckpt_tf = torch.load(
    os.path.join(SAVE_DIR, "transformer_shenzhen_best.pth"), map_location=device
)
model_tf.load_state_dict(ckpt_tf["model_state_dict"])

# ═══════════════════════════════════════════
# 测试集评估对比
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 Test Set Evaluation")
print("=" * 60)

# GRU 评估
y_pred_gru, y_true_gru = predict(model_gru, test_loader, device)
y_pred_gru_orig = np.zeros_like(y_pred_gru)
y_true_gru_orig = np.zeros_like(y_true_gru)
n_feat = scaler.n_features_in_
for i in range(PRED_LEN):
    d_p = np.zeros((y_pred_gru.shape[0], n_feat))
    d_t = np.zeros((y_true_gru.shape[0], n_feat))
    d_p[:, target_idx] = y_pred_gru[:, i]
    d_t[:, target_idx] = y_true_gru[:, i]
    y_pred_gru_orig[:, i] = scaler.inverse_transform(d_p)[:, target_idx]
    y_true_gru_orig[:, i] = scaler.inverse_transform(d_t)[:, target_idx]

mae_gru = np.mean(np.abs(y_true_gru_orig - y_pred_gru_orig))
rmse_gru = np.sqrt(np.mean((y_true_gru_orig - y_pred_gru_orig) ** 2))

# Transformer 评估
y_pred_tf, y_true_tf = predict(model_tf, test_loader, device)
y_pred_tf_orig = np.zeros_like(y_pred_tf)
y_true_tf_orig = np.zeros_like(y_true_tf)
for i in range(PRED_LEN):
    d_p = np.zeros((y_pred_tf.shape[0], n_feat))
    d_t = np.zeros((y_true_tf.shape[0], n_feat))
    d_p[:, target_idx] = y_pred_tf[:, i]
    d_t[:, target_idx] = y_true_tf[:, i]
    y_pred_tf_orig[:, i] = scaler.inverse_transform(d_p)[:, target_idx]
    y_true_tf_orig[:, i] = scaler.inverse_transform(d_t)[:, target_idx]

mae_tf = np.mean(np.abs(y_true_tf_orig - y_pred_tf_orig))
rmse_tf = np.sqrt(np.mean((y_true_tf_orig - y_pred_tf_orig) ** 2))

# 结果汇总
test_results = {
    "GRU": {"mae": round(float(mae_gru), 3), "rmse": round(float(rmse_gru), 3)},
    "Transformer": {"mae": round(float(mae_tf), 3), "rmse": round(float(rmse_tf), 3)},
}

print(f"\n├─ GRU         │ MAE={mae_gru:.3f}°C  RMSE={rmse_gru:.3f}°C")
print(f"├─ Transformer │ MAE={mae_tf:.3f}°C  RMSE={rmse_tf:.3f}°C")
print(f"└─ Winner      │ {'🟢 GRU' if mae_gru < mae_tf else '🔴 Transformer'} (by MAE)")

# ═══════════════════════════════════════════
# 🎯 预测今天（2026-05-05）的天气
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("🎯 Predicting TODAY's Weather (2026-05-05)")
print("=" * 60)

# 用过去 7 天（168h）预测今天 24h
n_available = len(today_data)
if n_available >= 24:
    print(f"\nToday has {n_available} hours of actual data available.")

# 取训练集最后 168 小时作为输入，预测今天
# 标准化
input_data = scaler.transform(train_val_data[-SEQ_LEN:])
x_input = torch.from_numpy(input_data).unsqueeze(0).to(device)  # (1, 168, 15)

model_gru.eval()
model_tf.eval()

with torch.no_grad():
    pred_gru = model_gru(x_input).cpu().numpy().squeeze()
    pred_tf = model_tf(x_input).cpu().numpy().squeeze()

# 反标准化预测值
pred_gru_orig_list = []
pred_tf_orig_list = []
for i in range(PRED_LEN):
    d_g = np.zeros((1, n_feat))
    d_t = np.zeros((1, n_feat))
    d_g[0, target_idx] = pred_gru[i]
    d_t[0, target_idx] = pred_tf[i]
    pred_gru_orig_list.append(scaler.inverse_transform(d_g)[0, target_idx])
    pred_tf_orig_list.append(scaler.inverse_transform(d_t)[0, target_idx])

pred_gru_c = np.array(pred_gru_orig_list)
pred_tf_c = np.array(pred_tf_orig_list)

# 如果今天有实际数据，取前24h对比
if n_available >= 24:
    today_actual = today_data[:24, target_idx]
    # 反标准化的是标准化过的 today_data
    d_actual = np.zeros((24, n_feat))
    d_actual[:, target_idx] = today_actual
    today_actual_orig = scaler.inverse_transform(d_actual)[:, target_idx]

    today_mae_gru = np.mean(np.abs(today_actual_orig - pred_gru_c))
    today_mae_tf = np.mean(np.abs(today_actual_orig - pred_tf_c))
    print(f"\n✅ Today's actual data matches!")
    print(f"   GRU  MAE today: {today_mae_gru:.2f}°C")
    print(f"   TF   MAE today: {today_mae_tf:.2f}°C")
else:
    today_actual_orig = None

# ═══════════════════════════════════════════
# 打印今日预测
# ═══════════════════════════════════════════
print(f"\n🌡️ {'Hour':<8} {'GRU Pred':<12} {'Transformer Pred':<18}", end="")
if today_actual_orig is not None:
    print(f"{'Actual':<10} {'GRU Err':<10} {'TF Err':<10}")
else:
    print()

print("-" * 70)
for i in range(min(24, PRED_LEN)):
    now = pd.Timestamp("2026-05-05")
    hour = now + pd.Timedelta(hours=i)
    hour_str = hour.strftime("%H:00")

    # 判断时段和温度范围给 emoji
    temp_avg = (pred_gru_c[i] + pred_tf_c[i]) / 2
    if 6 <= hour.hour < 18:
        emoji = "☀️" if temp_avg > 25 else "🌤️" if temp_avg > 18 else "🌥️"
    else:
        emoji = "🌙" if temp_avg > 20 else "🌜"

    line = f"   {emoji} {hour_str:<4}  {pred_gru_c[i]:>5.1f}°C      {pred_tf_c[i]:>5.1f}°C        "
    if today_actual_orig is not None and i < len(today_actual_orig):
        err_g = abs(today_actual_orig[i] - pred_gru_c[i])
        err_t = abs(today_actual_orig[i] - pred_tf_c[i])
        line += f"{today_actual_orig[i]:>5.1f}°C   {err_g:>5.1f}°C    {err_t:>5.1f}°C"
    print(line)

# 摘要
print(f"\n📋 Today's Forecast Summary:")
print(f"   GRU:         Avg {pred_gru_c.mean():.1f}°C  (Min {pred_gru_c.min():.1f} / Max {pred_gru_c.max():.1f})")
print(f"   Transformer: Avg {pred_tf_c.mean():.1f}°C  (Min {pred_tf_c.min():.1f} / Max {pred_tf_c.max():.1f})")

if today_actual_orig is not None:
    print(f"   Actual:      Avg {today_actual_orig.mean():.1f}°C  (Min {today_actual_orig.min():.1f} / Max {today_actual_orig.max():.1f})")

# ═══════════════════════════════════════════
# 生成可视化图表
# ═══════════════════════════════════════════
print("\n📈 Generating plots...")

# 训练历史
plot_training_history(
    history_gru,
    save_path=f"{SAVE_DIR}/shenzhen_gru_training.png",
    title="GRU Training — Shenzhen Weather",
)
plot_training_history(
    history_tf,
    save_path=f"{SAVE_DIR}/shenzhen_transformer_training.png",
    title="Transformer Training — Shenzhen Weather",
)

# 测试集预测示例
plot_predictions(
    y_true_gru_orig,
    y_pred_gru_orig,
    sample_idx=50,
    save_path=f"{SAVE_DIR}/shenzhen_gru_prediction.png",
    title="GRU — Shenzhen 24h Temperature Forecast",
)
plot_predictions(
    y_true_tf_orig,
    y_pred_tf_orig,
    sample_idx=50,
    save_path=f"{SAVE_DIR}/shenzhen_transformer_prediction.png",
    title="Transformer — Shenzhen 24h Temperature Forecast",
)

# 误差随预测时长变化
plot_error_by_horizon(
    y_true_gru_orig,
    y_pred_gru_orig,
    save_path=f"{SAVE_DIR}/shenzhen_gru_error_horizon.png",
    title="GRU — Shenzhen Forecast Error by Horizon",
)
plot_error_by_horizon(
    y_true_tf_orig,
    y_pred_tf_orig,
    save_path=f"{SAVE_DIR}/shenzhen_transformer_error_horizon.png",
    title="Transformer — Shenzhen Forecast Error by Horizon",
)

# 模型对比
compare_models(
    test_results,
    save_path=f"{SAVE_DIR}/shenzhen_model_comparison.png",
)

# ── 今日对比图 ──
if today_actual_orig is not None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = np.arange(24)
    plt.figure(figsize=(12, 5))
    plt.plot(hours, today_actual_orig, "k-o", label="Actual", linewidth=2, markersize=5)
    plt.plot(hours, pred_gru_c, "b--s", label="GRU", alpha=0.8, markersize=4)
    plt.plot(hours, pred_tf_c, "r--^", label="Transformer", alpha=0.8, markersize=4)
    plt.fill_between(
        hours, today_actual_orig, pred_gru_c, alpha=0.08, color="blue"
    )
    plt.fill_between(
        hours, today_actual_orig, pred_tf_c, alpha=0.08, color="red"
    )
    plt.xlabel("Hour (2026-05-05)", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.title("Today's Forecast: GRU vs Transformer vs Actual — Shenzhen", fontsize=14)
    plt.xticks(hours, [f"{h:02d}:00" for h in hours], rotation=45)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/shenzhen_today_forecast.png", dpi=150)
    plt.close()
    print(f"   ✓ Today's comparison chart saved")

# 保存结果摘要
summary = {
    "test_set": test_results,
    "today_forecast": {
        "gru_mean": round(float(pred_gru_c.mean()), 1),
        "gru_min": round(float(pred_gru_c.min()), 1),
        "gru_max": round(float(pred_gru_c.max()), 1),
        "transformer_mean": round(float(pred_tf_c.mean()), 1),
        "transformer_min": round(float(pred_tf_c.min()), 1),
        "transformer_max": round(float(pred_tf_c.max()), 1),
    },
}

if today_actual_orig is not None:
    summary["today_actual"] = {
        "mean": round(float(today_actual_orig.mean()), 1),
        "min": round(float(today_actual_orig.min()), 1),
        "max": round(float(today_actual_orig.max()), 1),
    }
    summary["today_mae"] = {
        "gru": round(float(today_mae_gru), 2),
        "transformer": round(float(today_mae_tf), 2),
    }

with open(f"{SAVE_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"✅ ALL DONE! Results saved to '{SAVE_DIR}/'")
print(f"{'='*60}")
print(json.dumps(summary, indent=2, ensure_ascii=False))
