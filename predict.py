#!/usr/bin/env python3
"""
天气预测脚本 — 使用训练好的模型进行未来天气预测。
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from utils.data_loader import WeatherDataset
from models.gru_model import WeatherGRU, WeatherGRUWithAttention
from models.transformer_model import WeatherTransformer, PatchTSTStyleTransformer


def main():
    parser = argparse.ArgumentParser(description="Predict future weather")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "gru_attn", "transformer", "patchtst"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_data", type=str, help="Path to recent weather data CSV (optional)")
    parser.add_argument("--seq_len", type=int, default=168, help="Input sequence length")
    parser.add_argument("--pred_len", type=int, default=24, help="Prediction length")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    input_dim = 14  # Jena 数据集的 14 个特征

    if args.model == "gru":
        model = WeatherGRU(input_dim=input_dim, output_dim=args.pred_len)
    elif args.model == "gru_attn":
        model = WeatherGRUWithAttention(input_dim=input_dim, output_dim=args.pred_len)
    elif args.model in ("transformer", "patchtst"):
        model = WeatherTransformer(input_dim=input_dim, output_dim=args.pred_len)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"\n✅ Model loaded from {args.checkpoint}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}, Val MAE: {checkpoint.get('val_mae', 'N/A')}")

    # 如果有输入数据，进行预测
    if args.input_data:
        df = pd.read_csv(args.input_data)
        recent_data = df.values[-args.seq_len:].astype(np.float32)
        x = torch.from_numpy(recent_data).unsqueeze(0).to(device)  # (1, seq, features)

        with torch.no_grad():
            pred = model(x).cpu().numpy().squeeze()

        print(f"\n🌡️ Predicted temperatures for next {args.pred_len} hours:")
        for i, temp in enumerate(pred):
            hour = (i + 1)
            emoji = "☀️" if temp > 20 else "🌤️" if temp > 10 else "❄️"
            print(f"   +{hour:2d}h: {temp:6.2f}°C {emoji}")

        print(f"\n📊 Summary:")
        print(f"   Mean:  {pred.mean():.2f}°C")
        print(f"   Max:   {pred.max():.2f}°C")
        print(f"   Min:   {pred.min():.2f}°C")
    else:
        print("\n💡 No input data provided. To predict, use: python predict.py --input_data recent_weather.csv")


if __name__ == "__main__":
    main()
