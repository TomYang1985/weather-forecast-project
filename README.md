# 🌤️ 天气时序预测项目 — Weather Time Series Forecasting

基于深度学习的天气预测项目，使用 **GRU** 和 **Transformer** 架构，利用历史气象时序数据预测未来气温。

## 项目结构

```
weather-forecast-project/
├── data/                    # 数据集（Jena Climate / 自定义数据）
├── models/                  # 模型定义
│   ├── gru_model.py         # GRU 天气预报模型
│   └── transformer_model.py # Transformer 天气预测模型
├── utils/                   # 工具函数
│   ├── data_loader.py       # 数据加载与预处理
│   ├── training.py          # 训练循环
│   └── evaluation.py        # 评估与可视化
├── results/                 # 训练结果和模型权重
├── papers/                  # 相关论文参考
├── train.py                 # 训练入口脚本
├── predict.py               # 预测入口脚本
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install torch numpy pandas matplotlib scikit-learn

# 训练 GRU 模型
python train.py --model gru --epochs 50

# 训练 Transformer 模型
python train.py --model transformer --epochs 50

# 预测
python predict.py --model gru --checkpoint results/gru_best.pth
```

## 数据来源

Jena Climate Dataset（德国耶拿气象站 2009-2016 年数据），包含 14 个气象特征，
每 10 分钟一条记录。

也可以使用中国气象数据网（data.cma.cn）或 NOAA 的公开数据。
