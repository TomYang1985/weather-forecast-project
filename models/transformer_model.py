"""
Transformer 天气预测模型
使用 Encoder-Only Transformer 架构进行多变量时间序列预测。

核心原理：
1. 输入嵌入：将多变量气象数据线性投影到 d_model 维度
2. 位置编码：为时序数据添加可学习的位置信息
3. Transformer Encoder：多头自注意力机制捕捉长程依赖
4. 输出投影：将编码器输出映射到预测窗口

为什么 Transformer 适合天气预测：
- 自注意力可以捕捉相隔很远的天气模式关联（如季风周期）
- 并行计算，比 RNN 训练更快
- 多头注意力可以从不同子空间理解数据

参考论文：
- Vaswani et al. "Attention Is All You Need" (2017)
- Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2020)
- Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021)
- Wu et al. "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (NeurIPS 2021)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码。

    原理：
    对于每个位置 pos 和每个维度 i：
    - PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    这使得模型能感知输入序列中每个时间步的位置，
    因为 Transformer 本身没有内置的序列顺序概念。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer（不参与训练的参数）
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch_size, seq_len, d_model)
        返回:
            加上位置编码的张量
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class WeatherTransformer(nn.Module):
    """
    基于 Transformer Encoder 的天气预测模型。

    架构流程：
    Input (batch, seq_len, features)
        ↓
    线性嵌入层：features → d_model
        ↓
    位置编码：添加时序位置信息
        ↓
    Transformer Encoder × N 层：
        ├── Multi-Head Self-Attention
        ├── Add & Norm (残差连接 + 层归一化)
        ├── Feed-Forward Network (两层全连接 + GELU)
        └── Add & Norm
        ↓
    全局平均池化 / 取 CLS token
        ↓
    MLP 解码器 → 预测序列

    参数：
        input_dim: 输入特征维度
        d_model: Transformer 隐藏维度
        nhead: 多头注意力头数
        num_layers: Encoder 层数
        dim_feedforward: 前馈网络维度
        output_dim: 预测步数
        dropout: Dropout 比例
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        output_dim: int = 24,
        dropout: float = 0.1,
        max_seq_len: int = 500,
    ):
        super(WeatherTransformer, self).__init__()

        self.d_model = d_model

        # 输入嵌入：将多变量特征投影到 d_model 维度
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # PyTorch 1.9+ 支持 batch_first
            norm_first=True,   # Pre-LN 架构，训练更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 全局池化 + 输出层
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch_size, seq_len, input_dim) — 历史天气数据
        返回:
            output: (batch_size, output_dim) — 未来温度预测
        """
        # 输入嵌入
        x = self.input_embedding(x)  # (batch, seq, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (batch, seq, d_model)

        # 全局平均池化：聚合所有时间步的信息
        x = x.mean(dim=1)  # (batch, d_model)

        # 输出预测
        output = self.output_fc(x)  # (batch, output_dim)

        return output


class PatchTSTStyleTransformer(nn.Module):
    """
    Patch-based Transformer（受 PatchTST 启发）。

    核心创新：
    - 将长序列切分成多个 patch（如每 6 小时一个 patch）
    - 每个 patch 独立嵌入后再输入 Transformer
    - 大幅减少序列长度，降低自注意力的 O(N²) 计算复杂度
    - 同时保留局部时间模式

    参考：
    - Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        output_dim: int = 24,
        patch_len: int = 6,  # 每个 patch 的时间步数
        stride: int = 3,     # patch 之间的步幅
        dropout: float = 0.1,
    ):
        super(PatchTSTStyleTransformer, self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Patch 嵌入：将 patch_len * input_dim 展平后投影到 d_model
        self.patch_embedding = nn.Linear(patch_len * input_dim, d_model)

        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

        # 输出投影
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
        )

    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入序列切分成 patches。

        参数:
            x: (batch, seq_len, features)
        返回:
            patches: (batch, num_patches, patch_len * features)
        """
        batch, seq_len, features = x.shape
        patches = []

        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i : i + self.patch_len, :]  # (batch, patch_len, features)
            patches.append(patch.reshape(batch, -1))  # (batch, patch_len * features)

        # 如果不够一个 patch 的余数也作为一个 patch
        if (seq_len - self.patch_len) % self.stride != 0:
            # 取最后 patch_len 个时间步
            patch = x[:, -self.patch_len :, :]
            patches.append(patch.reshape(batch, -1))

        return torch.stack(patches, dim=1)  # (batch, num_patches, patch_len*features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 创建 patches
        patches = self._create_patches(x)  # (batch, num_patches, patch_len*features)

        # Patch 嵌入
        x = self.patch_embedding(patches)  # (batch, num_patches, d_model)

        # 位置编码
        x = x + self.pos_embedding[:, : x.size(1), :]

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # 池化 + 输出
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.dropout(x)

        return self.output_fc(x)


class SimpleTimeSeriesTransformer(nn.Module):
    """
    极简版 Transformer，适合快速实验。

    与 WeatherTransformer 的区别：
    - 无位置编码（依赖输入嵌入隐式学习位置）
    - 更浅的层数
    - 更小的参数量
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        output_dim: int = 24,
        dropout: float = 0.1,
    ):
        super(SimpleTimeSeriesTransformer, self).__init__()

        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_fc(x)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.output_fc(x)
