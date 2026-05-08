"""
GRU 天气预测模型
基于多层 GRU + 全连接层的编码器-解码器架构，用于多步温度预测。

架构原理：
1. 输入：过去 N 小时的多变量气象数据 (温度、湿度、气压、风速等)
2. GRU 编码器：逐时间步处理输入序列，生成隐藏状态表示
3. 线性解码器：将最后的隐藏状态映射到未来 24 小时的预测值
4. 输出：未来 M 小时的温度预测

参考论文：
- Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014)
- Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
"""

import torch
import torch.nn as nn


class WeatherGRU(nn.Module):
    """
    用于天气时序预测的多层 GRU 模型。

    GRU 相比 LSTM 的优势：
    - 只有两个门（更新门 + 重置门），参数更少
    - 在中等规模数据上性能与 LSTM 相当
    - 训练速度更快，不易过拟合

    参数：
        input_dim: 输入特征维度（温度、湿度、气压等）
        hidden_dim: GRU 隐藏层维度
        num_layers: GRU 层数
        output_dim: 输出预测步数（如预测未来 24 小时）
        dropout: Dropout 比例，防止过拟合
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 24,
        dropout: float = 0.2,
    ):
        super(WeatherGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # GRU 层：batch_first=True 表示输入形状为 (batch, seq_len, features)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Dropout 正则化
        self.dropout = nn.Dropout(dropout)

        # 输出层：将 GRU 最终隐藏状态映射到预测序列
        # 这里使用两层全连接 + ReLU 激活
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """使用 Xavier 初始化权重"""
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 输入张量，形状 (batch_size, seq_len, input_dim)
               seq_len = 输入时间步数（如过去 168 小时）
               input_dim = 天气特征数量

        返回:
            output: 预测温度序列，形状 (batch_size, output_dim)
        """
        # GRU 前向传播
        # gru_out: (batch, seq_len, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        gru_out, h_n = self.gru(x)

        # 取最后一层的最后一个时间步的隐藏状态
        # h_n[-1] 形状: (batch, hidden_dim)
        last_hidden = h_n[-1]

        # Dropout
        last_hidden = self.dropout(last_hidden)

        # 全连接层输出预测
        output = self.fc(last_hidden)

        return output


class WeatherGRUWithAttention(nn.Module):
    """
    带注意力机制的 GRU 模型。
    使用 Bahdanau 注意力对 GRU 输出序列加权，而不是仅用最后一步。

    注意力机制原理：
    - 对 GRU 每个时间步的输出计算注意力权重
    - 加权求和得到上下文向量
    - 上下文向量与最后隐藏状态拼接后送入解码器

    这比纯 GRU 更好地捕捉长期依赖关系。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 24,
        dropout: float = 0.2,
    ):
        super(WeatherGRUWithAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # 注意力层
        self.attention = BahdanauAttention(hidden_dim)

        # 输出层：输入是 hidden_dim * 2（拼接了上下文向量）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, h_n = self.gru(x)  # gru_out: (batch, seq, hidden)

        # 计算注意力上下文向量
        context, _ = self.attention(gru_out)

        # 拼接最后隐藏状态和上下文向量
        combined = torch.cat([h_n[-1], context], dim=1)
        combined = self.dropout(combined)

        return self.fc(combined)


class BahdanauAttention(nn.Module):
    """
    Bahdanau (加性) 注意力机制。

    计算方式：
    1. score = v^T * tanh(W * h + U * h_last)
    2. alpha = softmax(score)
    3. context = sum(alpha_i * h_i)
    """

    def __init__(self, hidden_dim: int):
        super(BahdanauAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, encoder_outputs: torch.Tensor):
        """
        参数:
            encoder_outputs: (batch, seq_len, hidden_dim)
        返回:
            context: (batch, hidden_dim)
            attn_weights: (batch, seq_len)
        """
        # 用最后一个时间步的输出作为 query
        # encoder_outputs 的均值作为全局表示（也可用最后一个时间步）
        query = encoder_outputs.mean(dim=1)  # (batch, hidden)

        # 计算注意力分数
        energy = torch.tanh(self.attn(encoder_outputs))  # (batch, seq, hidden)
        query = query.unsqueeze(1)  # (batch, 1, hidden)
        scores = torch.sum(self.v * energy * query.squeeze(1).unsqueeze(1), dim=2)  # (batch, seq)
        attn_weights = torch.softmax(scores, dim=1)  # (batch, seq)

        # 加权求和得到上下文向量
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)  # (batch, hidden)

        return context, attn_weights
