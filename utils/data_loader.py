"""
数据加载与预处理工具
支持 Jena Climate Dataset 以及自定义 CSV 格式的天气数据。

数据集信息：
- Jena Climate: 德国耶拿气象站，2009-2016 年
- 14 个气象特征，每 10 分钟一条记录
- 可聚合为小时级数据
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import os


def load_jena_data(data_path: str = None) -> pd.DataFrame:
    """
    加载 Jena Climate 数据集。

    如果本地不存在，会自动从 TensorFlow 数据集下载。
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        import requests
        from io import BytesIO
        from zipfile import ZipFile

        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        print(f"Downloading Jena Climate dataset...")
        resp = requests.get(url, timeout=60)
        with ZipFile(BytesIO(resp.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            df = pd.read_csv(zf.open(csv_name))

    print(f"Loaded {len(df)} records, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_weather_data(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = "T (degC)",
    date_col: str = "Date Time",
    resample: str = "1h",
) -> pd.DataFrame:
    """
    准备天气数据：选择特征、处理日期、重采样。

    参数:
        df: 原始 DataFrame
        feature_cols: 使用的特征列列表，None 则自动选择数值列
        target_col: 预测目标列名
        date_col: 日期时间列名
        resample: 重采样频率（'1h'=每小时, '6h'=每6小时, None=不重采样）
    """
    df = df.copy()

    # 解析日期
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    # 自动选择数值特征列
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
            feature_cols = [target_col] + feature_cols  # target 放第一位
        print(f"Auto-selected {len(feature_cols)} features")

    df = df[feature_cols]

    # 重采样
    if resample:
        df = df.resample(resample).mean()
        print(f"Resampled to {resample}, now {len(df)} records")

    # 去除 NaN
    df = df.dropna()

    return df


class WeatherDataset(Dataset):
    """
    天气时序预测数据集。

    滑动窗口生成 (输入序列, 目标序列) 样本对：
    - 输入：过去 seq_len 个时间步的全部特征
    - 目标：未来 pred_len 个时间步的目标变量
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 168,    # 输入序列长度（如 168 小时 = 7 天）
        pred_len: int = 24,    # 预测长度（如 24 小时）
        target_idx: int = 0,   # 目标变量在特征中的索引
        scaler: object = None,
        fit_scaler: bool = False,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = target_idx

        # 标准化
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(data)
        elif scaler is not None:
            self.scaler = scaler
            self.data = scaler.transform(data)
        else:
            self.data = data
            self.scaler = None

        self.data = self.data.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入：seq_len 个时间步的所有特征
        x = self.data[idx : idx + self.seq_len]
        # 目标：接下来 pred_len 个时间步的目标变量
        y = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx
        ]

        return torch.from_numpy(x), torch.from_numpy(y)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """将标准化的目标值还原为原始尺度"""
        if self.scaler is None:
            return y

        # 构造完整形状以利用 scaler
        dummy = np.zeros((y.shape[0], self.scaler.n_features_in_))
        dummy[:, self.target_idx] = y.reshape(-1)
        return self.scaler.inverse_transform(dummy)[:, self.target_idx].reshape(y.shape)


def create_data_loaders(
    data: np.ndarray,
    seq_len: int = 168,
    pred_len: int = 24,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    target_idx: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    创建训练/验证/测试数据加载器。

    参数:
        data: 原始 numpy 数组 (n_samples, n_features)
        seq_len: 输入序列长度
        pred_len: 预测长度
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        target_idx: 目标列索引

    返回:
        train_loader, val_loader, test_loader, scaler
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # 在训练集上拟合 scaler
    scaler = StandardScaler()
    scaler.fit(data[:train_end])

    # 创建数据集
    train_ds = WeatherDataset(
        data[:train_end + pred_len],
        seq_len=seq_len,
        pred_len=pred_len,
        target_idx=target_idx,
        scaler=scaler,
    )

    val_ds = WeatherDataset(
        data[train_end - seq_len : val_end + pred_len],
        seq_len=seq_len,
        pred_len=pred_len,
        target_idx=target_idx,
        scaler=scaler,
    )

    test_ds = WeatherDataset(
        data[val_end - seq_len :],
        seq_len=seq_len,
        pred_len=pred_len,
        target_idx=target_idx,
        scaler=scaler,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples, Test: {len(test_ds)} samples")

    return train_loader, val_loader, test_loader, scaler
