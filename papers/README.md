# 相关论文参考 — Weather Time Series Forecasting

## 核心论文

### 1. Temporal Fusion Transformers (TFT)
- **标题**: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **作者**: Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister (Google Cloud AI)
- **发表**: 2019, arXiv:1912.09363
- **链接**: https://arxiv.org/abs/1912.09363
- **代码**: https://github.com/google-research/google-research/tree/master/tft
- **核心贡献**:
  - 结合 LSTM (GRU) 的局部处理能力和 Transformer 自注意力的长程依赖捕捉
  - 可解释性：Variable Selection Networks 自动选择重要特征
  - 处理三类输入：静态协变量、已知未来输入、历史观测外生变量
  - 在多个真实数据集上达到 SOTA

### 2. Attention Is All You Need (原始 Transformer)
- **标题**: Attention Is All You Need
- **作者**: Vaswani et al. (Google Brain)
- **发表**: NeurIPS 2017
- **链接**: https://arxiv.org/abs/1706.03762
- **核心贡献**: 提出多头自注意力机制和 Transformer 架构

### 3. Informer
- **标题**: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- **作者**: Zhou et al.
- **发表**: AAAI 2021
- **链接**: https://arxiv.org/abs/2012.07436
- **代码**: https://github.com/zhouhaoyi/Informer2020
- **核心贡献**:
  - ProbSparse 自注意力：降低复杂度从 O(L²) 到 O(L·log L)
  - 自注意力蒸馏：逐层缩短序列长度
  - 生成式解码器：一步输出全部预测

### 4. Autoformer
- **标题**: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
- **作者**: Wu et al. (清华)
- **发表**: NeurIPS 2021
- **链接**: https://arxiv.org/abs/2106.13008
- **代码**: https://github.com/thuml/Autoformer
- **核心贡献**:
  - 序列分解模块：将时序分解为趋势和季节性
  - Auto-Correlation 替代自注意力

### 5. PatchTST
- **标题**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- **作者**: Nie et al.
- **发表**: ICLR 2023
- **链接**: https://arxiv.org/abs/2211.14730
- **核心贡献**:
  - Patch 划分：将长序列切成子序列 patch
  - 通道独立：每个变量独立建模

### 6. GRU 原始论文
- **标题**: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
- **作者**: Cho et al.
- **发表**: EMNLP 2014
- **链接**: https://arxiv.org/abs/1406.1078
- **核心贡献**: 提出 Gated Recurrent Unit (GRU)

### 7. GRU vs LSTM 实证对比
- **标题**: Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
- **作者**: Chung et al.
- **发表**: NIPS 2014 Workshop
- **链接**: https://arxiv.org/abs/1412.3555
- **核心发现**: GRU 在大多数任务上性能与 LSTM 相当，但参数更少

### 8. STELLA (大气时序预测)
- **标题**: On the Integration of Spatial-Temporal Knowledge: A Lightweight Approach to Atmospheric Time Series Forecasting
- **作者**: Fu et al. (中科院计算所)
- **发表**: 2024, arXiv:2408.09695
- **链接**: https://arxiv.org/abs/2408.09695
- **代码**: https://github.com/GestaltCogTeam/STELLA
- **核心贡献**:
  - 仅 10k 参数的轻量模型
  - 时空位置嵌入 (STPE) 替代 Transformer
  - 1 小时训练达到 SOTA

## 综述论文

### 9. Transformer 时间序列预测综述
- **标题**: Transformers for Time-Series Forecasting: A Comprehensive Survey through 2024
- **发表**: 2024
- **链接**: https://www.researchgate.net/publication/399324903

### 10. Deep Learning + Transformer + RNN 天气预测综述
- **发表**: 2025, PMC
- **链接**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

## 实操教程

### Keras 官方天气预测教程
- **链接**: https://keras.io/examples/timeseries/timeseries_weather_forecasting/
- 使用 Jena Climate Dataset 的 LSTM 实现

### GRU + Transformer 天气预测项目
- **链接**: https://github.com/jingwenshi-dev/Weather-Forecasting-by-GRU-Transformer
- 对比 GRU vs Transformer 做多伦多地区 24 小时温度预测

### PyTorch Forecasting (TFT 实现)
- **链接**: https://github.com/sktime/pytorch-forecasting
- 工业级 TFT 实现，支持 GPU 分布式训练

## 数据集

### Jena Climate Dataset
- **链接**: https://www.bgc-jena.mpg.de/wetter/
- 德国耶拿气象站，2009-2016，14个特征，每10分钟

### ERA5 (ECMWF Reanalysis)
- **链接**: https://cds.climate.copernicus.eu/
- 全球气象再分析数据，适合大规模训练

### 中国气象数据网
- **链接**: http://data.cma.cn/
- 中国地面气象站逐小时观测数据
