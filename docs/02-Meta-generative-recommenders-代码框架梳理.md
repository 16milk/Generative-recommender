# Meta `generative-recommenders` 代码框架梳理

> 本文档对 Meta 官方开源仓库 [`meta-recsys/generative-recommenders`](https://github.com/meta-recsys/generative-recommenders)（HSTU / M-FALCON 论文配套代码）的代码框架做系统梳理，目的是为「从零实现一套类似的生成式推荐系统」提供一份可对照的工程蓝图：搞清楚仓库由哪些模块组成、各模块职责、数据/训练如何串起来，以及若要复刻需要依次完成哪些工作。
>
> 对应论文：*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*（ICML 2024，[arXiv:2402.17152](https://arxiv.org/abs/2402.17152)）。
>
> 背景知识请先读 [`01-生成式推荐架构-背景与技术综述.md`](01-生成式推荐架构-背景与技术综述.md)。

---

## 目录

1. [一句话概览](#1-一句话概览)
2. [仓库的两条主线](#2-仓库的两条主线)
3. [顶层目录结构](#3-顶层目录结构)
4. [research：学术复现路线（best place to start）](#4-research学术复现路线best-place-to-start)
5. [modules + ops：生产级 HSTU 算子与模型库](#5-modules--ops生产级-hstu-算子与模型库)
6. [dlrm_v3：生产级训练 / 推理系统](#6-dlrm_v3生产级训练--推理系统)
7. [配置系统（gin）](#7-配置系统gin)
8. [端到端数据流与训练流程](#8-端到端数据流与训练流程)
9. [关键设计点](#9-关键设计点)
10. [技术栈与依赖](#10-技术栈与依赖)
11. [如果要实现一个类似项目：工作清单](#11-如果要实现一个类似项目工作清单)
12. [给本项目的落地建议](#12-给本项目的落地建议)

---

## 1. 一句话概览

这个仓库做的事情是：**把传统 DLRM 排序/召回，重构成「用户行为序列上的自回归序列转导（sequential transduction）」问题**，并给出：

- 一个新的主干网络 **HSTU（Hierarchical Sequential Transduction Unit）**，替代标准 Transformer 的 softmax 注意力；
- 配套的 **高性能算子**（Triton + CUDA/FlashAttention-V3 风格的 jagged attention 内核）；
- 两套可运行的工程：一套面向**论文复现**（`research/`，对标 SASRec/HSTU 在 MovieLens / Amazon 上的 next-item 召回），一套面向**工业生产**（`dlrm_v3/`，多任务排序 + torchrec 分片 + 流式训练 + MLPerf 推理）。

> 注意：这份代码的「生成式」体现在 **统一的序列转导 + 大规模 scaling + 自回归监督**，主干是 **HSTU**，物品用 **Item ID embedding**（不是 TIGER/OneRec 那种 RQ-VAE 语义 ID）。如果你的目标是「语义 ID + encoder-decoder 生成」，这个仓库提供的是「序列建模主干 + 工程化」这一半，语义 ID 那一半需要自己补（见第 11 节）。

---

## 2. 仓库的两条主线

整个仓库其实是**两套相对独立的代码**，共享论文思想但工程目标不同。理解这一点是读懂仓库的关键：

| 维度 | `research/`（学术线） | `modules/` + `ops/` + `dlrm_v3/`（生产线） |
|------|----------------------|-------------------------------------------|
| 目标 | 复现论文表格（HR@K / NDCG@K） | 工业级排序系统 + 训练/推理 benchmark |
| 任务 | next-item 召回（retrieval） | 多任务排序（CTR、时长等）+ 召回 |
| 模型 | `SASRec` / `HSTU`（精简实现） | `DlrmHSTU`（完整 HSTU transducer） |
| 物品表示 | 单机 `nn.Embedding` | torchrec 分片大表（`EmbeddingCollection`） |
| 张量 | padded dense `[B, N, D]` | **jagged / 变长**（`KeyedJaggedTensor`） |
| 算子 | 纯 PyTorch | Triton + CUDA 自定义内核 |
| 并行 | `DDP`（数据并行） | torchrec 模型并行 + DMP 分片 |
| 入口 | `main.py` + `configs/*.gin` | `dlrm_v3/train/train_ranker.py` + `*/gin/*.gin` |
| 适合 | **学习、快速起步、做实验** | 理解工业落地、压测、规模化 |

建议路径：**先吃透 `research/`（能在单卡跑通），再读 `modules/` 了解生产级 HSTU，最后看 `dlrm_v3/` 的工程化。**

---

## 3. 顶层目录结构

```
generative-recommenders/
├── main.py                      # 学术线训练入口（torch.multiprocessing + gin）
├── preprocess_public_data.py    # 下载/预处理 MovieLens、Amazon 数据
├── run_fractal_expansion.py     # 把 ML-20M 用 fractal expansion 扩成 ML-3B 合成数据
├── setup.py / requirements.txt  # 依赖（torch / fbgemm_gpu / torchrec / gin）
├── configs/                     # 学术线 gin 配置（按数据集 × 模型组织）
│   ├── ml-1m/  ml-20m/  ml-3b/  amzn-books/
│   │   ├── hstu-sampled-softmax-*.gin
│   │   └── sasrec-sampled-softmax-*.gin
│
└── generative_recommenders/
    ├── common.py                # HammerModule/HammerKernel 等公共基建（切换 kernel 后端）
    │
    ├── research/                # ===== 学术复现线 =====
    │   ├── data/                # 数据预处理与 Dataset
    │   ├── modeling/            # 模型（SASRec / HSTU / 相似度 / loss）
    │   │   ├── sequential/
    │   │   └── rails/           # 检索相关（MoL / MIPS 相似度与 top-k）
    │   ├── indexing/            # 召回索引（candidate index, top-k）
    │   └── trainer/             # 训练循环 + data loader
    │
    ├── modules/                 # ===== 生产级 HSTU 模型库 =====
    │   ├── dlrm_hstu.py         # 顶层模型 DlrmHSTU
    │   ├── hstu_transducer.py   # HSTU 序列转导器
    │   ├── stu.py / dynamic_stu.py   # STU 层（HSTU 的基本单元）
    │   ├── preprocessors.py / contextual_interleave_preprocessor.py
    │   ├── action_encoder.py / content_encoder.py / positional_encoder.py
    │   ├── contextualize_mlps.py / postprocessors.py
    │   └── multitask_module.py  # 多任务头（CTR/时长等）
    │
    ├── ops/                     # ===== 高性能算子 =====
    │   ├── pytorch/             # 纯 PyTorch 参考实现
    │   ├── triton/              # Triton 内核（attention / layernorm / jagged ...）
    │   ├── cpp/                 # CUDA 内核
    │   │   └── hstu_attention/  # 基于 FlashAttention-V3 的 HSTU attention（H100 优化）
    │   ├── hstu_attention.py / hstu_compute.py   # 算子 dispatch 封装
    │   └── jagged_tensors.py / layer_norm.py / mm.py / position.py
    │
    └── dlrm_v3/                 # ===== 生产级训练/推理系统 =====
        ├── datasets/            # movie_lens / kuairand / synthetic_streaming ...
        ├── train/               # train_ranker.py + 训练循环 utils + gin
        ├── inference/           # MLPerf loadgen 推理服务（含 C++ runner）
        ├── configs.py / checkpoint.py / utils.py
        └── preprocess_public_data.py / streaming_synthetic_data.py
```

三个值得记住的入口：

1. **`main.py`** → 学术线训练（读 `configs/*.gin`，跑 `research/trainer/train.py:train_fn`）。
2. **`dlrm_v3/train/train_ranker.py`** → 生产线训练（读 `dlrm_v3/train/gin/*.gin`）。
3. **`dlrm_v3/inference/main.py`** → 生产线 MLPerf 推理。

---

## 4. research：学术复现路线（best place to start）

这是最容易跑通、也最适合学习的部分。它实现了一个**单一序列模型做 next-item 召回**的完整闭环：数据 → 模型 → 采样负样本 → sampled-softmax 损失 → MIPS top-k 评估。

### 4.1 数据层 `research/data/`

| 文件 | 职责 |
|------|------|
| `preprocessor.py` | 各数据集的预处理器（`get_common_preprocessors()` 返回 `ml-1m`/`ml-20m`/`amzn-books` 等），负责把原始评分文件整理成「按用户聚合的行为序列 CSV」。 |
| `dataset.py` | `DatasetV2` / `MultiFileDatasetV2`：把每个用户的历史读成定长 padding 序列，支持 `chronological`（按时间排序）、`ignore_last_n`（留出 target）、`sample_ratio`（positional sampling）。 |
| `reco_dataset.py` | `get_reco_dataset()`：把上面拼成一个 `RecoDataset`（含 `train_dataset`/`eval_dataset`/`all_item_ids`/`max_item_id`），并为 MovieLens 构造 item side features（genres/title/year 哈希成 jagged 特征）。 |
| `item_features.py` | `ItemFeatures` 容器（物品侧多值特征）。 |
| `eval.py` | 评估逻辑：`get_eval_state` + `eval_metrics_v2_from_tensors` 计算 HR@K / NDCG@K / MRR。 |

> 关键点：序列被组织成 `past_ids`（历史物品 ID）+ `past_lengths`（真实长度）+ `past_payloads`（时间戳/评分等），target 通过 `scatter_` 拼到序列末尾。这就是「行为序列即 token 序列」的具体落地。

### 4.2 建模层 `research/modeling/`

```
modeling/
├── sequential/
│   ├── embedding_modules.py            # LocalEmbeddingModule：单机 item embedding 表
│   ├── input_features_preprocessors.py # 加位置编码 + dropout（输入预处理）
│   ├── output_postprocessors.py        # L2Norm / LayerNorm 输出后处理
│   ├── sasrec.py                       # SASRec baseline（causal self-attention）
│   ├── hstu.py                         # HSTU 主干（research 版，~800 行，纯 PyTorch）
│   ├── autoregressive_losses.py        # BCELoss / 负采样器（InBatch / Local）
│   ├── losses/sampled_softmax.py       # SampledSoftmaxLoss（论文采用）
│   ├── encoder_utils.py                # get_sequential_encoder() 工厂
│   └── features.py                     # movielens_seq_features_from_row()
├── similarity_module.py / similarity_utils.py   # 相似度函数（DotProduct / MoL）
├── initialization.py
└── rails/                              # 检索加速（Retrieval with Approximate top-k）
    ├── similarities/   (dot_product, MoL: Mixture-of-Logits)
    └── indexing/       (mips_top_k, mol_top_k)
```

要点：

- **可插拔的「编码器 + 相似度 + 损失」三件套**：`main_module`（SASRec/HSTU）、`interaction_module_type`（DotProduct/MoL）、`loss_module`（BCE/SampledSoftmax）都由 gin 配置切换。
- **HSTU vs SASRec**：两者接口一致（同样的输入预处理 / 输出后处理 / 相似度 / 损失），只是主干 block 不同——方便公平对比。这是仓库做对照实验的设计精髓。
- **MoL（Mixture-of-Logits）**：`rails/` 下是论文里更强的相似度/检索方案，比单纯点积更有表达力。

### 4.3 训练层 `research/trainer/`

- `data_loader.py`：`create_data_loader()` 包装 `DistributedSampler` + `DataLoader`。
- `train.py`：核心训练函数 `train_fn`（`@gin.configurable`，约 530 行）。它把上述所有组件组装起来：

```
get_reco_dataset → create_data_loader
→ LocalEmbeddingModule + input_preproc + output_postproc
→ get_sequential_encoder(HSTU/SASRec)
→ SampledSoftmaxLoss + 负采样器(in-batch/local)
→ DDP + AdamW
→ 训练循环：每步 next-token 自回归监督；周期性 MIPS top-k 评估 HR/NDCG/MRR
→ TensorBoard 日志 + checkpoint
```

> 自回归监督的实现非常直观（`train.py`）：用 `seq_embeddings[:, :-1, :]` 预测 `supervision_ids[:, 1:]`，`ar_mask = supervision_ids[:, 1:] != 0` 屏蔽 padding。这正是「用户行为序列上的 next-token 预测」。

---

## 5. modules + ops：生产级 HSTU 算子与模型库

`research/` 为了可读性用纯 PyTorch + dense 张量；而 `modules/` + `ops/` 是**为工业规模重写的版本**，核心差异是：**变长（jagged）张量 + 自定义高性能内核 + torchrec 大表**。

### 5.1 `modules/`：模型组件

| 文件 | 职责 |
|------|------|
| `dlrm_hstu.py` | 顶层模型 **`DlrmHSTU`** + 配置 `DlrmHSTUConfig`。把 embedding 表、preprocessor、HSTU transducer、多任务头串成完整模型。配置里能看到全部超参：`hstu_attn_num_layers`、`hstu_attn_linear_dim`、`hstu_attn_qk_dim`、`max_seq_len=16384`、`multitask_configs` 等。 |
| `hstu_transducer.py` | **`HSTUTransducer`**：把预处理后的序列喂给一摞 STU 层，输出序列表示。 |
| `stu.py` | **STU / STULayer / STUStack / STULayerConfig**：HSTU 的基本计算单元（pointwise aggregated attention + gating，替代 softmax attention）。 |
| `dynamic_stu.py` | 动态/可变形态的 STU（用于流式/不同形状）。 |
| `preprocessors.py`、`contextual_interleave_preprocessor.py` | **序列构造的核心**：把「用户历史 item 序列」「候选 candidates」「上下文特征」交织（interleave）成统一 token 序列。对应论文 4.2 的 sequence formulation。 |
| `action_encoder.py` | 编码「行为类型」（点击/点赞/完播…），与 item 交织。 |
| `content_encoder.py` | 编码物品内容/侧信息特征。 |
| `positional_encoder.py` | `HSTUPositionalEncoder`：位置 + 时间编码。 |
| `contextualize_mlps.py` | 上下文化 MLP（把上下文特征注入序列）。 |
| `multitask_module.py` | **`DefaultMultitaskModule` / `TaskConfig` / `MultitaskTaskType`**：支持回归（如观看时长）+ 二分类（如点击）多个任务头，对应工业排序的多目标。 |
| `postprocessors.py` | 输出后处理（LayerNorm / 时间戳相关）。 |

### 5.2 `ops/`：高性能算子（这是仓库的「硬核」部分）

推荐的高基数、长序列（最长 16384）数据决定了**不能用 dense padding**，必须用 **jagged（变长拼接）张量**并配套专用内核。`ops/` 提供三套实现，通过 `HammerKernel`（见 `common.py`）在运行时切换：

```
ops/
├── pytorch/   # 纯 PyTorch 参考实现（正确性基准、CPU 可跑）
├── triton/    # Triton 内核（GPU 通用，含 HSTU attention / layernorm / addmm / jagged / swiglu）
└── cpp/       # CUDA 内核（最高性能）
    └── hstu_attention/   # 基于 FlashAttention-V3 的 HSTU attention，H100 SOTA
```

关键算子：

- **`hstu_attention.py` / `triton_hstu_attention.py` / `cpp/hstu_attention/`**：HSTU 的注意力计算，论文称比 FlashAttention2 的 Transformer 快 5.3–15.2×（长序列）。
- **`jagged_tensors.py` + `triton_jagged*.py` + `cpp/*jagged*.cu`**：变长序列的拼接、转置、padding 互转等（`concat_2D_jagged`、`split_1d_jagged_jagged` 等），是整个 jagged 数据流的基础。
- **`hstu_compute.py`**：把 layernorm + linear + attention 融合的计算路径。
- `layer_norm.py` / `mm.py`（addmm）/ `position.py`：各自的 triton/cpp/pytorch 三实现。
- `ops/benchmarks/` 与 `ops/cpp/benchmarks/`：各算子的性能基准脚本。

> 工程启示：**「变长序列 + 三套可切换内核（pytorch/triton/cuda）+ 严格的 op-level 单测和 benchmark」** 是这个仓库区别于一般科研代码的地方，也是把模型真正推到生产规模的关键投入。`ops/tests/` 下每个算子都有 parity 测试。

---

## 6. dlrm_v3：生产级训练 / 推理系统

`dlrm_v3/` 是把 `DlrmHSTU` 模型用 **torchrec**（Meta 的大规模推荐训练库）包起来的完整系统，目标是工业 RecSys 的训练与推理 benchmark。

### 6.1 训练 `dlrm_v3/train/`

入口 `train_ranker.py` 的结构很清晰（约 190 行）：

```python
# 命令: LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 .../train_ranker.py --dataset debug --mode train
setup(rank, world_size, ...)            # 初始化分布式
gin.parse_config_file(gin_file)         # 读配置
model, cfg, table_cfg = make_model()    # 构造 DlrmHSTU
model, opt = make_optimizer_and_shard() # torchrec 分片（模型并行）
train_dl, test_dl = make_train_test_dataloaders()
metrics = MetricsLogger(multitask_configs=...)
load_dmp_checkpoint(...)
# 四种模式：
train_loop / eval_loop / train_eval_loop / streaming_train_eval_loop
```

支持的数据集（`SUPPORTED_CONFIGS`）：`debug`、`kuairand-1k`、`movielens-1m/20m/13b/18b`、`streaming-400m/200b/100b`。可以看到它覆盖了从 1M 到 **2000 亿** 规模、以及**流式（streaming）**训练。

- `train/utils.py`：`make_model` / `make_optimizer_and_shard` / 各种 `*_loop`。
- `train/gin/*.gin`：每个规模一份配置。
- `checkpoint.py`：DMP（Distributed Model Parallel）checkpoint 存取。

### 6.2 数据 `dlrm_v3/datasets/`

| 文件 | 说明 |
|------|------|
| `dataset.py` | 基类，产出 torchrec `KeyedJaggedTensor` 批次。 |
| `movie_lens.py` / `kuairand.py` | 真实公开数据集。 |
| `synthetic_movie_lens.py` / `synthetic_streaming.py` | 合成数据（无需真实数据即可压测，`--dataset debug` 走这里）。 |
| `utils.py` | 特征处理工具。 |

### 6.3 推理 `dlrm_v3/inference/`

这是一个对接 **MLPerf loadgen** 的推理服务，用于标准化压测：

- `main.py`：推理入口。
- `model_family.py` / `inference_modules.py`：把训练好的模型拆成 `sparse_predict_module.py`（embedding 查表，CPU/分布式）+ `dense_predict_module.py`（HSTU dense 计算，GPU）两段，分别优化。
- `cpp/hstu_runner.cpp`：C++ 推理 runner。
- `data_producer.py` / `accuracy.py`：喂数据 + 验证精度。
- `thirdparty/loadgen/`：MLPerf 官方 loadgen（第三方，提供 server/offline/single-stream 等场景的压测）。
- `gin/*.gin`：推理配置（`movielens_13b`、`streaming_100b` 等）。

> 这部分体现了**「sparse / dense 解耦」**的经典工业推理范式：稀疏大表查询和稠密网络计算分开部署、各自扩缩容，对应论文里的 M-FALCON 推理摊销思想的工程落地。

---

## 7. 配置系统（gin）

整个仓库用 **[gin-config](https://github.com/google/gin-config)** 做配置注入，几乎所有「可调的东西」都是 `@gin.configurable` 函数的参数，配置文件直接给函数参数赋值。这让「换数据集 / 换模型 / 换超参」无需改代码。

学术线一个配置示例（`configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin`）：

```
train_fn.dataset_name = "ml-1m"
train_fn.max_sequence_length = 200
train_fn.main_module = "HSTU"
train_fn.item_embedding_dim = 50

hstu_encoder.num_blocks = 8
hstu_encoder.num_heads = 2
hstu_encoder.dqk = 25
hstu_encoder.dv = 25

train_fn.loss_module = "SampledSoftmaxLoss"
train_fn.num_negatives = 128
train_fn.sampling_strategy = "local"
train_fn.interaction_module_type = "DotProduct"
train_fn.top_k_method = "MIPSBruteForceTopK"
train_fn.temperature = 0.05
```

可以看到：**数据集、序列长度、主干、层数/头数、损失、负采样、相似度、top-k 方法**全部在一个文件里声明式配置。复刻时建议同样采用「配置驱动」，因为 RecSys 实验的核心成本就是大量超参/结构对比。

---

## 8. 端到端数据流与训练流程

以**学术线 next-item 召回**为例，把数据如何流过模型讲清楚（这是最值得先吃透的闭环）：

```
原始评分 (user, item, timestamp, rating)
   │  preprocess_public_data.py + research/data/preprocessor.py
   ▼
按用户聚合的行为序列 CSV
   │  DatasetV2（定长 padding、chronological、留出最后 n 个做 target）
   ▼
batch: past_ids[B,N] / past_lengths[B] / past_payloads(timestamps…) / target_ids[B]
   │  把 target scatter 到序列末尾 → 自回归监督对
   ▼
LocalEmbeddingModule: past_ids → past_embeddings[B,N,D]
   │  + LearnablePositionalEmbedding + dropout（输入预处理）
   ▼
主干 HSTU / SASRec: [B,N,D] → seq_embeddings[B,N,D]
   │  + L2Norm/LayerNorm（输出后处理）
   ▼
监督: 用 seq_embeddings[:, :-1] 预测 supervision_ids[:, 1:]
   │  负采样(InBatch/Local) + SampledSoftmaxLoss + DotProduct/MoL 相似度
   ▼
反向传播 (AdamW, DDP)
   │
   ▼
评估: 对 all_item_ids 做 MIPS top-k → HR@K / NDCG@K / MRR (TensorBoard)
```

生产线（`dlrm_v3`）的差异：输入是 `KeyedJaggedTensor`（变长、含 uih 历史 + candidates + 上下文特征），模型是 `DlrmHSTU`（多任务头同时算 CTR / 时长等），并行用 torchrec 分片，支持 `streaming` 持续训练。

---

## 9. 关键设计点

复刻时最值得借鉴 / 注意的工程设计：

1. **HSTU 主干替代标准注意力**：用 pointwise aggregated attention + gating（`stu.py`）替代 softmax attention，针对推荐的非平稳、高基数流式数据，且更易在长序列上加速。这是论文核心贡献，也是与「普通 Transformer 序列推荐」的最大区别。

2. **统一序列转导（召回 + 排序）**：通过 `preprocessors`/`contextual_interleave_preprocessor` 把 item、action、candidates、上下文交织成一条序列，让同一个模型既能做 next-item 召回（research 线），又能做多任务排序（dlrm_v3 线）。

3. **变长 jagged 张量 + 三套可切换内核**：`HammerKernel` 让同一段逻辑可在 `PYTORCH` / `TRITON` / `CUDA` 间切换——科研用 pytorch 验证正确性，生产用 cuda 求极致性能。每个算子配 parity 测试 + benchmark。

4. **Sampled Softmax + 负采样器抽象**：`InBatchNegativesSampler` / `LocalNegativesSampler` + `SampledSoftmaxLoss`，解决「物品基数巨大无法全 softmax」的问题（这是把召回做 scalable 的关键，源自 *Revisiting Neural Retrieval on Accelerators*）。

5. **相似度与 top-k 可插拔**：`DotProduct` vs `MoL`（Mixture-of-Logits）；`MIPSBruteForceTopK` 等。把「打分函数」和「检索」解耦。

6. **多任务头**：`multitask_module.py` 用 `TaskConfig` 声明任务（回归/二分类）、任务权重，支持工业排序的多目标（点击、时长…）。

7. **sparse / dense 解耦推理**：embedding 查表与 dense 网络分离部署（`sparse_predict_module` / `dense_predict_module`），便于扩缩容。

8. **配置驱动 + 多规模配置**：从 1M 到 200B 的 gin 配置 + 合成数据，使「scaling 实验」可复现。

---

## 10. 技术栈与依赖

`requirements.txt`（核心）：

```
torch>=2.6.0
fbgemm_gpu>=1.1.0      # Meta 的高性能稀疏/embedding 算子库
torchrec>=1.1.0        # 大规模推荐：分片 embedding、DMP、KeyedJaggedTensor
gin_config>=0.5.0      # 配置注入
pandas>=2.2.0          # 数据预处理
tensorboard>=2.19.0    # 训练可视化
pybind11               # C++/CUDA 扩展绑定
```

外加：

- **Triton**：GPU 内核（随 torch 提供）。
- **CUDA 12.x + nvcc**：编译 `ops/cpp/` 内核（`ops/cpp/setup.py`），HSTU attention 需要 H100（sm90）才能发挥 FlashAttention-V3 路径。
- **MLPerf loadgen**：推理压测（`dlrm_v3/inference/thirdparty/loadgen`，需单独编译）。
- 环境：官方测试为 Ubuntu 22.04 / CUDA 12.4 / Python 3.10 / 24GB+ HBM GPU。

> 提示：纯 PyTorch 路径（`ops/pytorch/`）+ 小数据集（ml-1m）可以在单卡甚至 CPU 上跑通学术线，**入门不需要 H100**。CUDA 内核与 dlrm_v3 大规模实验才需要强算力。

---

## 11. 如果要实现一个类似项目：工作清单

把上面拆出来的模块，翻译成「自己从零做需要依次完成的工作」。按依赖顺序分为四个阶段，每项标注对应仓库里的参考文件。

### 阶段 0：地基（必做）

| 工作 | 说明 | 参考 |
|------|------|------|
| 选定数据集 | 起步用 MovieLens-1M / Amazon Reviews，小、好跑、有公认 baseline。 | `preprocess_public_data.py` |
| 数据预处理 | 把交互日志按用户聚合成**时间有序的行为序列**；划分 train/eval（留最后 1 个做 target）。 | `research/data/preprocessor.py`、`dataset.py` |
| Dataset / DataLoader | 定长 padding + `past_lengths` + payload（时间戳）+ target。 | `research/data/reco_dataset.py`、`trainer/data_loader.py` |
| 配置系统 | 用 gin（或 Hydra/OmegaConf）做配置驱动，避免硬编码超参。 | `configs/*.gin` |
| 评估指标 | 实现 HR@K / NDCG@K / MRR + 一个 top-k 检索（先用 brute-force MIPS）。 | `research/data/eval.py`、`research/indexing/` |

### 阶段 1：能跑通的 baseline（MVP）

| 工作 | 说明 | 参考 |
|------|------|------|
| Item embedding 表 | 单机 `nn.Embedding`（`max_item_id × D`）。 | `embedding_modules.py` |
| 输入/输出处理 | 可学习位置编码 + dropout；输出 L2Norm/LayerNorm。 | `input_features_preprocessors.py`、`output_postprocessors.py` |
| SASRec baseline | 先实现 causal self-attention 序列模型（**强烈建议先做这个**，作为正确性与对照基线）。 | `sequential/sasrec.py` |
| 负采样 + 损失 | InBatch / Local 负采样 + Sampled Softmax。 | `autoregressive_losses.py`、`losses/sampled_softmax.py` |
| 训练循环 | next-token 自回归监督 + AdamW + TensorBoard + checkpoint。 | `trainer/train.py` |

> 完成阶段 1，你就有了一个**能复现 SASRec 数字的可工作系统**。这是后续一切的对照基准。

### 阶段 2：升级为 HSTU 生成式主干

| 工作 | 说明 | 参考 |
|------|------|------|
| HSTU block | 实现 STU 单元（pointwise aggregated attention + gating），替换 SASRec 主干，保持其余接口不变以便对比。 | `sequential/hstu.py`、`modules/stu.py` |
| 序列交织 | 把 action / 上下文特征与 item 交织进序列（从 next-item 走向更接近论文的设定）。 | `modules/preprocessors.py`、`action_encoder.py` |
| 更强相似度 | 在 DotProduct 之外尝试 MoL（Mixture-of-Logits）。 | `research/rails/` |
| Scaling 实验 | 用配置扫层数/维度/序列长度，观察指标随规模变化（验证 scaling）。 | `configs/*` 的 large 变体 |

### 阶段 3：工程化 / 规模化（按需）

| 工作 | 说明 | 参考 |
|------|------|------|
| 变长 jagged 张量 | 抛弃 dense padding，改用变长拼接以支持超长序列与高吞吐。 | `ops/jagged_tensors.py`、`modules/` |
| 高性能内核 | Triton / CUDA 实现 HSTU attention 等热点算子（先 Triton，CUDA 视算力而定）。 | `ops/triton/`、`ops/cpp/hstu_attention/` |
| 多任务排序 | 加多任务头（CTR + 时长 …），从召回扩展到排序。 | `modules/multitask_module.py`、`dlrm_hstu.py` |
| 分布式大表 | 用 torchrec 分片 embedding + 模型并行（物品/特征基数巨大时）。 | `dlrm_v3/train/utils.py` |
| 流式训练 | 持续/增量训练以应对非平稳数据。 | `dlrm_v3/.../streaming_train_eval_loop` |
| 推理服务 | sparse/dense 解耦 + 批摊销（M-FALCON 思想）+ 压测。 | `dlrm_v3/inference/` |

### 可选：语义 ID 路线（本仓库**没有**，但你的项目可能想要）

本仓库走的是 **Item ID embedding**。若想做 TIGER/OneRec 那种「语义 ID + 自回归生成 codeword」：

- 额外需要：物品 embedding（内容/协同）→ **RQ-VAE / RQ-Kmeans** 量化成分层 codeword → 用这些 codeword 作为模型词表 → decoder 自回归生成 + Beam Search 解码。
- 这部分本仓库不覆盖，需要单独实现（参考 `01-生成式推荐架构-背景与技术综述.md` 第 4.1 节）。

---

## 12. 给本项目的落地建议

结合本项目 README 的路线图，建议的最小可行路径（MVP → 进阶）：

1. **先抄学术线的闭环**：MovieLens-1M + 自己的 Dataset + SASRec + SampledSoftmax + HR/NDCG 评估。目标是**复现 SASRec 数字**，跑通「数据→训练→评估」。（对应本项目路线图的「数据预处理」「序列模型 baseline (SASRec)」「离线评估」）
2. **把主干换成 HSTU**：保持接口不变，仅替换 block，做 A/B 对比，验证「同等配置下 HSTU > SASRec」。（对应「生成式主干 (HSTU)」）
3. **决定是否走语义 ID**：若要对齐 TIGER/OneRec，再补 RQ-VAE/RQ-Kmeans + 自回归 codeword 生成（本仓库不提供，需自研）。（对应「物品语义 ID 生成」）
4. **再谈工程化**：jagged 张量、Triton 内核、多任务、torchrec、流式、推理服务——这些是规模化阶段才需要的，初期不要过早投入。（对应「推理与服务优化」）

> 一句话：**这个仓库给你的是「序列建模主干 + 大规模工程化」的工业级范本，从 `research/` 入手是最高性价比的学习与起步路径；语义 ID 生成那一半需要自己补。**

---

## 参考

- 仓库：<https://github.com/meta-recsys/generative-recommenders>
- 论文：*Actions Speak Louder than Words*，ICML 2024，[arXiv:2402.17152](https://arxiv.org/abs/2402.17152)
- 配套背景综述：[`01-生成式推荐架构-背景与技术综述.md`](01-生成式推荐架构-背景与技术综述.md)

