# Generative Recommender（生成式推荐架构）

一个从零搭建 **生成式推荐系统（Generative Recommender, GR）** 的项目。目标是把推荐从传统的「判别式打分 + 多级级联」范式，重构为「用户行为序列上的自回归生成」范式（参考 Meta HSTU、Google TIGER、快手 OneRec 等工作）。

## 项目结构

```
generative-recommender/
├── README.md                                   # 项目说明（本文件）
└── docs/
    ├── 01-生成式推荐架构-背景与技术综述.md       # 背景知识 / 技术栈 / 前沿进展 / 未来展望
    ├── 02-Meta-generative-recommenders-代码框架梳理.md  # 官方 HSTU 仓库代码框架拆解 + 复刻工作清单
    ├── 03-OneRec-与-OneRec-V2-论文梳理.md         # 快手 OneRec / OneRec-V2 论文方法与演进
    ├── 04-剪映生成式召回启发的GR框架设计.md       # 结合内部实践沉淀的可落地 GR 框架设计
    └── generative-recommender-architecture.html   # 精致 HTML 架构图与端到端链路展示
```

## 从这里开始

先阅读 [`docs/01-生成式推荐架构-背景与技术综述.md`](docs/01-生成式推荐架构-背景与技术综述.md)，它系统介绍了：

- 为什么需要生成式推荐（传统 DLRM 的瓶颈）
- 从判别式到生成式的范式转变
- 关键技术：物品 Tokenization（语义 ID / RQ-VAE / RQ-Kmeans）、序列建模、HSTU、对齐
- 完整技术栈（算法 / 框架 / 数据 / 服务 / 评估）
- 2024–2026 的前沿进展（TIGER、HSTU、OneRec / OneRec-V2 等）
- 挑战与未来展望

接着读 [`docs/02-Meta-generative-recommenders-代码框架梳理.md`](docs/02-Meta-generative-recommenders-代码框架梳理.md)，它把 Meta 官方 HSTU 仓库 [`meta-recsys/generative-recommenders`](https://github.com/meta-recsys/generative-recommenders) 的代码框架逐层拆解：

- 仓库的两条主线（学术复现线 `research/` vs 生产线 `dlrm_v3/`）
- 顶层目录结构与各模块职责
- 端到端数据流与训练流程
- 复刻一个类似项目需要分阶段完成的「工作清单」

如果想了解「语义 ID + encoder/decoder 生成」这条工业路线，读 [`docs/03-OneRec-与-OneRec-V2-论文梳理.md`](docs/03-OneRec-与-OneRec-V2-论文梳理.md)，它梳理了快手 OneRec 系列：

- OneRec（V1）：RQ-Kmeans 语义 ID + session-wise 生成 + Encoder-Decoder/MoE + DPO 偏好对齐
- OneRec-V2：Lazy Decoder-Only 架构（算力 −94%、可 scale 到 8B）+ 真实用户反馈对齐（GBPO）
- V1 → V2 的演进逻辑、线上效果与对自研系统的启示

最后读 [`docs/04-剪映生成式召回启发的GR框架设计.md`](docs/04-剪映生成式召回启发的GR框架设计.md)，它结合剪映/CapCut 生成式召回实践，把本项目需要建设的 GR 框架拆成数据、Tokenizer、模型、训练、Serving、监控六层，并配套一个可直接在浏览器打开的精致 HTML 架构图：[`docs/generative-recommender-architecture.html`](docs/generative-recommender-architecture.html)。

## 路线图（规划中）

- [ ] 数据预处理与用户行为序列构造
- [ ] 物品语义 ID 生成（RQ-VAE / RQ-Kmeans baseline）
- [ ] 序列模型 baseline（SASRec）
- [ ] 生成式主干（HSTU / decoder-only）实现
- [ ] 离线评估（HR@K / NDCG@K）
- [ ] 推理与服务优化
