# Generative Recommender（生成式推荐架构）

一个从零搭建 **生成式推荐系统（Generative Recommender, GR）** 的项目。目标是把推荐从传统的「判别式打分 + 多级级联」范式，重构为「用户行为序列上的自回归生成」范式（参考 Meta HSTU、Google TIGER、快手 OneRec 等工作）。

## 项目结构

```
generative-recommender/
├── README.md                                   # 项目说明（本文件）
└── docs/
    └── 01-生成式推荐架构-背景与技术综述.md       # 背景知识 / 技术栈 / 前沿进展 / 未来展望
```

## 从这里开始

先阅读 [`docs/01-生成式推荐架构-背景与技术综述.md`](docs/01-生成式推荐架构-背景与技术综述.md)，它系统介绍了：

- 为什么需要生成式推荐（传统 DLRM 的瓶颈）
- 从判别式到生成式的范式转变
- 关键技术：物品 Tokenization（语义 ID / RQ-VAE / RQ-Kmeans）、序列建模、HSTU、对齐
- 完整技术栈（算法 / 框架 / 数据 / 服务 / 评估）
- 2024–2026 的前沿进展（TIGER、HSTU、OneRec / OneRec-V2 等）
- 挑战与未来展望

## 路线图（规划中）

- [ ] 数据预处理与用户行为序列构造
- [ ] 物品语义 ID 生成（RQ-VAE / RQ-Kmeans baseline）
- [ ] 序列模型 baseline（SASRec）
- [ ] 生成式主干（HSTU / decoder-only）实现
- [ ] 离线评估（HR@K / NDCG@K）
- [ ] 推理与服务优化
