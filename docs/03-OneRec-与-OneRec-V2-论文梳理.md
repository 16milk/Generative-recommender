# OneRec 与 OneRec-V2 论文梳理

> 本文档梳理快手（Kuaishou）的两篇生成式推荐工作：**OneRec**（V1）与 **OneRec-V2**，目的是搞清楚它们各自解决什么问题、模型结构与训练方法、以及 V1 → V2 的演进逻辑，为本项目实现一套类似的「端到端生成式推荐」提供参考。
>
> - **OneRec**：*Unifying Retrieve and Rank with Generative Recommender and Preference Alignment*，[arXiv:2502.18965](https://arxiv.org/abs/2502.18965)（另有更详细的 *OneRec Technical Report*，[arXiv:2506.13695](https://arxiv.org/abs/2506.13695)）。
> - **OneRec-V2**：*OneRec-V2 Technical Report*，[arXiv:2508.20900](https://arxiv.org/abs/2508.20900)。
>
> 背景知识见 [`01-生成式推荐架构-背景与技术综述.md`](01-生成式推荐架构-背景与技术综述.md)；Meta HSTU 的代码框架见 [`02-Meta-generative-recommenders-代码框架梳理.md`](02-Meta-generative-recommenders-代码框架梳理.md)。

---

## 目录

1. [一图看懂两代演进](#1-一图看懂两代演进)
2. [共同的出发点：为什么要做端到端生成式推荐](#2-共同的出发点为什么要做端到端生成式推荐)
3. [OneRec（V1）详解](#3-onerecv1详解)
4. [OneRec-V2 详解](#4-onerec-v2-详解)
5. [V1 → V2 关键变化对比](#5-v1--v2-关键变化对比)
6. [线上效果与工程指标](#6-线上效果与工程指标)
7. [对实现一个类似系统的启示](#7-对实现一个类似系统的启示)
8. [术语速查](#8-术语速查)
9. [参考](#9-参考)

---

## 1. 一图看懂两代演进

```
                OneRec (V1)                         OneRec-V2
            ┌──────────────────┐              ┌──────────────────────┐
 架构        │ Encoder-Decoder  │  ───────▶    │ Lazy Decoder-Only     │
            │ (T5 风格) + MoE   │              │ (去 encoder, 简化      │
            │                  │              │  cross-attn) + MoE    │
            └──────────────────┘              └──────────────────────┘
 痛点        encoder 吃掉 97.66% 算力   ───▶   把算力集中到 target 解码
 规模        ~0.5B / 1B               ───▶    可 scale 到 8B（符合 scaling law）
 对齐        DPO + 奖励模型(RM)        ───▶    真实用户反馈 + GBPO 强化学习
            (IPA 迭代偏好对齐)                 (时长感知奖励整形)
 共享        语义 ID(RQ-Kmeans) + session-wise 生成 + 单阶段统一召回排序
```

一句话：**V1 证明了「单一生成模型可以端到端替代级联召回+排序，并超过它」；V2 解决了「V1 算力分配低效 + 仅靠奖励模型对齐」两大瓶颈，让模型能 scale 到 8B 并用真实用户反馈对齐。**

---

## 2. 共同的出发点：为什么要做端到端生成式推荐

两代工作针对的是同一个传统范式的痛点——**级联排序（cascade ranking）**：

```
召回(recall) → 粗排(pre-ranking) → 精排(ranking)
```

问题在于：

- **每一级独立优化**，上一级的效果成为下一级的「上界」，信息层层裁剪，难以全局最优。
- 即便有各种「让各级交互」的改进，本质仍是级联范式。
- 生成式检索（GR，如 TIGER）虽有潜力，但此前**只用于召回**，精度还达不到精心设计的多级排序器。

OneRec 系列的主张：**用一个统一的生成模型，端到端同时完成召回与排序**，直接优化最终目标，并借鉴 LLM 的 scaling law——把算力/参数堆上去就能持续涨点。这与 Meta HSTU 的思路一致，但 OneRec 走的是 **语义 ID（Semantic ID）+ encoder-decoder/decoder 生成** 路线，而非 HSTU 的 Item-ID 序列转导。

---

## 3. OneRec（V1）详解

OneRec V1 的完整框架分**两个阶段**：(i) session-wise 生成式预训练；(ii) IPA 偏好对齐（DPO + 奖励模型）。

### 3.1 物品 Tokenization：多级平衡量化（RQ-Kmeans）

每个视频先用**多模态 embedding** `e_i ∈ R^d`（已对齐真实 user-item 行为分布）表示，再量化成一组分层语义 ID。

- 不用 TIGER 那种 **RQ-VAE**，因为它会出现 **码本分布不均（hourglass 现象）**——少数 code 被频繁使用、多数闲置。
- 改用 **残差 K-means 量化（Residual K-Means / RQ-Kmeans）**，逐层做：
  - 第 1 层残差 `r_i^1 = e_i`；
  - 第 `l` 层找最近质心 `s_i^l = argmin_k ||r_i^l − c_k^l||²`；
  - 残差更新 `r_i^{l+1} = r_i^l − c_{s_i^l}^l`；
  - 共 `L` 层，得到分层语义 ID `(s_i^1, s_i^2, …, s_i^L)`。
- 关键是 **Balanced K-means（算法 1）**：把 `|V|` 个视频强制均分到 `K` 个簇（每簇恰好 `w = |V|/K` 个），最大化码本利用率，解决分布不均。

> 直觉：相似视频共享语义 ID 前缀 → 泛化好、缓解冷启动、词表从「亿级 item id」压到「几层 × 几千 code」。

### 3.2 序列构造与 Session-wise 生成

**输入（用户侧）**：用户正向历史行为序列 `H_u = {v_1^h, …, v_n^h}`（有效观看、点赞、关注、转发过的视频），每个视频用其语义 ID 表示。

**输出**：不是「下一个视频」，而是一个 **session（会话列表）** `S = {v_1, …, v_m}`，`m` 通常 5–10 个视频。

为什么是 session 而不是单个 item（**session-wise list generation**，核心创新之一）：

- 传统 point-by-point「逐个预测下一个」需要**手工规则**来保证列表的连贯性、多样性。
- session-wise 让模型**自主学习一整个列表的最优结构**（视频间的相对内容与顺序、用户兴趣、连贯性、多样性）。
- **高质量 session 的筛选标准**：用户实际观看 ≥ 5 个；总观看时长超阈值；存在点赞/收藏/分享等互动。

### 3.3 模型结构：Encoder-Decoder（T5 风格）+ MoE

- **Encoder**：堆叠多头自注意力 + FFN，编码用户历史 `H = Encoder(H_u)`。
- **Decoder**：以目标 session 的语义 ID 为输入，自回归生成；每个视频前加 `[BOS]` 起始 token 分隔。
- **MoE（Mixture-of-Experts）**：decoder 的 FFN 换成稀疏 MoE——`N_MoE` 个专家里每个 token 只激活 top-`K_MoE` 个，**在不成比例增加 FLOPs 的前提下扩大模型容量**（借鉴 LLM scaling）。
- **训练目标**：对目标 session 语义 ID 做 **next-token 预测（NTP）+ 交叉熵损失**（与 LLM 完全同构）。训练到一定程度得到种子模型 `M_t`。

### 3.4 IPA：迭代偏好对齐（DPO + 奖励模型）

预训练只学会了「什么是好 session」，V1 进一步用 **DPO（Direct Preference Optimization）** 提升生成质量。难点与做法：

- **难点**：推荐系统对每个请求**只有一次展示机会**，无法像 NLP 那样同时拿到「正/负」人工标注偏好对。且用户-物品交互数据稀疏。
- **奖励模型（RM）**：训练一个 session-wise reward model `R(u, S)` 来模拟用户、给候选 session 打分。
- **自构造 hard 偏好对**：借鉴 hard negative sampling，从模型自己的 **beam search 结果**里采样多个 session，用 RM 打分后取「最佳=chosen / 最差=rejected」组成偏好对（self-hard negatives），而非随机采负。
- **IPA（Iterative Preference Alignment，算法 2）**：迭代地「采样 → RM 打分 → 选 chosen/rejected → DPO 更新」，让模型自我改进。少量 DPO 样本即可显著对齐用户偏好。

### 3.5 V1 成果

- 部署在快手主场景（数亿日活），**观看时长 +1.6%**——这是相当显著的工业级提升。
- 据论文，是**首批用单一生成模型显著超越精心设计的多级级联系统**的工业方案之一。

---

## 4. OneRec-V2 详解

V2 是一份技术报告，针对 V1 暴露的两个瓶颈做改进：**(1) encoder-decoder 算力分配低效；(2) 强化学习仅依赖奖励模型**。

### 4.1 痛点定位：算力都浪费在「编码上下文」上

V2 把 transformer 的计算分成两类：

- **Context Encoding（上下文编码）**：处理用户特征序列——encoder 里的变换 + decoder 交叉注意力里对 context 的投影。
- **Target Decoding（目标解码）**：真正参与 loss 的目标物品语义 token 的计算——自注意力 + FFN + cross-attn 的 query/output 变换。

实测（1B 模型，context 长度 512）：

| 架构 | 目标解码占总算力比例 |
|------|----------------------|
| Encoder-Decoder（V1，0.5B:0.5B） | **2.34%**（context 长度 3000 时仅 0.41%） |
| Naive Decoder-Only（1B） | 2.85% |
| **Lazy Decoder-Only（1B，V2）** | **≈ 100%** |

也就是说，V1 把 **97.66%** 的算力花在「编码用户历史」上，真正做推荐决策的生成部分不到 3%。context 越长，这个比例越离谱。这严重限制了模型 scale 的性价比。

### 4.2 Lazy Decoder-Only 架构（核心创新）

把「上下文」当作**静态的条件信息**，只通过 cross-attention 访问，从而把算力几乎全部集中到目标解码上。

```
        ┌─────────────────────────────────────────────┐
        │ Context Processor                            │
        │ 异构用户特征(profile + 短期/长期行为) → 拼接   │
        │ → 按特征维切分 → 每层的 (k_l, v_l)（RMSNorm）  │
        │ → 产出 L_kv 组 layer-shared KV 对             │
        └─────────────────────────────────────────────┘
                          │ (静态 KV，供 cross-attn 复用)
                          ▼
        ┌─────────────────────────────────────────────┐
        │ Lazy Decoder（N_layer 个 block 堆叠）          │
        │ 输入: [BOS, s¹, s²]  (目标物品的语义 token)    │
        │ 每个 block:                                   │
        │   1) Lazy Cross-Attn（无 K/V 投影 + GQA）      │
        │   2) Causal Self-Attn                         │
        │   3) FFN（深层用 MoE）                         │
        │ 输出: 预测下一个语义 ID                         │
        └─────────────────────────────────────────────┘
```

三个关键设计：

1. **Context Processor**：把异构、多模态的用户信号（profile、短期/长期行为）拼成统一 context，并直接转换成各层的 **layer-shared key-value 对**（用 RMSNorm 归一化），无需 encoder 反复变换。
2. **Lazy Cross-Attention（极简交叉注意力）**：
   - **去掉 K/V 投影层**——context 直接当 KV 用，省掉大量参数和计算。
   - **KV 共享**：多个 decoder block 共享同一组 `(k_l, v_l)`（`l_kv = ⌊l·L_kv / N_layer⌋`）；甚至进一步令 `v_l = k_l`（tied KV）。
   - **GQA（Grouped Query Attention）**：query 用 `H_q` 个头，KV 只用 `G_kv` 组（`G_kv ≪ H_q`），大幅降显存。
3. **Tokenizer**：沿用 V1 的语义 tokenizer，每个目标物品 3 个语义 ID；训练时用前 2 个 + `[BOS]`。深层 FFN 用 MoE，并采用 **DeepSeek-V3 的无辅助损失负载均衡**策略。

**收益**：总计算量降 **94%**、训练资源降 **90%**，在同等算力预算下支持 **16× 更大模型（0.5B → 8B）**，且收敛 loss **紧贴 Chinchilla（Hoffmann 2022）scaling law** ——首次在生成式推荐上给出清晰的 scaling 经验+理论指引。

### 4.3 训练数据组织：只对「最新曝光」算 loss

为避免标准 NTP 的冗余与数据泄漏：

- **Naive 按曝光时间组织**：`A→B` 模式会在多次曝光里被重复训练（冗余）。
- **User-Centric（整段用户历史）**：有**时间泄漏**和流行度偏置风险。
- **V2 的做法**：按时间顺序组织，但**只对最新曝光的物品计算训练 loss**（更早的物品作为 context、不参与 NTP）。这正好契合 lazy decoder「context 静态、只解码目标」的设计。

### 4.4 后训练：用真实用户反馈做偏好对齐

V1 只靠奖励模型 RM，有两个问题：**采样效率低**（在线生成+打分耗算力，只能覆盖 ~1% 用户）、**reward hacking**（策略钻 RM 的空子）。V2 直接利用**真实用户反馈**：

1. **Duration-Aware Reward Shaping（时长感知奖励整形）**：原始观看时长有「视频越长、绝对时长越高」的偏置；整形后让奖励反映**内容质量而非单纯时长**。正样本取按时长感知奖励排序的 **top 25%**，负样本取显式负反馈（如「不感兴趣/dislike」）。
2. **GBPO（Gradient-Bounded Policy Optimization，梯度有界策略优化）**：V2 新提出的强化学习方法。
   - 与 PPO/GRPO/DAPO 等**靠 ratio 裁剪**的方法不同，GBPO **不丢弃任何样本的梯度**（充分利用全样本，鼓励多样探索）。
   - 对负样本，用 **BCE 损失的稳定梯度**给 RL 梯度设动态上界——解决「ratio=1 的样本也可能因负样本导致梯度爆炸/模型崩溃」的问题（V1 用的 ECPO 等裁剪法无法完全避免）。
3. **On-policy 自我迭代**：OneRec 已服务 25% 线上流量，可用**自己生成的曝光样本**做 on-policy RL，实现自我改进（self-improvement）。实验表明加入 OneRec 自产样本后，几乎所有指标都显著变好。

> V2 还对比了三种 RL 信号来源：**Reward Model / User Feedback / Hybrid**。结论：RM 偏向提升互动指标，真实用户反馈偏向提升 App 停留时长；二者结合（Hybrid）能更好平衡时长与互动、缓解「跷跷板效应」。线上部署版为简化系统**只用了 User Feedback Signals**。

---

## 5. V1 → V2 关键变化对比

| 维度 | OneRec（V1） | OneRec-V2 |
|------|--------------|-----------|
| 主干架构 | Encoder-Decoder（T5 风格）+ MoE | **Lazy Decoder-Only** + MoE |
| 上下文处理 | encoder 编码 + decoder cross-attn | **Context Processor** 产出 layer-shared 静态 KV |
| Cross-Attention | 标准（含 K/V 投影） | **去 K/V 投影 + KV 共享 + GQA** |
| 算力分配 | 97.66% 花在 context 编码 | ≈100% 集中到目标解码 |
| 模型规模 | ~0.5B / 1B | 可 scale 到 **8B**，符合 scaling law |
| 训练资源 | 基准 | 总算力 **−94%**、训练资源 **−90%** |
| 数据组织 | session-wise（曝光） | 时间序 + **只对最新曝光算 loss** |
| 偏好对齐 | DPO + 奖励模型（IPA 迭代） | **真实用户反馈** + 时长感知奖励整形 |
| RL 算法 | ECPO（早裁剪 GRPO） | **GBPO**（梯度有界，不丢样本） |
| 采样 | RM 在线打分，仅覆盖 ~1% 用户 | on-policy 自产样本（已占 25% 流量） |
| 物品 tokenization | RQ-Kmeans（Balanced K-means） | 沿用（每物品 3 个语义 ID） |
| 任务形式 | session-wise 自回归生成 | 同（next-item / session 语义 ID 生成） |

不变的内核：**语义 ID + 自回归生成 + 单阶段统一召回排序 + MoE 扩容**。V2 主要是「架构提效以便 scale」+「对齐方式从代理奖励转向真实反馈」。

---

## 6. 线上效果与工程指标

**OneRec V1**：快手主场景（数亿日活），观看时长 **+1.6%**。

**OneRec-V2**（对比 V1，Kuaishou / Kuaishou Lite，4 亿日活，5% 流量、1 周）：

| 指标 | Kuaishou | Kuaishou Lite |
|------|----------|---------------|
| App Stay Time（停留时长，最重要） | **+0.467%** | **+0.741%** |
| LT7（7 日留存） | +0.069% | +0.034% |
| Watch Time | +1.367% | +0.762% |
| Like / Follow / Comment / … | +3.9% ~ +5.4% | +5% ~ +8% |

线上部署：**1B 参数、context 长度 3000、beam size 512**，L20 GPU 上延迟 **36ms**、MFU（模型算力利用率）**62%**。在多目标上达到更好平衡、缓解「跷跷板效应」。

> ⚠️ V2 也暴露了**生态隐患**：关缓存的 1% 全量实验里，互动指标暴涨 9.6%~29.2%，但**冷启动视频曝光下降 36.7%~44.7%**、聚类密度上升——生成式系统可能放大热门偏置、挤压新内容曝光。这是落地时必须正视的副作用。

---

## 7. 对实现一个类似系统的启示

结合本项目「从零搭一套生成式推荐」的目标，从 OneRec 系列可提炼的工程要点：

1. **语义 ID 是地基，且量化方法很关键**：直接用 RQ-VAE 容易码本分布不均（hourglass）。优先考虑 **Balanced K-means / RQ-Kmeans** 以最大化码本利用率。这是 HSTU 仓库（doc 02）**没有覆盖**、需要自研的一半。
2. **session-wise 生成 > point-wise**：如果场景是 feed 流/一次返回多个，直接生成整个 list 比逐个生成 + 手工拼接更优雅，也更易学到多样性与连贯性。
3. **MoE 是低成本扩容手段**：想 scale 又不想线性增加 FLOPs，decoder FFN 换 MoE。
4. **算力要花在「产生 loss 的 token」上**：V1→V2 最大教训——别让上下文编码吃掉绝大部分算力。若用 decoder-only，考虑「context 静态化 + 简化 cross-attn（去 KV 投影 / KV 共享 / GQA）」。
5. **对齐：先 DPO/RM，再过渡到真实反馈**：起步可用奖励模型 + DPO（数据需求小）；规模上来、有自产流量后，转向**真实用户反馈 + on-policy RL**，并注意 RL 梯度稳定性（GBPO 思路）。
6. **奖励要去偏**：观看时长有视频长度偏置，需做 duration-aware 整形，否则模型会偏向长视频而非好内容。
7. **警惕生态副作用**：上线前评估冷启动曝光、内容多样性、热门偏置，不能只看互动/时长。

> 与 doc 02 的关系：Meta HSTU 仓库给你「序列建模主干 + 大规模工程化」的可运行代码（Item-ID 路线）；OneRec 系列给你「语义 ID + encoder/decoder 生成 + 偏好对齐」的工业方法论（论文，无开源代码）。两者结合能拼出较完整的实现蓝图。

---

## 8. 术语速查

- **GR (Generative Recommendation)**：生成式推荐，把推荐建模为自回归序列生成。
- **Semantic ID（语义 ID）**：物品 embedding 量化得到的分层离散 token，相似物品共享前缀。
- **RQ-VAE / RQ-Kmeans**：残差量化（VAE 版 / K-means 版）。OneRec 用后者 + Balanced K-means。
- **Session-wise generation**：一次生成一个视频列表（session），而非单个 item。
- **MoE (Mixture-of-Experts)**：稀疏专家网络，每 token 只激活部分专家，扩容不增 FLOPs。
- **DPO (Direct Preference Optimization)**：用偏好对直接优化策略，替代 RLHF。
- **IPA (Iterative Preference Alignment)**：V1 的迭代偏好对齐（采样→RM 打分→DPO）。
- **Lazy Decoder-Only**：V2 架构，context 静态化、简化 cross-attn，把算力集中到解码。
- **GQA (Grouped Query Attention)**：query 多头、KV 少组，降显存。
- **GBPO**：V2 的梯度有界策略优化，用 BCE 梯度给 RL 梯度设界、不丢样本。
- **MFU (Model FLOPs Utilization)**：模型算力利用率（V2 线上 62%）。
- **Reward Hacking**：策略钻奖励模型空子，提升代理奖励但无真实收益。

---

## 9. 参考

- OneRec：*Unifying Retrieve and Rank with Generative Recommender and Preference Alignment*，[arXiv:2502.18965](https://arxiv.org/abs/2502.18965)
- OneRec Technical Report：[arXiv:2506.13695](https://arxiv.org/abs/2506.13695)
- OneRec-V2 Technical Report：[arXiv:2508.20900](https://arxiv.org/abs/2508.20900)
- 关联背景：[`01-生成式推荐架构-背景与技术综述.md`](01-生成式推荐架构-背景与技术综述.md)、[`02-Meta-generative-recommenders-代码框架梳理.md`](02-Meta-generative-recommenders-代码框架梳理.md)
- 相关工作：TIGER（[arXiv:2305.05065](https://arxiv.org/abs/2305.05065)）、Meta HSTU（[arXiv:2402.17152](https://arxiv.org/abs/2402.17152)）、DeepSeek-V3（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）
