# BERT 与 GPT 架构对比

> 深入理解 Encoder-only 与 Decoder-only 两种范式的设计哲学

---

## 第一部分：BERT（Encoder-Only）

### 1. 概述

BERT（**B**idirectional **E**ncoder **R**epresentations from **T**ransformers）由 Google 于 2018 年提出，采用 **Transformer Encoder 堆叠** 架构，通过双向上下文建模实现深度的语言理解。

核心创新：**Masked Language Model（MLM）** 预训练 —— 随机遮盖输入中的部分 token，让模型从上下文中预测被遮盖的内容。

### 2. 模型架构

BERT 只使用 Transformer 的 Encoder 部分：

```
输入 → Token Emb + Segment Emb + Position Emb
     → Encoder Layer × 12 (BERT-Base) / × 24 (BERT-Large)
     → 上下文表示
```

两个标准规格：

| 规格 | 层数 $N$ | $d_{\text{model}}$ | 头数 $h$ | $d_{ff}$ | 参数量 |
|------|-----------|---------------------|-----------|-----------|--------|
| BERT-Base | 12 | 768 | 12 | 3072 | ~110M |
| BERT-Large | 24 | 1024 | 16 | 4096 | ~340M |

### 3. MLM 预训练（Masked Language Model）

#### 3.1 核心思想

类似完形填空：给定一个被部分遮盖的句子，让模型预测被遮盖的 token。

#### 3.2 Mask 策略

随机选择 **15%** 的 token 进行操作，对选中 token 的替换规则：

| 概率 | 操作 | 目的 |
|------|------|------|
| **80%** | 替换为 `[MASK]` | 主要训练信号 |
| **10%** | 替换为随机 token | 缓解预训练-微调 mismatch（微调时没有 `[MASK]`） |
| **10%** | 保持不变 | 让模型也学会输出原始 token（偏置向真实分布） |

**示例**：

```
原始句子:  "我 爱 人工 智能"
15%选中:   "爱", "智能"

80% [MASK]:   "我 [MASK] 人工 [MASK]"  → 预测 "爱", "智能"
10% 随机替换: "我 吃饭 人工 [MASK]"    → 预测 "爱", "智能"
10% 不变:     "我 爱 人工 [MASK]"      → 预测 "爱", "智能"
```

#### 3.3 为什么不全用 `[MASK]`

微调阶段没有 `[MASK]` token，若预训练阶段只见过 `[MASK]`，会导致 **pre-train/fine-tune 分布不一致**。混合策略使模型学会从真实 token 的上下文中预测，减少这种 mismatch。

#### 3.4 损失函数

仅对被遮盖位置的预测计算交叉熵损失：

$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P(w_i | \text{context})
$$

其中 $\mathcal{M}$ 为被选中遮盖的 token 集合。

### 4. NSP 任务（Next Sentence Prediction）

#### 4.1 任务定义

判断句子 B 是否为句子 A 的下一句：

- **正例（IsNext）**：从语料中连续抽取的两个句子
- **负例（NotNext）**：从语料中随机配对的不相关句子

正负例各占 50%。

#### 4.2 输出

使用 `[CLS]` token 的最终隐藏状态，经分类头输出二分类概率：

$$
P(\text{IsNext}|A, B) = \text{sigmoid}(W_{\text{NSP}} \cdot h_{[\text{CLS}]} + b_{\text{NSP}})
$$

#### 4.3 NSP 的争议与改进

RoBERTa 研究发现：去掉 NSP 任务，仅使用 MLM，并在更长序列上训练，效果反而更好。NSP 被认为过于简单，模型可能仅学到表面线索（如主题连贯性）。

ALBERT 提出 **SOP（Sentence Order Prediction）** 替代 NSP：判断两个句子是否顺序正确，任务更难，效果更好。

### 5. 输入表示

BERT 的输入由三种 Embedding 求和得到：

$$
X_{\text{input}} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}}
$$

#### 5.1 Token Embedding ($E_{\text{token}}$)

使用 WordPiece 分词（30,522 词表），每个 token 映射为 $d_{\text{model}}$ 维向量。特殊 token 包括：
- `[CLS]`：序列起始标记，其最终隐藏状态用作分类任务的聚合表示
- `[SEP]`：分隔句子 A 和 B
- `[PAD]`：填充标记
- `[MASK]`：遮盖标记
- `[UNK]`：未知词标记

#### 5.2 Segment Embedding ($E_{\text{segment}}$)

区分 token 属于句子 A（$E_A$）还是句子 B（$E_B$）。可学习的 $2 \times d_{\text{model}}$ 参数矩阵。对于单句任务，全部使用 $E_A$。

#### 5.3 Position Embedding ($E_{\text{position}}$)

可学习的位置嵌入，支持最大序列长度 512。与原始 Transformer 的正弦编码不同，BERT 的位置嵌入完全由数据学习。

#### 5.4 输入格式

```
[CLS] tok_A1 tok_A2 ... [SEP] tok_B1 tok_B2 ... [SEP]
Seg:   0     0     0   ...   0     1     1   ...   1
Pos:   0     1     2   ...   k    k+1   k+2 ...  511
```

### 6. Fine-tuning 方式

BERT 的预训练模型可直接适配多种下游任务，无需修改架构，仅需添加任务特定的输出层：

#### 6.1 单句分类（情感分析、文本分类）

$$
P(y|x) = \text{softmax}(W_c \cdot h_{[\text{CLS}]} + b_c)
$$

输入格式：`[CLS] 文本 [SEP]`

#### 6.2 句对分类（NLI、语义相似度）

$$
P(y|A, B) = \text{softmax}(W_c \cdot h_{[\text{CLS}]} + b_c)
$$

输入格式：`[CLS] A [SEP] B [SEP]`

#### 6.3 序列标注（NER、POS Tagging）

每个 token 独立分类：

$$
P(y_i|x) = \text{softmax}(W_s \cdot h_i + b_s), \quad \forall i
$$

输入格式：`[CLS] tok_1 tok_2 ... tok_n [SEP]`

#### 6.4 问答（SQuAD）

预测答案片段的起始和结束位置：

$$
P_{\text{start}}(i) = \text{softmax}(W_s \cdot h_i) ,\quad P_{\text{end}}(j) = \text{softmax}(W_e \cdot h_j)
$$

输入格式：`[CLS] 问题 [SEP] 段落 [SEP]`

#### 6.5 Fine-tuning 超参数建议

| 超参数 | 推荐值 |
|--------|--------|
| Batch Size | 16, 32 |
| 学习率 | 2e-5, 3e-5, 5e-5 |
| Epochs | 3, 4 |
| Max Seq Length | 128 / 512 |
| Warmup | 前 10% 步 |

### 7. BERT 的局限性

- **无法生成文本**：双向建模使其天然不适合自回归生成
- **`[MASK]` 假设独立性**：MLM 假设各被遮盖 token 相互独立，忽略其间的依赖
- **计算效率**：15% token 的预测贡献了损失，其余 85% 没有学习信号
- **固定长度**：最大 512 token 限制了长文档处理能力

---

## 第二部分：GPT（Decoder-Only）

### 1. 概述

GPT（**G**enerative **P**re-trained **T**ransformer）由 OpenAI 提出，采用 **Transformer Decoder 堆叠** 架构，通过大规模自回归语言建模实现通用文本生成能力。

设计哲学：**大规模无监督预训练 + 任务无关的迁移**，即让模型学会"预测下一个词"，从而隐式地学习语言知识。

### 2. 自回归预训练

#### 2.1 核心公式

给定语料 $\mathcal{U} = \{u_1, u_2, \dots, u_n\}$，最大化：

$$
\mathcal{L}(\mathcal{U}) = \sum_{i=1}^{n} \log P(u_i | u_{i-k}, \dots, u_{i-1}; \Theta)
$$

其中 $k$ 为上下文窗口大小，$\Theta$ 为模型参数。

等价于：

$$
\mathcal{L} = \sum_{t} \log P(x_t | x_{<t})
$$

#### 2.2 实现

使用 Masked Self-Attention（Causal Mask），确保每个位置只看历史信息：

$$
h_0 = U W_e + W_p
$$
$$
h_l = \text{transformer\_block}(h_{l-1}) \quad \forall l \in [1, N]
$$
$$
P(u) = \text{softmax}(h_N W_e^T)
$$

注意：输入和输出共享 Token Embedding 矩阵 $W_e$（Weight Tying），减少参数量并提高泛化性。

### 3. GPT 系列演进

#### 3.1 GPT-1（2018）

- 参数：117M
- 层数：12
- $d_{\text{model}} = 768$, 头数 12
- 训练数据：BooksCorpus（~5GB）
- 意义：证明了生成式预训练的有效性

#### 3.2 GPT-2（2019）

- 参数：1.5B（最大版本）
- 层数：48
- $d_{\text{model}} = 1600$
- 训练数据：WebText（~40GB），从 Reddit 高赞外链采集
- **关键发现**：语言模型是**无监督多任务学习器** —— 无需显式监督，只靠"预测下一个词"就可以学习翻译、摘要、问答等多种能力
- 提出 **Zero-shot Transfer**：用自然语言指令替代任务特定的格式

#### 3.3 GPT-3（2020）

- 参数：175B
- 层数：96
- $d_{\text{model}} = 12288$, 头数 96
- 训练数据：~570GB（CommonCrawl, WebText2, Books, Wikipedia）
- **核心贡献**：In-context Learning

#### 3.4 GPT-4（2023）

- 参数规模未公开（传闻 ~1.8T MoE）
- 多模态输入（文本 + 图像）
- 更强的推理、编码、指令遵循能力
- 引入更复杂的 RLHF 和安全对齐

### 4. In-context Learning

#### 4.1 三种范式

| 范式 | 定义 | 示例 |
|------|------|------|
| **Zero-shot** | 仅给出任务描述，无示例 | "将下面句子翻译成英文：" |
| **One-shot** | 给出 1 个示例 | 示例 + 任务输入 |
| **Few-shot** | 给出 $k$ 个示例（通常 $k$=10~64） | $k$ 个示例 + 任务输入 |

#### 4.2 核心发现

GPT-3 不需要梯度更新，仅通过**精心设计的上下文示例**就能显著提升各类 NLP 任务的表现。这种"通过示例来学习"的能力随模型规模扩大而涌现。

#### 4.3 工作原理（假说）

注意力机制使得模型能够在上下文中隐式地进行**模式匹配**和**类比推理**：
- 注意力头可能扮演类似于"梯度下降"的角色
- 上下文中的示例被编码为 Key-Value 对
- 测试输入作为 Query 进行检索式推理

### 5. RLHF（Reinforcement Learning from Human Feedback）

GPT-3.5 / GPT-4 的核心对齐技术，使模型输出更符合人类偏好。

#### 5.1 三步流程

**Step 1: Supervised Fine-Tuning (SFT)**

收集人类标注的高质量（prompt, response）对，对预训练模型进行监督微调：

$$
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y)\sim D_{\text{human}}}[\log P_\theta(y|x)]
$$

**Step 2: 训练奖励模型（Reward Model, RM）**

对同一 prompt 生成多个回答，让人类标注偏好排序。训练一个标量奖励模型 $r_\phi(x, y)$：

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)\sim D}[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]
$$

其中 $y_w$ 为偏好更高的回答，$y_l$ 为偏好更低的回答。这是一个 Bradley-Terry 成对比较模型。

**Step 3: PPO 优化**

使用 PPO（Proximal Policy Optimization）对 SFT 模型进行强化学习微调：

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_{x\sim D, y\sim \pi_\theta(y|x)}[r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(y|x) \| \pi_{\text{SFT}}(y|x))]
$$

其中：
- $\pi_\theta$ 为正在优化的策略（模型）
- $\beta$ 为 KL 散度惩罚系数，防止策略偏离 SFT 模型太远（避免"奖励黑客"）
- KL 散度项相当于行为克隆的正则化

### 6. DPO（Direct Preference Optimization）

#### 6.1 动机

RLHF 需要：
- 单独训练一个奖励模型
- PPO 训练不稳定，超参数敏感
- 需要在训练中持续采样并评估奖励

DPO 的目标：直接从偏好数据中优化策略，**绕过显式的奖励模型训练和强化学习**。

#### 6.2 核心公式

DPO 损失函数：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)\sim D}\left[\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} -
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right)\right]
$$

其中：
- $\pi_\theta$：正在优化的策略模型
- $\pi_{\text{ref}}$：参考模型（通常是 SFT 模型，参数冻结）
- $\beta$：控制偏离参考模型的程度
- $y_w$：偏好回答（chosen）
- $y_l$：非偏好回答（rejected）

#### 6.3 直观理解

DPO 的核心思想：
- 增加偏好回答 $y_w$ 相对于参考模型的概率
- 减少非偏好回答 $y_l$ 相对于参考模型的概率
- 当 $\beta$ 大时，倾向于更强的偏好对齐；当 $\beta$ 小时，更接近参考模型

#### 6.4 理论推导

由 Bradley-Terry 模型出发，最优奖励函数与策略之间满足：

$$
r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

将最优奖励代入偏好排序损失，消去配分函数 $Z(x)$，得到上述 DPO 损失。

#### 6.5 DPO vs RLHF 对比

| 维度 | RLHF (PPO) | DPO |
|------|------------|-----|
| 需要奖励模型 | 是 | 否 |
| 训练复杂度 | 高（4 个模型同时加载） | 低（2 个模型） |
| 训练稳定性 | 不稳定，需大量调参 | 稳定，标准分类式训练 |
| 在线/离线 | 通常在线（需采样） | 完全离线 |
| 性能 | 强基线 | 与 RLHF 相当或更好 |
| 代表模型 | InstructGPT, GPT-4 | LLaMA-2-Chat, Zephyr |

---

## 第三部分：BERT vs GPT 详细对比

### 7. 架构对比表

| 维度 | BERT | GPT |
|------|------|-----|
| **架构类型** | Encoder-Only | Decoder-Only |
| **基础单元** | Transformer Encoder (×12/24) | Transformer Decoder (×12/48/96) |
| **注意力方式** | Bidirectional（双向） | Causal / Masked（单向因果） |
| **预训练任务** | MLM + NSP | Next Token Prediction（自回归） |
| **预训练目标** | $- \log P(w_i|\text{context})$ | $- \sum_{t} \log P(x_t|x_{<t})$ |
| **输出形式** | 上下文表示向量 | 下一个 token 的概率分布 |
| **生成能力** | 不支持（需外部 Decoder） | 原生支持自回归生成 |
| **理解能力** | 强（双向上下文） | 较强（仅单向，但可通过上下文弥补） |
| **[CLS] token** | 有（聚合序列表示） | 无 |
| **位置编码** | 可学习 | 可学习（GPT-1/2/3）；RoPE（GPT-4，推测） |
| **典型参数量** | 110M ~ 340M | 117M ~ 1.76T (GPT-4，传闻 MoE) |
| **最大上下文** | 512 tokens | 2048（GPT-3）→ 128K（GPT-4 Turbo） |
| **适用任务** | 分类、NER、QA、语义匹配 | 生成、对话、代码、通用推理 |
| **代表模型** | BERT, RoBERTa, DeBERTa | GPT-1/2/3/4, LLaMA, Mistral |

### 8. 三类架构对比

#### 8.1 Encoder-Only（BERT 系）

| 代表 | 特点 | 优势 | 劣势 |
|------|------|------|------|
| BERT | 双向 Self-Attention | 深度上下文理解 | 无法生成 |
| RoBERTa | 去除 NSP，更大数据 | SOTA 理解性能 | 同上 |
| DeBERTa | 解耦注意力 + 增强 mask 解码器 | 最强理解能力 | 同上 |

**适用场景**：文本分类、情感分析、命名实体识别、关系抽取、语义搜索、文本匹配。

#### 8.2 Decoder-Only（GPT 系）

| 代表 | 特点 | 优势 | 劣势 |
|------|------|------|------|
| GPT-3/4 | 大规模自回归 | 通用生成、Few-shot | 单向建模 |
| LLaMA | 开源，高效架构 | 社区生态丰富 | 需要指令微调 |
| Mistral | 滑动窗口注意力 + MoE | 推理高效 | 新兴生态 |

**适用场景**：对话系统、文本生成、代码生成、机器翻译、创意写作、通用 AI 助手。

#### 8.3 Encoder-Decoder（T5 系）

| 代表 | 特点 | 优势 | 劣势 |
|------|------|------|------|
| T5 | Text-to-Text 统一框架 | 任务统一，可迁移 | 参数量大 |
| BART | 去噪自编码器 | 生成+理解兼顾 | 不如专项模型 |
| mT5 | 多语言 T5 | 跨语言迁移 | 训练昂贵 |

**适用场景**：机器翻译、文本摘要、问答生成、数据到文本。

#### 8.4 统一对比表

| 维度 | Encoder-Only | Decoder-Only | Encoder-Decoder |
|------|--------------|--------------|-----------------|
| **信息流** | 双向全可见 | 单向因果 | 编码器双向 + 解码器单向 |
| **预训练** | MLM 降噪 | 自回归 | Span Corruption / 去噪 |
| **生成** | ✗ | ✓ | ✓ |
| **理解** | ★★★ | ★★☆ | ★★☆ |
| **灵活性** | 低（需任务头） | 高（Prompt 驱动） | 中 |
| **零样本泛化** | 弱 | 强（涌现能力） | 中 |
| **当前趋势** | 衰退中 | **主导** | 特定领域 |

### 9. 思维链（Chain-of-Thought, CoT）

#### 9.1 定义

思维链是一种推理技术：在 few-shot 示例中不仅给出答案，还给出**中间的推理步骤**，引导模型逐步推理。

#### 9.2 效果

在数学推理（GSM8K）、常识推理（StrategyQA）、符号推理等任务上显著提升准确率。GPT-3 175B + CoT 在 GSM8K 上从 ~19% 提升至 ~58%。

#### 9.3 变体

- **Zero-shot CoT**：只需加上 "Let's think step by step" 即可触发
- **Self-Consistency**：多次采样 + 多数投票
- **Tree of Thoughts (ToT)**：搜索推理路径树，支持回溯
- **Graph of Thoughts (GoT)**：图结构的推理路径组合

#### 9.4 原理

CoT 将复杂问题分解为中间子问题，每个子问题适配了模型分布内的推理模式，累积增量正确性。

### 10. Scaling Laws

#### 10.1 Kaplan 等人 (2020)

针对自回归语言模型的损失，发现幂律关系：

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

其中：
- $N$：模型参数量
- $D$：训练数据量（tokens）
- $N_c, D_c$：临界常数
- $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$

**关键结论**：
- 模型性能随参数量、数据量、计算量的增加呈幂律提升（非饱和）
- 大模型比小模型更具样本效率（用更少数据达到相同性能）
- 最优模型大小和训练 token 数应同步扩大

#### 10.2 Chinchilla 定律 (Hoffmann et al., 2022)

DeepMind 的 Chinchilla 研究发现，Kaplan 定律**低估了数据的重要性**：

对于计算预算 $C$，**模型参数 $N$ 和训练 token 数 $D$ 应等比缩放**：

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

具体推荐：**每个参数约 20 个训练 token**。

例如：70B 模型约需 1.4T tokens 训练数据。

#### 10.3 涌现能力（Emergent Abilities）

某些能力在模型规模超过特定阈值后突然涌现（而非平滑增长）：

| 能力 | 涌现阈值 |
|------|----------|
| Few-shot 学习 | ~1B |
| CoT 推理 | ~100B |
| 算术推理 | ~10B |
| 指令遵循 | ~10B |
| 代码生成 | ~10B |
| 翻译 | ~100B |

涌现能力的机制尚未被完全理解，但对 AGI 研究有重要启示。

#### 10.4 最新进展

- **Llama 3 的发现**：用小模型 + 更多数据（15T tokens for 8B）也能达到强性能，挑战了"必须极大参数"的假设
- **数据质量 > 数据数量**：精心筛选的数据集（如 FineWeb）显著优于随机爬取
- **合成数据**：使用大模型生成训练数据，形成"数据飞轮"

---

## 第四部分：关键技术与概念补充

### 11. 参数量公式对比

**BERT/Encoder-Only**：

$$
P_{\text{BERT}} \approx 12N d_{\text{model}}^2 + V d_{\text{model}}
$$

（因为每层只有 Self-Attention + FFN，无 Cross-Attention）

**GPT/Decoder-Only**（近似）：

$$
P_{\text{GPT}} \approx 12N d_{\text{model}}^2 + V d_{\text{model}}
$$

（Decoder 层结构与 Encoder 参数基本相同，Weight Tying 节省了输出 Embedding 参数）

### 12. 训练效率对比

| 维度 | BERT 训练 | GPT 训练 |
|------|-----------|----------|
| 并行度 | 完全并行（双向注意力） | 完全并行（Causal Mask + Teacher Forcing） |
| 有效利用率 | 仅 15% token 有监督信号（MLM） | 100% token 有监督信号 |
| 预训练效率 | 相对较低 | 高（每个 token 都被学习） |
| 推理 | 一次前向（单输入） | 逐 token 生成，需 KV Cache |

### 13. 激活函数演进

| 模型 | 激活函数 |
|------|----------|
| Transformer (2017) | ReLU |
| BERT (2018) | GELU |
| GPT-2 / GPT-3 | GELU |
| LLaMA (2023) | SiLU (Swish) |
| LLaMA-2/3 | SiLU + SwiGLU (FFN) |
| PaLM | SwiGLU |

### 14. 归一化位置演进

| 模型 | 归一化方案 |
|------|-----------|
| Transformer (2017) | Post-LN |
| GPT-2 (2019) | Pre-LN |
| ViT (2020) | Pre-LN |
| LLaMA (2023) | Pre-LN + RMSNorm |
| Gemma (2024) | Pre-LN + RMSNorm |

---

## 第五部分：实际应用与选择指南

### 15. 什么时候选 BERT 系架构

- ✅ 需要深度理解文本语义（非生成）
- ✅ 需要对句子/段落级别分类
- ✅ 需要提取实体、关系
- ✅ 延迟敏感（单次前向即可）
- ✅ 计算资源有限（BERT-Base 仅 110M）

### 16. 什么时候选 GPT 系架构

- ✅ 需要文本生成能力
- ✅ 需要对话交互
- ✅ 需要代码生成
- ✅ 任务类型多变（通过 Prompt 统一）
- ✅ 需要 In-context Learning 能力
- ✅ 对生成质量要求高

### 17. 什么时候选 Encoder-Decoder 架构

- ✅ 序列转换任务（翻译、摘要、语音识别后处理）
- ✅ 输入和输出长度差异大
- ✅ 需要显式的对齐信息
- ✅ 对解码速度有较高要求（可用非自回归解码）

---

## 参考文献

1. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019
2. Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, arXiv 2019
3. Radford et al., *Improving Language Understanding by Generative Pre-Training*, 2018
4. Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019 (GPT-2)
5. Brown et al., *Language Models are Few-Shot Learners*, NeurIPS 2020 (GPT-3)
6. OpenAI, *GPT-4 Technical Report*, arXiv 2023
7. Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*, NeurIPS 2022 (InstructGPT)
8. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023
9. Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS 2022
10. Kaplan et al., *Scaling Laws for Neural Language Models*, arXiv 2020
11. Hoffmann et al., *Training Compute-Optimal Large Language Models*, NeurIPS 2022 (Chinchilla)
12. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
13. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2021
14. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention*, NeurIPS 2022
15. Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, arXiv 2023
16. Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2023
17. Jiang et al., *Mistral 7B*, arXiv 2023
18. Shazeer et al., *GLU Variants Improve Transformer*, arXiv 2020
19. Lan et al., *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*, ICLR 2020
20. Clark et al., *ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators*, ICLR 2020
