# GPT 系列详解

> **核心家族**: GPT-1 → GPT-2 → GPT-3 → GPT-3.5/ChatGPT → GPT-4 → GPT-4o
> **所属机构**: OpenAI
> **核心理念**: 自回归语言模型 + Decoder-only Transformer + Scaling Up

---

## 1. GPT 核心理念

### 1.1 自回归语言模型 (Autoregressive Language Model)

GPT 系列的核心思想是**自回归生成**：给定前文 $x_{<t} = (x_1, x_2, \dots, x_{t-1})$，模型预测下一个 token $x_t$ 的概率分布：

$$P(x) = \prod_{t=1}^{T} P(x_t \mid x_{<t}; \theta)$$

每次生成时，将当前生成的 token 拼接到已有序列末尾，作为下一步的输入，循环往复直到生成终止符 `<EOS>` 或达到最大长度。

### 1.2 Decoder-only Transformer

与 BERT 使用 Encoder-only、T5 使用 Encoder-Decoder 不同，GPT 采用纯粹的 **Decoder-only** 架构：

- **移除 Encoder-Decoder Cross-Attention**：仅有 Self-Attention 层
- **因果掩码 (Causal Mask)**：使用上三角掩码矩阵 $M$，保证位置 $t$ 只能注意到 $1 \sim t$（当前及之前），屏蔽未来信息
- **自回归生成能力天然内建**：训练即推理，不需要额外的适配结构

```
输入: [BOS] 今天    天气    真    好
        ↓      ↓      ↓     ↓     ↓
        ├──────┼──────┼─────┼─────┤
Mask:   可见   可见   可见   可见   可见  (当前token)
        ✗     可见   可见   可见   可见
        ✗     ✗     可见   可见   可见
        ✗     ✗     ✗     可见   可见
        ✗     ✗     ✗     ✗     可见
```

### 1.3 与 Encoder-Decoder 的对比

| 特性 | Decoder-only (GPT) | Encoder-Decoder (T5) | Encoder-only (BERT) |
|------|-------------------|---------------------|---------------------|
| 注意力类型 | 单向因果 | 双向+交叉 | 双向 |
| 文本生成 | 原生支持 | 需要解码器 | 不支持 |
| 上下文理解 | 单向受限 | 双向理解更好 | 双向理解最好 |
| 代表模型 | GPT-3/4, LLaMA | T5, BART | BERT, RoBERTa |
| 适用任务 | 生成、对话、补全 | 翻译、摘要 | 分类、NER、QA |

---

## 2. GPT-1（2018）：奠基之作

**论文**: *Improving Language Understanding by Generative Pre-Training*

### 2.1 模型架构

- **12 层 Transformer Decoder**（仅 decoder 部分，无 encoder）
- 隐藏维度 $d_{model} = 768$，12 个注意力头
- 参数量：约 **1.17 亿**
- 激活函数：GELU
- 词汇表大小：40,000 (BPE)

### 2.2 两阶段训练范式

**阶段一：无监督预训练 (Unsupervised Pre-training)**

在大规模无标注文本语料（BooksCorpus，约 7,000 本书）上进行语言建模：

$$L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)$$

其中 $k$ 为上下文窗口大小，$\Theta$ 为模型参数。

**阶段二：有监督微调 (Supervised Fine-tuning)**

在特定任务标注数据上微调，针对分类任务：

$$L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y \mid x^1, \dots, x^m)$$

并在微调时加入辅助语言建模目标以提升泛化性：

$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C})$$

### 2.3 任务适配

GPT-1 通过**文本拼接**将不同 NLP 任务统一为文本生成格式：
- 文本蕴含：`[文本A] [SEP] [文本B] [SEP]` → 分类
- 相似度：双向拼接 `[A][SEP][B]` 和 `[B][SEP][A]`
- 问答：`[文档][SEP][问题][SEP][答案]`

### 2.4 主要贡献

1. 证明了**大规模无监督预训练 → 有监督微调**范式的有效性
2. 统一了多种 NLP 任务的输入格式
3. 在 12 个 NLP 基准中的 9 个取得 SOTA

---

## 3. GPT-2（2019）：Zero-Shot 的曙光

**论文**: *Language Models are Unsupervised Multitask Learners*

### 3.1 核心理念变革

GPT-2 提出一个大胆假设：**语言模型本身就是无监督多任务学习器**。一个足够强大的语言模型，在足够大的数据集上训练后，无需任何微调即可完成多种下游任务。

$$p(\text{output} \mid \text{input}, \text{task})$$

其中 task 信息通过自然语言提示隐含地传入。

### 3.2 模型规格

- **15 亿参数** (GPT-2 XL)，48 层，$d_{model} = 1600$
- 训练数据：WebText（约 40GB，800 万网页）
- 强调**任务无关性 (Task-Agnostic)**：不针对任何下游任务做特殊设计
- 词表大小扩展至 **50,257** (BPE)

### 3.3 自回归训练目标

自回归语言模型的训练目标为最大化对数似然：

$$\mathcal{L}(\theta) = \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

参数 $\theta$ 通过梯度下降最小化负对数似然：

$$\theta^* = \arg\min_{\theta} -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

### 3.4 Zero-Shot 能力

GPT-2 在未见过任何微调样本的情况下，在多项任务上表现出竞争力：
- 阅读理解 (CoQA)：55 F1（无微调）
- 翻译 (WMT-14 En-Fr)：5 BLEU（虽低但非零，证明多语言能力已涌现）
- 摘要 (CNN/DailyMail)：生成式评价可行

### 3.5 局限性

- Zero-shot 性能仍显著弱于有监督 SOTA
- 生成长文本时连贯性不足
- 重复、矛盾问题频繁出现

---

## 4. GPT-3（2020）：规模涌现

**论文**: *Language Models are Few-Shot Learners*

### 4.1 模型规模

| 模型 | 层数 | $d_{model}$ | 头数 | 参数量 |
|------|------|-------------|------|--------|
| GPT-3 Small | 12 | 768 | 12 | 1.25 亿 |
| GPT-3 Medium | 24 | 1024 | 16 | 3.5 亿 |
| GPT-3 Large | 24 | 1536 | 16 | 7.6 亿 |
| GPT-3 XL | 24 | 2048 | 24 | 13 亿 |
| GPT-3 6.7B | 32 | 4096 | 32 | 67 亿 |
| GPT-3 13B | 40 | 5140 | 40 | 130 亿 |
| GPT-3 175B | 96 | 12288 | 96 | **1750 亿** |

### 4.2 Sparse Attention（稀疏注意力）

为处理长序列（2048 tokens），GPT-3 采用了 **Sparse Transformer** 中的稀疏注意力模式：

- **Strided Attention**：每个位置关注每隔 $\ell$ 步的前一个位置
- **Fixed Attention**：每个位置关注前 $k$ 个位置
- 将自注意力的 $O(n^2)$ 复杂度降低为 $O(n\sqrt{n})$

在 $n$ 层 Transformer 中，交替使用稠密和稀疏注意力层（每两层一个稠密），平衡效率与效果。

### 4.3 In-Context Learning（上下文学习）

GPT-3 最大的创新是发现了**上下文学习 (In-Context Learning)** 现象：模型无需梯度更新，仅通过上下文中的示例就能适应新任务。

**三种推断设定**：

#### Zero-shot
仅提供任务描述，无示例：
```
将英文翻译为法语：
cheese →
```

#### One-shot
提供一个示例：
```
将英文翻译为法语：
sea otter → loutre de mer
cheese →
```

#### Few-shot
提供多个示例（通常 10-100 个）：
```
将英文翻译为法语：
sea otter → loutre de mer
plush girafe → girafe peluche
cheese →
```

**关键发现**：模型规模越大，Few-shot 性能提升越显著，呈现**幂律增长**趋势。

### 4.4 Emergent Abilities（涌现能力）

> **涌现**：在较小模型中不存在或不显著，但在模型规模跨越某个临界点后突然出现的能力。

GPT-3 展现的涌现能力包括：

1. **少样本学习 (Few-shot Learning)**：通过上下文示例零梯度学习
2. **思维链推理 (Chain-of-Thought)**：多步推理能力在 ~100B 规模涌现
3. **代码生成**：从自然语言描述生成可执行代码
4. **算术推理**：多位数加减乘除
5. **翻译**：无需平行语料的跨语言翻译
6. **世界知识问答**：从训练语料中习得的事实性知识

---

## 5. GPT-3.5 / ChatGPT / InstructGPT

ChatGPT 背后的核心技术来自于 **InstructGPT** 论文。

### 5.1 训练三阶段

ChatGPT 的训练分为三个核心阶段：

```
预训练(Pre-training) → 监督微调(SFT) → 奖励模型(RM) → RLHF(PPO)
```

#### 阶段一：预训练 (Pre-training)

在海量互联网文本上进行自回归语言模型训练（同 GPT-3），目标为：

$$\mathcal{L}_{PT} = -\sum_{t} \log P(x_t \mid x_{<t})$$

#### 阶段二：监督微调 (Supervised Fine-Tuning, SFT)

使用人工标注的**指令-回复对 (Instruction-Response Pairs)** 进行微调：

$$\mathcal{L}_{SFT} = -\sum_{(x,y) \in \mathcal{D}_{SFT}} \log P_\theta(y \mid x)$$

标注数据包含多种任务：问答、摘要、翻译、代码、创意写作等。

#### 阶段三：奖励模型 (Reward Model, RM)

训练一个奖励模型来评估回复质量。对同一个 prompt $x$，收集多个回复 $\{y_1, y_2, \dots, y_K\}$，由人工标注偏好排序。

使用 **Bradley-Terry 模型** 建模偏好概率：

$$P(y_w \succ y_l \mid x) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))}$$

其中 $y_w$ 为更受偏好的回复，$y_l$ 为较差的回复。

奖励模型损失函数：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

其中 $\sigma$ 为 sigmoid 函数。直观理解：最大化 $r(y_w)$ 与 $r(y_l)$ 之间的差距。

### 5.2 RLHF（Reinforcement Learning from Human Feedback）

使用 **PPO (Proximal Policy Optimization)** 算法优化策略模型 $\pi_\theta$：

$$\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x, y) - \beta \cdot \text{KL}\big(\pi_\theta(\cdot \mid x) \;\|\; \pi_{\text{ref}}(\cdot \mid x)\big) \right]$$

其中：
- $r_\phi(x, y)$：奖励模型给出的分数
- $\beta$：KL 惩罚系数（控制与参考策略的偏离程度）
- $\pi_{\text{ref}}$：SFT 模型（参考策略，防止奖励黑客 Reward Hacking）
- $\text{KL}(\pi_\theta \|\| \pi_{\text{ref}})$：KL 散度，约束新策略不偏离 SFT 模型太远

**PPO 的 Clip 机制**：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 为概率比，$\hat{A}_t$ 为优势函数估计。

### 5.3 DPO（Direct Preference Optimization）

DPO 是 RLHF 的简化替代方案，**无需训练单独的奖励模型**，直接从偏好数据优化策略。

**核心推导**：从 Bradley-Terry 模型的最优奖励函数出发，推导出策略的目标函数。

最优奖励函数可表示为：

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

代入偏好损失，消去配分函数 $Z(x)$，得到 DPO 损失：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

**DPO 的优势**：
- 无需训练和部署独立的奖励模型
- 训练更稳定，超参更少
- 直接从偏好数据中学习，信息损失更小
- 相当于在策略空间直接做偏好排序

**DPO vs RLHF 对比**：

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 模型数量 | 4 个（SFT+RM+Policy+Ref） | 2 个（Policy+Ref） |
| 训练阶段 | 3 阶段 | 2 阶段 |
| 训练稳定性 | 需调参，易崩溃 | 更稳定 |
| 采样需求 | 在线采样 | 离线数据即可 |
| 理论保证 | 更灵活 | 等价于 BT 模型下的最优解 |

---

## 6. GPT-4（2023）：多模态突破

### 6.1 核心升级

- **多模态输入**：支持图像和文本联合输入（Vision + Language）
- **更长的上下文窗口**：初始 8K/32K，后续扩展至 128K
- **更强的推理能力**：在律师资格考试、SAT、GRE、AP 考试中取得人类顶尖水平
- **更好的指令遵循**和**事实准确性**
- **多语言能力**大幅提升（低资源语言表现改善显著）

### 6.2 推测性架构

GPT-4 的具体架构未公开，但业界推测：
- 采用 **Mixture of Experts (MoE)** 架构（传闻 8 个专家，每 token 激活 2 个）
- 总参数量估计 1.8 万亿
- 视觉编码器 + 语言模型的联合架构
- 大规模的多模态预训练数据（图文对、交错图文文档）

### 6.3 GPT-4o（Omni）

- **原生多模态**：文本、图像、音频统一建模
- **端到端训练**：不再级联独立模型（ASR → LLM → TTS），而是单一模型处理所有模态
- **极低延迟**：音频对话延迟 ~232ms（平均），接近真人对话速度
- **实时交互**：支持视频流理解、实时翻译、情感感知

---

## 7. Scaling Laws（规模法则）

**论文**: *Scaling Laws for Neural Language Models* (Kaplan et al., 2020)

### 7.1 核心公式

测试损失 $L$ 与模型参数量 $N$、数据量 $D$、计算量 $C$ 呈幂律关系：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$$

其中 $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$。

### 7.2 Chinchilla 最优

DeepMind 的 Chinchilla 论文 (2022) 修正了 Kaplan 的结论：

> **最优训练方案**：模型参数量与训练 token 数应成**等比例**增长。
> 即：对于每个参数，应有约 20 个训练 token。

Chinchilla 最优公式：

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

这意味着过去的大模型普遍**过度参数化而数据不足**（如 GPT-3 175B 仅训练了约 300B tokens，远低于 Chinchilla 最优的 ~3.5T）。

### 7.3 实践指导

- **给定计算预算**，应优先增加数据量而非仅堆参数
- LLaMA 系列采用了"小模型+大数据"策略（LLaMA-13B 训练 1T tokens，性能超过 GPT-3 175B）
- 数据质量比数据量更重要：经过精心过滤的高质量数据可以显著改善 Scaling Law 的常数因子

---

## 8. 关键贡献总结

| 模型 | 年份 | 关键创新 | 参数量 |
|------|------|---------|--------|
| GPT-1 | 2018 | 预训练+微调范式 | 1.17 亿 |
| GPT-2 | 2019 | Zero-shot 迁移，任务无关 | 15 亿 |
| GPT-3 | 2020 | In-Context Learning，涌现 | 1750 亿 |
| InstructGPT | 2022 | RLHF 对齐 | 1750 亿 |
| ChatGPT | 2022 | 对话式交互，广泛应用 | 未知 |
| GPT-4 | 2023 | 多模态，MoE 架构 | ~1.8T (传闻) |
| GPT-4o | 2024 | 原生多模态 Omni | 未知 |

---

## 9. 参考文献

1. Radford et al. "Improving Language Understanding by Generative Pre-Training" (2018)
2. Radford et al. "Language Models are Unsupervised Multitask Learners" (2019)
3. Brown et al. "Language Models are Few-Shot Learners" (2020)
4. Ouyang et al. "Training Language Models to Follow Instructions with Human Feedback" (2022)
5. Rafailov et al. "Direct Preference Optimization" (2023)
6. Kaplan et al. "Scaling Laws for Neural Language Models" (2020)
7. Hoffmann et al. "Training Compute-Optimal Large Language Models" (2022)
8. OpenAI. "GPT-4 Technical Report" (2023)
9. Wei et al. "Emergent Abilities of Large Language Models" (2022)
