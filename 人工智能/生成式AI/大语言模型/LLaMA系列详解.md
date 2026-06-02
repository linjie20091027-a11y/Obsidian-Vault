# LLaMA 系列详解

> **核心家族**: LLaMA 1 → LLaMA 2 → LLaMA 3 → LLaMA 3.1
> **所属机构**: Meta AI (FAIR)
> **定位**: 开源高效大语言模型，对标闭源 GPT 系列

---

## 1. LLaMA 系列定位

### 1.1 设计哲学

Meta 在 LLaMA 系列中的核心设计理念：

1. **开源开放**：完整公开模型权重、技术报告，推动社区发展
2. **高效训练**：在更少参数下获得更强性能，挑战"更大即更好"的理念
3. **数据驱动**：重视训练数据的质量和多样性，而非仅堆参数
4. **实用导向**：提供多种规模（7B / 13B / 33B / 65B / 70B / 405B），适配不同部署场景

### 1.2 与 GPT 系列的主要差异

| 维度 | GPT 系列 | LLaMA 系列 |
|------|---------|-----------|
| 开放性 | 闭源（API + ChatGPT） | 开源权重 |
| 架构 | Decoder-only（大致相似） | Decoder-only + 多项改进 |
| 归一化 | LayerNorm（后归一化） | RMSNorm（前归一化） |
| 激活函数 | GELU | SwiGLU |
| 位置编码 | 可学习绝对位置编码 | RoPE（旋转位置编码） |
| 训练数据 | 互联网文本为主 | 精选高质量多语言数据 |
| 上下文窗口 | 逐步扩展 | 原生大窗口（最大 128K+） |

---

## 2. LLaMA 1（2023）：架构奠基

**论文**: *LLaMA: Open and Efficient Foundation Language Models*

### 2.1 Pre-Normalization（前置归一化）

与标准 Transformer 的 Post-LN（残差后归一化）不同，LLaMA 采用 **Pre-Normalization**：

```
标准 Post-Norm:  x → LayerNorm(x + Sublayer(x))
LLaMA Pre-Norm:  x → x + Sublayer(RMSNorm(x))
```

**优势**：
- 训练更稳定，梯度流更顺畅
- 无需 Warm-up 学习率调度（或减少 Warm-up 步数）
- 缓解深层网络中的梯度消失/爆炸问题

### 2.2 RMSNorm（均方根归一化）

LLaMA 使用 **RMSNorm** 替代标准 LayerNorm，移除了均值中心化步骤：

**LayerNorm**：
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

其中 $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$，$\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2}$

**RMSNorm**：
$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}$$

**优势**：
- 减少一次均值计算（约 7-15% 加速）
- 实验表明效果与 LayerNorm 相当甚至更好
- 计算更简洁，推理更快

### 2.3 SwiGLU 激活函数

LLaMA 用 **SwiGLU** 替换标准 FFN 中的 ReLU/GELU：

**标准 FFN**：
$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

**SwiGLU FFN**：
$$\text{SwiGLU}(x) = (xW_1 \odot \text{Swish}(xW_2)) \cdot W_3$$

其中：
- $\text{Swish}(z) = z \cdot \sigma(z)$（其中 $\sigma$ 为 sigmoid）
- $\odot$ 为逐元素乘法（Hadamard 积）
- SwiGLU 有三组权重矩阵 $(W_1, W_2, W_3)$，而标准 FFN 只有两组

**完整展开**：
$$\text{SwiGLU}(x) = (x \cdot W_1) \odot (x \cdot W_2 \cdot \sigma(x \cdot W_2)) \cdot W_3$$

**优势**：SwiGLU 在多项 NLP 基准上优于 ReLU/GELU，特别是在大规模模型中。

### 2.4 Rotary Position Embedding (RoPE)

**论文**: *RoFormer: Enhanced Transformer with Rotary Position Embedding*

RoPE 是 LLaMA 使用的旋转位置编码，通过**旋转矩阵**将位置信息编码到注意力计算中。

#### 原理

对于位置 $m$ 的 Query $q_m$ 和位置 $n$ 的 Key $k_n$，RoPE 施加与位置相关的旋转变换：

$$\tilde{q}_m = q_m \cdot e^{im\theta}$$
$$\tilde{k}_n = k_n \cdot e^{in\theta}$$

其中 $\theta_j = 10000^{-2j/d}, \; j = 0, 1, \dots, d/2 - 1$，旋转在二维子空间中进行。

#### 相对位置编码性质

关键洞察：RoPE 后注意力分数的内积仅依赖**相对位置** $n-m$：

$$\tilde{q}_m^T \tilde{k}_n = q_m^T R_{n-m} k_n$$

其中 $R_{n-m}$ 是一个仅依赖于相对位置 $n-m$ 的旋转矩阵。

**优势**：
- 自然编码相对位置信息
- 具备一定的**长度外推能力**（可处理比训练时更长的序列）
- 随序列长度衰减的注意力权重，符合语言直觉

#### 具体实现

对于每个二维子空间 $(x_{2j}, x_{2j+1})$，施加旋转：

$$\begin{pmatrix} x_{2j}' \\\\ x_{2j+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\\\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} \begin{pmatrix} x_{2j} \\\\ x_{2j+1} \end{pmatrix}$$

### 2.5 LLaMA 1 模型规格

| 模型 | 层数 | 隐藏维度 | 头数 | 参数量 | 训练 Token |
|------|------|---------|------|--------|-----------|
| LLaMA-7B | 32 | 4096 | 32 | 6.7B | 1.0T |
| LLaMA-13B | 40 | 5120 | 40 | 13.0B | 1.0T |
| LLaMA-33B | 60 | 6656 | 52 | 32.5B | 1.4T |
| LLaMA-65B | 80 | 8192 | 64 | 65.2B | 1.4T |

**关键发现**：LLaMA-13B 在大多数基准上超过 GPT-3 175B，证明了**数据质量 > 模型规模**。

---

## 3. LLaMA 2（2023）：对话对齐

**论文**: *Llama 2: Open Foundation and Fine-Tuned Chat Models*

### 3.1 核心升级

- **2 万亿 token 训练**（LLaMA 1 的 1.4 倍）
- **上下文窗口扩展至 4096**
- **分组查询注意力 (GQA)** 在 70B 模型中使用
- 提供 **LLaMA 2-Chat** 版本，经过 RLHF 对齐

### 3.2 Ghost Attention (GAtt)

GAtt 是 LLaMA 2 提出的特殊注意力机制，用于增强**多轮对话的指令遵循**：

**问题**：在多轮对话中，模型容易遗忘初始的系统指令（如"始终保持礼貌"）。

**GAtt 方案**：在 SFT 训练时，对每轮对话的**所有用户消息前**拼接原始的 System Prompt，使模型在多轮对话中持续"看到"系统指令，从而在 RLHF 后习得持久化指令遵循的能力。

```
原始多轮：
[System] 保持简洁
[User] 解释量子力学
[Assistant] ...(长篇)
[User] 再简短点

GAtt 训练时改写：
[System] 保持简洁
[User] 保持简洁\n解释量子力学
[Assistant] ...(简短)
[User] 保持简洁\n再简短点
```

### 3.3 RLHF 对齐流程

LLaMA 2-Chat 的训练流程：

1. **预训练** → LLaMA 2 Base
2. **SFT**（约 27K 高质量指令数据）
3. **两个独立的 RM**（Helpfulness RM + Safety RM，分别训练约 1M 偏好数据）
4. **PPO + 拒绝采样**（多轮迭代，逐步优化）
5. **GAtt 增强**（改善多轮对话一致性）

### 3.4 安全对齐

LLaMA 2 特别强调**安全对齐**：
- Safety RM 独立训练，评估潜在有害回复
- 帮助性 (Helpfulness) 与安全性 (Safety) 的平衡
- 多轮对话中持续安全约束
- 上下文蒸馏 (Context Distillation)：从长上下文中提取安全相关知识

---

## 4. LLaMA 3（2024）：全面进化

### 4.1 LLaMA 3 / 3.1 核心特性

- **128K 上下文窗口**（原生支持，非后期扩展）
- **多语言能力大幅提升**：覆盖 30+ 语言
- **多模态架构**：早期融合视觉编码器
- **超大规模**：8B / 70B / 405B 三个版本
- **训练数据**：>15T tokens，7 倍于 LLaMA 2
- **Tokenizer 升级**：词汇量扩展至 128K（Tiktoken BPE）

### 4.2 LLaMA 3.1 405B

- **4050 亿参数**，开源最大稠密模型
- **MoE 探索**但 405B 采用稠密架构
- 性能逼近 GPT-4，部分基准超越
- 使用 **FP8 混合精度**训练

### 4.3 关键创新

1. **Data Curriculum**：分阶段的数据课程训练（通用 → 代码/数学 → 多语言 → 长上下文）
2. **Annealing**：退火阶段使用最高质量数据精细调优
3. **Post-training**：SFT + DPO + 拒绝采样混合策略

---

## 5. KV Cache 机制

### 5.1 问题背景

自回归生成过程中，每一步需要计算所有历史 token 的注意力。如果每一步都重新计算，复杂度为 $O(n^2)$（$n$ 为序列长度）。

### 5.2 KV Cache 原理

**核心思想**：将已计算的 Key ($K$) 和 Value ($V$) 缓存起来，生成新 token 时只需计算新 token 的 $Q, K, V$，然后与缓存的 $K, V$ 进行注意力计算。

$$K_{\text{cache}}^{(t+1)} = [K_{\text{cache}}^{(t)}; K^{(t+1)}]$$
$$V_{\text{cache}}^{(t+1)} = [V_{\text{cache}}^{(t)}; V^{(t+1)}]$$

注意力计算：
$$\text{Attention}(Q^{(t+1)}, K_{\text{cache}}, V_{\text{cache}})$$

**加速效果**：
- 第 $t$ 步的计算复杂度从 $O(t^2 \cdot d)$ 降低到 $O(t \cdot d)$
- 生成 1000 个 token 约**加速 10-20 倍**

### 5.3 内存分析

KV Cache 的内存占用：
$$\text{Memory}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L_{\text{seq}} \times \text{bytes\_per\_element}$$

对于 LLaMA-7B（FP16，2048 tokens）：
$$2 \times 32 \times 32 \times 128 \times 2048 \times 2 = 1.07 \text{ GB}$$

GQA 可以显著减少 KV Cache（见第7节）。

---

## 6. Flash Attention

**论文**: *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*

### 6.1 问题分析

标准注意力计算的瓶颈：
1. **HBM 带宽瓶颈**：注意力矩阵 $QK^T$ 大小为 $O(n^2)$，需反复读写 GPU HBM
2. **SRAM 利用不足**：GPU 片上 SRAM (~20MB) 远比 HBM (~40-80GB) 快，但容量小

### 6.2 Tiling 分块策略

Flash Attention 的核心技巧：**在 SRAM 中分块计算注意力，避免将完整注意力矩阵写入 HBM**。

**算法流程**：

1. 将 $Q, K, V$ 分成小块加载到 SRAM（各块大小受 SRAM 容量限制）
2. 在 SRAM 中计算局部注意力分数 $S_{ij} = Q_i K_j^T / \sqrt{d_k}$
3. 在线计算 Softmax（无需完整矩阵，使用数值稳定的 running max 技巧）
4. 将局部结果累积到输出 $O$

**在线 Softmax 的数值稳定技巧**：

标准 Softmax 需要两遍扫描（一次求 max，一次求 sum），Flash Attention 用 running statistics 合并为一遍：

$$m_i^{(new)} = \max(m_i, m_j)$$
$$\alpha_i = e^{m_i - m_i^{(new)}} \cdot \alpha_i, \quad \alpha_j = e^{m_j - m_i^{(new)}} \cdot \alpha_j$$
$$O_i^{(new)} = \alpha_i \cdot O_i + \alpha_j \cdot P_{ij} V_j$$

### 6.3 效果

- **内存**：$O(n)$ 替代 $O(n^2)$（不存储完整注意力矩阵）
- **速度**：训练加速 2-4 倍，推理加速 1.5-3 倍
- **精度**：精确注意力，无近似（与稀疏注意力不同）
- **支持长序列**：可在单 GPU 上训练 64K+ 序列长度的模型

### 6.4 Flash Attention 2

- 改进并行策略（序列长度维度并行）
- 减少非矩阵乘法操作
- 更好的 warp 调度
- 进一步提速 ~2x

---

## 7. Grouped Query Attention (GQA)

### 7.1 从 MHA 到 GQA

| 变体 | Query 头数 | KV 头数 | 共享方式 |
|------|-----------|---------|---------|
| MHA (Multi-Head) | $h$ | $h$ | 不共享 |
| MQA (Multi-Query) | $h$ | $1$ | 所有 Q 共享 KV |
| GQA (Grouped-Query) | $h$ | $g$ ($1 < g < h$) | $h/g$ 个 Q 共享一对 KV |

### 7.2 GQA 的动机

- **MHA**：效果最好，但 KV Cache 内存大（每层存 $2h$ 个矩阵）
- **MQA**：KV Cache 最小（每层仅 2 个矩阵），但质量有明显下降
- **GQA**：中间的折中方案，在质量与效率间平衡

### 7.3 内存对比

对于 $n_{\text{layers}}$ 层，每层 $h$ 个头，每头维度 $d_h$，序列长度 $L$：

$$\text{MHA KV Cache} = 2 \cdot n_{\text{layers}} \cdot h \cdot d_h \cdot L$$
$$\text{GQA KV Cache} = 2 \cdot n_{\text{layers}} \cdot g \cdot d_h \cdot L$$
$$\text{MQA KV Cache} = 2 \cdot n_{\text{layers}} \cdot 1 \cdot d_h \cdot L$$

当 $g = h/4$（如 LLaMA 2 70B：64 个 Q 头，8 组 KV 头），KV Cache 减少 **75%**。

### 7.4 LLaMA 系列采用情况

- LLaMA 1：全系列 MHA
- LLaMA 2：34B/70B 使用 GQA（$g = 8$），7B/13B 使用 MHA
- LLaMA 3：全系列使用 GQA

---

## 8. 模型量化

### 8.1 量化基础

将 FP16/FP32 模型参数压缩为低精度整数（INT8/INT4），减少显存占用和推理延迟。

**均匀量化**：
$$x_{\text{quant}} = \text{round}\left(\frac{x - \text{zero}}{\text{scale}}\right)$$
$$x_{\text{dequant}} = x_{\text{quant}} \cdot \text{scale} + \text{zero}$$

### 8.2 GPTQ

**GPTQ (Post-Training Quantization for GPT)**：

- 基于 **OBQ (Optimal Brain Quantization)** 的逐层量化
- 使用二阶 Hessian 信息补偿量化误差
- 对每一层的权重按列量化，每次量化一列后用剩余列的 Hessian 信息更新未量化权重
- 支持 INT4/INT3 量化，INT4 下精度损失极小

**核心思想**：量化误差在 Hessian 矩阵的引导下"重新分配"到未量化的权重上，从而最小化输出误差。

### 8.3 AWQ

**AWQ (Activation-aware Weight Quantization)**：

- 核心发现：并非所有权重同等重要 —— **显著权重**（对应大激活值的权重通道）对精度影响更大
- 策略：保留 1% 显著权重为 FP16，其余量化为 INT4
- 通过对显著权重通道做 **per-channel scaling** 来减少量化误差
- 无需反向传播，仅基于激活值统计

$$
s^* = \arg\min_s \| \mathcal{Q}(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} - W \|_F^2
$$

### 8.4 GGUF / llama.cpp

**GGUF (GPT-Generated Unified Format)**：

- llama.cpp 生态的量化格式
- 支持从 2-bit 到 8-bit 的多种量化级别
- **K-Quant** 策略：对重要层用更多 bit，次要层用更少 bit
- CPU 推理优化，无需 GPU
- 代表格式：Q4_K_M, Q5_K_M, Q8_0 等

### 8.5 QLoRA 中的 NF4

**4-bit NormalFloat (NF4)**：

- 假设权重服从正态分布，设计信息论最优的 4-bit 量化格式
- 量化区间根据正态分布的分位数划分，而非均匀划分
- 与 FP4 相比，NF4 更好地保持了模型精度

| 量化方法 | 精度 | 模型大小（7B） | 内存占用 | 质量损失 |
|---------|------|-------------|---------|---------|
| FP16 | 16-bit | ~13 GB | ~14 GB | 无 |
| INT8 | 8-bit | ~7 GB | ~8 GB | <0.5% |
| GPTQ INT4 | 4-bit | ~3.9 GB | ~5 GB | <1% |
| AWQ INT4 | 4-bit | ~3.9 GB | ~5 GB | <1% |
| GGUF Q4_K_M | 4-bit | ~4.1 GB | ~4.5 GB (CPU) | <1% |

---

## 9. SFT+RLHF vs DPO 对比

### 9.1 训练流程对比

**RLHF 路径** (LLaMA 2)：
```
Base Model → SFT → Reward Model → PPO + 拒绝采样 → Chat Model
                 ↘ Reference Model ↗ (用于 KL 惩罚)
```
- 需要 4 个模型在训练中同时存在（Policy, Reference, RM, Value/Critic）
- 显存需求大
- 训练不稳定，需大量调参

**DPO 路径** (LLaMA 3 混合使用)：
```
Base Model → SFT → DPO → Chat Model
                 ↘ Reference Model ↗
```
- 仅需 2 个模型（Policy + Reference）
- 训练稳定，收敛快
- 离线偏好数据即可

### 9.2 优劣对比

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 计算成本 | 高（+RM + 在线采样） | 低 |
| 显存占用 | ~4x 模型大小 | ~2x 模型大小 |
| 训练稳定性 | 需精细调参 | 较稳定 |
| 在线/离线 | 需在线生成 | 纯离线 |
| 最优性 | 渐进最优 | BT 模型下最优 |
| 灵活性 | 高（可加约束） | 受限 |
| 实践趋势 | 顶级闭源模型 | 开源/中规模模型 |

许多团队采用**混合策略**：先用 DPO 快速迭代，再用 PPO 做最终对齐。

---

## 10. 参考总结

LLaMA 系列通过以下关键设计，在开源生态中建立了统治地位：

1. **Pre-Norm + RMSNorm**：稳定训练
2. **SwiGLU + RoPE**：提升模型质量
3. **GQA + KV Cache**：降低推理成本
4. **高质量数据 + 充分训练**：以少胜多
5. **多种量化方案**：降低部署门槛
6. **持续开源**：推动社区繁荣

---

## 11. 参考文献

1. Touvron et al. "LLaMA: Open and Efficient Foundation Language Models" (2023)
2. Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023)
3. Meta AI. "The Llama 3 Herd of Models" (2024)
4. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
5. Shazeer. "GLU Variants Improve Transformer" (2020)
6. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
7. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer" (2023)
8. Frantar et al. "GPTQ: Accurate Post-Training Quantization" (2022)
9. Lin et al. "AWQ: Activation-aware Weight Quantization" (2023)
10. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
