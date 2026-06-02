# Transformer 核心原理

> 论文: *Attention Is All You Need* (Vaswani et al., NeurIPS 2017)

---

## 1. 核心思想

Transformer 彻底抛弃了 RNN/LSTM 的循环结构，**完全基于注意力机制（Attention Mechanism）** 来建模序列数据。其核心信念是：序列中**任意两个位置的依赖关系都可以通过一次注意力计算直接捕获**，无需像 RNN 那样逐步递推。

RNN 的根本缺陷：
- 第 $t$ 步必须等待第 $t-1$ 步完成，**无法并行**
- 长距离依赖受限于梯度消失/爆炸，信息随距离衰减

Transformer 的解决方案：
- 所有位置**并行计算**
- 任意两个位置直接交互，信息路径长度为 $O(1)$

---

## 2. 整体架构：Encoder-Decoder

```
输入序列 → Encoder (×N) → 编码表示 → Decoder (×N) → 输出序列
```

- **Encoder**：将输入序列 $(x_1, \dots, x_n)$ 映射为连续表示序列 $(z_1, \dots, z_n)$
- **Decoder**：根据 Encoder 输出 + 已生成的输出，自回归地生成 $(y_1, \dots, y_m)$

每个 Encoder Layer 包含两个子层：
1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network

每个 Decoder Layer 包含三个子层：
1. Masked Multi-Head Self-Attention
2. Cross-Attention（对 Encoder 输出做注意力）
3. Position-wise Feed-Forward Network

每个子层后都有残差连接 + Layer Normalization。

---

## 3. Self-Attention 详细推导

### 3.1 符号定义

设输入矩阵 $X \in \mathbb{R}^{n \times d_{\text{model}}}$，其中 $n$ 为序列长度，$d_{\text{model}}$ 为模型维度。

### 3.2 Q, K, V 的线性投影

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

其中：
- $W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$（Query 投影矩阵）
- $W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$（Key 投影矩阵）
- $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$（Value 投影矩阵）

通常 $d_k = d_v = d_{\text{model}} / h$，其中 $h$ 为头数。

### 3.3 注意力分数矩阵

$$
S = QK^T \in \mathbb{R}^{n \times n}
$$

$S_{ij}$ 表示第 $i$ 个 Query 与第 $j$ 个 Key 的原始相关性分数。

### 3.4 缩放

$$
S' = \frac{S}{\sqrt{d_k}}
$$

除以 $\sqrt{d_k}$ 的原因见第 4 节。

### 3.5 Softmax 归一化

$$
A = \text{softmax}(S') = \frac{\exp(S'_{ij})}{\sum_{j=1}^{n} \exp(S'_{ij})}
$$

$A_{ij}$ 表示位置 $i$ 对位置 $j$ 的注意力权重，满足 $\sum_{j} A_{ij} = 1$。

### 3.6 加权输出

$$
\text{Output} = AV \in \mathbb{R}^{n \times d_v}
$$

每个位置的输出是所有 Value 的加权和，权重由注意力分布决定。

### 3.7 完整公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

---

## 4. 缩放因子 $\sqrt{d_k}$ 的方差分析

### 4.1 问题来源

假设 $Q$ 和 $K$ 的每个分量独立同分布，均值为 $0$，方差为 $1$。

则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的：

- **均值**：$\mathbb{E}[q \cdot k] = \sum \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$
- **方差**：$\text{Var}(q \cdot k) = \sum \text{Var}(q_i k_i) = \sum \text{Var}(q_i) \text{Var}(k_i) = d_k$

因此点积的方差随 $d_k$ 线性增长，$S = QK^T$ 中的元素标准差达到 $\sqrt{d_k}$。

### 4.2 后果

当 $d_k$ 很大时，点积值的量级变大，经过 Softmax 后会导致**梯度极度稀疏**——Softmax 输出趋近于 one-hot 向量，大部分位置的梯度接近 0，模型难以训练。

### 4.3 解决方案

除以 $\sqrt{d_k}$ 使得缩放后点积的方差回到 $1$：

$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \cdot d_k = 1
$$

这保证了 Softmax 输入处于合理范围，梯度流动正常。

---

## 5. Multi-Head Attention

### 5.1 动机

单头注意力的表达能力有限。多头注意力让模型在**不同的表示子空间**中同时关注不同位置、不同类型的关系。

### 5.2 公式

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

其中每个头：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

投影矩阵维度：
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$

### 5.3 典型参数

原始论文：$h = 8$, $d_k = d_v = d_{\text{model}} / h = 64$, $d_{\text{model}} = 512$

### 5.4 计算复杂度

- Self-Attention：$O(n^2 \cdot d_{\text{model}})$（由 $QK^T$ 的 $n \times n$ 矩阵决定）
- 当 $n \gg d_{\text{model}}$ 时，注意力成为瓶颈，这也是后续线性注意力研究的动机。

---

## 6. Positional Encoding

### 6.1 为什么需要位置编码

注意力机制本身是**置换等变的（permutation equivariant）**——若打乱输入顺序，输出也会相应打乱，但不会改变各位置的表示内容。这意味着模型无法感知 token 之间的顺序。而序列顺序对语言理解至关重要。

### 6.2 正弦位置编码

原始论文使用固定的正弦/余弦函数：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

其中：
- $pos$：token 在序列中的位置
- $i$：维度的索引（$0 \le i < d_{\text{model}}/2$）
- $10000^{2i/d_{\text{model}}}$：频率缩放因子，低频对应低维，高频对应高维

### 6.3 正弦编码的性质

1. **相对位置可线性表示**：对于任意固定偏移 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性变换。这使得模型可能学习到相对位置关系。

2. **值域有界**：$PE \in [-1, 1]$，数值稳定。

3. **可外推**：训练时未见过的位置也能编码。

### 6.4 可学习位置编码

后续工作（如 BERT、GPT）使用**可学习的位置嵌入**：

$$
X_{\text{input}} = X_{\text{token}} + E_{\text{position}}
$$

其中 $E_{\text{position}} \in \mathbb{R}^{n \times d_{\text{model}}}$ 是随机初始化、随训练更新的参数矩阵。

### 6.5 其他位置编码方案

| 方案 | 提出者 | 特点 |
|------|--------|------|
| Sinusoidal | Vaswani et al. (2017) | 固定，可外推 |
| Learned Absolute | BERT / GPT | 可学习，简单 |
| Relative Position | Shaw et al. (2018) | 建模相对距离 |
| RoPE | Su et al. (2021) | 旋转位置编码，广泛用于 LLM |
| ALiBi | Press et al. (2022) | 线性偏置，极强外推能力 |

---

## 7. Feed-Forward Network (FFN)

### 7.1 标准形式（ReLU）

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

其中 $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$。

典型设置：$d_{ff} = 4 \times d_{\text{model}} = 2048$（$d_{\text{model}} = 512$ 时）。

### 7.2 GELU 版本

BERT 和许多后继模型使用 GELU 激活函数替代 ReLU：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

或近似形式：

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)
$$

GELU 相比 ReLU 的优势：处处可微，在零点附近平滑过渡而非硬截断，实践中训练更稳定。

### 7.3 SwiGLU 版本

现代 LLM（LLaMA 等）普遍使用 SwiGLU：

$$
\text{SwiGLU}(x) = (xW_1 \cdot \text{SiLU}(xW_2))W_3
$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$（SiLU 即 Swish 激活函数）。

---

## 8. Layer Normalization & 残差连接

### 8.1 子层包装

每个子层的输出经过以下处理：

$$
\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

即：**残差连接** + **Layer Normalization**。

### 8.2 Post-LN vs Pre-LN

**Post-LN**（原始论文）：
$$
x_{l+1} = \text{LN}(x_l + \text{Sublayer}(x_l))
$$
LayerNorm 放在残差连接**之后**。

**Pre-LN**（现代实践）：
$$
x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))
$$
LayerNorm 放在残差连接**之前**。

Pre-LN 的优势：
- 训练更稳定，无需 warm-up 即可收敛
- 梯度流更顺畅，对学习率不敏感
- 现已成为默认选择

### 8.3 Layer Normalization 公式

对输入 $x \in \mathbb{R}^{d_{\text{model}}}$ 的每个样本独立归一化：

$$
\mu = \frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}} x_i
$$

$$
\sigma^2 = \frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
\text{LN}(x) = \gamma \odot \hat{x} + \beta
$$

其中 $\gamma, \beta \in \mathbb{R}^{d_{\text{model}}}$ 是可学习的缩放和平移参数，$\epsilon$ 为数值稳定项。

### 8.4 RMS LayerNorm

LLaMA 等模型使用更简洁的 RMSNorm：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i}x_i^2 + \epsilon}} \odot \gamma
$$

取消了均值中心化步骤，计算量更小。

---

## 9. Encoder 与 Decoder 的区别

| 特性 | Encoder | Decoder |
|------|---------|---------|
| Self-Attention 类型 | Bidirectional（双向） | Masked / Causal（单向/因果） |
| 能否看到未来 token | 能（全部位置可见） | 不能（掩码遮盖未来） |
| Cross-Attention | 无 | 有（对 Encoder 输出做注意力） |
| 输出 | 上下文表示序列 | 自回归生成的下一个 token |
| 并行度 | 完全并行 | 训练时并行（Teacher Forcing），推理时串行 |

### 9.1 Cross-Attention 详解

在 Decoder 的 Cross-Attention 层中：
- **Query** 来自 Decoder 自身的上一子层输出
- **Key** 和 **Value** 来自 Encoder 的最终输出

$$
\text{CrossAttn} = \text{softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}
$$

这使得 Decoder 在每个生成步骤都能**关注输入序列中最相关的部分**。

---

## 10. Mask 机制

### 10.1 Padding Mask

处理不定长序列时，短序列需要填充到固定长度（通常用 `<PAD>` token）。Padding Mask 确保模型**不对填充位置做注意力**。

**实现**：在 Softmax 之前，将填充位置对应的分数置为 $-\infty$（实践中用一个极大的负数如 $-10^9$），使得 Softmax 后该位置的权重为 $0$。

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{\text{pad}}\right)
$$

其中 $M_{\text{pad}}$ 在填充位置为 $-\infty$，其他位置为 $0$。

### 10.2 Sequence Mask（Look-ahead Mask / Causal Mask）

Decoder 的自回归生成要求位置 $i$ 只能关注位置 $1 \sim i$，不能看到 $i+1$ 及之后的内容（防止"作弊"）。

**实现**：使用上三角掩码矩阵：

$$
M_{ij} = \begin{cases}
0 & \text{if } i \ge j \\
-\infty & \text{if } i < j
\end{cases}
$$

即：
$$
M = \begin{bmatrix}
0 & -\infty & -\infty & \cdots & -\infty \\
0 & 0 & -\infty & \cdots & -\infty \\
0 & 0 & 0 & \cdots & -\infty \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 0
\end{bmatrix}
$$

训练时使用上三角掩码实现 Teacher Forcing——一次前向传播即可计算整个序列的损失，因为所有 ground-truth 已知。

---

## 11. 参数量分析

### 11.1 各组件参数量

以基础 Transformer 为例：$d_{\text{model}} = 512$, $h = 8$, $d_k = d_v = 64$, $d_{ff} = 2048$, $N = 6$（encoder 层数 = decoder 层数 = 6）。

**Embedding 层**：
- Token Embedding: $V \times d_{\text{model}}$（其中 $V$ 为词表大小）
- Position Embedding（可学习）：$n_{\text{max}} \times d_{\text{model}}$

**单个 Multi-Head Attention**：
- $W^Q, W^K, W^V$ 各为 $d_{\text{model}} \times d_{\text{model}} = 512 \times 512$
- $W^O$ 也为 $512 \times 512$
- 合计：$4 \times 512^2 \approx 1.05\text{M}$ 参数

**单个 FFN**：
- $W_1$: $512 \times 2048 \approx 1.05\text{M}$
- $W_2$: $2048 \times 512 \approx 1.05\text{M}$
- 合计：$\approx 2.1\text{M}$ 参数

**每层合计**：$\approx 3.15\text{M}$

**总参数量**（base 模型，$N=6\times2=12$ 层注意力+FFN，不含 Embedding）：
- Encoder：$6 \times 3.15\text{M} = 18.9\text{M}$
- Decoder：$6 \times 3.15\text{M} = 18.9\text{M}$（实际更多，因为有 Cross-Attention 的额外 $W^K, W^V$）

加上 Embedding 层，**base 模型总参数量约 65M**。

### 11.2 参数量公式

一般地，Transformer 的参数量约为：

$$
P \approx 2 \cdot N \cdot d_{\text{model}}^2 \cdot \left(4 + 2 \cdot \frac{d_{ff}}{d_{\text{model}}}\right) + V \cdot d_{\text{model}}
$$

当 $d_{ff} = 4 d_{\text{model}}$ 时：
$$
P \approx 24 \cdot N \cdot d_{\text{model}}^2 + V \cdot d_{\text{model}}
$$

### 11.3 典型模型参数规模

| 模型 | $d_{\text{model}}$ | $N$ | $h$ | $d_{ff}$ | 参数量 |
|------|---------------------|-----|------|-----------|--------|
| Transformer-Base | 512 | 6 | 8 | 2048 | ~65M |
| Transformer-Big | 1024 | 6 | 16 | 4096 | ~213M |
| BERT-Base | 768 | 12 | 12 | 3072 | ~110M |
| BERT-Large | 1024 | 24 | 16 | 4096 | ~340M |
| GPT-2 Small | 768 | 12 | 12 | 3072 | ~124M |
| GPT-2 XL | 1600 | 48 | 25 | 6400 | ~1.5B |

---

## 12. 训练技巧

### 12.1 Adam 优化器 + Warm-up

原始论文使用 Adam 优化器，特殊之处在于学习率调度：

$$
lr = d_{\text{model}}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$

- **Warm-up 阶段**（前 $warmup\_steps$ 步）：学习率线性增长
- **衰减阶段**：学习率按 $1/\sqrt{step}$ 衰减

Warm-up 对 Post-LN 架构至关重要——初期残差分支输出接近零，LayerNorm 的梯度可能爆炸，需要小学习率过渡。Pre-LN 架构对 warm-up 的依赖大大降低。

### 12.2 Label Smoothing

在 Cross-Entropy 损失中使用标签平滑（$\epsilon_{ls} = 0.1$）：

$$
L = -(1 - \epsilon_{ls})\log P(y_{\text{true}}) - \frac{\epsilon_{ls}}{V-1} \sum_{y \neq y_{\text{true}}} \log P(y)
$$

效果：防止模型过度自信，提高泛化能力，但可能略微损害困惑度（perplexity）。

### 12.3 Dropout

原始论文在三个位置使用 Dropout（$p = 0.1$）：
1. 每个子层的输出（残差连接之前）
2. 注意力权重 $A$（Softmax 之后）
3. Embedding 层输出

现代实践中，Dropout 逐渐被其他正则化技术取代。

### 12.4 混合精度训练

现代 Transformer 训练普遍使用 FP16/BF16 混合精度：
- 前向/反向传播使用半精度
- 权重更新使用单精度 Master Copy
- 对注意力 Softmax 等敏感操作保持高精度
- BF16 相比 FP16 有更大的动态范围（与 FP32 相同指数位），几乎不需要 loss scaling

### 12.5 梯度累积

GPU 显存不足以支持大 batch size 时，通过多次小 batch 前向传播累积梯度后再更新参数，模拟大 batch 训练效果。

### 12.6 分布式训练

- **数据并行（DP/DDP）**：每张 GPU 持有完整模型副本，处理不同数据子集
- **模型并行（MP）**：单层参数分片到多张 GPU
- **流水线并行（PP）**：不同层分配到不同 GPU
- **ZeRO（DeepSpeed）**：优化器状态、梯度、参数分片
- **张量并行（Megatron-LM）**：矩阵乘法切分到多 GPU
- **序列并行**：长序列的激活值分片

### 12.7 Flash Attention

一种 IO-aware 的精确注意力算法，通过分块（tiling）和重计算（recomputation）：
- 将 $O(n^2)$ 的内存读写降至 $O(n)$ 级别
- 显存占用大幅减少，速度提升 2-4 倍
- 计算结果是精确的（非近似）
- 已成为训练大模型的标准配置

### 12.8 梯度裁剪（Gradient Clipping）

限制梯度的全局范数，防止梯度爆炸：

$$
\tilde{g} = g \cdot \min\left(1, \frac{\text{max\_norm}}{\|g\|_2}\right)
$$

典型阈值：$\text{max\_norm} = 1.0$。

---

## 13. Transformer 的数学模型总结

### 13.1 Encoder 完整流程

```
Input:  x ∈ R^{n × d_model}

1. Embedding:        h0 = TokenEmbed(x) + PosEmbed
2. For l = 1 to N:
     h'_l = MultiHead(LN_{pre}(h_{l-1}), LN_{pre}(h_{l-1}), LN_{pre}(h_{l-1}))
     h_l = h_{l-1} + h'_l                              (残差)
     h''_l = FFN(LN_{pre}(h_l))
     h_l = h_l + h''_l                                  (残差)
3. Output:            z = h_N
```

### 13.2 Decoder 完整流程

```
Input:  y ∈ R^{m × d_model}  (已生成序列)

1. Embedding:        s0 = TokenEmbed(y) + PosEmbed
2. For l = 1 to N:
     s'_l = MaskedMultiHead(LN(s_{l-1}), LN(s_{l-1}), LN(s_{l-1}))
     s_l = s_{l-1} + s'_l                              (残差)
     s''_l = CrossMultiHead(LN(s_l), LN(z), LN(z))       (Cross-Attn)
     s_l = s_l + s''_l                                  (残差)
     s'''_l = FFN(LN(s_l))
     s_l = s_l + s'''_l                                 (残差)
3. Output:            o = Linear(s_N); P = softmax(o)
```

---

## 14. 推理过程

### 14.1 训练

- Encoder 一次性处理整个输入序列
- Decoder 使用 Teacher Forcing，以 ground-truth 序列偏移一位作为输入
- 输出层使用 Softmax 计算所有词的概率分布
- 损失函数：标准交叉熵

### 14.2 推理（自回归生成）

1. Encoder 编码输入序列（仅一次）
2. Decoder 从 `<BOS>` 起始
3. 每步生成一个 token，将其拼接到已生成序列末尾
4. 重复直到生成 `<EOS>` 或达到最大长度
5. 可使用 Beam Search / Top-k Sampling / Top-p (Nucleus) Sampling 等策略

### 14.3 KV Cache

推理时，已生成 token 的 Key 和 Value 可以缓存复用：

- 第 $t$ 步只需计算当前 token 的 $Q_t, K_t, V_t$
- $K_t, V_t$ 拼接到先前缓存的 $K_{1:t-1}, V_{1:t-1}$
- 避免了重复计算，将推理复杂度从 $O(n^3)$ 降至 $O(n^2)$

---

## 15. 常见变体与相关扩展

### 15.1 模型分支

| 范式 | 代表模型 | 特点 |
|------|----------|------|
| Encoder-Only | BERT, RoBERTa, DeBERTa | 双向编码，适合理解任务 |
| Decoder-Only | GPT 系列, LLaMA, Mistral | 自回归生成，通用 LLM |
| Encoder-Decoder | T5, BART, mT5 | 适合序列转换（翻译、摘要） |

### 15.2 高效注意力变体

| 方案 | 复杂度 | 思路 |
|------|--------|------|
| Sparse Attention | $O(n \sqrt{n})$ | 局部窗口 + 全局采样 |
| Linformer | $O(n)$ | 低秩分解 K, V |
| Performer | $O(n)$ | 核函数近似 Softmax |
| Reformer | $O(n \log n)$ | LSH 哈希 |
| FlashAttention | $O(n^2)$（但 IO 最优） | IO-aware 精确计算 |

---

## 参考文献

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
2. Ba et al., *Layer Normalization*, arXiv 2016
3. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
4. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, NAACL 2019
5. Radford et al., *Improving Language Understanding by Generative Pre-Training*, 2018
6. Brown et al., *Language Models are Few-Shot Learners*, NeurIPS 2020
7. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention*, NeurIPS 2022
8. Shazeer et al., *GLU Variants Improve Transformer*, arXiv 2020
9. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2021
10. Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020
