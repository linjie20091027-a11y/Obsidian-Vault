# 长短期记忆网络（LSTM）

## 1. 引言

长短期记忆网络 (Long Short-Term Memory, LSTM) 由 Hochreiter 和 Schmidhuber 于 1997 年提出，是解决 RNN 梯度消失问题的最经典架构。LSTM 的核心创新是引入了**门控机制** (gating mechanism) 和**细胞状态** (cell state)，使得网络有能力选择性地记忆和遗忘信息，从而有效捕获长距离依赖。

---

## 2. 核心思想

### 2.1 双通道设计

标准 RNN 只有一个隐藏状态 $h_t$ 同时承担"记忆"和"输出"两个功能。LSTM 将其解耦为两个通道：

| 通道 | 符号 | 作用 | 特性 |
|------|------|------|------|
| **细胞状态** (Cell State) | $c_t$ | 长期记忆的传输带，贯穿整个序列 | 梯度流动畅通，几乎无衰减 |
| **隐藏状态** (Hidden State) | $h_t$ | 当前时间步的输出 | 受细胞状态制约，经由输出门调控 |

### 2.2 门控机制

LSTM 使用三个"门"来控制信息流。每个门本质上是一个以 sigmoid 为激活的全连接层，输出值在 $[0, 1]$ 之间，表示"允许多少信息通过"（$0$ = 完全阻断，$1$ = 完全通过）。

三个门及其功能：

1. **遗忘门 (Forget Gate)** $f_t$：决定从细胞状态中丢弃哪些旧信息
2. **输入门 (Input Gate)** $i_t$：决定将哪些新信息存入细胞状态
3. **输出门 (Output Gate)** $o_t$：决定从细胞状态中输出哪些信息

---

## 3. 详细公式与结构

### 3.1 符号约定

- $x_t \in \mathbb{R}^{d_x}$：时刻 $t$ 的输入向量
- $h_{t-1} \in \mathbb{R}^{d_h}$：上一时刻的隐藏状态
- $c_{t-1} \in \mathbb{R}^{d_h}$：上一时刻的细胞状态
- $W_* \in \mathbb{R}^{d_h \times (d_h + d_x)}$：各类权重矩阵
- $b_* \in \mathbb{R}^{d_h}$：各类偏置向量
- $\sigma$：sigmoid 激活函数，$\sigma(x) = \frac{1}{1+e^{-x}}$
- $\odot$：逐元素乘积 (Hadamard product)

### 3.2 遗忘门 (Forget Gate)

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**物理意义**：通过观察当前的输入 $x_t$ 和上一时刻的隐藏状态 $h_{t-1}$，为细胞状态 $c_{t-1}$ 的每个维度输出一个 $[0, 1]$ 之间的值——接近 $1$ 表示"保留此信息"，接近 $0$ 表示"遗忘此信息"。这是 LSTM 解决梯度消失的关键所在。

直观理解：在一段文本中，当遇到一个句子的句号时，遗忘门可能会选择性地清除与上一句主语相关的信息。

### 3.3 输入门 (Input Gate)

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**物理意义**：决定当前输入中哪些新信息值得写进细胞状态。$i_t$ 的每个维度取值 $[0, 1]$，起到"写入过滤器"的作用。

### 3.4 候选记忆 (Candidate Memory)

$$
\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

**物理意义**：基于当前输入和上一隐藏状态，生成**候选的新记忆内容**。使用 $\tanh$ 激活将值压缩到 $[-1, 1]$，使其与细胞状态的数值范围一致。这个候选内容还**不会直接写入**，需要经过输入门的筛选。

$\tilde{c}_t$ 可以理解为"当前时间步产生的新知识的草稿"，而输入门 $i_t$ 决定了这份草稿中哪些部分值得被长期记住。

### 3.5 细胞状态更新 (Cell State Update)

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

**物理意义**：这是 LSTM 最核心的操作——以逐元素的方式将旧记忆与新记忆线性组合。

- $f_t \odot c_{t-1}$：遗忘门控制的"选择性遗忘"——保留旧记忆中仍有用的部分
- $i_t \odot \tilde{c}_t$：输入门控制的"选择性写入"——加入当前有价值的新信息

注意这里用的是**加法** ($+$) 而非乘法或矩阵乘法——这意味着梯度可以沿 $c_t$ 这条路径**直接相加回流**，不会因为 $f_t$ 的某种取值而完全消失（只要 $f_t$ 不恒为零）。

### 3.6 输出门 (Output Gate)

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

**物理意义**：决定细胞状态中的哪些部分应当作为当前时间步的输出（隐藏状态）。$o_t$ 选择性地暴露细胞状态的内容，用于下游任务（分类、生成下一词等），同时也传给下一时间步。

### 3.7 隐藏状态 (Hidden State)

$$
h_t = o_t \odot \tanh(c_t)
$$

**物理意义**：首先通过 $\tanh$ 将细胞状态压缩到 $[-1, 1]$（归一化信息强度），然后通过输出门 $o_t$ 对其进行筛选。最终 $h_t$ 既作为当前时间步的输出，也作为下一时间步的输入。

---

## 4. LSTM 前向传播完整流程

```
输入: x_t, h_{t-1}, c_{t-1}
输出: h_t, c_t

步骤1: 遗忘门
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

步骤2: 输入门
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

步骤3: 候选记忆
    c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

步骤4: 更新细胞状态
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

步骤5: 输出门
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

步骤6: 更新隐藏状态
    h_t = o_t ⊙ tanh(c_t)
```

### 4.1 结构示意图（文字描述）

```
                    ┌──────────────────────────────────────┐
                    │            细胞状态 c_t               │
                    │   c_{t-1} ──→[×]─→(+)──→ c_t        │
                    │              ↑     ↑                │
                    │            f_t  i_t⊙c̃_t             │
                    │              ↑     ↑                │
  x_t ──────────►───┤              │     │                │
  h_{t-1} ──────►───┤              │     │                │
                    │   ┌──────────┼─────┼────────────┐   │
                    │   │ 遗忘门   │ 输入门│  候选记忆   │   │
                    │   │ f_t=σ    │i_t=σ │c̃_t=tanh    │   │
                    │   └──────────┼─────┼────────────┘   │
                    │              │     │                │
                    │              │     │                │
                    │              │  输出门               │
                    │              │  o_t=σ               │
                    │              │     │                │
                    │              │  h_t = o_t⊙tanh(c_t) │
                    │              │     ↓                │
                    │              │    h_t               │
                    └──────────────────────────────────────┘
```

---

## 5. 为什么 LSTM 能缓解梯度消失

### 5.1 核心数学分析

在标准 RNN 中，隐藏状态的梯度包含连乘项：

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}[\tanh'] \cdot W_h
$$

每次连乘，范数可能小于 $1$（导致梯度消失）或大于 $1$（导致梯度爆炸）。

在 LSTM 中，细胞状态提供了另一条梯度传播路径：

$$
\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}[f_t] + \frac{\partial(\cdots)}{\partial c_{t-1}}
$$

**关键洞察**：$f_t$ 的输出在 $[0, 1]$，但**不是**通过一个雅可比矩阵连乘的形式——而是通过逐元素乘积。当 $f_t \approx 1$（即遗忘门选择"记住"时），$\frac{\partial c_t}{\partial c_{t-1}} \approx 1$，梯度可以近乎无损地传递。

更重要的是，**细胞状态的更新使用了加法** ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$)，这意味着梯度随时间的反向传播路径中**不存在连乘的权重矩阵**——这是与标准 RNN ($h_t = \tanh(W_h h_{t-1} + \cdots)$) 最根本的区别。

### 5.2 与 RNN 的对比

| 特性 | 标准 RNN | LSTM |
|------|----------|------|
| 状态更新方式 | 全矩阵乘 + 非线性 | 逐元素门控 + 加法 |
| 梯度传播路径 | $\frac{\partial h_t}{\partial h_{t-1}}$ 含 $W_h$ 连乘 | $\frac{\partial c_t}{\partial c_{t-1}} \approx f_t$ |
| $f_t=1$ 时梯度 | 仍然受 $\|W_h\|$ 和 $\tanh'$ 影响 | 梯度几乎无损传递 |
| 学习长距离依赖 | 非常困难 (通常 $<20$ 步) | 可以捕获数百步以上的依赖 |

### 5.3 遗忘门偏置初始化技巧

实践中，通常将遗忘门的偏置 $b_f$ 初始化为较大正值（如 $1.0$），使得 $f_t$ 在训练初期接近 $1$，鼓励模型先"记住"信息，再由训练过程学习何时遗忘。这显著改善了长序列训练的稳定性。

---

## 6. 参数总量与计算复杂度

### 6.1 参数矩阵

LSTM 有 4 组参数矩阵（分别对应遗忘门、输入门、候选记忆、输出门），每组包含两个子矩阵：

$$
W_* = [W_{*h} \mid W_{*x}]
$$

其中 $W_{*h} \in \mathbb{R}^{d_h \times d_h}$，$W_{*x} \in \mathbb{R}^{d_h \times d_x}$。

### 6.2 参数总量

$$
\text{Params} = 4 \times \left(d_h \times d_h + d_h \times d_x + d_h\right)
$$

例如，当 $d_x = 300$, $d_h = 256$ 时：

$$
\begin{aligned}
\text{Params} &= 4 \times (256 \times 256 + 256 \times 300 + 256) \\
&= 4 \times (65536 + 76800 + 256) \\
&= 4 \times 142592 = 570368
\end{aligned}
$$

### 6.3 计算复杂度

单个时间步的计算复杂度为：

$$
O(4 \times d_h \times (d_h + d_x))
$$

约为标准 RNN 的 4 倍。对整个长度为 $T$ 的序列，总复杂度为 $O(T \cdot d_h \cdot (d_h + d_x))$。

---

## 7. LSTM 变体

### 7.1 Peephole Connection（窥视孔连接）

标准 LSTM 中，门的计算仅依赖于 $h_{t-1}$ 和 $x_t$。Peephole LSTM (Gers & Schmidhuber, 2000) 让门也能"窥视"细胞状态 $c_{t-1}$（或当前 $c_t$）：

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [c_{t-1}, h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i \cdot [c_{t-1}, h_{t-1}, x_t] + b_i) \\
o_t &= \sigma(W_o \cdot [c_t, h_{t-1}, x_t] + b_o)
\end{aligned}
$$

Peephole 连接使门在决策时可以"看到"记忆单元的实际数值，在某些精确时序任务（如学习间隔计数）上表现更好。

### 7.2 Coupled Forget-Input Gate（耦合遗忘-输入门）

将遗忘门和输入门耦合为互补关系：

$$
i_t = 1 - f_t
$$

这使细胞状态始终容纳相同的信息总量（$c_t = f_t \odot c_{t-1} + (1-f_t) \odot \tilde{c}_t$），减少了参数量。这种设计与 GRU 的更新门机制非常相似。

### 7.3 Layer Normalization LSTM

在 LSTM 内部的每个门的线性变换后、激活函数前加入 Layer Normalization：

$$
f_t = \sigma(\text{LayerNorm}(W_f \cdot [h_{t-1}, x_t]) + b_f)
$$

这有助于稳定训练动态，加速收敛。

---

## 8. 与 GRU 的对比

| 维度 | LSTM | GRU |
|------|------|-----|
| **提出年份** | 1997 | 2014 |
| **门数量** | 3（遗忘、输入、输出） | 2（重置、更新） |
| **状态通道** | 2（$c_t$ 和 $h_t$） | 1（仅 $h_t$） |
| **遗忘机制** | 显式遗忘门 $f_t$ | 隐式通过 $(1-z_t)$ |
| **参数矩阵数** | 4 | 3 |
| **参数量** | $4 \times d_h \times (d_h+d_x)$ | $3 \times d_h \times (d_h+d_x)$ |
| **计算速度** | 较慢 | 约快 25% |
| **长序列表现** | 通常略优 | 通常相当 |
| **小数据集** | 易过拟合 | 稍好（参数少） |

### 8.1 选择建议

- **资源充足、追求极致性能** → LSTM
- **资源受限、需要快速迭代** → GRU
- **经验法则**：两者差异通常不显著，超参数调整的影响远大于 LSTM vs GRU 的选择。建议两者都尝试。

---

## 9. 应用场景

### 9.1 机器翻译 (Machine Translation)

Seq2Seq 架构中，LSTM 作为编码器和解码器的标准组件：

- **编码器 LSTM**：将源语言句子编码为固定维度的上下文向量（或注意力机制下的变量集）
- **解码器 LSTM**：从上下文向量自回归生成目标语言句子

在注意力机制被引入后，LSTM Seq2Seq + Attention 在 Transformer 出现前长期是机器翻译的 SOTA 方案。

### 9.2 语音识别 (Speech Recognition)

- **声学模型**：将声学特征（MFCC、FBANK 等）序列映射为音素或字符序列
- **DeepSpeech** (百度) 使用了多层双向 LSTM 作为声学模型的核心
- 现代端到端语音识别（如 LAS: Listen, Attend and Spell）仍大量使用 LSTM

### 9.3 命名实体识别 (NER)

BiLSTM-CRF 是 NER 任务的经典架构：

1. 输入：词向量序列
2. 双向 LSTM：提取每个词的上下文特征
3. CRF 层：在 LSTM 输出之上建模标签序列的转移约束（如 I-ORG 不能出现在 B-PER 之后）

这种组合在 Transformer 普及之前长期霸榜各类序列标注任务。

### 9.4 其他应用

- **文本生成 / 语言建模**：字符级 LSTM (如 Andrej Karpathy 的 char-rnn)
- **情感分析**：多对一结构，取 $h_T$ 或注意力池化
- **时间序列异常检测**：LSTM 自编码器 → 重构误差
- **视频描述生成**：CNN 编码视频帧 → LSTM 生成描述
- **股票预测**：利用历史量价序列建模
- **手写识别**：输入笔迹坐标序列，输出文本

---

## 10. 总结

### 10.1 LSTM 的核心贡献

1. **细胞状态 + 门控机制**：创造了一条梯度可以无衰减传播的"高速公路"
2. **加法更新**：$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ 从根本上避免了连乘性梯度衰减
3. **可学习的遗忘**：网络可以自适应地决定何时记住、何时遗忘

### 10.2 关键公式速查

| 组件     | 公式                                                               |
| ------ | ---------------------------------------------------------------- |
| 遗忘门    | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$                   |
| 输入门    | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$                   |
| 候选记忆   | $\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$            |
| 细胞状态更新 | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$                |
| 输出门    | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$                   |
| 隐藏状态   | $h_t = o_t \odot \tanh(c_t)$                                     |
| 细胞状态梯度 | $\frac{\partial c_t}{\partial c_{t-1}} \approx \text{diag}[f_t]$ |

---

## 参考文献

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory". *Neural Computation*.
2. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). "Learning to Forget: Continual Prediction with LSTM". *Neural Computation*.
3. Gers, F. A., & Schmidhuber, J. (2000). "Recurrent Nets that Time and Count". *IJCNN*.
4. Graves, A. (2013). "Generating Sequences With Recurrent Neural Networks". *arXiv:1308.0850*.
5. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to Sequence Learning with Neural Networks". *NIPS*.
6. Chung, J., et al. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling". *arXiv:1412.3555*.
7. Greff, K., et al. (2017). "LSTM: A Search Space Odyssey". *IEEE TNNLS*.
