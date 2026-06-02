# 循环神经网络（RNN）

## 1. 核心思想

循环神经网络 (Recurrent Neural Network) 是专门用于处理**序列数据** (sequence data) 的神经网络架构。其核心理念是：通过一个**隐藏状态** (hidden state) 向量 $h_t$，将历史信息沿时间步向后传递，使得网络在处理当前时刻输入时能够保留之前的上下文。

与标准前馈网络的根本区别在于：RNN 中隐藏层之间的连接形成了一条沿时间方向传播的**循环边**，这使得网络的行为等价于一个随时间展开的深层网络。

### 1.1 数学形式

$$
h_t = \sigma(W_h h_{t-1} + W_x x_t + b)
$$

其中：
- $h_t \in \mathbb{R}^{d_h}$：时刻 $t$ 的隐藏状态
- $x_t \in \mathbb{R}^{d_x}$：时刻 $t$ 的输入
- $W_h \in \mathbb{R}^{d_h \times d_h}$：隐藏到隐藏的权重矩阵（**跨时间共享**）
- $W_x \in \mathbb{R}^{d_h \times d_x}$：输入到隐藏的权重矩阵（**跨时间共享**）
- $b \in \mathbb{R}^{d_h}$：偏置项
- $\sigma$：非线性激活函数（通常为 $\tanh$ 或 $\text{sigmoid}$）

### 1.2 关键特性：参数共享

RNN 最核心的设计原则是**时间步上的参数共享** (parameter sharing across time steps)：无论序列有多长，所有时间步都使用同一组权重矩阵 $W_h$, $W_x$。这带来三个重要优势：

1. **泛化到不同长度的序列**：模型可以处理任意长度的输入
2. **平移等变性**：相同的模式出现在序列的不同位置时，网络可以识别
3. **参数效率**：参数量与序列长度无关，避免组合爆炸

---

## 2. 展开结构 (Unfolding)

RNN 的循环结构可以沿时间轴展开，形成一个权值共享的深层前馈网络。

### 2.1 时间展开示意图（文字描述）

```
时刻:     t=1          t=2          t=3          ...          t=T
输入:     x1           x2           x3                       x_T
           |            |            |                        |
隐藏:  h0->[RNN]->h1->[RNN]->h2->[RNN]->h3  ->  ...  ->  [RNN]-> h_T
           |            |            |                        |
输出:     y1           y2           y3                       y_T
           |            |            |                        |
目标:     y1           y2           y3                       y_T
```

其中：
- $h_0$ 通常初始化为零向量
- 每个 `[RNN]` 块使用**完全相同的参数** $(W_h, W_x, W_y, b_h, b_y)$
- 信息从 $t=1$ 沿箭头方向流到 $t=T$

### 2.2 等价视角

展开后的 RNN 等价于一个深度为 $T$ 的前馈网络，但所有层的权重都被约束为相同（shared weights）。这种视角对于理解 BPTT 算法至关重要。

---

## 3. 前向传播公式

一个完整的 RNN 前向传播包含隐藏状态更新和输出计算两个步骤。

### 3.1 基础 RNN 前向传播

**隐藏状态更新：**

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)
$$

**输出计算（分类任务）：**

$$
\hat{y}_t = \text{softmax}(W_y h_t + b_y)
$$

### 3.2 损失函数

对于序列任务，损失函数通常是各时间步损失的累加：

$$
L = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)
$$

其中单个时间步的损失 $L_t$ 常用负对数似然（交叉熵）：

$$
L_t = -\sum_{c=1}^{C} y_{t,c} \log(\hat{y}_{t,c})
$$

$C$ 为类别数，$y_{t,c}$ 为 one-hot 标签。

### 3.3 多对一 / 一对多 / 多对多结构

| 模式 | 描述 | 典型应用 |
|------|------|----------|
| **多对一** | 输入整个序列，输出一个标签 | 情感分类 |
| **一对多** | 输入一个向量，输出一个序列 | 图像描述生成 |
| **多对多（同步）** | 每个输入对应一个输出 | 词性标注、NER |
| **多对多（异步）** | 编码器-解码器结构 | 机器翻译 |

---

## 4. BPTT（通过时间反向传播）

BPTT (Backpropagation Through Time) 是 RNN 的训练算法，本质上是将标准反向传播应用到 RNN 的时间展开图上。

### 4.1 基本思路

由于参数 $W_h, W_x, W_y$ 在所有时间步共享，我们需要累积所有时间步对参数的梯度：

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}
$$

其中每一项 $\frac{\partial L_t}{\partial W}$ 都需要沿时间方向展开链式法则。

### 4.2 详细推导

#### 4.2.1 对 $W_y$ 的梯度

输出层权重 $W_y$ 不参与循环，梯度计算与普通前馈网络一致：

$$
\frac{\partial L_t}{\partial W_y} = \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial W_y}
$$

#### 4.2.2 对 $W_h$ 的梯度

这是 BPTT 的核心难点。以最后一个时间步的损失 $L_T$ 为例，它对 $W_h$ 的梯度需要回溯 $h_T$ 在所有历史时间步中对 $W_h$ 的依赖：

$$
\frac{\partial L_T}{\partial W_h} = \sum_{k=1}^{T} \frac{\partial L_T}{\partial h_T} \cdot \left( \prod_{j=k+1}^{T} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_h}
$$

#### 4.2.3 雅可比矩阵 $\frac{\partial h_j}{\partial h_{j-1}}$

对于 $\tanh$ 激活的 RNN：

$$
h_j = \tanh(W_h h_{j-1} + W_x x_j + b_h)
$$

令 $a_j = W_h h_{j-1} + W_x x_j + b_h$，则：

$$
\frac{\partial h_j}{\partial h_{j-1}} = \frac{\partial h_j}{\partial a_j} \cdot \frac{\partial a_j}{\partial h_{j-1}} = \text{diag}[\tanh'(a_j)] \cdot W_h
$$

其中 $\tanh'(x) = 1 - \tanh^2(x) \in (0, 1]$，最大值为 $1$（在 $x=0$ 处）。

#### 4.2.4 截断 BPTT (Truncated BPTT)

实践中，当序列非常长时，完整展开所有时间步计算代价过高。截断 BPTT 每隔固定的步数截断梯度传播：

- 将长序列分割为长度为 $k$ 的块
- 在每个块内执行标准 BPTT
- 前一个块的最终隐藏状态传给下一个块作为初始状态，但不传梯度

### 4.3 BPTT 伪代码

```
输入: 序列 x[1..T], 标签 y[1..T]
输出: 梯度 dL/dW_h, dL/dW_x, dL/dW_y, dL/db_h, dL/db_y

// 前向传播
for t = 1 to T:
    a[t] = W_h @ h[t-1] + W_x @ x[t] + b_h
    h[t] = tanh(a[t])
    o[t] = W_y @ h[t] + b_y
    y_hat[t] = softmax(o[t])

// 反向传播
dh_next = 0
for t = T downto 1:
    do[t] = y_hat[t] - y[t]                       // softmax + 交叉熵的梯度
    dW_y += do[t] @ h[t]^T
    db_y += do[t]
    dh[t] = W_y^T @ do[t] + dh_next
    da[t] = dh[t] * (1 - h[t]^2)                  // tanh 的导数
    dW_h += da[t] @ h[t-1]^T
    dW_x += da[t] @ x[t]^T
    db_h += da[t]
    dh_next = W_h^T @ da[t]
```

---

## 5. 梯度消失与梯度爆炸

### 5.1 数学根源

从 BPTT 推导可知，梯度中包含连乘项：

$$
\prod_{j=k+1}^{T} \frac{\partial h_j}{\partial h_{j-1}}
$$

每一步的雅可比矩阵近似为：

$$
\left\| \frac{\partial h_j}{\partial h_{j-1}} \right\| \leq \|W_h\| \cdot \|\text{diag}[\sigma'(a_j)]\|
$$

### 5.2 梯度消失

当 $\|\frac{\partial h_j}{\partial h_{j-1}}\| < 1$ 时，$T-k$ 次连乘后梯度趋近于零：

$$
\lim_{T \to \infty} \prod_{j=k+1}^{T} \left\| \frac{\partial h_j}{\partial h_{j-1}} \right\| = 0
$$

**具体条件**：对于 $\tanh$ 激活，$\tanh'(x) \leq 1$（仅在 $x=0$ 处取到 $1$），对于 $\text{sigmoid}$，$\sigma'(x) \leq 1/4$。若 $\|W_h\| < 1$，梯度必然衰减。即便 $\|W_h\| = 1$，由于激活函数导数的压缩效应，梯度也会衰减。

**后果**：长距离依赖 (long-term dependency) 无法被学习——远处的输入对当前梯度几乎无影响。

### 5.3 梯度爆炸

当 $\|\frac{\partial h_j}{\partial h_{j-1}}\| > 1$ 时：

$$
\lim_{T \to \infty} \prod_{j=k+1}^{T} \left\| \frac{\partial h_j}{\partial h_{j-1}} \right\| = \infty
$$

具体条件：$\|W_h\| \gg 1$ 且激活函数导数不充分抑制。

**后果**：参数更新步长过大，训练发散，损失变为 NaN。

### 5.4 数值示例

考虑一个简单情形：$h_t = \sigma(w h_{t-1})$，其中 $\sigma = \text{sigmoid}$：

- 若 $w = 0.5$：$\sigma'(x) \leq 0.25$ → $\frac{\partial h_t}{\partial h_{t-1}} \leq 0.125$ → 10 步后梯度约为 $10^{-9}$
- 若 $w = 10$：$\frac{\partial h_t}{\partial h_{t-1}}$ 可能 $\gg 1$ → 10 步后梯度约为 $10^{10}$

---

## 6. 梯度裁剪 (Gradient Clipping)

梯度裁剪是应对梯度爆炸的实用技巧，对梯度爆炸非常有效（对梯度消失无效）。

### 6.1 范数裁剪

计算所有参数梯度的全局范数 $\|g\|$，若超过阈值 $\text{threshold}$，则等比例缩放：

$$
\text{if } \|g\|_2 > \text{threshold}, \quad g \leftarrow \frac{\text{threshold}}{\|g\|_2} \cdot g
$$

其中 $g = \frac{\partial L}{\partial \theta}$ 是所有参数梯度的拼接向量。

### 6.2 元素裁剪

更简单的方法，将每个梯度元素裁剪到 $[-c, c]$ 区间：

$$
g_i \leftarrow \max(-c, \min(c, g_i))
$$

### 6.3 PyTorch 实现

```python
# 范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

### 6.4 工作原理

梯度爆炸时，参数空间的损失曲面存在陡峭区域。梯度裁剪将更新方向限制在球内，防止参数被一步推入不可恢复的区域，同时保留了有益的梯度方向信息。

---

## 7. 双向 RNN (Bidirectional RNN)

标准 RNN 只能利用过去的信息（正向传播）。双向 RNN 同时利用过去和未来的上下文。

### 7.1 结构

双向 RNN 由两个独立的 RNN 组成：

- **前向 RNN**：从左到右处理序列，产生 $\overrightarrow{h_t}$
- **后向 RNN**：从右到左处理序列，产生 $\overleftarrow{h_t}$

### 7.2 数学形式

$$
\begin{aligned}
\overrightarrow{h_t} &= \tanh(\overrightarrow{W_h} \overrightarrow{h_{t-1}} + \overrightarrow{W_x} x_t + \overrightarrow{b_h}) \\
\overleftarrow{h_t} &= \tanh(\overleftarrow{W_h} \overleftarrow{h_{t+1}} + \overleftarrow{W_x} x_t + \overleftarrow{b_h})
\end{aligned}
$$

最终的隐藏状态是两者拼接：

$$
h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}] \in \mathbb{R}^{2d_h}
$$

输出层：

$$
\hat{y}_t = \text{softmax}(W_y h_t + b_y)
$$

### 7.3 适用场景

双向 RNN 适用于**可以看到完整序列**的任务：
- 词性标注 (POS Tagging)
- 命名实体识别 (NER)
- 语音识别（离线）
- 文本分类

**不适用**于在线 / 因果任务（如实时预测、语言模型自回归生成）。

---

## 8. 深层 RNN (Deep/Stacked RNN)

类似于前馈网络可以通过增加层数提高表达能力，RNN 也可以通过堆叠多层来增强模型容量。

### 8.1 结构

第 $l$ 层的隐藏状态：

$$
h_t^{(l)} = \tanh\left(W_h^{(l)} h_{t-1}^{(l)} + W_x^{(l)} h_t^{(l-1)} + b_h^{(l)}\right)
$$

其中 $h_t^{(0)} = x_t$，$h_t^{(L)}$ 为最终隐藏状态。

### 8.2 关键特性

- 不同层使用**不同的参数** $W_h^{(l)}, W_x^{(l)}, b_h^{(l)}$
- 同一层内跨时间步共享参数
- 通常 2~4 层效果较好，过多层可能加剧梯度消失

### 8.3 训练技巧

- 层间使用残差连接（Residual RNN）
- 交替使用 Dropout（但不在循环连接上）
- 配合 Layer Normalization

---

## 9. 应用场景

### 9.1 自然语言处理 (NLP)

- **语言模型**：给定前文预测下一个词，$P(w_t | w_1, \dots, w_{t-1})$
- **文本分类**：多对一结构，取 $h_T$ 或池化后分类
- **序列标注**：词性标注、命名实体识别
- **文本生成**：字符级或词级自回归生成

### 9.2 时间序列预测

- **单变量预测**：利用历史值预测未来的值
- **多变量预测**：使用多个相关序列协同预测
- **异常检测**：预测值与观测值的偏差超过阈值判定为异常

典型公式：$\hat{x}_{t+1} = f_\theta(x_{t-k+1}, \dots, x_t)$

### 9.3 语音处理

- **语音识别 (ASR)**：声学特征序列映射为文本序列
- **语音合成 (TTS)**：文本序列映射为声学特征序列
- **说话人识别**：声纹特征提取

### 9.4 其他应用

- **音乐生成**：MIDI 音符序列建模
- **视频理解**：帧序列的动作识别
- **生物信息学**：DNA / 蛋白质序列分析
- **金融**：股价序列建模、算法交易

---

## 10. 总结

| 方面 | 说明 |
|------|------|
| **优势** | 可处理变长序列、参数共享、理论上有记忆长序列的能力 |
| **核心缺陷** | 梯度消失/爆炸限制了实际可学习的序列长度 |
| **改进方向** | LSTM、GRU（门控机制）、残差连接、注意力机制 |
| **后继模型** | Seq2Seq + Attention → Transformer |

### 10.1 RNN 的局限性

| 问题 | 描述 |
|------|------|
| 梯度消失 | 无法捕获长距离依赖（超过 10 步），BPTT 连乘导致梯度指数衰减 |
| 梯度爆炸 | 大规模连乘导致梯度剧增，训练不稳定（可通过梯度裁剪缓解） |
| 串行计算 | 每个时间步的计算依赖前一步结果，无法并行化，训练速度慢 |
| 记忆容量有限 | 隐藏状态维度固定，压缩所有历史信息到固定向量中导致信息瓶颈 |
| 初始隐状态敏感 | $h_0$ 的初始化质量影响早期时间步的学习效果 |

### 10.2 关键公式速查

| 名称 | 公式 |
|------|------|
| 隐藏状态更新 | $h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$ |
| 输出计算 | $\hat{y}_t = \text{softmax}(W_y h_t + b_y)$ |
| 总损失 | $L = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)$ |
| BPTT 核心 | $\frac{\partial L_T}{\partial W_h} = \sum_{k=1}^{T} \frac{\partial L_T}{\partial h_T} \prod_{j=k+1}^{T} \frac{\partial h_j}{\partial h_{j-1}} \frac{\partial h_k}{\partial W_h}$ |
| 梯度爆炸条件 | $\|\frac{\partial h_j}{\partial h_{j-1}}\| > 1$ |
| 梯度消失条件 | $\|\frac{\partial h_j}{\partial h_{j-1}}\| < 1$ |
| 梯度裁剪 | $g \leftarrow \frac{\text{threshold}}{\|g\|_2} \cdot g \quad \text{if } \|g\|_2 > \text{threshold}$ |
| 双向拼接 | $h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$ |

---

## 参考文献

1. Elman, J. L. (1990). "Finding Structure in Time". *Cognitive Science*.
2. Werbos, P. J. (1990). "Backpropagation Through Time: What It Does and How to Do It".
3. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). "On the difficulty of training recurrent neural networks". *ICML*.
4. Hochreiter, S. (1991). "Untersuchungen zu dynamischen neuronalen Netzen". Diploma thesis.
5. Bengio, Y., Simard, P., & Frasconi, P. (1994). "Learning long-term dependencies with gradient descent is difficult". *IEEE TNN*.
