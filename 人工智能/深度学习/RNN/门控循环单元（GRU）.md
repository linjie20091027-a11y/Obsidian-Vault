# 门控循环单元（GRU）

## 1. 引言

门控循环单元 (Gated Recurrent Unit, GRU) 由 Cho 等人于 2014 年在论文 *"Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"* 中提出。GRU 可以视为 **LSTM 的简化变体**，在保持类似长距离记忆能力的同时减少了门控数量和参数量，从而获得了更快的训练速度和更低的过拟合风险。

---

## 2. 核心思想

### 2.1 设计哲学：少即是多

GRU 的设计基于一个观察：LSTM 的三个门中，遗忘门和输入门的角色存在一定冗余——一个负责忘记，一个负责写入，两者本质上都在调控细胞状态的更新幅度。GRU 的解决方案是：

1. **合并遗忘门和输入门**为一个**更新门** (Update Gate)
2. **合并细胞状态 $c_t$ 和隐藏状态 $h_t$**为单一状态 $h_t$
3. 引入**重置门** (Reset Gate) 来控制历史信息对新候选状态的贡献

最终只保留两个门，参数量约为 LSTM 的 3/4。

### 2.2 两门设计

| 门 | 符号 | 取值范围 | 功能 |
|----|------|----------|------|
| **更新门** (Update Gate) | $z_t$ | $[0, 1]$ | 控制多大程度保留旧信息 vs 引入新信息 |
| **重置门** (Reset Gate) | $r_t$ | $[0, 1]$ | 控制多大程度忽略过去的隐藏状态 |

---

## 3. 详细公式与推导

### 3.1 符号约定

- $x_t \in \mathbb{R}^{d_x}$：时刻 $t$ 的输入向量
- $h_{t-1} \in \mathbb{R}^{d_h}$：上一时刻的隐藏状态
- $W_z, W_r, W \in \mathbb{R}^{d_h \times (d_h + d_x)}$：权重矩阵
- $b_z, b_r, b \in \mathbb{R}^{d_h}$：偏置向量
- $\sigma$：sigmoid 激活函数
- $\odot$：逐元素乘积 (Hadamard product)

### 3.2 更新门 (Update Gate)

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
$$

（为简洁起见，偏置项 $b_z$ 隐含在 $W_z$ 的矩阵乘法中，实际使用时应加上。）

**物理意义**：更新门 $z_t$ 控制着旧状态 $h_{t-1}$ 和新候选状态 $\tilde{h}_t$ 之间的**线性插值系数**。$z_t$ 的每个维度决定了该维度的信息是更多地保留历史（$z_t \to 0$）还是更多地采用新信息（$z_t \to 1$）。

可以想象成一个"滑动比例尺"：当 $z_t$ 的某个元素接近 $0.3$ 时，该维度保留 $70\%$ 的旧信息和 $30\%$ 的新信息。

### 3.3 重置门 (Reset Gate)

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
$$

**物理意义**：重置门 $r_t$ 控制着在计算候选状态时，**有多大程度忽略**过去的隐藏状态。当 $r_t$ 的某个维度接近 $0$ 时，该维度对应的历史信息几乎被完全忽略——这等效于"重置"了该维度的记忆，让模型可以像读一个新句子开头一样重新开始。

这一机制在处理"句子边界"或"语义转折"时非常重要：模型需要知道何时丢掉前文的句法结构，重新理解一个全新的语义单元。

重置门是 GRU 区别于简单 RNN 的关键——它赋予了模型**自适应地决定历史信息留存范围**的能力。

### 3.4 候选隐藏状态 (Candidate Hidden State)

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])
$$

**物理意义**：候选隐藏状态 $\tilde{h}_t$ 表示"如果忽略历史约束，当前时间步最合理的隐藏状态"。关键点：

- $r_t \odot h_{t-1}$：重置门先对旧状态进行**选择性清洗**（$r_t$ 中接近 $0$ 的维度会屏蔽对应的历史信息）
- 清洗后的历史信息与 $x_t$ 拼接，经过 $\tanh$ 激活后成为候选
- 这个候选**不会直接成为最终输出**，还需要经过更新门的插值

### 3.5 最终隐藏状态 (Final Hidden State)

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

**物理意义**：这是 GRU 最优雅的设计——通过一个简单的**线性插值**同时完成了 LSTM 中"遗忘"和"写入"两个操作：

- $(1 - z_t) \odot h_{t-1}$：保留的旧信息（$z_t \to 0$ 时完全保留）
- $z_t \odot \tilde{h}_t$：引入的新信息（$z_t \to 1$ 时完全刷新）

注意 $(1 - z_t)$ 和 $z_t$ 之和恒为 $1$，这构成了一个**凸组合** (convex combination)，意味着 $h_t$ 始终在 $h_{t-1}$ 和 $\tilde{h}_t$ 张成的空间内。

**极端情况分析**：
- $z_t = 0$：$h_t = h_{t-1}$——完全保持旧状态，不更新（类似于 LSTM 中 $f_t=1, i_t=0$）
- $z_t = 1$：$h_t = \tilde{h}_t$——完全丢弃旧状态，使用新候选（类似于 LSTM 中 $f_t=0, i_t=1$）
- $r_t = 0$：$\tilde{h}_t = \tanh(W \cdot [\mathbf{0}, x_t])$——候选状态仅依赖当前输入，等效于"硬重置"

---

## 4. GRU 前向传播完整流程

```
输入: x_t, h_{t-1}
输出: h_t

步骤1: 更新门
    z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

步骤2: 重置门
    r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

步骤3: 候选隐藏状态
    h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t] + b)

步骤4: 最终隐藏状态（线性插值）
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### 4.1 结构示意图（文字描述）

```
         x_t ─────────────────────────┐
         h_{t-1} ─────────────────────┤
                                      │
         ┌────────────────────────────┼──────────────┐
         │                            │              │
         │    ┌───────────────────────┤              │
         │    │   重置门 r_t          │   更新门 z_t  │
         │    │   σ(W_r·[h,x])       │   σ(W_z·[h,x])│
         │    │        │              │        │      │
         │    │   r_t ⊙ h_{t-1}      │        │      │
         │    │        │              │        │      │
         │    │   ┌────┴────┐         │        │      │
         │    │   │ 候选状态 │         │        │      │
         │    │   │ h̃_t=tanh│         │        │      │
         │    │   └────┬────┘         │        │      │
         │    │        │              │        │      │
         │    │        └──────┬───────┘        │      │
         │    │               │                │      │
         │    │    h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t   │
         │    │               │                │      │
         │    │              h_t               │      │
         └────────────────────────────────────────────┘
```

---

## 5. 与 LSTM 的门映射关系

### 5.1 结构对应

GRU 的两个门可以"解码"为 LSTM 三个门的功能组合：

| GRU 组件 | LSTM 等效 | 说明 |
|----------|-----------|------|
| $z_t$（更新门） | $f_t + i_t$（耦合） | 遗忘和写入通过 $(1-z_t, z_t)$ 被约束为互补关系 |
| $r_t$（重置门） | 近似 $f_t$ | 控制历史信息在候选计算中的贡献；在 LSTM 中此角色由 $f_t$ 部分承担 |
| $\tilde{h}_t$（候选状态） | $\tilde{c}_t$ | 候选新信息，但 GRU 直接生成隐藏候选而非细胞候选 |
| $h_t$（最终状态） | $c_t$ 和 $h_t$ 的合并 | GRU 不区分长期记忆和输出通道 |

### 5.2 数学对应

LSTM 的细胞状态更新：

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

GRU 的隐藏状态更新：

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

对比可见，如果令 $f_t = 1 - z_t$，$i_t = z_t$（即强制 $f_t + i_t = 1$），以及 $c_t = h_t$，则两者完全等价。因此 GRU 可以看作 LSTM 在 **"遗忘门 + 输入门 = 1"约束下的特化版本**。

这种约束并非弱点——实践中，GRU 的"耦合遗忘-输入"设计在许多任务上表现与 LSTM 相当甚至更优，因为它消除了两个门之间的冗余自由度，减少了过拟合风险。

---

## 6. 梯度流分析

### 6.1 隐藏状态梯度传播

对 GRU 最终状态更新公式求偏导：

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}[1 - z_t] + \frac{\partial (z_t \odot \tilde{h}_t)}{\partial h_{t-1}}
$$

展开第二部分：

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}[1 - z_t] + \tilde{h}_t \odot \frac{\partial z_t}{\partial h_{t-1}} + z_t \odot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}
$$

**关键观察**：
- 第一项 $\text{diag}[1 - z_t]$ 是直接梯度通路。当 $z_t \to 0$ 时，该项趋近于单位阵，梯度可以近乎无损地回传
- 第二项和第三项涉及门和候选状态的间接梯度，但它们的存在保证了模型仍然可以学习调节信息流

该结构确保**至少存在一条梯度无衰减的路径**（当 $z_t$ 足够小时），其数学效果与 LSTM 中 $f_t \approx 1$ 的场景类似，但实现更简洁。

### 6.2 与 LSTM 梯度流的对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 主要梯度通路 | $\frac{\partial c_t}{\partial c_{t-1}} \approx f_t$ | $\frac{\partial h_t}{\partial h_{t-1}} \approx 1 - z_t$ |
| 通路复杂度 | 仅细胞状态通道 | 单一隐藏状态通道（但也承担梯度通路） |
| 线性成分 | $f_t \odot c_{t-1}$ | $(1-z_t) \odot h_{t-1}$ |
| 非线性干扰 | $\tanh(c_t)$ 通过输出门 $o_t$ 传递 | 无额外非线性（$h_t$ 直接是线性的凸组合） |

GRU 中 $h_t$ 直接由线性插值得到，意味着隐藏状态本身就没有经过 $\tanh$ 压缩——这进一步减少了梯度衰减的因素。

### 6.3 梯度消失风险分析

GRU 仍然存在梯度消失的可能——当 $z_t \to 1$ 长期持续时，$1-z_t \to 0$，主梯度通路被切断，梯度将依赖图中的间接近回路径（通过 $z_t$ 和 $\tilde{h}_t$ 的计算），这些路径仍然涉及 $\tanh'$ 和权重矩阵连乘，可能衰减。

但实践中，更新门 $z_t$ 很少在所有维度上同时饱和到 $1$——模型通常会学习让某些维度保持较小的 $z_t$ 值以维护长距离记忆通道。

---

## 7. GRU vs LSTM 全面对比

### 7.1 定量对比

| 维度 | LSTM | GRU | 比率 |
|------|------|-----|------|
| 门数量 | 3 | 2 | 67% |
| 状态向量数 | 2 ($c_t$, $h_t$) | 1 ($h_t$) | 50% |
| 参数矩阵数 | 4 | 3 | 75% |
| 总参数量 | $4d_h(d_h+d_x)$ | $3d_h(d_h+d_x)$ | 75% |
| 每个时间步的矩阵乘法 | 8 (4 gates × 2 matrices) | 6 (3 gates × 2 matrices) | 75% |
| 非线性激活 | 5 ($\sigma \times 3$, $\tanh \times 2$) | 3 ($\sigma \times 2$, $\tanh \times 1$) | 60% |
| 逐元素运算 | 3 ($\odot \times 2$, $+$) | 2 ($\odot$, $+$) + 凸组合 | 基本相当 |

### 7.2 定性对比

| 维度 | LSTM | GRU |
|------|------|-----|
| **表达能力** | 略强（更多自由度） | 轻微受限（$f_t + i_t$ 不强制为 1） |
| **训练速度** | 较慢 | 约快 20-30% |
| **收敛速度** | 相似 | 通常略快 |
| **小数据集** | 易过拟合 | 较鲁棒（参数少） |
| **大数据集** | 可能略优 | 通常相当 |
| **实现复杂度** | 中等 | 较低 |
| **调参难度** | 中等 | 略低 |

### 7.3 经验结论

Chung 等人 (2014) 的大规模实证研究表明：

> GRU 和 LSTM 在各种序列建模任务上的性能差异通常不显著。在大多数情况下，两者的性能处于同一水平，细微的差异取决于具体任务和超参数设置。建议在实践中将两种模型都纳入实验范围。

---

## 8. GRU 变体

### 8.1 最小门控单元 (Minimal Gated Unit, MGU)

Zhou 等人 (2016) 提出了更激进的简化方案——只保留**一个门**，将遗忘门的功能也耦合进更新门：

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
\tilde{h}_t &= \tanh(W \cdot [f_t \odot h_{t-1}, x_t] + b) \\
h_t &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t
\end{aligned}
$$

MGU 只有一个门 $f_t$，同时承担 GRU 中 $z_t$（更新门）和 $r_t$（重置门）的功能。参数量减少到约 LSTM 的 50%，在某些任务上性能与 GRU 相当。

### 8.2 Light GRU (Li-GRU)

Ravanelli 等人 (2018) 提出了完全移除 $\tanh$ 的轻量 GRU：

$$
\begin{aligned}
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \\
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \\
\tilde{h}_t &= W \cdot [r_t \odot h_{t-1}, x_t] + b \quad \text{(无 tanh)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

完全线性化的候选计算极大加速了训练，在某些语音任务上性能几乎没有损失。

### 8.3 Bayesian GRU

通过在 GRU 的权重上施加先验分布（如 Gaussian）并估计后验，使 GRU 具备不确定性量化能力。在医疗、金融等对不确定性敏感的场景中有重要应用。

### 8.4 Conv-GRU

将 GRU 中矩阵乘法替换为卷积操作，用于处理具有空间结构的序列（如视频帧、气象雷达图）：

$$
\begin{aligned}
z_t &= \sigma(W_z * [h_{t-1}, x_t] + b_z) \\
r_t &= \sigma(W_r * [h_{t-1}, x_t] + b_r) \\
\tilde{h}_t &= \tanh(W * [r_t \odot h_{t-1}, x_t] + b) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

---

## 9. 应用场景

### 9.1 机器翻译 (Machine Translation)

GRU 的首次提出就是为了 SMT（统计机器翻译）中的 RNN Encoder–Decoder 架构。在英法翻译任务上，GRU Encoder–Decoder 显著优于传统的基于短语的 SMT 系统。

如今在 Transformer 主导的时代，GRU 仍偶有应用于：
- 低资源场景的轻量翻译模型
- 移动端 / 边缘设备的翻译系统（参数少、推理快）

### 9.2 语音识别

GRU 在语音任务中表现出色，尤其适合：
- **在线 / 流式识别**：单方向 GRU 而非双向，满足实时性要求
- **关键词检测 (Keyword Spotting)**：轻量 GRU 可部署在低功耗设备上

DeepSpeech 2 等系统使用 GRU 作为 RNN 层的首选单元。

### 9.3 时间序列预测

GRU 在时间序列任务中的优势：
- 相比 LSTM 更快的训练速度允许更大的超参数搜索空间
- 更少的参数降低了小样本金融 / 传感器数据上的过拟合风险
- 典型的预测框架：滑动窗口 GRU → 全连接层 → 单点 / 多点预测

### 9.4 推荐系统

会话推荐 (Session-based Recommendation) 中，GRU 用于建模用户的点击序列：

- 输入：用户点击的物品 ID 序列
- 输出：下一个可能点击的物品概率分布
- 代表性工作：GRU4Rec (Hidasi et al., 2016)

### 9.5 其他应用

- **音乐生成**：旋律序列建模
- **蛋白质结构预测**：氨基酸序列特征提取
- **自然语言理解**：意图识别、槽位填充
- **异常检测**：重构误差基线

---

## 10. 总结

### 10.1 GRU 的核心贡献

1. **极简设计**：用两个门实现了与 LSTM 相当的长距离记忆能力
2. **线性插值更新**：$(1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ 构建了梯度有效的凸组合通路
3. **参数量减少 25%**：在保持性能的前提下降低了计算开销和过拟合风险

### 10.2 关键公式速查

| 组件 | 公式 |
|------|------|
| 更新门 | $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$ |
| 重置门 | $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$ |
| 候选隐藏状态 | $\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$ |
| 最终隐藏状态 | $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ |
| 梯度传播（主通路） | $\frac{\partial h_t}{\partial h_{t-1}} \approx \text{diag}[1 - z_t]$ |

### 10.3 三者关系图

```
                    RNN
                     │
                     │ 解决梯度消失
                     ▼
              ┌──────┴──────┐
              │             │
              ▼             ▼
            LSTM          简化
         (1997, 3门)         │
              │             ▼
              │           GRU
              │        (2014, 2门)
              │             │
              │             │ 进一步简化
              │             ▼
              │           MGU
              │        (2016, 1门)
              │
              │ 增强
              ▼
        Peephole LSTM
        BiLSTM / Stacked LSTM
```

---

## 参考文献

1. Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation". *EMNLP*.
2. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling". *arXiv:1412.3555*.
3. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). "An Empirical Exploration of Recurrent Network Architectures". *ICML*.
4. Zhou, G. B., et al. (2016). "Minimal Gated Unit for Recurrent Neural Networks". *Information Sciences*.
5. Ravanelli, M., et al. (2018). "Light Gated Recurrent Units for Speech Recognition". *IEEE TETCI*.
6. Hidasi, B., et al. (2016). "Session-based Recommendations with Recurrent Neural Networks". *ICLR*.
7. Greff, K., et al. (2017). "LSTM: A Search Space Odyssey". *IEEE TNNLS*.
