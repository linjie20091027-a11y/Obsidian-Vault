# ResNet 残差网络

---

## 1. 背景与动机

2015 年，何恺明等人在《Deep Residual Learning for Image Recognition》中提出了 ResNet（Residual Network），并一举夺得 ILSVRC 2015 分类、检测、定位等多项冠军。

在此之前的共识是"网络越深，效果越好"，但实验揭示了一个令人困扰的矛盾现象。

---

## 2. 退化问题

### 2.1 问题描述

直觉上，更深的网络应该至少不差于浅层网络——因为深层网络可以通过将额外层学成恒等映射来实现与浅层网络完全相同的表达。然而实验发现：

> **网络深度增加到一定程度后，继续加深不仅不会提升准确率，训练误差和测试误差反而都会显著升高。**

这表明深层网络**无法被简单地优化**。

### 2.2 退化 vs 过拟合

| 现象 | 表现 | 原因 |
|------|------|------|
| **退化问题** | 训练误差升高（不只是测试误差） | 优化困难——深层网络难以有效学习恒等映射 |
| **过拟合** | 训练误差低但测试误差高 | 模型过于复杂，泛化能力差 |

这是两个本质不同的问题。退化问题说明深层网络连训练集都无法较好地拟合，是**优化问题**而非泛化问题。

### 2.3 深层网络为何难以优化

- **梯度消失/爆炸**：反向传播中梯度逐层连乘，导致前层梯度指数级衰减或爆炸（Batch Normalization 已有一定缓解）
- **恒等映射难以学习**：让堆叠的非线性层去逼近恒等映射 $\mathcal{H}(x) = x$ 在数值上是不稳定的，多层复合的非线性变换容易偏离恒等函数
- **信息衰减**：前向传播中信息经过多层非线性变换逐渐失真

---

## 3. 残差学习

### 3.1 核心思想

ResNet 的核心思想是**残差学习**。假设我们希望几层神经网络去拟合一个目标映射 $\mathcal{H}(x)$，不让它直接学习 $\mathcal{H}(x)$，而让它学习残差：

$$\mathcal{F}(x) = \mathcal{H}(x) - x$$

于是原始映射变为：

$$\mathcal{H}(x) = \mathcal{F}(x) + x$$

当 $\mathcal{H}(x)$ 近似恒等映射时（$\mathcal{H}(x) \approx x$），让网络将 $\mathcal{F}(x)$ 推向零比让网络直接用非线性层逼近恒等映射**容易得多**。

### 3.2 残差学习的形式化

给定一个深层网络的某几层（通常2-3层）构成一个残差块：

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

其中 $\mathcal{F}(\mathbf{x}, \{W_i\})$ 是待学习的残差映射，$+ \mathbf{x}$ 为**跳跃连接**（Shortcut Connection / Skip Connection）。

如果残差块的输入和输出维度不匹配，需要通过线性投影 $W_s$ 进行匹配：

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}$$

### 3.3 为什么残差学习有效

- **恒等映射成为合理的先验**：如果恒等映射已足够好，网络只需将残差部分推至零（比学习复杂的非线性恒等映射容易得多）
- **梯度高速公路**：跳跃连接为梯度提供了直接通道
- **更平滑的损失曲面**：残差结构使优化目标函数更加平滑

---

## 4. 残差块结构

### 4.1 基础残差块（Basic Block）

用于 ResNet-18/34，由两个 $3 \times 3$ 卷积层堆叠而成：

```
    x (输入)
    │
    ├─────────────── shortcut (恒等映射)
    │
    ▼
Conv2d(3×3, stride=s)
    ▼
BatchNorm
    ▼
ReLU
    ▼
Conv2d(3×3, stride=1)
    ▼
BatchNorm
    ▼
  + ◄────────────── shortcut
    ▼
ReLU
    ▼
  y (输出)
```

**前向传播公式**：

$$\mathbf{y} = \text{ReLU}(\text{BN}(\text{Conv}_2(\text{ReLU}(\text{BN}(\text{Conv}_1(\mathbf{x}))))) + \mathbf{x})$$

### 4.2 瓶颈残差块（Bottleneck Block）

用于 ResNet-50/101/152，使用 $1 \times 1 \to 3 \times 3 \to 1 \times 1$ 的瓶型结构：

```
    x (输入，通道数 C_in)
    │
    ├────────────────── shortcut
    │
    ▼
Conv2d(1×1, C_in → C_in/4)     ← 降维
    ▼
BatchNorm + ReLU
    ▼
Conv2d(3×3, C_in/4 → C_in/4)   ← 空间卷积
    ▼
BatchNorm + ReLU
    ▼
Conv2d(1×1, C_in/4 → C_in)     ← 升维
    ▼
BatchNorm
    ▼
  + ◄────────────────────── shortcut
    ▼
ReLU
    ▼
  y (输出)
```

**计算量分析**：

设输入特征图尺寸为 $H \times W \times 256$：

| 方案 | 结构 | 参数量 | 计算量 |
|------|------|--------|--------|
| 两个3×3卷积 (Basic) | `3×3,256 → 3×3,256` | $3^2\times 256\times 256 \times 2 = 1.18\text{M}$ | $H \times W \times 9 \times 256 \times 256 \times 2$ |
| 瓶颈块 (Bottleneck) | `1×1,256→64 → 3×3,64→64 → 1×1,64→256` | $\approx 70\text{K}$ | $H \times W \times (256\times 64 + 9\times 64\times 64 + 64\times 256)$ |

瓶颈结构使参数和计算量显著下降，但仍保持相当的表达能力，因此可用于构建非常深的网络。

### 4.3 Shortcut 连接的两种类型

| 类型 | 条件 | 实现 |
|------|------|------|
| **Identity Shortcut (A)** | 输入输出维度相同 | $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ |
| **Projection Shortcut (B)** | 维度变化（步长>1 或通道数变化） | $\mathbf{y} = \mathcal{F}(\mathbf{x}) + W_s \mathbf{x}$，$W_s$ 为 $1 \times 1$ 卷积 |

ResNet 论文中的选项 (B) 仅在维度改变时使用投影，其他情况使用恒等映射。

---

## 5. 恒等映射与梯度流动

### 5.1 梯度反向传播分析

考虑一个由 $L$ 个残差块组成的网络，每个块输出为：

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$$

递归展开，第 $L$ 层的特征可以表示为：

$$\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$$

这表明任何深层单元可以分解为**浅层特征 $\mathbf{x}_l$** 加上其所有中间层的**残差函数之和**。

### 5.2 反向传播中的梯度

对损失函数 $\mathcal{L}$，第 $l$ 层的梯度：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \left( 1 + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i) \right)$$

**关键观察**：

1. 括号中的 **$1$** 来自恒等连接的梯度，确保梯度无衰减地从深层直接传回浅层
2. $\frac{\partial}{\partial \mathbf{x}_l} \sum \mathcal{F}$ 项即使在极端情况下（权重极小）也很难让所有样本的梯度都为零
3. 梯度永远不会完全消失，因为 $1$ 的存在保证了信息传递

这解释了为什么 ResNet 能有效训练 1000 层以上的超深网络。

### 5.3 比 Plain Network 的优势

在 Plain Network（无跳跃连接）中：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \prod_{i=l}^{L-1} \frac{\partial \mathbf{x}_{i+1}}{\partial \mathbf{x}_i}$$

当 $\frac{\partial \mathbf{x}_{i+1}}{\partial \mathbf{x}_i} < 1$ 时，深层网络的梯度以指数级衰减——梯度消失不可避免。

---

## 6. Batch Normalization 的使用

### 6.1 在残差块中的位置

ResNet 采用**Pre-Activation**（先 BN 后 Conv）或**原始方案**（先 Conv 后 BN）两种方式。

**原始 ResNet（原论文）**：
```
Conv → BN → ReLU → Conv → BN → + shortcut → ReLU
```

**Pre-Activation ResNet（ResNet v2）**：
```
BN → ReLU → Conv → BN → ReLU → Conv → + shortcut
```

Pre-Activation 结构使恒等映射路径从输入到输出完全无阻碍，梯度流动更加顺畅。

### 6.2 Batch Normalization 公式

对于批次中的每个通道：

$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_{\mathcal{B}}^{(k)}}{\sqrt{\sigma_{\mathcal{B}}^{(k)^2} + \epsilon}}$$

$$y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$

其中 $\mu_{\mathcal{B}}$ 和 $\sigma_{\mathcal{B}}^2$ 为批次均值和方差，$\gamma$ 和 $\beta$ 为可学习参数。

---

## 7. ResNet 架构对比

### 7.1 完整架构表

| 层名 | 输出尺寸 | ResNet-18 | ResNet-34 | ResNet-50 | ResNet-101 | ResNet-152 |
|------|----------|-----------|-----------|-----------|------------|------------|
| conv1 | 112×112 | 7×7, 64, stride 2 | 同左 | 同左 | 同左 | 同左 |
| pool1 | 56×56 | 3×3 max pool, stride 2 | 同左 | 同左 | 同左 | 同左 |
| conv2_x | 56×56 | $\begin{bmatrix}3\times3, 64 \\ 3\times3, 64\end{bmatrix}\times 2$ | $\begin{bmatrix}3\times3, 64 \\ 3\times3, 64\end{bmatrix}\times 3$ | $\begin{bmatrix}1\times1, 64 \\ 3\times3, 64 \\ 1\times1, 256\end{bmatrix}\times 3$ | $\begin{bmatrix}1\times1, 64 \\ 3\times3, 64 \\ 1\times1, 256\end{bmatrix}\times 3$ | $\begin{bmatrix}1\times1, 64 \\ 3\times3, 64 \\ 1\times1, 256\end{bmatrix}\times 3$ |
| conv3_x | 28×28 | $\begin{bmatrix}3\times3, 128 \\ 3\times3, 128\end{bmatrix}\times 2$ | $\begin{bmatrix}3\times3, 128 \\ 3\times3, 128\end{bmatrix}\times 4$ | $\begin{bmatrix}1\times1, 128 \\ 3\times3, 128 \\ 1\times1, 512\end{bmatrix}\times 4$ | $\begin{bmatrix}1\times1, 128 \\ 3\times3, 128 \\ 1\times1, 512\end{bmatrix}\times 4$ | $\begin{bmatrix}1\times1, 128 \\ 3\times3, 128 \\ 1\times1, 512\end{bmatrix}\times 8$ |
| conv4_x | 14×14 | $\begin{bmatrix}3\times3, 256 \\ 3\times3, 256\end{bmatrix}\times 2$ | $\begin{bmatrix}3\times3, 256 \\ 3\times3, 256\end{bmatrix}\times 6$ | $\begin{bmatrix}1\times1, 256 \\ 3\times3, 256 \\ 1\times1, 1024\end{bmatrix}\times 6$ | $\begin{bmatrix}1\times1, 256 \\ 3\times3, 256 \\ 1\times1, 1024\end{bmatrix}\times 23$ | $\begin{bmatrix}1\times1, 256 \\ 3\times3, 256 \\ 1\times1, 1024\end{bmatrix}\times 36$ |
| conv5_x | 7×7 | $\begin{bmatrix}3\times3, 512 \\ 3\times3, 512\end{bmatrix}\times 2$ | $\begin{bmatrix}3\times3, 512 \\ 3\times3, 512\end{bmatrix}\times 3$ | $\begin{bmatrix}1\times1, 512 \\ 3\times3, 512 \\ 1\times1, 2048\end{bmatrix}\times 3$ | $\begin{bmatrix}1\times1, 512 \\ 3\times3, 512 \\ 1\times1, 2048\end{bmatrix}\times 3$ | $\begin{bmatrix}1\times1, 512 \\ 3\times3, 512 \\ 1\times1, 2048\end{bmatrix}\times 3$ |
| 输出 | 1×1 | 全局平均池化 → 1000-d fc → softmax | 同左 | 同左 | 同左 | 同左 |
| **总层数** | | **18** | **34** | **50** | **101** | **152** |
| **FLOPs** | | 1.8×10⁹ | 3.6×10⁹ | 3.8×10⁹ | 7.6×10⁹ | 11.3×10⁹ |

### 7.2 参数量对比

| 模型 | 参数量 |
|------|--------|
| ResNet-18 | 11.7M |
| ResNet-34 | 21.8M |
| ResNet-50 | 25.6M |
| ResNet-101 | 44.5M |
| ResNet-152 | 60.2M |

注意 ResNet-50 的参数量比 ResNet-34 略多，但 FLOPs 持平——瓶颈结构以更深的层数实现了更高的精度而不显著增加计算量。

---

## 8. 深度极限：ResNet-1001

ResNet 的理论框架支持极深网络的训练。He 等人实验验证了 ResNet-1001（1001 层）可在 CIFAR-10 上成功训练并取得良好结果，而同等深度的 Plain Network 完全无法收敛。

这也印证了：深度的限制主要在于**优化难度**而非表示能力，残差连接正是解决这一瓶颈的核心手段。

---

## 9. 训练策略

### 9.1 数据增强

- **随机水平翻转**
- **随机裁剪**：将图像调整至 $256 \times 256$ 后随机裁剪 $224 \times 224$
- **颜色抖动**
- **PCA 颜色增强**（AlexNet 提出，ResNet 也采用）

### 9.2 学习率调度

- 初始学习率 0.1
- 当验证误差 plateau 时将学习率除以 10
- 使用 Momentum SGD（momentum = 0.9）
- 权重衰减（Weight Decay）= 0.0001

### 9.3 其他技巧

- **Batch Normalization**：加速收敛，允许更大学习率
- **Xavier / Kaiming 初始化**：He 初始化（MSRA init）专门为 ReLU 设计
- **Label Smoothing**：软标签正则化

---

## 10. 变体与扩展

### 10.1 ResNeXt

引入**分组卷积**和**基数**（Cardinality）概念。将残差块中的路径分割为多个相同的并行分支（"Network-in-Neuron"）。

$$\mathbf{y} = \mathbf{x} + \sum_{i=1}^{C} \mathcal{T}_i(\mathbf{x})$$

其中 $C$ 为基数，$\mathcal{T}_i$ 为拓扑相同的变换。ResNeXt 证明了在相同计算量下，增加基数（宽度方向的分组数）比增加深度或宽度更有效。

### 10.2 Wide ResNet（WRN）

减少深度但大幅增加宽度（通道数），降低计算并行度上的瓶颈，训练更快且效果更佳。

### 10.3 ResNet-D / ResNet v2

- **Pre-Activation**：BN → ReLU → Conv 取代 Conv → BN → ReLU
- 下采样时用 $2 \times 2$ 平均池化配合 $1 \times 1$ 卷积替代 stride-2 的 $1 \times 1$ 投影，减少信息丢失

### 10.4 DenseNet

见"现代CNN架构"文档，密集连接可看作残差连接的泛化：每一层与所有前置层连接。

---

## 11. ResNet 的影响与意义

1. **解决了极深网络的优化问题**：首次让超过 100 层的网络训练成为现实
2. **残差连接成为深度学习基础设施**：后续几乎所有架构 (Transformer, UNet, BERT, GPT) 都采用了残差连接
3. **概念统一**：残差学习将恒等映射视为默认假设，强调了"学习增量变化"的思想
4. **工业落地**：ResNet-50/101 至今仍是计算机视觉任务中最常用的 Backbone 之一

---

## 12. 代码示意（PyTorch）

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

---

## 总结

ResNet 通过**残差学习**和**恒等跳跃连接**，从根本上解决了深层神经网络的优化难题。其核心思想——让梯度通过恒等映射无衰减传播——已成为现代深度学习的基石，影响深远。
