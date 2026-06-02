# 现代 CNN 架构：DenseNet、MobileNet、EfficientNet

---

## 第一部分：DenseNet

### 1. 概述

DenseNet（Densely Connected Convolutional Networks, 2017 CVPR Best Paper）由黄高等人提出，核心思想是**密集连接**：每一层都接收其前面所有层的输出作为输入。

$$x_l = H_l([x_0, x_1, x_2, ..., x_{l-1}])$$

其中 $[x_0, x_1, ..., x_{l-1}]$ 表示对第 $0$ 到第 $l-1$ 层特征图在通道维度上的**拼接**（Concatenation），$H_l(\cdot)$ 为复合函数（BN → ReLU → Conv）。

### 2. 核心设计：密集连接

#### 2.1 与传统网络的对比

| 网络类型 | 连接方式 | 第 $l$ 层的连接数 |
|----------|----------|-------------------|
| **Plain CNN** | $x_l = H_l(x_{l-1})$ | 1 |
| **ResNet（加性）** | $x_l = H_l(x_{l-1}) + x_{l-1}$ | 1（加法不增加连接） |
| **DenseNet（拼接）** | $x_l = H_l([x_0, x_1, ..., x_{l-1}])$ | $l$ |

#### 2.2 密集连接的公式化

在 DenseNet 的每一个密集块（Dense Block）中：

- 第 $l$ 层的输入是前面所有层输出的**通道维度拼接**
- 若每一层产生 $k$ 个特征图（$k$ 称为**增长率**），则第 $l$ 层的输入通道数为 $k_0 + k \times (l-1)$

$$
x_l = H_l\left([x_0 \;|\; x_1 \;|\; ... \;|\; x_{l-1}]\right)
$$

其中 $[\cdot|\cdot]$ 表示通道维度上的拼接。

#### 2.3 Dense Block 内部结构

```
输入 x_l（通道数 k₀ + k×(l-1)）
    │
    ▼
Batch Normalization
    ▼
ReLU
    ▼
Conv 1×1（降维到 4k 通道）
    ▼
Batch Normalization
    ▼
ReLU
    ▼
Conv 3×3（输出 k 个通道）
    ▼
输出 k 个特征图 → 拼接到已有的特征图集合中
```

注意：实际代码中通常在 $3 \times 3$ Conv 前加 $1 \times 1$ Conv 降维（Bottleneck 层），形成 **DenseNet-B** 变体。

### 3. 增长率 k

**增长率（Growth Rate）** $k$ 是每个 $H_l$ 贡献的新特征图数量。

$$k = \text{# of output channels per layer}$$

DenseNet 的一个反直觉特性是：$k$ 可以非常小（典型值 $k = 12, 24, 32$ 或 $40$），因为每一层都能访问所有前置层的"集体知识"。

**小增长率的优势**：

- 每一层向"集体知识库"添加少量新信息
- 全局状态和新增信息的明确区分
- 非常窄的层（$k = 12$），但总通道数依然增长，信息流极其丰富

### 4. 过渡层

在密集块之间插入**过渡层**（Transition Layer）进行降维和下采样：

$$\text{Transition: BN → ReLU → 1×1 Conv → 2×2 Average Pooling}$$

关键参数 $\theta$（压缩因子，$0 < \theta \leq 1$）：

若密集块输出 $m$ 个特征图，过渡层的 $1 \times 1$ 卷积输出 $\lfloor \theta m \rfloor$ 个特征图。

- $\theta = 1$：不压缩（标准 DenseNet）
- $\theta < 1$：压缩变体（**DenseNet-C**）

通常 $\theta = 0.5$，即过渡层将特征图数量减半。

**组合变体**：
| 名称 | 1×1 Bottleneck | 过渡层压缩 |
|------|---------------|-----------|
| DenseNet | ✗ | ✗ |
| DenseNet-B | ✓ ($H_l$ 中使用 1×1) | ✗ |
| DenseNet-C | ✗ | ✓ ($\theta < 1$) |
| DenseNet-BC | ✓ | ✓ |

### 5. DenseNet 的优势

#### 5.1 缓解梯度消失

每一层直接连接到损失函数的梯度信号（通过更短的路径），改善了深层网络的梯度流动：

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} = \frac{\partial L}{\partial x_L} \cdot \frac{\partial H_L}{\partial H_l} \cdot \frac{\partial H_l}{\partial x_l} + \text{（更多短路径）}$$

密集连接创建了从损失函数到每一层的**直接监督信号**。

#### 5.2 特征复用

不同于 ResNet 中残差分支学习"增量"，DenseNet 中每一层都可以直接访问所有前置层的原始输出（而非求和混合后的结果），实现了更彻底的特征复用。

#### 5.3 参数效率

由于每层非常窄（$k = 12$），且特征被高度复用，DenseNet 达到同等性能所需的参数量远小于 ResNet：

| 模型 | 参数量 | CIFAR-10 错误率 (无数据增强) |
|------|--------|------------------------------|
| DenseNet-100 (k=12) | 0.8M | 5.77% |
| ResNet-1001 | 10.2M | 4.92% |
| DenseNet-250 (k=24) | 15.3M | 3.62% |

### 6. DenseNet 架构表

| 层 | 输出尺寸 | DenseNet-121 | DenseNet-169 | DenseNet-201 | DenseNet-264 |
|----|----------|--------------|--------------|--------------|--------------|
| Conv | 112×112 | 7×7 conv, stride 2 | 同左 | 同左 | 同左 |
| Pool | 56×56 | 3×3 max pool, stride 2 | 同左 | 同左 | 同左 |
| Block1 | 56×56 | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 6$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 6$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 6$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 6$ |
| Trans1 | 28×28 | 1×1 conv, 2×2 avg pool | 同左 | 同左 | 同左 |
| Block2 | 28×28 | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 12$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 12$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 12$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 12$ |
| Trans2 | 14×14 | 同左 | 同左 | 同左 | 同左 |
| Block3 | 14×14 | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 24$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 32$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 48$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 64$ |
| Trans3 | 7×7 | 同左 | 同左 | 同左 | 同左 |
| Block4 | 7×7 | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 16$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 32$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 32$ | $\begin{bmatrix}1\times1 \\ 3\times3\end{bmatrix}\times 48$ |
| Classify | 1×1 | 7×7 GAP → FC 1000 | 同左 | 同左 | 同左 |

---

## 第二部分：MobileNet 系列

### 7. MobileNet v1（2017）

#### 7.1 深度可分离卷积

MobileNet v1 的核心是**深度可分离卷积**（Depthwise Separable Convolution），将标准卷积分解为两个步骤：

**步骤 1：深度卷积**
$$\hat{G}_{k,l,m} = \sum_{i,j} \hat{K}_{i,j,m} \cdot F_{k+i-1, l+j-1, m}$$

每个通道独立进行空间卷积，卷积核为 $D_K \times D_K \times 1$。

**步骤 2：逐点卷积**
$$G_{k,l,n} = \sum_{m} \hat{G}_{k,l,m} \cdot P_{m,n}$$

用 $1 \times 1$ 卷积将深度卷积的输出按通道混合，组合为新的特征图。

#### 7.2 计算量对比

设输入为 $D_F \times D_F \times M$，输出为 $D_F \times D_F \times N$，卷积核尺寸 $D_K \times D_K$。

**标准卷积**：
$$\text{Cost}_{std} = D_K \times D_K \times M \times N \times D_F \times D_F$$

**深度可分离卷积**：
$$\text{Cost}_{ds} = \underbrace{D_K \times D_K \times M \times D_F \times D_F}_{\text{深度卷积}} + \underbrace{M \times N \times D_F \times D_F}_{\text{逐点卷积}}$$

**计算量缩减比**：
$$\frac{\text{Cost}_{ds}}{\text{Cost}_{std}} = \frac{D_K^2 M D_F^2 + M N D_F^2}{D_K^2 M N D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}$$

以 $D_K = 3$ 为例，计算量减少约 **8~9 倍**，精度损失极小（ImageNet 上约 1%）。

#### 7.3 MobileNet v1 架构

```
Conv / s=2 → DW Conv / s=1 → 1×1 Conv  (×1)
DW Conv / s=2 → 1×1 Conv              (×1)
DW Conv / s=1 → 1×1 Conv              (×2)
DW Conv / s=2 → 1×1 Conv              (×1)
DW Conv / s=1 → 1×1 Conv              (×2)
DW Conv / s=2 → 1×1 Conv              (×1)
DW Conv / s=1 → 1×1 Conv              (×6)
DW Conv / s=2 → 1×1 Conv              (×1)
DW Conv / s=1 → 1×1 Conv              (×2)
Avg Pool → FC → Softmax
```

所有层后接 BN + ReLU。

#### 7.4 两个超参数：宽度乘子与分辨率乘子

**宽度乘子 $\alpha \in (0, 1]$**：

控制每一层的通道数缩放，将输入/输出通道 $M$ 变为 $\alpha M$。

$$\text{Cost}_{\alpha} = D_K \times D_K \times \alpha M \times D_F \times D_F + \alpha M \times \alpha N \times D_F \times D_F$$

典型值：$\alpha = 1, 0.75, 0.5, 0.25$，计算量约缩减为 $\alpha^2$。

**分辨率乘子 $\rho \in (0, 1]$**：

将输入分辨率 $D_F$ 变为 $\rho D_F$。

$$\text{Cost}_{\alpha,\rho} = D_K \times D_K \times \alpha M \times \rho D_F \times \rho D_F + \alpha M \times \alpha N \times \rho D_F \times \rho D_F$$

典型值：$\rho = 1, 0.857, 0.714, 0.571$（对应 224, 192, 160, 128 分辨率）。

### 8. MobileNet v2（2018）

#### 8.1 线性瓶颈

MobileNet v2 基于一个重要观察：**在低维空间中，非线性（ReLU）会破坏信息**。

对于一个 $n$ 维子空间（低维流形），经过 ReLU 变换后，信息被压缩在非负半空间的一个子集中。如果输入流形恰好位于输入空间的低维子空间，ReLU 可以保持信息完整性——但前提是输入流形纬度必须足够低。

**解决方案**：将 ReLU 去除，在瓶颈层使用**线性激活**。

#### 8.2 倒残差结构

与 ResNet 的瓶颈块相反（宽→窄→宽），MobileNet v2 使用**倒残差**（窄→宽→窄）：

```
输入 (低维, k 通道)
    │
    ▼
1×1 Conv (扩展为 t×k 通道)
    ▼
BN + ReLU6
    ▼
3×3 DW Conv (深度卷积, stride=s)
    ▼
BN + ReLU6
    ▼
1×1 Conv (压缩为 k' 通道)
    ▼
BN (线性激活, 无ReLU!)
    ▼
  + ◄──────── shortcut (当 s=1 且 k=k' 时)
```

其中 $t$ 为**扩展比**（Expansion Ratio），典型值 $t = 6$。

**ReLU6**：$f(x) = \min(\max(0, x), 6)$，在低精度计算中具有更好的数值鲁棒性。

#### 8.3 为什么使用线性瓶颈

在低维空间应用 ReLU 会导致信息不可逆的丢失（因为 ReLU 将所有负值置零）。在瓶颈处使用线性激活，可以保留低维流形的完整信息，提升模型表达能力。

从数学角度：设瓶颈输出维度为 $d$，输入流形的本征维度为 $d_{man}$。当 $d < d_{man}$ 时，ReLU 不可避免地丢失信息；线性瓶颈不引入这种信息损失。

#### 8.4 MobileNet v2 完整架构

| 输入尺寸 | 操作 | $t$ | $c$ | $n$ | $s$ |
|----------|------|-----|-----|-----|-----|
| 224²×3 | Conv2d | - | 32 | 1 | 2 |
| 112²×32 | Bottleneck | 1 | 16 | 1 | 1 |
| 112²×16 | Bottleneck | 6 | 24 | 2 | 2 |
| 56²×24 | Bottleneck | 6 | 32 | 3 | 2 |
| 28²×32 | Bottleneck | 6 | 64 | 4 | 2 |
| 14²×64 | Bottleneck | 6 | 96 | 3 | 1 |
| 14²×96 | Bottleneck | 6 | 160 | 3 | 2 |
| 7²×160 | Bottleneck | 6 | 320 | 1 | 1 |
| 7²×320 | Conv2d 1×1 | - | 1280 | 1 | 1 |
| 7²×1280 | AvgPool 7×7 | - | - | 1 | - |
| 1²×1280 | Conv2d 1×1 | - | k | - | - |

其中 $t$ = 扩展比，$c$ = 输出通道数，$n$ = 重复次数，$s$ = 步长。

#### 8.5 性能对比

| 模型 | Top-1 Accuracy | Params | Mult-Adds (M) |
|------|----------------|--------|---------------|
| MobileNet v1 | 70.6% | 4.2M | 575 |
| MobileNet v2 | 72.0% | 3.4M | **300** |
| MobileNet v2 (1.4×) | 74.7% | 6.9M | 585 |
| ShuffleNet v2 (2×) | 74.9% | ~7.4M | 591 |

MobileNet v2 以更少的参数和更低的计算量（300M MAdds）超越了 MobileNet v1（575M MAdds）的精度。

### 9. MobileNet v3 (2019)

结合**神经架构搜索（NAS）+ NetAdapt** 优化网络结构，并使用 **h-swish** 替代 ReLU6：

$$\text{h-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$

h-swish 在深层效果好，但在计算上更友好（相比标准 swish 需要计算 sigmoid）。MobileNet v3 还引入 Squeeze-and-Excitation (SE) 模块。

---

## 第三部分：EfficientNet

### 10. 概述

EfficientNet（Tan & Le, 2019）回答了一个关键问题：**是否存在一种系统性的方法来放大 CNN 以获得更好的精度和效率？**

### 11. 复合缩放方法

传统的网络放大方式有三种独立的维度：

- **深度 $d$**：增加网络层数（如 ResNet-18 → ResNet-200）
- **宽度 $w$**：增加每层通道数（如 Wide ResNet）
- **分辨率 $r$**：增大输入图像尺寸（如 224² → 299² → 380²）

EfficientNet 的核心观察是：**这三个维度不是独立的，应协同缩放**。

#### 11.1 复合缩放公式

$$\begin{aligned}
\text{depth:} \quad & d = \alpha^{\phi} \\
\text{width:} \quad & w = \beta^{\phi} \\
\text{resolution:} \quad & r = \gamma^{\phi}
\end{aligned}$$

满足约束：
$$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$$
$$\alpha \geq 1, \beta \geq 1, \gamma \geq 1$$

其中 $\phi$ 是**复合系数**（用户指定），控制模型整体扩大倍数；$\alpha, \beta, \gamma$ 通过在小基线模型上的**网格搜索**确定。

**理解约束条件**：深度、宽度和分辨率分别以 $\alpha^{\phi}, \beta^{\phi}, \gamma^{\phi}$ 的倍数缩放，对应的 FLOPS 近似以 $(\alpha \cdot \beta^2 \cdot \gamma^2)^{\phi}$ 的倍数增长。约束 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ 确保当 $\phi$ 增加 1 时，总计算量约翻倍。

#### 11.2 搜索到的缩放系数

对于 EfficientNet-B0（基线模型）：

| 系数 | 值 | 含义 |
|------|-----|------|
| $\alpha$ | 1.2 | 深度缩放系数 |
| $\beta$ | 1.1 | 宽度缩放系数 |
| $\gamma$ | 1.15 | 分辨率缩放系数 |

### 12. EfficientNet-B0 基线架构

EfficientNet-B0 通过 NAS（多目标神经网络架构搜索）发现，以准确率和 FLOPS 为优化目标：

- 使用 **MBConv 块**（Mobile Inverted Bottleneck Conv，即 MobileNet v2 + SE 模块）
- 共 18 个 MBConv 层 + 2 个标准卷积
- 约 5.3M 参数

**MBConv 块结构**：

```
输入
  │
  ▼
1×1 Conv (扩展, 升维)
  ▼
BN + Swish
  ▼
k×k DW Conv (深度卷积)
  ▼
BN + Swish
  ▼
Squeeze-and-Excitation (SE) 模块
  ▼
1×1 Conv (投影, 降维)
  ▼
BN + Dropout
  ▼
  + ◄──────── shortcut (如果维度匹配且 stride=1)
```

**Swish 激活函数**：
$$\text{swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

### 13. EfficientNet 系列

| 模型 | $\phi$ | 输入分辨率 | 参数量 | FLOPs | ImageNet Top-1 |
|------|--------|-----------|--------|-------|-----------------|
| B0 | 0 | 224² | 5.3M | 0.39G | 77.1% |
| B1 | 1 | 240² | 7.8M | 0.70G | 79.1% |
| B2 | 2 | 260² | 9.2M | 1.0G | 80.1% |
| B3 | 3 | 300² | 12M | 1.8G | 81.6% |
| B4 | 4 | 380² | 19M | 4.2G | 82.9% |
| B5 | 5 | 456² | 30M | 9.9G | 83.6% |
| B6 | 6 | 528² | 43M | 19G | 84.0% |
| B7 | 7 | 600² | 66M | 37G | **84.3%** |
| L2 (Noisy Student) | - | 800² | 480M | - | 88.4% |

EfficientNet 以显著更少的参数和 FLOPS 达到了与当时 SOTA 相当甚至更好的精度。

### 14. 性能对比总览

| 模型 | 参数量 | FLOPs | ImageNet Top-1 | 年份 |
|------|--------|-------|----------------|------|
| ResNet-50 | 25.6M | 3.8G | 76.2% | 2015 |
| DenseNet-201 | 20.0M | 4.3G | 77.4% | 2017 |
| MobileNet v2 | 3.4M | 0.3G | 72.0% | 2018 |
| MobileNet v3 Large | 5.4M | 0.22G | 75.2% | 2019 |
| EfficientNet-B0 | 5.3M | 0.39G | 77.1% | 2019 |
| EfficientNet-B4 | 19M | 4.2G | 82.9% | 2019 |
| EfficientNet-B7 | 66M | 37G | 84.3% | 2019 |

---

## 第四部分：深度可分离卷积深入

### 15. 标准卷积 vs 深度可分离卷积（可视化）

**标准卷积**（Conv $D_K \times D_K \times M \times N$）：
```
一个卷积核对所有输入通道做空间卷积，所有通道结果求和得到一个输出通道。
N 个卷积核 → N 个输出通道。
```

**深度可分离卷积**：
```
步骤1 [深度卷积]: M 个 D_K×D_K×1 的卷积核，各对各通道做空间滤波。
步骤2 [逐点卷积]: N 个 1×1×M 的卷积核，混合各通道信息。
```

### 16. 实际计算量公式对比

设输入 $D_F \times D_F \times M$，输出 $D_F \times D_F \times N$，$D_K = 3$。

| 类型 | 计算公式 | 典型值 ($N=128, M=64, D_F=56$) |
|------|----------|------|
| 标准卷积 | $D_K^2 \times M \times N \times D_F^2$ | $9 \times 64 \times 128 \times 3136 \approx 231\text{M}$ |
| 深度卷积 | $D_K^2 \times M \times D_F^2$ | $9 \times 64 \times 3136 \approx 1.8\text{M}$ |
| 逐点卷积 | $M \times N \times D_F^2$ | $64 \times 128 \times 3136 \approx 25.7\text{M}$ |
| **合计 (DS Conv)** | **约 27.5M** | **减少约 8.4 倍** |

---

## 总结

| 架构 | 核心思想 | 技术特点 |
|------|----------|----------|
| **DenseNet** | 密集连接 (特征拼接) | 参数效率极高，梯度流动最优，特征高度复用 |
| **MobileNet v1** | 深度可分离卷积 | 计算量降至标准卷积的 ~1/9，适合移动端 |
| **MobileNet v2** | 线性瓶颈 + 倒残差 | 低维信息保护，比 v1 更少参数、更高精度 |
| **EfficientNet** | 复合缩放 (depth × width × resolution) | NAS 搜索基线 + 维度协调放大，极致的效率-精度平衡 |

四种架构代表了 CNN 从"做大做深"到"做精做高效"的发展脉络，共同体现了现代 CNN 设计的核心理念：**在给定的计算/存储预算下，最大化信息提取和特征复用效率**。
