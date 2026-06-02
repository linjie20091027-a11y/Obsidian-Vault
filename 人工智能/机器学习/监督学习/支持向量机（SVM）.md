# 支持向量机（SVM）

## 一、核心思想

支持向量机（Support Vector Machine, SVM）的核心思想是：**在特征空间中寻找一个最优超平面，使得不同类别的样本点之间的距离（即分类间隔 Margin）最大化**。

直观理解：
- 给定二分类数据集，存在无数个超平面可以将两类样本分开
- SVM 选择使"分类最鲁棒"的那个超平面——离超平面最近的样本点（支持向量）到超平面的距离最大
- 决策边界仅由少数"支持向量"决定，与其余样本点无关

数学上，SVM 可归结为一个**凸二次规划（Quadratic Programming）**问题，存在全局最优解。

---

## 二、线性可分 SVM（硬间隔）

### 2.1 问题定义

给定训练集 $D = \{(x_i, y_i)\}_{i=1}^N$，其中 $x_i \in \mathbb{R}^d$，$y_i \in \{+1, -1\}$。

超平面方程为：
$$w^T x + b = 0$$

分类决策函数：
$$f(x) = \text{sign}(w^T x + b)$$

### 2.2 函数间隔与几何间隔

**函数间隔**（Functional Margin）：
$$\hat{\gamma}_i = y_i (w^T x_i + b)$$

函数间隔可随 $w, b$ 的等比例缩放而改变，不利于优化。

**几何间隔**（Geometric Margin）：
$$\gamma_i = \frac{y_i (w^T x_i + b)}{\|w\|}$$

几何间隔是样本点到超平面的真实欧氏距离，具有尺度不变性。

### 2.3 最大间隔优化问题

目标：最大化最小几何间隔。令最小函数间隔为 1，则原始问题为：

$$
\begin{aligned}
\min_{w, b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & y_i (w^T x_i + b) \geq 1, \quad i = 1, 2, \dots, N
\end{aligned}
$$

其中 $\frac{1}{2}\|w\|^2$ 的使用是为了求导方便（二次函数），约束条件保证了所有样本到超平面的函数间隔至少为 1。

**最大化 $\frac{1}{\|w\|}$ 等价于最小化 $\frac{1}{2}\|w\|^2$。**

---

## 三、拉格朗日对偶推导

### 3.1 拉格朗日函数

对每个约束引入拉格朗日乘子 $\alpha_i \geq 0$：

$$L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{N} \alpha_i \big[ y_i (w^T x_i + b) - 1 \big]$$

原始问题等价于：
$$\min_{w, b} \max_{\alpha: \alpha_i \geq 0} L(w, b, \alpha)$$

### 3.2 对偶问题推导

令 $L$ 对 $w, b$ 的偏导数为零：

$$\frac{\partial L}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \quad \Rightarrow \quad w = \sum_{i=1}^{N} \alpha_i y_i x_i$$

$$\frac{\partial L}{\partial b} = -\sum_i \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^{N} \alpha_i y_i = 0$$

代入 $L$ 消去 $w, b$：

$$
\begin{aligned}
L(w, b, \alpha) &= \frac{1}{2} \left\| \sum_i \alpha_i y_i x_i \right\|^2 - \sum_i \alpha_i \big[ y_i (w^T x_i + b) - 1 \big] \\
&= \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^T x_j) - \sum_i \alpha_i y_i w^T x_i - b \sum_i \alpha_i y_i + \sum_i \alpha_i \\
&= \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^T x_j) - \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^T x_j) + \sum_i \alpha_i \\
&= \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^T x_j)
\end{aligned}
$$

### 3.3 对偶问题形式

$$
\begin{aligned}
\max_{\alpha} \quad & \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i^T x_j) \\
\text{s.t.} \quad & \alpha_i \geq 0, \quad i = 1, \dots, N \\
& \sum_{i=1}^{N} \alpha_i y_i = 0
\end{aligned}
$$

**优点**：对偶问题的维数只依赖于样本数 $N$ 而非特征维度 $d$，且目标函数仅涉及样本间的内积，为核技巧埋下伏笔。

### 3.4 决策函数

求得最优 $\alpha^*$ 后：
$$w^* = \sum_i \alpha_i^* y_i x_i$$

$$b^* = y_j - \sum_i \alpha_i^* y_i (x_i^T x_j), \quad \text{其中 } \alpha_j^* > 0$$

最终分类决策函数：
$$f(x) = \text{sign}\left( \sum_{i=1}^{N} \alpha_i^* y_i (x_i^T x) + b^* \right)$$

---

## 四、支持向量

**支持向量（Support Vector）**定义为 $\alpha_i > 0$ 的样本点。

由 KKT 互补松弛条件：
$$\alpha_i \big[ y_i (w^T x_i + b) - 1 \big] = 0$$

可知：
- 若 $\alpha_i > 0$，则 $y_i (w^T x_i + b) = 1$，即样本落在间隔边界上
- 若 $\alpha_i = 0$，则 $y_i (w^T x_i + b) \geq 1$，即样本在间隔之外，不出现在最终的 $w$ 中

这意味着：**SVM 的解是稀疏的——只有少部分样本（支持向量）决定了决策边界**，其余样本可以被丢弃而不影响模型。

---

## 五、软间隔 SVM（线性不可分情形）

现实数据往往不是线性可分的。引入**松弛变量** $\xi_i \geq 0$ 和**惩罚参数** $C > 0$：

### 5.1 软间隔原始问题

$$
\begin{aligned}
\min_{w, b, \xi} \quad & \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \xi_i \\
\text{s.t.} \quad & y_i (w^T x_i + b) \geq 1 - \xi_i, \quad i = 1, \dots, N \\
& \xi_i \geq 0, \quad i = 1, \dots, N
\end{aligned}
$$

其中：
- $\xi_i$ 衡量第 $i$ 个样本违反约束的程度
- $C$ 是正则化参数，控制"间隔最大化"与"分类误差最小化"之间的权衡
  - $C \to \infty$：退化为硬间隔 SVM，不允许任何分类错误
  - $C$ 较小：允许更多样本落在间隔内或被错误分类，模型泛化能力更强

### 5.2 软间隔对偶问题

推导结果仅在 $\alpha_i$ 的约束上发生变化：

$$
\begin{aligned}
\max_{\alpha} \quad & \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i^T x_j) \\
\text{s.t.} \quad & 0 \leq \alpha_i \leq C, \quad i = 1, \dots, N \\
& \sum_{i=1}^{N} \alpha_i y_i = 0
\end{aligned}
$$

唯一区别：$\alpha_i$ 的上限从无穷变为 $C$。
- $\alpha_i = 0$：非支持向量
- $0 < \alpha_i < C$：位于间隔边界上的支持向量（$\xi_i = 0$）
- $\alpha_i = C$：位于间隔内部或被错误分类的支持向量（$\xi_i > 0$）

---

## 六、核技巧（Kernel Trick）

### 6.1 动机

线性 SVM 无法处理非线性可分数据。解决方案：**将数据映射到高维空间，在高维空间中寻找线性超平面**。

设映射函数 $\phi: \mathbb{R}^d \to \mathcal{H}$（高维特征空间），则样本内积变为 $\phi(x_i)^T \phi(x_j)$。

### 6.2 核函数定义

**核函数**直接计算低维空间中函数值，而无需显式计算高维内积：

$$K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle = \phi(x_i)^T \phi(x_j)$$

对偶问题中的内积 $(x_i^T x_j)$ 替换为 $K(x_i, x_j)$：

$$\max_{\alpha} \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

决策函数：
$$f(x) = \text{sign}\left( \sum_i \alpha_i^* y_i K(x_i, x) + b^* \right)$$

### 6.3 Mercer 定理

一个对称函数 $K(x, z)$ 可以作为核函数的充要条件：对于任意有限样本，对应的 Gram 矩阵是半正定的。

---

## 七、常用核函数对比

| 核函数 | 表达式 | 参数 | 特点 |
|--------|--------|------|------|
| **线性核** | $K(x, y) = x^T y$ | 无 | 最简单，等价于线性 SVM；适用于高维稀疏数据（如文本） |
| **多项式核** | $K(x, y) = (x^T y + c)^d$ | $d$：阶数，$c$：常数 | $d$ 越大映射空间维度越高，易过拟合；$d=1$ 退化为线性核 |
| **RBF（高斯）核** | $K(x, y) = \exp\left(-\gamma \|x - y\|^2\right)$ | $\gamma > 0$ | 最常用；可映射到无穷维空间；$\gamma$ 越大决策边界越复杂 |
| **Sigmoid 核** | $K(x, y) = \tanh(\kappa x^T y + \theta)$ | $\kappa > 0, \theta < 0$ | 源自神经网络激活函数；不总是半正定（使用时需谨慎） |

### 参数影响

- **$C$ （惩罚参数）**：$C$ 越大 → 分类错误惩罚越大 → 决策边界试图正确分类所有样本 → 可能过拟合
- **$\gamma$（RBF 参数）**：$\gamma$ 越大 → 单个样本影响范围越小 → 决策边界越扭曲 → 可能过拟合

通常通过交叉验证选择最优 $(C, \gamma)$ 组合。

---

## 八、SMO 算法简介

SMO（Sequential Minimal Optimization）由 Platt (1998) 提出，是求解 SVM 对偶问题的高效算法。

### 核心思想

将大型 QP 问题分解为一系列最小的 QP 子问题（每次只优化两个 $\alpha_i$），解析求解二变量 QP 问题。

### 算法步骤

1. **选择两个 $\alpha_i, \alpha_j$** 作为优化变量（启发式选择：一个是违反 KKT 条件最严重的，另一个由最大步长期望决定）
2. **固定其他 $\alpha$**，对这两个变量求解析解（利用约束 $\sum \alpha_i y_i = 0$ 的二变量形式）
3. **更新阈值 $b$** 和误差缓存
4. **重复**直到所有 $\alpha$ 满足 KKT 条件

### 优势

- 每次迭代计算量小（O(1) 解析解），无需矩阵存储
- 内存占用 O(N)，适合大规模问题
- 与核技巧天然兼容

---

## 九、多分类 SVM

SVM 本质是二分类器。扩展到多类常用两种策略：

1. **一对多（One-vs-Rest, OvR）**：训练 $K$ 个 SVM，每个将第 $k$ 类与其余类分开。预测时取置信度最高的类别。
2. **一对一（One-vs-One, OvO）**：训练 $K(K-1)/2$ 个 SVM，每两个类之间训练一个。预测时投票决定。

---

## 十、优缺点

### 优点

- **全局最优解**：凸优化问题，不存在局部极小值
- **稀疏解**：仅依赖支持向量，模型存储和预测高效
- **泛化能力强**：以最大化间隔为目标，基于结构风险最小化
- **核技巧**：可处理非线性问题，无需显式特征映射
- **理论完备**：有 VC 维理论支撑
- **高维数据表现好**：适合特征维度远大于样本数的情况

### 缺点

- **对参数敏感**：$C$ 和核参数需要通过交叉验证精细调优
- **核矩阵计算量大**：$O(N^2)$ 的存储和计算，不适合大规模数据（>$10^5$ 样本）
- **可解释性差**：相比决策树和线性模型，非线性 SVM 的决策机制不易解释
- **概率输出困难**：需要额外的 Platt Scaling 等后处理才能输出概率
- **对缺失值敏感**：需要预处理填充缺失值
- **类别不平衡**：SVM 对类别不平衡较为敏感，需要设置类别权重

---

## 十一、与其他算法的联系

| 对比算法 | 联系与区别 |
|----------|-----------|
| **逻辑回归** | 都是线性分类器。LR 最小化交叉熵损失（概率输出），SVM 最大化间隔（Hinge Loss）；SVM 只关心边界附近样本，LR 受所有样本影响 |
| **感知机** | 感知机只要求正确分类，解不唯一且由初值和样本顺序决定；SVM 求解唯一且最大化间隔 |
| **kNN** | kNN 是非参数懒惰学习（无显式训练），SVM 是有参数 eager 学习；高维空间中 SVM 优于 kNN |
| **神经网络** | 单层 SVM 等价于单层神经网络；核 SVM 可视为有隐层的神经网络（隐层大小 = 支持向量数）；深层网络学习层级特征，SVM 需要手工选核 |
| **高斯过程** | 对于特定核函数，SVM 与 GP 有等价性，但 SVM 给出稀疏点预测，GP 给出完整的预测分布 |
| **L2 正则化** | SVM 的 $\frac{1}{2}\|w\|^2$ 本质上就是 L2 正则化，与岭回归的目标类似 |

---

## 十二、应用场景

| 领域 | 应用 |
|------|------|
| **文本分类** | 垃圾邮件检测、新闻分类、情感分析（线性 SVM + TF-IDF 特征） |
| **图像识别** | 手写数字识别（MNIST）、人脸识别（早期方法，现多为 CNN 取代） |
| **生物信息学** | 基因表达分类、蛋白质结构预测、疾病诊断 |
| **异常检测** | One-Class SVM 用于异常检测和离群点发现 |
| **自然语言处理** | 命名实体识别、文本分块、句法分析 |
| **时间序列** | SVM 回归（SVR）用于金融时间序列预测、能源负荷预测 |

---

## 十三、核心公式速查

### Hinge Loss 形式

SVM 原始问题与 Hinge Loss 等价：
$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \max\big(0, 1 - y_i (w^T x_i + b)\big)$$

### 核化决策函数

$$f(x) = \text{sign}\left( \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b \right)$$

其中 $SV$ 表示支持向量下标集合。

### RBF 核的重要性质

$$\exp\left(-\gamma \|x - y\|^2\right) = \exp\big(-\gamma(\|x\|^2 + \|y\|^2 - 2x^T y)\big)$$

当 $\gamma$ 取 $\frac{1}{2\sigma^2}$ 时，RBF 核等价于高斯分布的内积形式。

---

## 十四、SVR（支持向量回归）简述

SVM 也可用于回归，称为 SVR（Support Vector Regression）：

目标：在 $\varepsilon$ 不敏感损失函数下，寻找一个扁平（flat）函数，使其与所有训练样本的偏差在 $\varepsilon$ 以内。

$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} L_{\varepsilon}\big(y_i - f(x_i)\big)$$

其中 $L_{\varepsilon}(u) = \max(0, |u| - \varepsilon)$ 称为 $\varepsilon$-不敏感损失。

---

> **参考文献**：
> - Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
> - Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines.
> - Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
> - 李航. (2019). 《统计学习方法》(第2版). 清华大学出版社.
