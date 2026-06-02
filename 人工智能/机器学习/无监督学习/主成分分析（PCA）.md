# 主成分分析（PCA）

## 1. 核心思想

主成分分析（Principal Component Analysis, PCA）是一种线性降维方法。其核心思想是：**通过正交变换，将原始数据投影到方差最大的方向上，在尽可能保留数据信息的前提下实现降维**。

直观理解：如果数据在某个方向上散布得很开（方差大），这个方向就包含更多信息；如果数据在某个方向上几乎不变（方差小），这个方向可以忽略。

---

## 2. 数学推导

### 2.1 数据中心化

给定数据矩阵 $X \in \mathbb{R}^{n \times d}$，$n$ 个样本，$d$ 维特征。首先将数据中心化：

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

$$\tilde{x}_i = x_i - \bar{x}$$

中心化后的数据矩阵记为 $\tilde{X}$，满足 $\sum_{i=1}^{n} \tilde{x}_i = 0$。

### 2.2 协方差矩阵

计算中心化数据的协方差矩阵：

$$C = \frac{1}{n} \tilde{X}^T \tilde{X}$$

其中 $C \in \mathbb{R}^{d \times d}$，$C_{ij}$ 表示第 $i$ 个特征与第 $j$ 个特征之间的协方差。

协方差矩阵的性质：
- 对称矩阵：$C = C^T$
- 半正定矩阵：$v^T C v \geq 0$，对任意 $v$
- 对角线元素为各特征的方差

### 2.3 特征值分解

对协方差矩阵 $C$ 做特征值分解：

$$C v = \lambda v$$

其中：
- $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$ 为特征值
- $v_1, v_2, \ldots, v_d$ 为对应的标准正交特征向量（$v_i^T v_j = \delta_{ij}$）

### 2.4 选取前 k 个主成分

取前 $k$ 个最大特征值对应的特征向量 $V_k = [v_1, v_2, \ldots, v_k] \in \mathbb{R}^{d \times k}$，则降维后的数据为：

$$Z = \tilde{X} V_k$$

其中 $Z \in \mathbb{R}^{n \times k}$，实现了从 $d$ 维到 $k$ 维的降维（$k \ll d$）。

---

## 3. 方差最大化视角

PCA 的目标是找到投影方向 $w$，使得投影后数据的方差最大化。

投影后数据：$z_i = w^T \tilde{x}_i$

投影后方差：

$$\text{Var}(z) = \frac{1}{n} \sum_{i=1}^{n} (w^T \tilde{x}_i)^2 = \frac{1}{n} w^T \tilde{X}^T \tilde{X} w = w^T C w$$

目标为最大化 $w^T C w$，约束 $w^T w = 1$（单位向量，方向唯一）。

### 拉格朗日乘子法

构造拉格朗日函数：

$$\mathcal{L}(w, \lambda) = w^T C w - \lambda (w^T w - 1)$$

对 $w$ 求偏导并令其为零：

$$\frac{\partial \mathcal{L}}{\partial w} = 2C w - 2\lambda w = 0$$

$$\Rightarrow C w = \lambda w$$

这正是协方差矩阵的特征值方程。取 $\lambda$ 最大对应的特征向量就是第一个主成分方向。推广到多个主成分，取前 $k$ 个最大特征值对应的特征向量。

此时最大方差值等于对应的特征值：$w_1^T C w_1 = \lambda_1$。

---

## 4. 最小重构误差视角

PCA 也可以从最小化重构误差的角度来理解。

设 $W_k \in \mathbb{R}^{d \times k}$ 为前 $k$ 个主成分构成的正交矩阵（$W_k^T W_k = I_k$）。

投影：$Z = \tilde{X} W_k$（编码）
重构：$\hat{X} = Z W_k^T = \tilde{X} W_k W_k^T$（解码）

重构误差：

$$\min_{W_k} \|\tilde{X} - \tilde{X} W_k W_k^T\|_F^2$$

其中 $\|\cdot\|_F$ 为 Frobenius 范数。

可以证明，这个优化问题的最优解正是取前 $k$ 个最大特征值对应的特征向量（Eckart-Young 定理的特殊情形）。也就是说，**方差最大化** 与 **重构误差最小化** 是等价的目标。

---

## 5. 方差解释率

第 $i$ 个主成分的方差解释率：

$$\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

前 $k$ 个主成分的累计方差解释率：

$$\text{Cumulative EVR}_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

---

## 6. 奇异值分解（SVD）视角

对中心化数据矩阵 $\tilde{X}$ 做 SVD：

$$\tilde{X} = U \Sigma V^T$$

其中：
- $U \in \mathbb{R}^{n \times n}$：左奇异向量矩阵
- $\Sigma \in \mathbb{R}^{n \times d}$：奇异值对角矩阵（对角元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(n,d)} \geq 0$）
- $V \in \mathbb{R}^{d \times d}$：右奇异向量矩阵

协方差矩阵与 SVD 的关系：

$$C = \frac{1}{n} \tilde{X}^T \tilde{X} = \frac{1}{n} V \Sigma^T U^T U \Sigma V^T = \frac{1}{n} V \Sigma^T \Sigma V^T$$

可见：$C$ 的特征向量就是右奇异向量 $V$，特征值 $\lambda_i = \frac{\sigma_i^2}{n}$。

降维后的数据：

$$Z = \tilde{X} V_k = U \Sigma V^T V_k = U_k \Sigma_k$$

因此 PCA 降维等价于取 SVD 的前 $k$ 个右奇异向量作为投影方向。实际计算中常使用 SVD，因为它数值更稳定，且不需要显式计算协方差矩阵。

---

## 7. 核 PCA（Kernel PCA）

当数据在原始空间中不可线性分离时（如同心圆、螺旋结构），标准 PCA 效果不佳。核 PCA 通过核技巧将数据映射到高维特征空间再做 PCA。

### 7.1 基本原理

设非线性映射 $\phi: \mathbb{R}^d \to \mathcal{H}$（特征空间），在 $\mathcal{H}$ 中做 PCA。

核函数：

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

常见的核函数：
- 多项式核：$K(x_i, x_j) = (x_i^T x_j + c)^p$
- RBF 核：$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$
- Sigmoid 核：$K(x_i, x_j) = \tanh(\alpha x_i^T x_j + c)$

### 7.2 核 PCA 步骤

1. 计算核矩阵 $K_{ij} = K(x_i, x_j)$
2. 中心化核矩阵：$\tilde{K} = K - 1_n K - K 1_n + 1_n K 1_n$，其中 $1_n = \frac{1}{n} \mathbf{1}\mathbf{1}^T$
3. 对 $\tilde{K}$ 做特征值分解：$\tilde{K} = \alpha \Lambda \alpha^T$
4. 取前 $k$ 个最大特征值对应的特征向量 $\alpha_k$
5. 新样本 $x$ 的第 $j$ 个主成分：$z_j = \sum_{i=1}^{n} \alpha_{ij} K(x_i, x)$

核 PCA 能够发现非线性结构，但计算开销较大（$O(n^3)$），且无法显式重构原空间的数据。

---

## 8. 如何选择 k（主成分个数）

### 方法一：累计方差解释率阈值

选择满足以下条件的最小 $k$：

$$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i} \geq \tau$$

常用阈值 $\tau = 0.95$（保留 95% 方差）或 $\tau = 0.99$。

### 方法二：碎石图（Scree Plot）

绘制特征值随主成分编号的下降曲线，寻找"肘部"（elbow point）—— 特征值急剧下降后趋于平缓的转折点。

### 方法三：交叉验证

将降维作为预处理步骤，在后续任务（如分类）上做交叉验证，选择使下游性能最优的 $k$。

---

## 9. 与 t-SNE 对比

| 特性 | PCA | t-SNE |
|------|-----|-------|
| 类型 | 线性降维 | 非线性降维 |
| 目标 | 保留全局方差结构 | 保留局部邻域结构 |
| 计算复杂度 | $O(\min(n^2 d, n d^2))$ | $O(n^2)$（未优化） |
| 可解释性 | 高（有明确线性变换） | 低（黑盒映射） |
| 随机性 | 确定性 | 随机（需设随机种子） |
| 适用于 | 特征工程、数据预处理、去噪 | 数据可视化（2D/3D） |
| 可逆性 | 可通过 $W_k W_k^T$ 近似重构 | 不可逆 |
| 大数据 | 可增量学习 | 不适合直接处理 |

**总结**：PCA 适合需要保留全局结构、可解释性强的场景（如特征提取）；t-SNE 适合高维数据的可视化探索。

---

## 10. 优缺点

### 优点

1. **计算简单高效**：仅需特征值分解或 SVD，有成熟高效的数值算法
2. **数学基础坚实**：有明确的优化目标和统计解释
3. **可解释性强**：每个主成分是原始特征的线性组合，可以分析哪些特征的贡献大
4. **可逆性好**：能近似重构原始数据，可用于去噪和压缩
5. **消除多重共线性**：主成分之间互相正交，去除了特征间的相关性
6. **无超参数**（除 $k$ 外）：不需要学习率、迭代次数等

### 缺点

1. **仅能捕获线性结构**：对非线性流形数据（如瑞士卷）效果不佳
2. **对尺度敏感**：不同特征量纲不同时，方差大的特征会主导主成分方向（通常需先标准化）
3. **方差 ≠ 信息**：方差小的方向可能恰好是判别信息所在（如 LDA 考虑标签信息）
4. **主成分可解释性有限**：主成分是原始特征的线性组合，物理含义可能模糊
5. **假设高斯分布**：PCA 隐含假设数据服从多元高斯分布

---

## 11. 应用场景

1. **数据可视化**：将高维数据降到 2D/3D 进行可视化
2. **特征提取与降维**：去除冗余特征，加速后续算法（如聚类、分类）
3. **去噪**：保留前 $k$ 个主成分重构数据，抛弃小方差方向（常为噪声）
4. **图像压缩**：特征脸（Eigenfaces）—— 用少数特征脸重构人脸图像
5. **基因数据分析**：高维基因表达数据的降维与可视化
6. **金融因子分析**：从多个资产收益中提取少数共同因子
7. **推荐系统**：SVD 在协同过滤中的应用（矩阵分解）
8. **异常检测**：异常点在低方差方向上的重构误差较大

---

## 12. 相关公式汇总

| 公式 | 含义 |
|------|------|
| $\tilde{X} = X - \bar{x}$ | 数据中心化 |
| $C = \frac{1}{n}\tilde{X}^T\tilde{X}$ | 协方差矩阵 |
| $C v = \lambda v$ | 特征值分解 |
| $Z = \tilde{X} V_k$ | 降维投影 |
| $\hat{X} = Z V_k^T$ | 数据重构 |
| $\frac{\lambda_i}{\sum \lambda_j}$ | 方差解释率 |
| $\tilde{X} = U\Sigma V^T$ | SVD 分解 |
| $\lambda_i = \sigma_i^2 / n$ | 特征值与奇异值关系 |

---

## 参考文献

- Jolliffe, I. T. (2002). Principal Component Analysis. Springer.
- Shlens, J. (2014). A Tutorial on Principal Component Analysis.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Chapter 12.
