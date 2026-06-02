# 高斯混合模型（GMM）

## 1. 核心思想

高斯混合模型（Gaussian Mixture Model, GMM）假设数据是由 $K$ 个高斯分布混合生成的。其核心表达式为：

$$P(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

其中：
- $K$：高斯成分的个数（需预先指定）
- $\pi_k$：第 $k$ 个成分的混合系数（先验概率），满足 $0 \leq \pi_k \leq 1$ 且 $\sum_{k=1}^{K} \pi_k = 1$
- $\mu_k, \Sigma_k$：第 $k$ 个高斯分布的均值和协方差矩阵

> GMM 是一种**软聚类**（soft clustering）方法：每个样本以一定概率属于每个簇，而非硬性地分配到某个单一簇。

---

## 2. 单个高斯分布

多元高斯分布的数学表达式：

$$\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

其中：
- $x \in \mathbb{R}^D$：$D$ 维数据点
- $\mu \in \mathbb{R}^D$：均值向量（分布的中心）
- $\Sigma \in \mathbb{R}^{D \times D}$：协方差矩阵（分布的形状和方向）
- $|\Sigma|$：协方差矩阵的行列式
- $(x-\mu)^T \Sigma^{-1}(x-\mu)$：马氏距离（Mahalanobis Distance）的平方

指数部分中的二次型 $(x-\mu)^T \Sigma^{-1}(x-\mu)$ 定义了等概率密度的椭球面。

---

## 3. 隐变量模型

GMM 可以形式化为含隐变量的概率模型。

引入隐变量 $z \in \{0, 1\}^K$，$\sum_k z_k = 1$，表示样本来自哪个高斯成分：

$$P(z_k = 1) = \pi_k \quad \Rightarrow \quad P(z) = \prod_{k=1}^{K} \pi_k^{z_k}$$

给定隐变量后的条件分布：

$$P(x \mid z_k = 1) = \mathcal{N}(x \mid \mu_k, \Sigma_k) \quad \Rightarrow \quad P(x \mid z) = \prod_{k=1}^{K} \mathcal{N}(x \mid \mu_k, \Sigma_k)^{z_k}$$

联合分布与边缘分布：

$$P(x, z) = P(z) P(x \mid z) = \prod_{k=1}^{K} \left[\pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)\right]^{z_k}$$

$$P(x) = \sum_{z} P(x, z) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

---

## 4. EM 算法推导

由于存在隐变量 $z$，直接最大化对数似然函数很困难：

$$\ln P(X \mid \pi, \mu, \Sigma) = \sum_{n=1}^{N} \ln \left( \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k) \right)$$

对数内有求和，无闭式解。EM 算法提供了一种迭代求解方法。

### 4.1 E 步（期望步 —— Expectation）

计算每个样本属于每个高斯成分的后验概率（**责任值, Responsibility**）：

$$\gamma(z_{nk}) = P(z_{nk} = 1 \mid x_n) = \frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}$$

其中 $\gamma(z_{nk})$ 表示第 $n$ 个样本属于第 $k$ 个簇的概率（软分配），满足 $\sum_{k} \gamma(z_{nk}) = 1$。

### 4.2 M 步（最大化步 —— Maximization）

使用 E 步计算的责任值来更新参数（最大化完整数据的期望对数似然）：

**有效样本数**：

$$N_k = \sum_{n=1}^{N} \gamma(z_{nk})$$

$N_k$ 可解释为"软归属于第 $k$ 个簇的有效样本数"，满足 $\sum_{k} N_k = N$。

**更新混合系数**：

$$\pi_k = \frac{N_k}{N}$$

**更新均值**：

$$\mu_k = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) \, x_n$$

**更新协方差矩阵**：

$$\Sigma_k = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) \, (x_n - \mu_k)(x_n - \mu_k)^T$$

---

## 5. EM 算法完整流程

| 步骤 | 操作 |
|------|------|
| **初始化** | 随机初始化 $\pi_k, \mu_k, \Sigma_k$ 或用 K-Means 结果初始化 |
| **E 步** | 用当前参数计算后验概率 $\gamma(z_{nk})$ |
| **M 步** | 用 $\gamma(z_{nk})$ 更新 $\pi_k, \mu_k, \Sigma_k$ |
| **检查收敛** | 计算对数似然，若增量小于阈值（如 $10^{-6}$）或达到最大迭代次数则停止 |
| **重复** | 否则回到 E 步 |

### 对数似然计算

$$\ln P(X \mid \pi, \mu, \Sigma) = \sum_{n=1}^{N} \ln \left( \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k) \right)$$

---

## 6. EM 算法收敛性

### 6.1 对数似然单调不减

EM 算法的关键性质：**每次 E-M 迭代，观测数据的对数似然函数值单调不减**。

$$\ln P(X \mid \theta^{(t+1)}) \geq \ln P(X \mid \theta^{(t)})$$

### 6.2 证明思路

$$\ln P(X\mid\theta) = \underbrace{\mathcal{L}(q,\theta)}_{\text{下界}} + \underbrace{KL(q \parallel p)}_{\geq 0}$$

其中 $\mathcal{L}(q,\theta) = \sum_z q(z) \ln \frac{P(X,z\mid\theta)}{q(z)}$

E 步：令 $q(z) = P(z \mid X, \theta^{(t)})$，使 KL 散度为零，下界紧贴对数似然。
M 步：最大化 $\mathcal{L}(q,\theta)$ 关于 $\theta$，使下界提升。

### 6.3 注意

- EM 保证收敛到**局部极大值**，不保证全局最优
- 结果依赖初始化，建议多次随机初始化选最好结果
- 可能出现奇异解（某个成分坍缩到单个样本点，$\Sigma_k \to 0$）

---

## 7. 协方差矩阵类型

GMM 中每个高斯成分的协方差矩阵可以有不同形式的约束：

| 类型 | 参数说明 | 几何形状 | 参数个数（每成分） |
|------|----------|----------|---------------------|
| **full** | 任意正定矩阵 | 任意方向的椭球 | $D(D+1)/2$ |
| **tied** | 所有成分共享同一协方差 | 相同形状、不同中心 | $D(D+1)/2$（共享） |
| **diag** | 对角矩阵 | 轴对齐的椭球 | $D$ |
| **spherical** | $\sigma_k^2 I$ | 球体 | $1$ |

### 选择指南

- `full`：参数最多，拟合能力最强，但容易过拟合，对小样本不友好
- `diag`：平衡灵活性（允许各维度不同方差）与参数效率（无协方差项）
- `spherical`：类似 K-Means（各向同性），计算最快
- `tied`：假设所有簇形状相似

---

## 8. 模型选择：BIC 与 AIC

由于 $K$ 需预先指定，如何选择合适的簇数？

### 8.1 BIC（Bayesian Information Criterion）

$$BIC = k \ln(n) - 2 \ln(\hat{L})$$

### 8.2 AIC（Akaike Information Criterion）

$$AIC = 2k - 2 \ln(\hat{L})$$

其中：
- $k$：模型自由参数的总数（$\pi_k$ 占 $K-1$ 个，$\mu_k$ 占 $K\cdot D$ 个，$\Sigma_k$ 取决于类型）
- $n$：样本数
- $\hat{L}$：模型的最大似然值（$P(X \mid \hat{\theta})$）

### 8.3 使用原则

- 较小值对应更好的模型（在拟合优度与复杂度之间平衡）
- BIC 对模型复杂度的惩罚大于 AIC（因为 $\ln(n) > 2$，当 $n \geq 8$）
- BIC 倾向于选择更简单的模型
- 实际操作：对候选 $K$ 值分别训练 GMM，选择 BIC/AIC 最小的 $K$

---

## 9. 与 K-Means 对比

| 特性 | K-Means | GMM |
|------|---------|-----|
| 分配方式 | 硬分配（hard assignment） | 软分配（soft assignment） |
| 簇形状 | 球形（各向同性） | 可椭球（取决于协方差类型） |
| 概率框架 | 无（几何距离） | 有（概率生成模型） |
| 不确定度 | 不提供 | 提供（后验概率） |
| 优化方法 | 坐标下降 | EM 算法 |
| 初始化敏感度 | 敏感 | 敏感 |

### K-Means 是 GMM 的特例

当满足以下条件时，GMM 退化为 K-Means：
1. 所有协方差矩阵为 $\sigma^2 I$，且 $\sigma \to 0$（方差极大 → 硬分配极限）
2. 所有混合系数相等：$\pi_k = 1/K$（均匀先验）
3. 后验概率 $\gamma(z_{nk})$ 趋于 one-hot 向量（每个样本严格属于一个簇）

---

## 10. GMM 用于异常检测

通过概率密度阈值进行异常检测：

1. 训练 GMM 拟合正常数据的分布
2. 对于新样本 $x_{\text{new}}$，计算 $\log P(x_{\text{new}})$
3. 若 $\log P(x_{\text{new}}) < \text{threshold}$，则判定为异常

阈值可通过正常数据的对数似然分布（如取某个低分位数）来确定。

---

## 11. 优缺点

### 优点

1. **软聚类**：提供每个样本属于各簇的概率，而非硬性二值分配，包含不确定性信息
2. **簇形状灵活**：通过不同协方差约束，可以拟合球形、椭球形等各种簇形状
3. **概率生成模型**：可以生成新样本（从拟合的 GMM 采样），可用于数据增强
4. **数学理论坚实**：有明确的统计解释和收敛性保证
5. **密度估计**：GMM 本质是概率密度估计器，可用于异常检测
6. **适用于多种数据类型**：只要选择合适的分布族（不限于高斯），可扩展为一般混合模型

### 缺点

1. **需指定簇数 K**：是超参数，需通过 BIC/AIC 或领域知识确定
2. **对初始化敏感**：收敛到局部最优而非全局最优，需多次初始化
3. **奇异解问题**：某个成分可能坍缩到单一样本点，导致协方差矩阵奇异（可通过正则化缓解）
4. **假设高斯分布**：如果真实数据不符合高斯分布，拟合效果不佳
5. **高维问题**：协方差矩阵参数随维度平方增长（full 型）；数值计算不稳定，易奇异
6. **计算开销**：EM 迭代需要反复计算后验概率和更新参数，在大数据集上较慢
7. **对离群点敏感**：离群点会拉动高斯分布的均值和协方差（因为用的是欧氏距离）

---

## 12. 变种与扩展

1. **贝叶斯高斯混合模型（BGMM）**：为参数 $\pi, \mu, \Sigma$ 设先验分布（Dirichlet + Normal-Inverse-Wishart），用变分推断或 MCMC 自动确定 $K$
2. **狄利克雷过程高斯混合模型（DPGMM）**：非参数贝叶斯方法，$K$ 自动从数据中推断
3. **增量 GMM**：适用于流式数据/在线学习场景，不需要一次性加载全部数据
4. **混合专家模型（MoE）**：每个高斯成分后接一个线性回归模型
5. **因子分析器混合模型（MFA）**：对协方差矩阵施加低秩约束 $\Sigma_k = W_k W_k^T + \Psi_k$，适合高维数据

---

## 13. 核心公式汇总

| 公式 | 含义 |
|------|------|
| $P(x) = \sum \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ | GMM 概率密度 |
| $\mathcal{N}(x\mid\mu,\Sigma)$ | 多元高斯分布 |
| $\gamma(z_{nk})$ | E 步：后验责任值 |
| $N_k = \sum \gamma(z_{nk})$ | 有效样本数 |
| $\mu_k = \frac{1}{N_k} \sum \gamma(z_{nk}) x_n$ | M 步：更新均值 |
| $\Sigma_k = \frac{1}{N_k} \sum \gamma(z_{nk})(x_n-\mu_k)(x_n-\mu_k)^T$ | M 步：更新协方差 |
| $\pi_k = N_k / N$ | M 步：更新混合系数 |
| $BIC = k\ln(n) - 2\ln(\hat{L})$ | 贝叶斯信息准则 |
| $AIC = 2k - 2\ln(\hat{L})$ | 赤池信息准则 |

---

## 参考文献

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 9.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. *JRSS-B*.
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. Chapter 11.
