# 梯度提升 (GBDT / XGBoost / LightGBM)

## 1. Boosting 思想

Boosting（提升法）是一类将**弱学习器**串行组合成**强学习器**的集成学习方法。其核心思想是：

> 每一轮训练一个新的弱学习器，专门去修正前面所有学习器累计产生的错误。

与 Bagging（如随机森林）并行训练、投票平均不同，Boosting 是**串行、逐轮纠错**的过程。每一轮新模型都聚焦于之前模型表现最差的样本。

---

## 2. 加法模型框架

梯度提升的数学框架是一个**前向分步加法模型**：

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

其中：
- $F_m(\mathbf{x})$ —— 经过 $m$ 轮迭代后的集成模型。
- $F_{m-1}(\mathbf{x})$ —— 前 $m-1$ 轮的集成模型。
- $h_m(\mathbf{x})$ —— 第 $m$ 轮训练的弱学习器（通常为决策树）。
- $\eta$ —— **学习率（Shrinkage）**，用于控制每棵树的贡献程度。

最终模型为所有弱学习器的加权和：

$$F(\mathbf{x}) = \sum_{m=1}^{M} \eta \cdot h_m(\mathbf{x})$$

---

## 3. GBDT 核心原理

### 3.1 用负梯度近似残差

GBDT (Gradient Boosting Decision Tree) 的核心洞察：**在函数空间中做梯度下降**。

对于可微损失函数 $L(y, F(\mathbf{x}))$，我们想要找到使损失最小化的 $F$。在第 $m$ 轮，当前模型的"残差"无法直接计算（不像线性回归），因此 GBDT 使用损失函数关于当前预测值的**负梯度**作为残差的近似：

$$r_{im} = -\left[ \frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F = F_{m-1}}$$

然后训练一个新的弱学习器 $h_m$ 去拟合这些伪残差 $r_{im}$，再通过线搜索或固定学习率将其加入模型。

---

### 3.2 回归任务

**损失函数**（平方误差）：

$$L(y, F) = \frac{1}{2}(y - F)^2$$

**负梯度**（即伪残差）：

$$-\frac{\partial L}{\partial F} = y - F$$

在回归任务中，负梯度恰好等于真实残差 $y - F_{m-1}(\mathbf{x})$，因此 GBDT 在回归中的每一轮实际上就是在拟合前一轮的预测误差。

---

### 3.3 二分类任务

**损失函数**（负二项对数似然 / Deviance）：

$$L(y, F) = \log\left(1 + \exp(-2yF)\right), \quad y \in \{-1, +1\}$$

**梯度推导**：

令 $p = \frac{1}{1 + e^{-2F}}$（为预测为正类的概率，通过 sigmoid 映射），求导得：

$$\frac{\partial L}{\partial F} = \frac{-2y \cdot e^{-2yF}}{1 + e^{-2yF}} = -2y \cdot \frac{1}{1 + e^{2yF}}$$

整理后，**负梯度（伪残差）** 为：

$$r_{im} = \frac{2y_i}{1 + \exp(2y_i F_{m-1}(\mathbf{x}_i))}$$

在多分类任务中，常用 Softmax 交叉熵损失，每一类训练一棵树（One-vs-Rest 或 K-class 同时优化）。

---

## 4. 收缩（Shrinkage / 学习率）

学习率 $\eta$（通常 $0.01 \sim 0.1$）是 GBDT 最重要的正则化手段之一：

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

- **$\eta$ 较小**：每棵树贡献小 → 需要更多树才能收敛（训练慢），但泛化能力更强。
- **$\eta$ 较大**：每棵树贡献大 → 收敛快，但容易过拟合。

经验法则：较小的学习率配合较大的迭代次数（`n_estimators`）通常效果更好。通常 $\eta$ 和 `n_estimators` 之间存在折衷关系，实践中常通过早停（Early Stopping）确定最优迭代次数。

---

## 5. XGBoost 的改进

XGBoost (eXtreme Gradient Boosting) 在 GBDT 基础上引入了多项关键改进，使其成为 Kaggle 竞赛和工业界的首选工具。

### 5.1 二阶泰勒展开

传统 GBDT 只用一阶梯度信息，XGBoost 使用损失函数的**二阶泰勒展开**来近似目标函数，从而更精确地指导树的生长方向：

在第 $t$ 轮，目标函数近似为：

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

其中：
- $g_i = \frac{\partial L(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$ —— **一阶梯度**。
- $h_i = \frac{\partial^2 L(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$ —— **二阶梯度（Hessian）**。
- $\Omega(f_t)$ —— 正则化项。

二阶信息使得 XGBoost 能更准确地选择分裂点，提升收敛速度与精度。

---

### 5.2 正则化项

XGBoost 在目标函数中显式加入正则化项来控制模型复杂度：

$$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

其中：
- $T$ —— 树的叶子节点数量。
- $w_j$ —— 第 $j$ 个叶子节点的预测权重。
- $\gamma$ —— 叶子节点数量的惩罚系数（剪枝阈值，越大树越浅）。
- $\lambda$ —— 叶子权重的 L2 正则化系数。

展开叶子权重的闭式解：

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

对应的最优目标值（用作分裂增益的度量）：

$$\mathcal{L}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$

---

### 5.3 列采样 (Column Subsampling)

受随机森林启发，XGBoost 支持**按列（特征）采样**：
- 每棵树随机只使用一部分特征，类似于随机森林的 `max_features`。
- 有效减少过拟合，同时加速训练（尤其在特征维度很高时）。

---

### 5.4 加权分位点近似 (Weighted Quantile Sketch)

传统 GBDT 需要对每个特征的所有分裂点进行穷举搜索，XGBoost 使用**加权分位点**算法，以 Hessian 作为样本权重，智能选取候选分裂点，大幅减少搜索空间，在处理大规模数据时尤为有效。

基本原理：将每个样本的二阶梯度 $h_i$ 作为该样本的权重，在寻找分位点时，使相邻分位点间的 Hessian 和近似相等，从而在信息量大的区域放置更多候选点。

---

### 5.5 稀疏感知 (Sparsity Awareness)

XGBoost 原生处理缺失值：在训练每个分裂节点时，自动学习将缺失值路由到左子树还是右子树（选择增益更大的一侧）。预测时同样自动处理缺失特征，无需复杂的缺失值填充预处理。

### 5.6 XGBoost 其他特性

- **正则化目标函数**：比传统 GBDT 多一层正则化保护。
- **支持自定义损失函数**：只需提供一阶和二阶梯度。
- **并行化**：特征粒度上的并行计算（寻找最佳分裂点时并行扫描特征）。
- **缓存感知访问**：高效利用 CPU 缓存，加速计算。
- **核外计算（Out-of-Core）**：支持无法完全装入内存的数据集。

---

## 6. LightGBM

LightGBM (Light Gradient Boosting Machine) 由微软开源，专门针对**大数据量 + 高维特征**场景优化，在训练速度和内存占用上有显著优势。

### 6.1 基于直方图的算法 (Histogram-based Algorithm)

将连续特征值离散化为 $k$ 个分桶（bins，默认 255），在寻找最佳分裂点时只需遍历分桶边界而非所有样本值，大幅降低计算量：

- 时间复杂度从 $O(\text{n\_data} \times \text{n\_features})$ 降至 $O(\text{n\_bins} \times \text{n\_features})$。
- 内存占用显著减少（存储桶索引而非浮点值）。

### 6.2 GOSS (Gradient-based One-Side Sampling)

**单边梯度采样**：根据梯度的绝对值对样本进行降采样，但保留所有大梯度样本（训练不足、误差大的样本），仅随机丢弃一些小梯度样本。这样在不损失太多信息的前提下大幅减少训练数据量。

核心思想：梯度大的样本对信息增益贡献更大，梯度小的样本已经"训练得不错"，可以适当丢弃。

### 6.3 EFB (Exclusive Feature Bundling)

**互斥特征捆绑**：将互斥的稀疏特征（即不同时非零的特征）捆绑为一个特征，以减少特征维度。

在现实数据中，高维稀疏特征常常是互斥或近似互斥的（例如 One-Hot 编码后的类别特征），EFB 能有效降维且几乎不损失信息。

### 6.4 Leaf-wise 生长策略

不同于 XGBoost 的**Level-wise（逐层生长）**策略，LightGBM 采用 **Leaf-wise（按叶生长）**：

- **Level-wise**：每层所有节点同时分裂，更加平衡，但可能在不必要的节点上浪费计算。
- **Leaf-wise**：每次选择收益（分裂增益）最大的叶子节点进行分裂。

在相同分裂次数下，Leaf-wise 能获得更低的损失，但容易长成深树导致过拟合；通常配合 `max_depth` 或 `min_data_in_leaf` 进行约束。

---

## 7. 与随机森林的对比

| 维度 | 随机森林 (Bagging) | 梯度提升 (Boosting) |
|------|---------------------|----------------------|
| 训练方式 | 并行（每棵树独立） | 串行（每棵树依赖前一棵） |
| 核心思想 | 降低方差（平均多个高方差模型） | 降低偏差（逐步拟合残差） |
| 基学习器 | 深度较大、高方差树 | 浅层树（弱学习器） |
| 过拟合风险 | 较低（方差降低） | 较高（需正则化、早停） |
| 训练速度 | 快（天然可并行） | 较慢（串行约束） |
| 预测速度 | 多棵树投票，较慢 | 累加每棵树，与树数成正比 |
| 异常值敏感度 | 低 | 较高（梯度受影响） |
| 典型应用 | 通用分类/回归基线 | 追求最高精度的竞赛与工业场景 |

---

## 8. 核心超参数

### GBDT / XGBoost / LightGBM 通用超参数

| 参数 | 含义 | 典型范围 |
|------|------|----------|
| `n_estimators` | 迭代轮数（树的数量） | 100 ~ 1000+ |
| `learning_rate` / `eta` | 学习率（Shrinkage） | 0.01 ~ 0.3 |
| `max_depth` | 树的最大深度 | 3 ~ 10 |
| `subsample` | 行采样比例（随机梯度提升） | 0.5 ~ 1.0 |
| `colsample_bytree` | 列（特征）采样比例 | 0.5 ~ 1.0 |
| `min_child_weight` | 叶子节点的最小样本权重和 | 1 ~ 10 |

### XGBoost 特有参数

| 参数 | 含义 |
|------|------|
| `gamma` / `min_split_loss` | 分裂所需的最小损失减小量 |
| `lambda` / `reg_lambda` | L2 正则化系数（叶子权重） |
| `alpha` / `reg_alpha` | L1 正则化系数 |
| `tree_method` | 树构建方法 (`hist` / `gpu_hist` / `approx`) |

### LightGBM 特有参数

| 参数 | 含义 |
|------|------|
| `num_leaves` | 最大叶子数（控制树复杂度，非深度） |
| `min_data_in_leaf` | 叶子节点的最小样本数 |
| `max_bin` | 直方图分桶数（默认 255） |
| `boosting_type` | `gbdt` / `dart` / `goss` / `rf` |
| `feature_fraction` | 特征采样比例 |

---

## 9. 优缺点

### 优点

| 优点 | 说明 |
|------|------|
| 极高的预测精度 | Boosting 系列在结构化/tabular 数据上长期统治 Kaggle 和工业界 |
| 自动特征组合 | 决策树天然能捕捉非线性关系和特征交互 |
| 灵活的损失函数 | 支持回归、分类（二分类/多分类）、排序等 |
| 鲁棒的正则化 | XGBoost/LightGBM 的多重正则化有效防止过拟合 |
| 特征重要性输出 | 可解释每个特征对模型的贡献（gain / split / cover） |
| 处理混合类型特征 | 不需要太多预处理，连续和离散特征均可直接使用 |
| LightGBM 的高效性 | 在大数据（亿级样本）上仍可快速训练 |

### 缺点

| 缺点 | 说明 |
|------|------|
| 训练速度较慢 | Boosting 的串行性使其天然慢于 Bagging |
| 需要仔细调参 | 超参数多且敏感，不当配置会导致过拟合或欠拟合 |
| 对异常值敏感 | 回归任务中，梯度受异常值影响较大 |
| 不擅长高维稀疏数据 | 相比线性模型或朴素贝叶斯，Boosting 在超高维稀疏场景效果不如 |
| 模型体积较大 | 多棵树集成后，模型文件可能很大 |
| 不易并行训练 | 树间无法并行（但树内可分特征或数据并行） |

---

## 10. 常用实现与代码示例

### Python 库

```python
# GBDT (scikit-learn)
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3
)

# XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1, reg_alpha=0,
    tree_method='hist'  # 使用直方图算法加速
)

# LightGBM
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.1,
    num_leaves=31, max_depth=-1,
    subsample=0.8, colsample_bytree=0.8,
    boosting_type='gbdt'
)
```

---

## 11. 总结

梯度提升树（GBDT）及其现代变体 XGBoost 和 LightGBM 是结构化数据建模的"王者算法"。三者关系如下：

- **GBDT** 奠定了梯度提升的理论基础（函数空间梯度下降）。
- **XGBoost** 在 GBDT 上引入二阶导数、显式正则化、列采样等，极大提升了精度和工程实用性。
- **LightGBM** 则在 XGBoost 基础上进一步优化训练效率（直方图、GOSS、EFB、Leaf-wise），使得海量数据训练成为可能。

三者共同构成了现代机器学习竞赛和工业应用中梯度提升方法的完整图谱。
