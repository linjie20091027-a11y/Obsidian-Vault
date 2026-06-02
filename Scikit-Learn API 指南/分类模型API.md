# 分类模型 API 详解

> 实战手册 | Scikit-Learn 分类器全收录 | 每个 API 附带代码、参数表、踩坑记录

---

## 1. 线性分类模型

### 1.1 LogisticRegression

#### 功能说明
逻辑回归。虽然名字带"回归"，但它是一个**线性分类器**。核心原理：对特征线性组合后通过 sigmoid 函数映射到 (0,1)，得到类别概率；训练时最小化交叉熵损失（对数损失）。

数学形式（二分类）：
```
P(y=1|x) = 1 / (1 + exp(-(w·x + b)))
```

多分类有两种策略：
- **OvR（一对多）**：为每个类别训练一个二分类器
- **Multinomial**：直接使用 softmax + 交叉熵

#### 什么时候用
- **二分类/多分类的首选 baseline**——花 1 分钟跑一下，就能知道问题难度
- 特征数量很大（>10 万）时尤其合适，因为模型简单、训练快
- 需要概率输出时（`predict_proba` 给出校准较好的概率）
- 需要特征可解释性时（`coef_` 直接反映每个特征的影响方向和强度）
- 线性可分或近似线性可分的问题

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `penalty` | str | `'l2'` | 正则化类型：`'l1'`/`'l2'`/`'elasticnet'`/`None`。L1 产生稀疏解（特征选择）；L2 防止过拟合；elasticnet 是 L1+L2 混合；None 不做正则 |
| `C` | float | `1.0` | 正则化强度的**倒数**。越小正则越强，越大模型越自由。典型搜索范围：`[0.001, 0.01, 0.1, 1, 10, 100]` |
| `solver` | str | `'lbfgs'` | 优化算法，见下方选择指南 |
| `class_weight` | str/dict | `None` | `'balanced'` 自动按类别样本数反比加权，处理不平衡数据。也可手动传 `{0: 0.3, 1: 0.7}` |
| `max_iter` | int | `100` | 最大迭代次数。**经常需要调大**，否则报 `ConvergenceWarning` |
| `multi_class` | str | `'auto'` | `'ovr'`：一对多策略；`'multinomial'`：softmax 多分类（需要 `solver` 支持） |
| `fit_intercept` | bool | `True` | 是否拟合截距项。数据已中心化时可设 `False` |
| `tol` | float | `1e-4` | 收敛阈值，调小可提高精度但训练变慢 |
| `random_state` | int | `None` | 随机种子（影响 `solver='sag'/'saga'` 和 `multi_class='multinomial'`） |
| `l1_ratio` | float | `None` | 仅 `penalty='elasticnet'` 时有效，L1 占比。0 = 全 L2，1 = 全 L1 |
| `n_jobs` | int | `None` | 并行线程数（`multi_class='ovr'` 时可用） |

#### solver 选择指南

| solver | 支持的正则化 | 适用场景 | 说明 |
|--------|-------------|----------|------|
| `'lbfgs'` | L2 / None | **默认，首选**。中小数据集（<10 万） | 拟牛顿法，收敛快。不支持 L1，不支持 'ovr' 的 `n_jobs` |
| `'liblinear'` | L1 / L2 | 小数据集、需要 L1 正则 | 坐标下降法。只支持 OvR。**已标记为将来弃用** |
| `'newton-cg'` | L2 / None | 中等数据集 | 牛顿法变体，和 `'lbfgs'` 类似但更慢 |
| `'sag'` | L2 / None | **大数据集**（>10 万） | 随机平均梯度，收敛快于 sgd |
| `'saga'` | L1/L2/elasticnet/None | **大数据 + 需要 L1 / elasticnet** | `'sag'` 的改进版，支持所有正则化类型。**最通用、最推荐** |

> **经验法则**：优先用 `solver='lbfgs'`（默认），遇到 `ConvergenceWarning` 则加大 `max_iter`；数据量大时换 `'saga'`；需要 L1 正则时必须用 `'liblinear'` 或 `'saga'`。

#### 代码示例

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score

# ---------- 生成数据 ----------
X, y = make_classification(n_samples=5000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 基础用法 ----------
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]
print("基础 AUC:", roc_auc_score(y_test, y_proba))

# ---------- 带标准化的 Pipeline（重要！）----------
# 线性模型 + 正则化 = 必须先标准化
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
pipe.fit(X_train, y_train)
print("Pipeline AUC:", roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))

# ---------- GridSearchCV 调参 ----------
param_grid = {
    'clf__penalty': ['l1', 'l2', 'elasticnet'],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__solver': ['saga'],  # saga 支持所有 penalty
    'clf__l1_ratio': [0, 0.5, 1],  # 仅 elasticnet 时生效
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV AUC:", grid.best_score_)

# ---------- 处理不平衡数据 ----------
lr_balanced = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_balanced.fit(X_train, y_train)
print("Balanced AUC:", roc_auc_score(y_test, lr_balanced.predict_proba(X_test)[:, 1]))

# ---------- 获取特征重要性 ----------
coef = lr.coef_  # shape: (n_classes, n_features) 或 (1, n_features) 二分类
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
importance = np.abs(coef[0])
sorted_idx = np.argsort(importance)[::-1]
for i in sorted_idx[:10]:
    print(f"{feature_names[i]:>12}: {coef[0, i]:+.4f}")
```

#### 常见报错与解决

| 报错/警告 | 原因 | 解决 |
|-----------|------|------|
| `ConvergenceWarning: lbfgs failed to converge` | 迭代次数不够 | 增大 `max_iter`（如 3000-5000）。或换 `solver='saga'`。也可调大 `tol` |
| `Solver lbfgs supports only 'l2' or None penalties` | `penalty='l1'` 但用了 lbfgs | 换 `solver='liblinear'` 或 `'saga'` |
| `This solver needs samples of at least 2 classes` | 训练数据只有一种标签 | 检查数据划分和标签分布 |
| 模型效果很差 | 没有做特征缩放 | 加上 `StandardScaler` |
| `predict_proba` 概率极度接近 0 或 1 | C 太小导致模型欠拟合，或 C 太大过拟合 | 调整 C 值，检查是否过拟合 |

#### 注意事项 / 踩坑点

1. **必须做特征缩放**：正则化对特征尺度敏感，不做 `StandardScaler` 效果会很差。唯一的例外是 `penalty=None`。
2. **C 越小正则越强**：这是反直觉的，记住 C = 1/λ。
3. **多分类默认策略**：从 sklearn 0.22 起，`multi_class='auto'` 在 `solver='lbfgs'` 时默认选择 `'multinomial'`。如果你想要 OvR，需要显式指定。
4. **`predict_proba` 的校准**：逻辑回归的概率输出通常校准得不错，但如果有严重过拟合则不准。
5. **liblinear 已被标记弃用**：新项目应使用 `solver='saga'` 代替 `'liblinear'`。
6. **`class_weight='balanced'` 只调整损失中的权重**，不影响 `predict_proba` 的概率值。
7. **共线特征**：逻辑回归对共线性没有 Ridge 那么鲁棒，高维数据建议先做特征选择或换 `RidgeClassifier`。

---

### 1.2 SGDClassifier

#### 功能说明
随机梯度下降分类器。用 SGD 方式优化线性模型，通过 `loss` 参数可以等价于不同的线性分类器：
- `loss='hinge'` → 线性 SVM
- `loss='log_loss'` → 逻辑回归
- `loss='modified_huber'` → 带概率输出的平滑 hinge
- `loss='perceptron'` → 感知机

#### 什么时候用
- **数据量 > 10 万样本**，`LogisticRegression` 太慢时
- 在线学习 / 增量训练场景（`partial_fit`）
- 内存紧张，需要 out-of-core 训练
- 对训练速度要求高于精度的场景

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `loss` | str | `'hinge'` | 损失函数，决定等价模型 |
| `penalty` | str | `'l2'` | 正则化类型：`'l2'`/`'l1'`/`'elasticnet'` |
| `alpha` | float | `0.0001` | 正则化强度（**直接比例**，不像 C 是倒数） |
| `l1_ratio` | float | `0.15` | elasticnet 中 L1 比例 |
| `max_iter` | int | `1000` | 最大迭代次数（对数据的 epoch 数） |
| `tol` | float | `1e-3` | 收敛阈值 |
| `learning_rate` | str | `'optimal'` | `'constant'`/`'optimal'`/`'invscaling'`/`'adaptive'` |
| `eta0` | float | `0.0` | 初始学习率（`'constant'`/`'invscaling'`/`'adaptive'` 时有效） |
| `class_weight` | str/dict | `None` | 类别权重 |
| `random_state` | int | `None` | 随机种子 |
| `n_jobs` | int | `None` | 并行数（仅 OvR 多分类时可用） |
| `early_stopping` | bool | `False` | 是否早停（需要指定 validation set） |
| `shuffle` | bool | `True` | 每个 epoch 后是否打乱数据 |

#### 代码示例

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------- 基础用法（线性 SVM 等价）----------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42))
])
pipe.fit(X_train, y_train)
print("SGD hinge accuracy:", pipe.score(X_test, y_test))

# ---------- 逻辑回归等价（带概率输出）----------
pipe_log = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SGDClassifier(loss='log_loss', max_iter=1000, random_state=42))
])
pipe_log.fit(X_train, y_train)
# loss='log_loss' 支持 predict_proba
print("SGD log_loss AUC:", roc_auc_score(y_test, pipe_log.predict_proba(X_test)[:, 1]))

# ---------- 增量训练 ----------
clf = SGDClassifier(loss='log_loss', random_state=42)
scaler = StandardScaler()

# 模拟批次数据
batch_size = 500
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    X_batch_scaled = scaler.partial_fit(X_batch).transform(X_batch) if i == 0 else scaler.transform(X_batch)
    clf.partial_fit(X_batch_scaled, y_batch, classes=np.unique(y))

print("Incremental accuracy:", clf.score(scaler.transform(X_test), y_test))
```

#### 注意事项 / 踩坑点

1. **数据必须打乱 + 缩放**：SGD 对数据顺序和尺度都极为敏感。
2. **`alpha` 不是 `C` 的倒数**：`alpha` 越大正则越强，和 `LogisticRegression` 的 `C` 相反。
3. **默认 `loss='hinge'` 没有 `predict_proba`**：需要概率输出时必须用 `loss='log_loss'` 或 `'modified_huber'`。
4. **对学习率很敏感**：默认 `learning_rate='optimal'` 通常够用，但如果效果不好需要调 `eta0`。
5. **`partial_fit` 第一次调用必须传所有类别**：否则后续遇到新类别会报错。

---

### 1.3 RidgeClassifier

#### 功能说明
岭分类器。用 Ridge 回归（L2 正则化最小二乘）做分类——将目标标签转换为 {-1, +1} 然后用回归拟合，预测时取符号。本质上是**最小二乘 SVM 的效率近似**。

#### 什么时候用
- **特征多（> 样本数）** + 线性可分：比 LogisticRegression 更稳定
- 共线性严重的问题（Ridge 对共线性很鲁棒）
- 只关心分类标签不关心概率（`RidgeClassifier` 没有 `predict_proba`）
- 文本分类的快速 baseline（配合 TF-IDF）

#### 与 LogisticRegression 的区别

| 特性 | LogisticRegression | RidgeClassifier |
|------|-------------------|-----------------|
| 损失函数 | 交叉熵 | 最小二乘 |
| 概率输出 | `predict_proba` 可用 | 无（需用 `RidgeClassifierCV` + CalibratedClassifierCV） |
| 共线性 | 较敏感 | 鲁棒 |
| 训练速度 | 较慢（迭代求解） | 较快（闭式解） |
| L1 正则 | 支持 | 不支持 |

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alpha` | float | `1.0` | 正则化强度，越大正则越强 |
| `fit_intercept` | bool | `True` | 是否拟合截距 |
| `solver` | str | `'auto'` | `'auto'`/`'svd'`/`'cholesky'`/`'lsqr'`/`'sparse_cg'`/`'sag'`/`'saga'` |

#### 代码示例

```python
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RidgeClassifier(alpha=1.0))
])
pipe.fit(X_train, y_train)
print("Ridge accuracy:", pipe.score(X_test, y_test))

# 自动选 alpha
rcv = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0], cv=5)
rcv.fit(StandardScaler().fit_transform(X_train), y_train)
print("Best alpha:", rcv.alpha_)
```

---

### 1.4 Perceptron

#### 功能说明
经典感知机算法。最简单的神经网络（单层，无隐藏层），使用感知机损失 + SGD 优化。

#### 什么时候用
- 教学演示 / 历史原因
- 数据完全线性可分且不需要概率输出

**实际工程中不推荐使用**——用 `SGDClassifier(loss='perceptron')` 或 `LogisticRegression` 代替。

```python
from sklearn.linear_model import Perceptron
clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train, y_train)
```

---

## 2. 支持向量机

### 2.1 SVC

#### 功能说明
支持向量分类（Support Vector Classification）。基于核技巧，在高维特征空间中寻找最大间隔分离超平面。核函数使得 SVC 能够处理**非线性**决策边界。

核心公式：
```
min  1/2 ||w||^2 + C * Σ ξ_i
s.t. y_i (w·φ(x_i) + b) ≥ 1 - ξ_i,  ξ_i ≥ 0
```

其中 C 控制间隔最大化 vs 误分类惩罚的权衡。

#### 什么时候用
- **小到中等数据量**（< 1 万样本）——再大就会很慢
- 特征维数高（比如文本 TF-IDF 10000 维）——SVM 在高维空间表现优异
- **需要非线性决策边界**——配合 RBF kernel 非常灵活
- 图像分类、文本分类的传统强 baseline
- 数据干净、异常值少时效果好

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `C` | float | `1.0` | 正则化参数（和 LogisticRegression 一样是倒数逻辑）。**C 越大，间隔越窄，越努力拟合每个点（容易过拟合）**；C 越小，间隔越宽，偏差越大 |
| `kernel` | str | `'rbf'` | 核函数：`'linear'`/`'poly'`/`'rbf'`/`'sigmoid'`/`'precomputed'` |
| `gamma` | str/float | `'scale'` | 核函数系数。`'scale'` = 1/(n_features * X.var())，`'auto'` = 1/n_features。**gamma 越大，决策边界越复杂，越容易过拟合** |
| `degree` | int | `3` | 多项式核的阶数（仅 `kernel='poly'` 时有效） |
| `coef0` | float | `0.0` | 核函数中的独立项（`'poly'` 和 `'sigmoid'` 时使用） |
| `probability` | bool | `False` | 是否启用概率估计。**设为 True 会显著增加训练时间**（内部做 5-fold CV 校准） |
| `class_weight` | str/dict | `None` | 类别权重 |
| `tol` | float | `1e-3` | 停止准则的容忍度 |
| `max_iter` | int | `-1` | 最大迭代次数，`-1` 表示无限制 |
| `decision_function_shape` | str | `'ovr'` | 多分类策略：`'ovo'`（一对一）或 `'ovr'`（一对多）。ovo 构建 n*(n-1)/2 个分类器 |
| `cache_size` | float | `200` | 核缓存大小（MB），大数据时调大可以加速 |

#### kernel 选择指南

| kernel | 适用场景 | 关键参数 | 说明 |
|--------|----------|----------|------|
| `'linear'` | 线性可分数据、高维稀疏数据（文本） | C | 等价于 `LinearSVC` 但更慢，**大数据请用 LinearSVC** |
| `'rbf'` | **默认首选**，通用非线性问题 | C, gamma | 最灵活最常用。调参范围：C ∈ [0.1, 1, 10, 100]，gamma ∈ [0.001, 0.01, 0.1, 1, 'scale'] |
| `'poly'` | 已知数据有多项式关系 | degree, gamma, coef0 | 参数多难调，一般不优先使用 |
| `'sigmoid'` | 近似两层神经网络 | gamma, coef0 | 实践中很少使用 |

#### 代码示例

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# ---------- 基础用法 ----------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=42))
])
pipe.fit(X_train, y_train)
print("Default SVC accuracy:", pipe.score(X_test, y_test))

# ---------- 完整调参 ----------
param_grid = {
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'svc__class_weight': [None, 'balanced'],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

# ---------- 获取支持向量 ----------
best_svc = grid.best_estimator_.named_steps['svc']
print("支持向量数量:", best_svc.n_support_)
print("支持向量索引:", best_svc.support_[:10])
print("每类支持向量数:", best_svc.n_support_)  # 多分类时各有多少个

# ---------- 概率输出（注意：训练会变慢 5 倍以上）----------
svc_prob = SVC(probability=True, random_state=42)
svc_prob.fit(StandardScaler().fit_transform(X_train[:2000]), y_train[:2000])
proba = svc_prob.predict_proba(StandardScaler().transform(X_test[:500]))
```

#### 注意事项 / 踩坑点

1. **必须做特征缩放**：SVM 对特征尺度极度敏感，不做 StandardScaler 几乎等于自杀。
2. **大数据非常慢**：训练复杂度 O(n² ~ n³)。超过 1 万样本请考虑 `LinearSVC` 或 `SGDClassifier`。
3. **`probability=True` 很慢**：内部做 5-fold 交叉验证来校准概率，训练时间×5 以上。不要在你需要速度时开启。
4. **C 和 gamma 同时调**：二者强相关。C 大 + gamma 大 = 严重过拟合。推荐的调参策略是先用粗粒度网格，再细化。
5. **二分类/多分类的 `decision_function_shape`**：`'ovo'` 默认会创建 n*(n-1)/2 个二分类器，类别多时很慢。
6. **`gamma='scale'`（默认）**：这是 0.22 版本后的默认值，优于旧的 `'auto'`，数据方差大时自动降低 gamma。

---

### 2.2 LinearSVC

#### 功能说明
线性支持向量分类。基于 LIBLINEAR，使用坐标下降法优化 hinge loss。比 `SVC(kernel='linear')` **快得多**（尤其是数据量大时）。

#### 什么时候用
- 线性可分的大数据问题（> 1 万样本）
- 文本分类（高维稀疏特征 + 线性核）
- 不需要非线性决策边界

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `C` | float | `1.0` | 正则化参数 |
| `penalty` | str | `'l2'` | 正则化：`'l1'`/`'l2'`。**L1 正则化产生稀疏解（特征选择）** |
| `loss` | str | `'squared_hinge'` | `'hinge'` 或 `'squared_hinge'`（后者收敛更快，默认） |
| `dual` | str/bool | `'auto'` | `n_samples > n_features` 时自动设为 `False`，加速训练 |
| `class_weight` | str/dict | `None` | 类别权重 |
| `max_iter` | int | `1000` | 最大迭代次数 |
| `tol` | float | `1e-4` | 收敛容忍度 |
| `fit_intercept` | bool | `True` | 是否拟合截距 |

#### 代码示例

```python
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# ---------- LinearSVC 没有 predict_proba，需要校准 ----------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LinearSVC(C=1.0, max_iter=10000, random_state=42))
])
pipe.fit(X_train, y_train)

# 如果需要概率输出，用 CalibratedClassifierCV 包装
calibrated = CalibratedClassifierCV(LinearSVC(max_iter=10000), cv=5, method='sigmoid')
calibrated.fit(X_train, y_train)
proba = calibrated.predict_proba(X_test)

# ---------- L1 正则（特征选择）----------
l1_svc = LinearSVC(penalty='l1', C=0.1, dual=False, max_iter=10000)
# penalty='l1' 时 dual 必须为 False
l1_svc.fit(X_train, y_train)
print("非零系数数量:", (l1_svc.coef_ != 0).sum())
```

#### 注意事项

1. **`LinearSVC` 没有 `predict_proba`**，需要概率用 `CalibratedClassifierCV` 包装。
2. **`penalty='l1'` 时 `dual` 必须为 `False`**，否则报错。
3. **比 `SVC(kernel='linear')` 快 10-100 倍**，线性问题优先用这个。

---

### 2.3 NuSVC

#### 功能说明
ν-SVM 分类器。与 `SVC` 类似，但用参数 `nu` 代替 `C` 来控制支持向量的比例和训练误差。

- `nu` ∈ (0, 1]：训练误差的上界和支持向量比例的下界
- 相比 C-SVC，`nu` 的物理含义更直观（你可以直接说"我希望最多 10% 的训练误差"）

**实践中很少使用**，除非有特殊的 `nu` 控制需求。调参比 C-SVC 更不直观。

```python
from sklearn.svm import NuSVC
clf = NuSVC(nu=0.5, kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)
```

---

## 3. 树模型

### 3.1 DecisionTreeClassifier

#### 功能说明
决策树分类器。基于特征的条件划分递归构建树结构，叶节点输出最多出现的类别。

#### 什么时候用
- **可解释性第一优先**：需要向非技术人员解释模型决策逻辑
- 特征交互复杂，线性模型无法捕捉
- 作为集成模型（随机森林、GBDT）的基学习器
- 数据有类别特征（需先做编码）
- 特征尺度不统一但不想做缩放

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `criterion` | str | `'gini'` | 分裂标准：`'gini'`（基尼不纯度）或 `'entropy'`（信息增益）。差别通常很小 |
| `max_depth` | int | `None` | 树的最大深度。**防止过拟合最重要的参数**。`None` 会一直分裂到纯叶节点 |
| `min_samples_split` | int/float | `2` | 内部节点再分裂所需的最小样本数。int = 样本数，float = 比例 |
| `min_samples_leaf` | int/float | `1` | 叶节点最少样本数。**比 min_samples_split 更平滑地防过拟合** |
| `max_features` | int/float/str | `None` | 分裂时考虑的特征数。`'sqrt'`/`'log2'`/int/float。减少可增加树之间的多样性 |
| `class_weight` | str/dict | `None` | `'balanced'` 处理不平衡数据 |
| `min_impurity_decrease` | float | `0.0` | 分裂必须带来的最小不纯度减少量 |
| `ccp_alpha` | float | `0.0` | 成本复杂度剪枝参数（Minimal Cost-Complexity Pruning）。**0.22 版新增，推荐用于后剪枝** |
| `random_state` | int | `None` | 随机种子 |

#### 代码示例

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ---------- 基础用法 ----------
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)
print("DT accuracy:", dt.score(X_test, y_test))

# ---------- 可视化树（需要安装 graphviz）----------
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=[f"f{i}" for i in range(X.shape[1])],
          class_names=['Class0', 'Class1'], max_depth=3)
plt.tight_layout()
# plt.show()

# ---------- 调参防止过拟合 ----------
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1],
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

# ---------- 特征重要性 ----------
importances = dt.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for i in sorted_idx[:10]:
    print(f"Feature {i}: {importances[i]:.4f}")

# ---------- 查看剪枝路径 ----------
path = dt.cost_complexity_pruning_path(X_train, y_train)
print("ccp_alphas:", path.ccp_alphas[:5])
print("impurities:", path.impurities[:5])
```

#### 注意事项 / 踩坑点

1. **默认不限制深度，极度容易过拟合**：训练准确率 100% 但测试一塌糊涂。**永远加 `max_depth` 约束**。
2. **对小数据变化敏感**：旋转数据、轻微扰动可能产生完全不同的树结构。
3. **偏向于取值多的特征**：如果某特征有大量唯一值（如 ID），树会过度利用它。用 `max_features` 可以部分缓解。
4. **特征重要性偏向高基数特征**：`feature_importances_` 会高估取值多的连续特征的重要性。
5. **`ccp_alpha` 是很好的后剪枝方法**：0.22+ 版本推荐用它替代手动调 `max_depth`。

---

### 3.2 RandomForestClassifier

#### 功能说明
随机森林分类器。并行训练多棵决策树（Bagging + 随机特征子集），通过投票聚合结果。是**表格数据最强的开箱即用 baseline**。

核心思想：每棵树在**自助采样 + 随机特征子集**上训练，降低方差而不显著增加偏差。

#### 什么时候用
- **表格数据分类的"万能 baseline"**——无需特征缩放、不需大量调参即可获得不错结果
- 特征维度高（> 几百维），需要筛选特征（`feature_importances_` 天然提供特征选择能力）
- 数据有噪音，需要降低方差的模型
- 需要概率输出（`predict_proba` 基于多棵树投票比例，比你想象的靠谱）
- OOB 评估（`oob_score=True`）可以在不划分验证集的情况下评估泛化能力

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_estimators` | int | `100` | 树的数量。**越大效果越好（但边际递减），训练时间线性增加**。100-500 是常用范围 |
| `max_depth` | int | `None` | 树的最大深度。默认不限制，但建议设一个合理值（如 10-30） |
| `min_samples_split` | int/float | `2` | 内部节点再划分所需最小样本数 |
| `min_samples_leaf` | int/float | `1` | 叶节点最少样本数。增大可以减少过拟合 |
| `max_features` | int/float/str | `'sqrt'` | 每棵树分裂时考虑的最大特征数。**分类默认 `'sqrt'`，回归默认 `None`（全部）**。典型选择：`'sqrt'`/`'log2'`/0.3 |
| `bootstrap` | bool | `True` | 是否使用自助采样（Bagging）。设 `False` 则用全部数据 |
| `oob_score` | bool | `False` | 是否计算袋外分数（Out-of-Bag）。**免费获得验证集评估！**设为 `True` 后用 `.oob_score_` 获取 |
| `n_jobs` | int | `None` | 并行线程数。**设 `-1` 使用全部 CPU**，大量加速 |
| `class_weight` | str/dict | `None` | `'balanced'` 或 `'balanced_subsample'`（每棵树的 bootstrap 内平衡） |
| `criterion` | str | `'gini'` | 同决策树 |
| `max_samples` | int/float | `None` | 每棵树使用的样本数（bootstrap 时）。`None` = 全部，`float` = 比例 |
| `ccp_alpha` | float | `0.0` | 复杂度剪枝参数 |
| `random_state` | int | `None` | 随机种子 |
| `verbose` | int | `0` | 控制训练过程输出，`1` 显示进度 |

#### 代码示例

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# ---------- 基础用法 ----------
rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print("RF accuracy:", rf.score(X_test, y_test))
print("RF AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# ---------- OOB 评估（省一个验证集）----------
rf_oob = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1, random_state=42)
rf_oob.fit(X_train, y_train)
print("OOB score:", rf_oob.oob_score_)
print("Test score:", rf_oob.score(X_test, y_test))

# ---------- 特征重要性 ----------
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]
print("Top 10 features:")
for i in range(10):
    print(f"  {indices[i]}: {feature_importances[indices[i]]:.4f}")

# ---------- RandomizedSearchCV 调参 ----------
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=50, cv=5, scoring='roc_auc',
    n_jobs=-1, verbose=1, random_state=42
)
random_search.fit(X_train, y_train)
print("Best params:", random_search.best_params_)
print("Best CV AUC:", random_search.best_score_)

# ---------- 使用最佳模型 ----------
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]
```

#### 特征重要性深入

```python
import pandas as pd

# 创建特征重要性 DataFrame
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
fi_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(fi_df.head(15))

# 累积重要性（看前多少个特征能解释 95%）
fi_df['cumsum'] = fi_df['importance'].cumsum()
n_95 = (fi_df['cumsum'] <= 0.95).sum() + 1
print(f"前 {n_95} 个特征解释了 95% 的重要性")
```

#### 注意事项 / 踩坑点

1. **`n_estimators` 不是越大越好**：100 棵树通常就够了，500 棵后收益递减。每棵树的训练独立，所以并行可以弥补时间。
2. **OOB score 是免费的验证指标**：当数据量小、不想分验证集时，`oob_score=True` 非常实用。但 OOB 分数通常略低于 CV 分数，正常现象。
3. **特征重要性有偏差**：`feature_importances_` 基于 impurity reduction，会偏向高基数连续特征和类别特征。更可靠的方法是 permutation importance（`sklearn.inspection.permutation_importance`）。
4. **不需要缩放**：树模型基于分裂，对特征的线性变换不敏感。这是它相比 SVM/逻辑回归的巨大优势。
5. **`n_jobs=-1` 在 Windows 上可能有 overhead**：在 Notebook 中有时会因多进程 fork 问题卡住。如果遇到，显式设 `n_jobs=4`。
6. **`class_weight='balanced_subsample'`** 是随机森林特有选项，对极度不平衡数据在每棵树的 bootstrap 子集内平衡，比普通 `'balanced'` 能增加树的多样性。
7. **随机森林不算快**：1000 棵树 × 大数据 = 训练很慢。大数据场景优先考虑 `HistGradientBoostingClassifier`。

---

### 3.3 ExtraTreesClassifier

#### 功能说明
极端随机树分类器。与随机森林的区别：
- **分裂点完全随机选择**（而非搜索最佳分裂点）
- **不进行 Bootstrap 采样**（默认用全部数据）

#### 与 RandomForest 的区别

| 特性 | RandomForest | ExtraTrees |
|------|-------------|------------|
| 分裂方式 | 搜索最优分裂点 | 随机选择分裂点 |
| 采样 | Bootstrap（默认） | 全量数据（默认） |
| 训练速度 | 较慢 | **更快** |
| 方差 | 较低 | **更低**（随机性更大） |
| 偏差 | 较低 | 稍高 |

**什么时候用**：训练速度优先、特征维度特别高（随机分裂反而有用）、想进一步降低方差。

```python
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
et.fit(X_train, y_train)
print("ET accuracy:", et.score(X_test, y_test))
```

---

### 3.4 GradientBoostingClassifier

#### 功能说明
梯度提升分类器。**串行**训练决策树（每棵树拟合前序模型的负梯度/残差），通过逐步"修正错误"来提升精度。

核心公式：每棵新树去拟合之前模型的**负梯度方向**（回归中使用残差，分类中使用对数似然的梯度）。

#### 什么时候用
- **需要最高精度**（通常优于随机森林）
- 数据量中等（< 10 万），GBDT 比 XGBoost/LightGBM 慢很多
- 需要单调性约束（0.23 版新增 `monotonic_cst`）
- 与 XGBoost/LightGBM 对标的 sklearn 原生方案

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_estimators` | int | `100` | 提升阶段数（树的数量）。**learning_rate 小则需要更多** |
| `learning_rate` | float | `0.1` | 学习率（shrinkage）。**每棵树的贡献缩放到此比例**。越小需要越多 n_estimators，泛化通常更好 |
| `max_depth` | int | `3` | 树的深度。GBDT 中树通常很浅（3-8） |
| `subsample` | float | `1.0` | 训练每棵树使用的样本比例。**< 1.0 可以防止过拟合（随机梯度提升）** |
| `min_samples_split` | int/float | `2` | 内部节点最小样本数 |
| `min_samples_leaf` | int/float | `1` | 叶节点最小样本数 |
| `max_features` | int/float/str | `None` | 每棵树使用的特征比例，同随机森林 |
| `loss` | str | `'log_loss'` | 损失函数：`'log_loss'`（分类标准）/`'exponential'`（AdaBoost 等价）/`'deviance'`（同 `'log_loss'`，已弃用） |
| `validation_fraction` | float | `0.1` | 早停用的验证集比例 |
| `n_iter_no_change` | int | `None` | 早停：连续 N 轮验证集无改善则停止。**建议设 5-10** |
| `tol` | float | `1e-4` | 早停容忍度 |
| `ccp_alpha` | float | `0.0` | 剪枝参数 |
| `random_state` | int | `None` | 随机种子 |

#### 与 XGBoost/LightGBM/CatBoost 的关系

| 特性 | sklearn GBDT | XGBoost | LightGBM | CatBoost |
|------|-------------|---------|----------|----------|
| 速度 | 慢 | 快 | **最快** | 中等 |
| 正则化 | 基础 | L1/L2 | L1/L2 | L2 |
| 缺失值 | 需预填充 | 自动处理 | 自动处理 | 自动处理 |
| 类别特征 | 需编码 | 需编码 | 原生支持 | **原生支持** |
| GPU | 无 | 有 | 有 | 有 |
| 分布式 | 无 | 有 | 有 | 有 |
| 安装 | 自带 | 需装 | 需装 | 需装 |

> **结论**：sklearn 新版中推荐用 `HistGradientBoostingClassifier` 替代传统 `GradientBoostingClassifier`。

#### 代码示例

```python
from sklearn.ensemble import GradientBoostingClassifier

# ---------- 基础用法 + 早停 ----------
gbc = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    validation_fraction=0.1,
    n_iter_no_change=10,  # 连续 10 轮无改善则停止
    random_state=42
)
gbc.fit(X_train, y_train)
print("GBDT accuracy:", gbc.score(X_test, y_test))
print("实际用了", gbc.n_estimators_, "棵树")  # 可能因为早停而 < 500

# ---------- 学习曲线 ----------
test_scores = []
for pred in gbc.staged_predict(X_test):
    test_scores.append(roc_auc_score(y_test, pred))

stage_best = np.argmax(test_scores)
print(f"最佳阶段: {stage_best}, AUC = {test_scores[stage_best]:.4f}")

# ---------- 特征重要性 ----------
for i, imp in enumerate(gbc.feature_importances_[:10]):
    print(f"Feature {i}: {imp:.4f}")
```

#### 注意事项 / 踩坑点

1. **容易过拟合**：`n_estimators` 太大 + `learning_rate` 太大 = 训练集完美、测试集崩塌。**一定要用早停**。
2. **`n_estimators` 和 `learning_rate` 负相关**：`learning_rate=0.1` 配合 100 棵树 ≈ `learning_rate=0.01` 配合 1000 棵树。调大学习率 + 早停通常更优。
3. **串行训练，无法并行**：不像随机森林，GBDT 树是顺序的。`n_jobs` 无法加速整体训练（只能在单棵树内并行特征搜索）。
4. **不支持 `class_weight`**：需要处理不平衡数据时，用 `sample_weight` 或过采样/欠采样。
5. **`staged_predict` 和 `staged_predict_proba`** 可以查看每增加一棵树的效果变化，非常有用。
6. **严格不给概率校准**：GBDT 的 `predict_proba` 通常偏差较大（过于集中在 0 和 1），需要 `CalibratedClassifierCV` 校准。

---

### 3.5 HistGradientBoostingClassifier（重点！）

#### 功能说明
基于直方图的梯度提升分类器。**sklearn 0.21 新增，0.22 稳定。是 LightGBM 的 sklearn 原生实现。**

核心创新：
1. **直方图分箱**：将连续特征离散化为 256 个 bin，大幅加速分裂搜索
2. **原生缺失值支持**：训练时自动学习缺失值的最佳处理方向
3. **原生类别特征支持**：`categorical_features` 参数，不需要 One-Hot 编码
4. **大量性能优化**：比传统 GBDT 快 10-100 倍

#### 什么时候用
- **大数据量分类（> 10 万样本）**：性能远超 `GradientBoostingClassifier`
- **数据有缺失值**：不需要预先填充，省掉一道工序
- **特征列里有高基数类别变量**（如城市 ID = 1000 个值）：原生支持，不需要 OHE 膨胀维度
- **需要比 LightGBM/XGBoost 更简单的依赖**：sklearn 自带，不需要额外安装

#### 与 LightGBM 的对比

| 特性 | HistGradientBoostingClassifier | LightGBM |
|------|------------------------------|----------|
| 缺失值 | 自动学习 | 自动学习 |
| 类别特征 | 原生支持 | 原生支持 |
| GPU | 0.24 后无 | 有 |
| 分布式 | 无 | 有 |
| 速度 | 接近 LightGBM（略慢） | 最快 |
| Categorical split 算法 | 二分 | 独有 one-vs-other |
| 安装 | pip install scikit-learn | pip install lightgbm |
| 社区/文档 | sklearn 官方 | 更大更丰富 |
| 参数数量 | 少，简单 | 多，灵活 |

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `loss` | str | `'log_loss'` | 损失函数，分类用 `'log_loss'` |
| `learning_rate` | float | `0.1` | 学习率 |
| `max_iter` | int | `100` | **提升迭代次数（等于树的数量）**。sklearn 这里不用 `n_estimators`！ |
| `max_leaf_nodes` | int | `31` | 每棵树最大叶节点数。**替代 `max_depth` 控制树复杂度**，比用深度更灵活 |
| `max_depth` | int | `None` | 树的最大深度（也可以用，但推荐用 `max_leaf_nodes`） |
| `min_samples_leaf` | int | `20` | 叶节点最少样本数。**默认 20，比传统 GBDT 更保守** |
| `l2_regularization` | float | `0.0` | L2 正则化强度（λ）。增大以防止过拟合 |
| `max_bins` | int | `255` | 直方图的 bin 数量。**越大精度越高但越慢**。最大约 255 |
| `categorical_features` | array-like | `None` | 类别特征的列索引。**原生支持，不需要 One-Hot！** |
| `early_stopping` | str/bool | `'auto'` | 是否早停。`'auto'` 自动使用 10% 验证集 |
| `validation_fraction` | int/float | `0.1` | 用于早停的验证集比例 |
| `n_iter_no_change` | int | `10` | 连续多少轮无改善后停止 |
| `scoring` | str | `'loss'` | 早停的评分标准（`'loss'` = 训练损失） |
| `tol` | float | `1e-7` | 早停容忍度 |
| `class_weight` | str/dict | `None` | `'balanced'` 或手动权重 |
| `random_state` | int | `None` | 随机种子 |
| `verbose` | int | `0` | `1` 显示进度 |

#### 代码示例

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# ---------- 基础用法 ----------
hgbc = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
hgbc.fit(X_train, y_train)
print("HistGB accuracy:", hgbc.score(X_test, y_test))
print("实际迭代次数:", hgbc.n_iter_)

# ---------- 原生缺失值处理 ----------
X_missing = X_train.copy()
X_missing[np.random.rand(*X_missing.shape) < 0.1] = np.nan
X_test_missing = X_test.copy()
X_test_missing[np.random.rand(*X_test_missing.shape) < 0.1] = np.nan

hgbc_missing = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.1, early_stopping=True, random_state=42
)
hgbc_missing.fit(X_missing, y_train)
print("Missing data accuracy:", hgbc_missing.score(X_test_missing, y_test))
# 不需要 fillna！直接 pass NaN 即可

# ---------- 原生类别特征 ----------
# 假设第 5、8、12 列是类别特征
categorical_cols = [5, 8, 12]
hgbc_cat = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    categorical_features=categorical_cols,  # 直接指定，不需要 OHE
    random_state=42
)
hgbc_cat.fit(X_train, y_train)

# ---------- L2 正则化 ----------
hgbc_reg = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.1,
    l2_regularization=1.0,  # L2 正则化防过拟合
    random_state=42
)
hgbc_reg.fit(X_train, y_train)

# ---------- 调参建议 ----------
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [200, 300, 500],
    'max_leaf_nodes': [15, 31, 63],
    'min_samples_leaf': [10, 20, 30],
    'l2_regularization': [0.0, 0.1, 1.0],
}
search = RandomizedSearchCV(
    HistGradientBoostingClassifier(early_stopping=True, random_state=42),
    param_dist, n_iter=30, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
```

#### 注意事项 / 踩坑点

1. **参数名不一样**：`max_iter` 不是 `n_estimators`！`max_leaf_nodes` 代替 `max_depth`。
2. **`max_bins` 上限 255**：数据量大且需要高精度时可设 255；追求速度可设 64 或 128。
3. **`categorical_features` 需要 int 编码**：类别特征的值必须是 0 到 n-1 的整数。可以用 `OrdinalEncoder`。注意缺失值用 -1 表示。
4. **`early_stopping` 默认开启**：用 10% 数据做验证，这是很好的默认行为。如果你的数据已经很小，考虑设为 `False`。
5. **sklearn 版本要求**：需要 sklearn ≥ 0.22。建议 1.0+ 以获得最佳体验。
6. **交互约束**：1.0 版本新增 `interaction_cst` 参数，支持指定哪些特征可以交互。
7. **处理大数据时内存友好**：基于直方图的方式比传统的预排序方法内存使用少很多。

---

## 4. 近邻模型

### 4.1 KNeighborsClassifier

#### 功能说明
K 近邻分类器。对于新样本，找出训练集中距离最近的 K 个点，投票决定类别。**无显式训练过程（惰性学习）**。

#### 什么时候用
- **小数据、低维（< 20 维）**——维度一高，距离就失去意义
- 决策边界非常不规则，参数模型无法拟合
- 快速 baseline（不需要训练）
- 多类别 + 样本量小的场景

#### 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_neighbors` | int | `5` | K 值。**太小过拟合，太大欠拟合**。奇数避免平票 |
| `weights` | str | `'uniform'` | `'uniform'`（等权重）/`'distance'`（距离反比，近的权重大） |
| `algorithm` | str | `'auto'` | `'brute'`/`'kd_tree'`/`'ball_tree'`/`'auto'`。几乎永远用 `'auto'` |
| `metric` | str | `'minkowski'` | 距离度量。`'euclidean'`/`'manhattan'`/`'chebyshev'`/`'cosine'` |
| `p` | int | `2` | Minkowski 距离的 p。`2` = 欧氏，`1` = 曼哈顿 |
| `leaf_size` | int | `30` | 树结构的叶节点大小，影响构建和查询速度 |
| `n_jobs` | int | `None` | 并行数 |

#### 代码示例

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ---------- KNN 必须缩放 ----------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='uniform'))
])
pipe.fit(X_train, y_train)
print("KNN accuracy:", pipe.score(X_test, y_test))

# ---------- 选择最佳 K ----------
k_scores = []
for k in range(1, 31, 2):  # 奇数避免平票
    pipe.set_params(knn__n_neighbors=k)
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    k_scores.append((k, scores.mean()))

best_k = max(k_scores, key=lambda x: x[1])
print(f"Best K = {best_k[0]}, CV accuracy = {best_k[1]:.4f}")

# ---------- 距离加权 ----------
knn_dist = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
])
knn_dist.fit(X_train, y_train)
print("Distance-weighted KNN accuracy:", knn_dist.score(X_test, y_test))
```

#### 注意事项 / 踩坑点

1. **维度诅咒**：特征维数 > 20 时距离趋于相同，KNN 基本失效。先降维（PCA）再用。
2. **必须缩放**：不等 `StandardScaler` 直接用 KNN，尺度大的特征将主导距离计算。
3. **预测慢**：训练是 O(1)（只是记住数据），但**预测是 O(n)**（需要扫描所有训练样本）。大数据预测时体验极差。
4. **内存大**：需要保存全部训练数据。10 万样本 × 1000 维 = 内存爆炸。
5. **K 值选择**：常用经验 `K ≈ sqrt(n_samples)`。在验证集上搜索确认。
6. **`metric='cosine'` 时不需要缩放**：余弦距离对尺度免疫。

---

## 5. 朴素贝叶斯

### 5.1 GaussianNB

#### 功能说明
高斯朴素贝叶斯。假设每个类别的特征服从正态分布，根据训练数据估计均值和方差，预测时用贝叶斯定理计算后验概率。

#### 什么时候用
- 特征近似正态分布
- 特征维数高、样本少（维度 >> 样本数也能用）
- 在线学习（支持 `partial_fit`）
- 快速 baseline，尤其是在维度很高时

```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(var_smoothing=1e-9)  # var_smoothing 防止方差为 0
gnb.fit(X_train, y_train)
print("GNB accuracy:", gnb.score(X_test, y_test))
```

### 5.2 MultinomialNB

#### 功能说明
多项式朴素贝叶斯。假设特征是从多项分布中抽取的（适合整数计数特征，如词频、TF-IDF）。

#### 什么时候用
- **文本分类（经典搭配）**：TF-IDF 特征 + MultinomialNB 是文本分类的标准 baseline
- 特征是非负整数（计数数据）

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# 文本分类示例
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(texts)
mnb = MultinomialNB(alpha=1.0)  # alpha 是拉普拉斯平滑参数
mnb.fit(X_tfidf, y_labels)
```

| 参数 | 说明 |
|------|------|
| `alpha` | 平滑参数（拉普拉斯/贝叶斯平滑）。默认 1.0。设为 0 不平滑，可能导致零概率 |
| `fit_prior` | 是否学习类先验。默认 `True` |
| `class_prior` | 手动指定类先验概率 |

### 5.3 BernoulliNB

#### 什么时候用
- **二值特征**（0/1 表示是否出现）：短文本分类、购物篮分析
- 特征明确是"出现/不出现"而非"出现多少次"

```python
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB(alpha=1.0, binarize=0.5)  # binarize 阈值
bnb.fit(X_binary, y_labels)
```

**朴素贝叶斯三种模型对比**：

| 类型 | 特征假设 | 典型场景 | 分布假设 |
|------|---------|---------|---------|
| GaussianNB | 连续，正态分布 | 通用分类、低维 | N(μ, σ²) |
| MultinomialNB | 非负整数计数 | 文本分类 | 多项分布 |
| BernoulliNB | 二值 0/1 | 短文本、布尔特征 | 伯努利分布 |

---

## 6. 集成方法

### 6.1 AdaBoostClassifier

#### 功能说明
自适应增强。**串行**训练一系列弱学习器（默认是深度为 1 的决策树桩），后续学习器聚焦于之前分错的样本（提高错分样本的权重）。

```python
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,  # 每棵树的贡献缩放
    random_state=42
)
ada.fit(X_train, y_train)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_estimators` | `50` | 弱学习器数量 |
| `learning_rate` | `1.0` | 每个学习器的贡献缩放。越小需要越多 `n_estimators` |
| `algorithm` | `'SAMME.R'` | `'SAMME'`（离散，不能输出概率）/`'SAMME.R'`（实数，支持概率，更快，**推荐**） |
| `estimator` | `None` | 默认用 `max_depth=1` 的决策树。可以换成其他弱学习器 |

**注意**：AdaBoost 对离群点和噪声敏感（会不断提高错分样本的权重）。

### 6.2 BaggingClassifier

#### 功能说明
自主聚合分类器。对训练数据做 Bootstrap 采样，用同一种基学习器训练多个模型，投票/平均聚合。

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

# 用 SVM 作为基学习器的 Bagging
bag = BaggingClassifier(
    estimator=SVC(),  # 基学习器
    n_estimators=50,
    max_samples=0.8,  # 每个学习器用 80% 数据
    max_features=0.8,  # 每个学习器用 80% 特征
    bootstrap=True,    # 有放回采样
    bootstrap_features=False,  # 特征无放回采样
    n_jobs=-1,
    random_state=42
)
bag.fit(X_train, y_train)

# oob_score 也可以用在 BaggingClassifier
bag_oob = BaggingClassifier(n_estimators=100, oob_score=True, random_state=42)
bag_oob.fit(X_train, y_train)
print("OOB score:", bag_oob.oob_score_)
```

**什么时候用**：降低高方差模型（如深度决策树、SVM）的方差。基学习器越不稳定（方差大），Bagging 的提升越大。

### 6.3 VotingClassifier（投票集成）

#### 功能说明
投票集成。组合多个**不同类型**的分类器，通过投票（硬投票）或概率平均（软投票）做出最终预测。

#### 硬投票 vs 软投票

| 方式 | 原理 | 何时用 |
|------|------|--------|
| 硬投票（Hard） | 每个分类器投 1 票，多数决定 | 分类器没有 `predict_proba` |
| 软投票（Soft） | 平均每个分类器的预测概率 | 所有分类器都有 `predict_proba`，**通常效果更好** |

#### 代码示例

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# ---------- 构造多个异构分类器 ----------
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
# SVC 默认没有 predict_proba，需要开启
svc = SVC(probability=True, random_state=42)

# ---------- 软投票 ----------
voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('gb', gb),
    ],
    voting='soft',  # 软投票
    weights=[1, 2, 2],  # 给 rf 和 gb 更高权重
    n_jobs=-1
)
voting_clf.fit(X_train, y_train)
print("Voting accuracy:", voting_clf.score(X_test, y_test))

# ---------- 硬投票 ----------
voting_hard = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gb', gb), ('svc', svc)],
    voting='hard',
    weights=[1, 2, 2, 1],
    n_jobs=-1
)
voting_hard.fit(X_train, y_train)
```

#### 注意事项

1. **软投票通常优于硬投票**：概率中包含更多信息（置信度），比离散的类别投票更精细。
2. **`weights` 需要手动调**：基于各分类器的 CV 表现分配权重。通常给树模型更高的权重。
3. **`n_jobs` 用于并行训练各子分类器**：如果基学习器本身就并行（如 `RandomForest`），注意不要过度竞争 CPU。
4. **所有基学习器都要 fit 过**：VotingClassifier 调用 fit 时会依次训练每一个。

### 6.4 StackingClassifier（重点！）

#### 功能说明
堆叠分类器。训练多个**第一层分类器**（基学习器），再用**第二层分类器**（元学习器，通常是逻辑回归）把第一层的输出作为输入来做最终预测。

**Kaggle 竞赛中常见的高级集成方法**，通常能略优于 VotingClassifier。

#### 代码示例

```python
from sklearn.ensemble import StackingClassifier

# ---------- 第一层：异构基学习器 ----------
base_estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
]

# 元学习器：默认是逻辑回归
stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),  # 第二层
    cv=5,               # 用 5 折交叉验证生成第一层的"干净"预测（防止数据泄露）
    stack_method='predict_proba',  # 用概率而非类别作为第二层的输入。'auto' 会自动选择
    n_jobs=-1,
    passthrough=False   # True = 把原始特征也传给元学习器
)
stacking.fit(X_train, y_train)
print("Stacking accuracy:", stacking.score(X_test, y_test))

# ---------- 如果 passthrough=True ----------
stacking_pt = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=True,  # 原始特征 + 第一层输出 → 第二层
    n_jobs=-1
)
stacking_pt.fit(X_train, y_train)
print("Stacking (passthrough) accuracy:", stacking_pt.score(X_test, y_test))
```

#### 注意事项 / 踩坑点

1. **`cv` 很重要**：必须用交叉验证生成第一层的 out-of-fold 预测，否则会导致严重数据泄露（用训练集预测训练集 → 第二层学到的是第一层的过拟合模式）。
2. **`stack_method`**：`'predict_proba'` 给第二层更多信息（每类的概率），推荐用于分类。但需要所有基学习器都支持 `predict_proba`。
3. **元学习器推荐**：分类用 `LogisticRegression`（简单、可解释）；回归用 `Ridge`。
4. **训练慢**：CV×基学习器数量 的训练，比如 `cv=5` + 5 个基学习器 = 训练 25 次。`n_jobs=-1` 是必选项。
5. **基学习器多样性比单个的精度更重要**：Stacking 的核心是"不同模型犯不同的错误"。

---

## 7. 其他

### 7.1 MLPClassifier（多层感知机）

#### 功能说明
多层感知机分类器（sklearn 的神经网络实现）。一个或多个隐藏层的全连接神经网络。

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 两层隐藏层：100 神经元 + 50 神经元
    activation='relu',
    solver='adam',
    alpha=0.0001,          # L2 正则化
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
mlp.fit(X_train, y_train)
print("MLP accuracy:", mlp.score(X_test, y_test))
```

#### 什么时候用、局限性

- **什么时候用**：非线性关系复杂、其他模型已经尝试过、需要一个"万能拟合器"
- **局限性**：
  - **必须做特征缩放**：不缩放等于自杀
  - **参数极度敏感**：`hidden_layer_sizes`、`alpha`、`learning_rate_init` 都需要仔细调
  - **训练不稳定**：每次训练结果可能不同
  - **可解释性差**：黑盒模型
  - **小数据不出彩**：样本少于几千时很容易过拟合
  - **大数据推荐用 PyTorch/TensorFlow**：MLPClassifier 无法 GPU 加速、不支持 CNN/RNN

### 7.2 DummyClassifier

#### 功能说明
虚拟分类器。完全不看特征、只依赖标签分布的"智能猜答案"分类器。用作**性能下限 baseline**。

```python
from sklearn.dummy import DummyClassifier

# 策略对比
for strategy in ['most_frequent', 'stratified', 'uniform', 'constant']:
    dummy = DummyClassifier(strategy=strategy, constant=1 if strategy == 'constant' else None, random_state=42)
    dummy.fit(X_train, y_train)
    print(f"Dummy ({strategy:>15}): {dummy.score(X_test, y_test):.4f}")
```

| strategy | 预测方式 |
|----------|---------|
| `'most_frequent'` | 永远预测出现最多的类别 |
| `'stratified'` | 按训练集类别比例随机预测 |
| `'uniform'` | 均匀随机预测所有类别 |
| `'constant'` | 永远预测指定类别（`constant` 参数） |

**重要性**：如果花了好几天调参的复杂模型只比 `DummyClassifier` 好 2%，该反思数据质量或特征了。

---

## 8. 分类模型选择速查表

| 模型 | 数据规模 | 需缩放 | 缺失值 | 可解释性 | 训练速度 | 分类精度 | 概率输出 | 典型场景 |
|------|---------|--------|--------|---------|---------|---------|---------|---------|
| LogisticRegression | 小 - 大 | **是** | 否 | **高**（coef\_） | 快 | 中 | 是 | baseline |
| SGDClassifier | **大**（>10万） | **是** | 否 | 高 | **快** | 中 | loss='log_loss' 时 | 在线学习 |
| RidgeClassifier | 小 - 中 | 是 | 否 | 高 | **快** | 中 | 否（需校准） | 共线性 |
| SVC | **小**（<1万） | **是** | 否 | 中（核难以解释） | 慢 | **高** | 需开启 | 非线性、高维 |
| LinearSVC | 小 - 大 | 是 | 否 | 高 | 快 | 中 - 高 | 否（需校准） | 文本分类 |
| DecisionTree | 小 - 中 | 否 | 否 | **最高** | 快 | 低 - 中 | 是 | 需要可解释时 |
| RandomForest | 中 - 大 | 否 | 否 | 高（feature_importances\_） | 中 | **高** | 是 | **万能 baseline** |
| ExtraTrees | 中 - 大 | 否 | 否 | 高 | 中 - 快 | 高 | 是 | 速度优先的 RF |
| GradientBoosting | 小 - 中 | 否 | 否 | 中 | 慢 | **很高** | 是（需校准） | 精度优先 |
| **HistGradientBoosting** | **中 - 大** | 否 | **原生支持** | 中 | **快** | **很高** | 是 | **大数据 + 缺失值** |
| KNeighbors | **小** | **是** | 否 | 低 | 训练快/预测慢 | 低 - 中 | 是 | 低维小数据 |
| GaussianNB | 小 - 中 | 否 | 否 | 中 | **极快** | 低 | 是 | 高维、独立性假设 |
| MultinomialNB | 小 - 大 | 否 | 否 | 中 | **极快** | 中 | 是 | 文本分类 |
| AdaBoost | 小 - 中 | 否 | 否 | 中 | 中 | 中 | 是 | 弱学习器提升 |
| VotingClassifier | 任意 | 按基学习器 | 按基学习器 | 低 | 慢（取决于基） | **很高** | 是 | 各异构模型集成 |
| StackingClassifier | 中 - 大 | 按基学习器 | 按基学习器 | 低 | 很慢 | **最高** | 是 | Kaggle 竞赛 |
| MLPClassifier | 中 - 大 | **是** | 否 | 低（黑盒） | 慢 | 高 | 是 | 非线性模式 |
| DummyClassifier | 任意 | 否 | 否 | — | 瞬间 | 下限 | 是 | baseline 对比 |

---

## 相关概念

- [[回归模型API]] — 回归任务的所有 sklearn 模型详解
- [[数据预处理API]] — StandardScaler, OneHotEncoder, OrdinalEncoder 等
- [[模型选择与评估API]] — GridSearchCV, RandomizedSearchCV, cross_val_score, 分类指标
- [[特征选择API]] — SelectKBest, RFE, SelectFromModel
- [[管道与特征工程]] — Pipeline, ColumnTransformer, FunctionTransformer
- [[校准曲线与概率校准]] — CalibratedClassifierCV, calibration_curve
