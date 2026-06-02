# Scikit-Learn 概述与核心设计

---

## 1. Scikit-Learn 是什么

### 功能说明

Scikit-Learn 是 Python 生态中最流行的传统机器学习库，构建在 NumPy、SciPy 和 Matplotlib 之上。它提供了一整套统一 API 的机器学习算法实现，覆盖从数据预处理到模型部署的完整工作流。

### 什么时候用

| 场景 | 是否适合 Sklearn |
|------|:---:|
| 表格数据的分类/回归/聚类 | ✅ 首选 |
| 传统 ML 算法（SVM、随机森林、XGBoost 风格） | ✅ 核心场景 |
| 特征工程与数据预处理 | ✅ 最佳实践 |
| 模型评估与超参数调优 | ✅ 内置完善 |
| Pipeline 构建生产级 ML 流水线 | ✅ 核心能力 |
| 深度学习（CNN/RNN/Transformer） | ❌ 用 PyTorch/TF |
| 超大规模数据（TB 级） | ❌ 考虑 Spark MLlib |
| 在线学习 / 流式学习 | ⚠️ 部分支持（SGD 系列） |
| 图像/文本/语音原始数据处理 | ⚠️ 需配合其他库 |

### 核心特点

1. **统一 API**：所有模型遵循同一套接口约定（`fit` / `predict` / `transform`），降低切换成本
2. **纯 NumPy 实现**：大部分算法用 Cython 优化，速度快且可审计
3. **完善的文档**：每个 API 都有 Gallery 示例 + 理论说明
4. **活跃的社区**：GitHub 56k+ stars，每月数百万下载
5. **BSD 协议**：商用友好，无版权顾虑

---

## 2. 核心设计哲学——Estimator API

Scikit-Learn 最核心的设计就是 **Estimator API**：所有对象（模型、预处理器、特征选择器）都共享同一套接口约定。这使得 Pipeline、GridSearchCV 等高级功能可以无缝组合任意组件。

### 2.1 三大接口

#### 2.1.1 Estimator（估计器）—— 所有模型的基类

**功能说明**：能够从数据中"学习"参数的对象。所有 sklearn 模型（分类器、回归器、聚类器）和预处理器都属于 Estimator。

**核心方法**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `fit(X, y)` | `self.fit(X, y)` | 从训练数据中学习，返回 `self` |
| `get_params(deep=True)` | 返回 `dict` | 获取构造参数 |
| `set_params(**params)` | 返回 `self` | 重新设置构造参数 |

**代码示例**：

```python
from sklearn.linear_model import LogisticRegression

# 创建估计器（此时还没有学习任何东西）
clf = LogisticRegression(C=1.0, penalty='l2')

# 获取当前参数
print(clf.get_params())
# {'C': 1.0, 'class_weight': None, 'dual': False, ...}

# 从数据中学习
clf.fit(X_train, y_train)

# fit 返回 self，所以可以链式调用
# clf.fit(X_train, y_train).predict(X_test)  # 合法但不推荐（可读性差）
```

**注意事项**：
- `fit` 返回 `self` 是为了支持链式调用，但**不要滥用链式调用**——分离 fit 和 predict 便于调试
- `get_params` 只返回构造参数（`__init__` 中定义的参数），不返回学到的参数（如 `coef_`）

#### 2.1.2 Transformer（转换器）—— 数据变换器

**功能说明**：能够对数据进行某种"变换"的 Estimator。典型例子：标准化、编码、特征选择、降维。

**核心方法**（继承 Estimator 后新增）：

| 方法 | 签名 | 说明 |
|------|------|------|
| `transform(X)` | `X_new = self.transform(X)` | 将学到变换应用到 X，返回变换后的数据 |
| `fit_transform(X, y=None)` | `X_new = self.fit(X, y).transform(X)` | 先 fit 再 transform 的便捷方法 |

**代码示例**：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 方式一：分步操作（推荐用于教学/调试）
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 方式二：fit_transform 快捷写法（推荐用于训练集）
X_train_scaled = scaler.fit_transform(X_train)  # 一次性完成 fit + transform
X_test_scaled = scaler.transform(X_test)        # 测试集只能用 transform！
```

**注意事项/踩坑点**：

1. **永远不要在测试集上 `fit` 或 `fit_transform`** —— 这是数据泄漏的经典来源：

   ```python
   # ❌ 错误！测试集上用了 fit_transform，相当于偷看测试数据
   X_test_scaled = scaler.fit_transform(X_test)

   # ✅ 正确：测试集只用 transform，使用训练集学到的 μ 和 σ
   X_test_scaled = scaler.transform(X_test)
   ```

2. **`fit_transform` vs `fit` + `transform` 的区别**：
   - 在大多数 Transformer 中，`fit_transform` 只是快捷方式，结果完全相同
   - 但在**某些特殊 Transformer** 中（如 `PCA` 的 `randomized` 模式），`fit_transform` 可能因为算法优化而产生微小差异
   - 此外，`fit_transform` 返回的是 **新的 numpy 数组**，而 `fit` + `transform` 允许你检查中间状态

3. **Transformer 的 `transform` 输入必须与 `fit` 时的特征维度一致**：

   ```python
   scaler.fit(X_train)  # X_train.shape = (800, 5)
   scaler.transform(X_test)  # ✅ X_test.shape = (200, 5)
   scaler.transform(X_test[:, :3])  # ❌ 只有 3 列，会报错！
   ```

#### 2.1.3 Predictor（预测器）—— 能做预测的模型

**功能说明**：在 fit 之后能够对新数据进行预测的 Estimator。

**核心方法**（继承 Estimator 后新增）：

| 方法 | 签名 | 说明 |
|------|------|------|
| `predict(X)` | `y_pred = self.predict(X)` | 返回预测标签（分类）或预测值（回归） |
| `predict_proba(X)` | `y_proba = self.predict_proba(X)` | 返回各个类别的概率（仅分类器） |
| `predict_log_proba(X)` | `y_log_proba = self.predict_log_proba(X)` | 返回对数概率，数值更稳定 |
| `score(X, y)` | `score = self.score(X, y)` | 返回默认评估指标（分类器用 accuracy，回归器用 R²） |
| `decision_function(X)` | `scores = self.decision_function(X)` | 返回决策函数的原始分数（仅部分分类器） |

**代码示例**：

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测类别
y_pred = clf.predict(X_test)

# 预测概率（用于需要置信度的场景，如阈值调优）
y_proba = clf.predict_proba(X_test)
# y_proba.shape = (n_samples, n_classes)

# 取"最可能是正类"的概率
positive_proba = y_proba[:, 1]

# 决策函数分数（用于 ROC 曲线等，不需要概率的场景更快）
scores = clf.decision_function(X_test)

# 默认评分
print(f"准确率: {clf.score(X_test, y_test):.4f}")
```

**注意事项/踩坑点**：

1. **`predict_proba` 并不总是校准过的** — 即使概率输出是合理的，也不代表概率是"真的概率"。如果需要校准概率，使用 `CalibratedClassifierCV`：

   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrated_clf = CalibratedClassifierCV(RandomForestClassifier(), cv=5)
   calibrated_clf.fit(X_train, y_train)
   ```

2. **有些模型没有 `predict_proba`**（如 `SVC` 默认 `probability=False`），需要显式设置参数：

   ```python
   svc = SVC(probability=True)  # 启用概率输出（会降低训练速度）
   ```

3. **`score` 方法的默认指标不代表问题的真实需求** — 分类默认用准确率，但如果类别不平衡，准确率没有意义。始终用 `sklearn.metrics` 中的专用函数手动评估。

---

### 2.2 fit / transform / predict 模式——数据流全景

#### 设计原理

三个方法的分工反映了机器学习工作流中**信息流向**的约束：

```
┌──────────────────────────────────────────────┐
│                  原始数据                      │
│          X_raw.shape = (n, p)                 │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
     ┌─────────────────────────┐
     │   Transformer.fit(X_train) │  ← 计算 μ、σ、分位数等统计量
     │   X_train_trans = transform │  ← 应用变换到训练集
     └─────────────────────────┘
                   │
                   ▼
     ┌─────────────────────────┐
     │   Estimator.fit(X_train_trans, y)│ ← 学习模型参数（权重、树结构）
     └─────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   predict(X_test)      score(X_test, y_test)
   生成预测值              评估性能
```

#### 为什么 fit 只能在训练集上做（防止数据泄漏）

**数据泄漏**是指测试数据的信息以任何形式"污染"了训练过程，导致模型在测试集上的表现虚高，但在生产环境中失效。

**经典泄漏场景**：

```python
# ❌ 泄漏场景 1：在整个数据集上做标准化再划分
X_scaled = StandardScaler().fit_transform(X)  # fit 看到了全部数据！
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# ❌ 泄漏场景 2：在整个数据集上做特征选择
selector = SelectKBest(k=10).fit(X, y)  # fit 看到了测试集的 y！
X_selected = selector.transform(X)
X_train, X_test = train_test_split(X_selected, y, test_size=0.2)

# ✅ 正确流程：先划分，再对训练集 fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler().fit(X_train)           # 只 fit 训练集
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)          # 用训练集的 μ 和 σ 变换测试集
```

**用 Pipeline 防止泄漏**：

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Pipeline 在交叉验证的每一折中自动只在训练 fold 上 fit
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('clf', LogisticRegression())
])

# 每一折都正确隔离了数据泄漏
scores = cross_val_score(pipe, X, y, cv=5)
print(f"CV scores: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

### 2.3 参数约定

#### 构造参数（超参数）

**功能说明**：在 `__init__` 中设置，控制模型的结构和行为。这些参数在 `fit` 之前确定，`fit` 过程中不变。

**关键约定**：
- 通过 `__init__` 传入
- 可通过 `get_params()` 获取
- 可通过 `set_params()` 修改（必须在 `fit` 之前）
- 命名通常全小写

**代码示例**：

```python
from sklearn.linear_model import Ridge

# 构造参数在 __init__ 中设置
ridge = Ridge(alpha=1.0, fit_intercept=True, solver='auto')

# 查看所有构造参数
params = ridge.get_params()
print(params['alpha'])  # 1.0

# 在 fit 之前修改构造参数
ridge.set_params(alpha=10.0)

# 现在 fit 会用 alpha=10.0
ridge.fit(X_train, y_train)
```

#### 学到的参数（attributes_）

**功能说明**：在 `fit` 过程中从数据中学到的参数，以**下划线 `_` 结尾**。

**代码示例**：

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
print(hasattr(lr, 'coef_'))  # False —— 还没 fit

lr.fit(X_train, y_train)

print(lr.coef_)      # [1.2, -0.5, 3.1]  学到的权重
print(lr.intercept_)  # -0.8              学到的截距
print(lr.n_features_in_)  # 3              学习时输入的特征数
print(lr.feature_names_in_)  # ['a', 'b', 'c']  如果输入是 DataFrame

# 不同模型的学到参数示例：
# StandardScaler: mean_, var_, scale_, n_features_in_
# PCA:           components_, explained_variance_, mean_, n_components_
# KMeans:        cluster_centers_, labels_, inertia_, n_iter_
# DecisionTreeClassifier: feature_importances_, tree_, n_classes_
```

**注意事项/踩坑点**：

1. **检查是否有下划线确定是否已 fit**：
   ```python
   from sklearn.utils.validation import check_is_fitted
   check_is_fitted(scaler)  # 如果没 fit 会抛出 NotFittedError
   ```

2. **`get_params()` 不返回学到的参数**：
   ```python
   clf = LogisticRegression().fit(X, y)
   clf.get_params()    # {'C': 1.0, 'penalty': 'l2', ...}
   clf.coef_           # array([[0.5, -1.2, 3.0]])  ← 这不在 get_params 返回中
   ```

3. **`set_params` 后必须重新 `fit`**：
   ```python
   clf = RandomForestClassifier(n_estimators=100).fit(X, y)
   clf.set_params(n_estimators=200)  # 修改了构造参数
   # clf 此时处于不一致状态——内部仍是 100 棵树，但 get_params 返回 200
   clf.fit(X, y)  # 必须重新 fit
   ```

---

## 3. 数据格式约定

### 3.1 X 的格式规范

**规定**：`(n_samples, n_features)` 的 **二维** 数组。

| 数据类型 | 示例 | 是否支持 |
|----------|------|:---:|
| NumPy ndarray (2D) | `np.array([[1,2],[3,4]])` | ✅ |
| Pandas DataFrame | `pd.DataFrame({...})` | ✅ (1.0+ 原生支持) |
| SciPy sparse matrix | `scipy.sparse.csr_matrix(...)` | ✅ (部分算法) |
| Python list of lists | `[[1,2],[3,4]]` | ⚠️ 自动转换 |
| 一维数组 | `np.array([1,2,3])` | ❌ 必须 reshape |

**常见坑点——一维特征向量**：

```python
# ❌ 错误：单特征时容易误写成一维
X = np.array([1, 2, 3, 4, 5])  # shape = (5,)
# 大部分 sklearn API 会警告甚至报错

# ✅ 正确
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # shape = (5, 1)
```

### 3.2 y 的格式规范

**规定**：`(n_samples,)` 的 **一维** 数组或 Series。

```python
# ✅ 合法的 y 格式
y = np.array([0, 1, 1, 0])
y = pd.Series([0, 1, 1, 0])
y = [0, 1, 1, 0]

# ⚠️ 不推荐但通常也能工作
y = np.array([0, 1, 1, 0]).reshape(-1, 1)  # 二维 (4,1) - sklearn 会给出警告
```

### 3.3 稀疏矩阵支持

```python
from scipy.sparse import csr_matrix

# 文本特征通常很稀疏（大量 0）
X_sparse = csr_matrix(X_dense)

# 大部分算法都能处理稀疏矩阵，但少数不行（如某些核方法）
scaler = StandardScaler(with_mean=False)  # 稀疏矩阵不能去均值
X_scaled = scaler.fit_transform(X_sparse)  # 返回的仍是稀疏矩阵
```

**哪些操作支持稀疏矩阵**：
- ✅ `MaxAbsScaler`、`MinMaxScaler`（`with_mean=False` 的 `StandardScaler`）
- ✅ `OneHotEncoder(sparse_output=True)`
- ✅ `HashingVectorizer`、`TfidfVectorizer`
- ❌ PCA（不能直接处理稀疏，需要用 `TruncatedSVD`）
- ❌ `StandardScaler(with_mean=True)` —— 零均值化会破坏稀疏性

### 3.4 pandas DataFrame 集成（set_output）

```python
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
import pandas as pd

# 全局设置：所有 Transformer 返回 DataFrame
set_config(transform_output="pandas")

# 或者针对单个 Transformer
scaler = StandardScaler().set_output(transform="pandas")

X_train = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
X_transformed = scaler.fit_transform(X_train)

print(type(X_transformed))  # <class 'pandas.core.frame.DataFrame'>
print(X_transformed.columns.tolist())  # ['a', 'b']  ← 保持列名
```

---

## 4. 安装与版本

### 4.1 安装

```bash
# 标准安装
pip install scikit-learn

# 指定版本
pip install scikit-learn==1.5.0

# conda 安装（推荐，自动处理 C 依赖）
conda install scikit-learn

# 从源码安装最新开发版
pip install --pre --extra-index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn

# 验证安装
python -c "import sklearn; print(sklearn.__version__)"
```

### 4.2 依赖关系

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | ≥ 3.9 | — |
| NumPy | ≥ 1.19.5 | 数组计算核心 |
| SciPy | ≥ 1.6.0 | 稀疏矩阵、优化、统计 |
| joblib | ≥ 1.2.0 | 并行计算、模型持久化 |
| threadpoolctl | ≥ 3.1.0 | 控制底层线程数 |

### 4.3 常用 import 约定

```python
# 数据集
from sklearn import datasets
from sklearn.datasets import load_iris, load_digits, fetch_california_housing

# 预处理
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# 模型选择
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit

# 线性模型
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso

# 集成学习
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 评估指标
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# 管道
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

# 模型持久化
import joblib
```

### 4.4 版本查看与兼容性

```python
import sklearn
print(sklearn.__version__)  # 例如 '1.5.0'

# 查看各依赖版本
import sklearn; sklearn.show_versions()
# 输出：
# System:
#     python: 3.10.0
#    machine: Windows-10...
# Python dependencies:
#       numpy: 1.26.0
#       scipy: 1.12.0
# ...
```

---

## 5. 模块速查表

| 模块 | 用途 | 典型 API | 一句话说明 |
|------|------|----------|-----------|
| `sklearn.preprocessing` | 数据预处理 | `StandardScaler`, `LabelEncoder`, `OneHotEncoder`, `PolynomialFeatures` | 缩放、编码、离散化、多项式生成 |
| `sklearn.impute` | 缺失值填充 | `SimpleImputer`, `KNNImputer`, `IterativeImputer`, `MissingIndicator` | 均值/中位数/KNN/多重插补 |
| `sklearn.feature_extraction` | 特征提取 | `TfidfVectorizer`, `CountVectorizer`, `DictVectorizer` | 文本转特征、字典转特征 |
| `sklearn.feature_extraction.text` | 文本特征 | `CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer` | 词袋、TF-IDF、特征哈希 |
| `sklearn.feature_extraction.image` | 图像特征 | `extract_patches_2d`, `PatchExtractor` | 图像 patch 提取 |
| `sklearn.feature_selection` | 特征选择 | `SelectKBest`, `RFE`, `SelectFromModel`, `VarianceThreshold` | 过滤式、包裹式、嵌入式的特征选择 |
| `sklearn.decomposition` | 降维/矩阵分解 | `PCA`, `TruncatedSVD`, `NMF`, `FastICA`, `FactorAnalysis`, `LatentDirichletAllocation` | 线性降维、主题模型 |
| `sklearn.manifold` | 流形学习 | `TSNE`, `MDS`, `Isomap`, `LocallyLinearEmbedding` | 非线性降维（主要用于可视化） |
| `sklearn.discriminant_analysis` | 判别分析 | `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` | LDA 降维 + 分类 |
| `sklearn.linear_model` | 线性模型 | `LogisticRegression`, `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDClassifier`, `SGDRegressor`, `Perceptron`, `PassiveAggressiveClassifier` | 分类、回归、正则化、在线学习 |
| `sklearn.tree` | 决策树 | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `plot_tree` | 树模型及可视化 |
| `sklearn.ensemble` | 集成方法 | `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `AdaBoostClassifier`, `AdaBoostRegressor`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `VotingClassifier`, `StackingClassifier`, `BaggingClassifier` | Bagging、Boosting、Stacking、Voting |
| `sklearn.svm` | 支持向量机 | `SVC`, `SVR`, `LinearSVC`, `LinearSVR`, `NuSVC`, `OneClassSVM` | 分类、回归、异常检测 |
| `sklearn.neighbors` | 近邻方法 | `KNeighborsClassifier`, `KNeighborsRegressor`, `NearestNeighbors`, `RadiusNeighborsClassifier`, `LocalOutlierFactor`, `NeighborhoodComponentsAnalysis` | k-NN 分类/回归、局部异常因子 |
| `sklearn.naive_bayes` | 朴素贝叶斯 | `GaussianNB`, `MultinomialNB`, `ComplementNB`, `BernoulliNB`, `CategoricalNB` | 文本分类、连续/离散特征 |
| `sklearn.cluster` | 聚类 | `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `HDBSCAN`, `OPTICS`, `AgglomerativeClustering`, `Birch`, `MeanShift`, `SpectralClustering`, `AffinityPropagation` | 划分聚类、密度聚类、层次聚类 |
| `sklearn.mixture` | 高斯混合模型 | `GaussianMixture`, `BayesianGaussianMixture` | 软聚类、密度估计 |
| `sklearn.neural_network` | 神经网络 | `MLPClassifier`, `MLPRegressor`, `BernoulliRBM` | 全连接网络（浅层，非深度学习） |
| `sklearn.kernel_approximation` | 核近似 | `RBFSampler`, `Nystroem`, `AdditiveChi2Sampler` | 大样本下近似核方法 |
| `sklearn.kernel_ridge` | 核岭回归 | `KernelRidge` | 核化的 Ridge 回归 |
| `sklearn.gaussian_process` | 高斯过程 | `GaussianProcessClassifier`, `GaussianProcessRegressor`, 各种 `Kernel` | 回归+不确定性估计、贝叶斯优化 |
| `sklearn.cross_decomposition` | 交叉分解 | `PLSRegression`, `PLSCanonical`, `CCA` | 偏最小二乘、典型相关分析 |
| `sklearn.isotonic` | 保序回归 | `IsotonicRegression` | 单调约束回归 |
| `sklearn.covariance` | 协方差估计 | `EllipticEnvelope`, `EmpiricalCovariance`, `MinCovDet`, `LedoitWolf`, `OAS`, `ShrunkCovariance`, `GraphicalLasso` | 异常检测（基于马氏距离）、稀疏逆协方差 |
| `sklearn.semi_supervised` | 半监督学习 | `SelfTrainingClassifier`, `LabelPropagation`, `LabelSpreading` | 利用未标注数据 |
| `sklearn.multiclass` | 多分类策略 | `OneVsRestClassifier`, `OneVsOneClassifier`, `OutputCodeClassifier` | 将二分类器扩展为多分类 |
| `sklearn.multioutput` | 多输出 | `MultiOutputClassifier`, `MultiOutputRegressor`, `ClassifierChain`, `RegressorChain` | 多标签/多目标学习 |
| `sklearn.calibration` | 概率校准 | `CalibratedClassifierCV`, `calibration_curve` | Platt scaling、保序回归校准 |
| `sklearn.metrics` | 评估指标 | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `log_loss`, `mean_squared_error`, `r2_score`, `mean_absolute_error`, `confusion_matrix`, `classification_report`, `roc_curve`, `precision_recall_curve`, `silhouette_score`, `adjusted_rand_score` | 分类、回归、聚类、排序全部指标 |
| `sklearn.model_selection` | 模型选择 | `train_test_split`, `cross_val_score`, `cross_validate`, `GridSearchCV`, `RandomizedSearchCV`, `KFold`, `StratifiedKFold`, `RepeatedKFold`, `LeaveOneOut`, `TimeSeriesSplit`, `ShuffleSplit`, `learning_curve`, `validation_curve`, `ParameterGrid`, `ParameterSampler` | 数据划分、交叉验证、超参搜索 |
| `sklearn.pipeline` | 流水线 | `Pipeline`, `make_pipeline`, `FeatureUnion`, `make_union` | 串联多个步骤，一体化 fit/predict |
| `sklearn.compose` | 组合器 | `ColumnTransformer`, `make_column_transformer`, `TransformedTargetRegressor`, `make_column_selector` | 对不同列应用不同预处理 |
| `sklearn.inspection` | 模型诊断 | `partial_dependence`, `PartialDependenceDisplay`, `permutation_importance`, `DecisionBoundaryDisplay` | 特征重要性、偏依赖图、决策边界 |
| `sklearn.dummy` | 基准模型 | `DummyClassifier`, `DummyRegressor` | 始终需要一个 baseline |
| `sklearn.datasets` | 内置数据 | `load_iris`, `load_digits`, `fetch_california_housing`, `make_classification`, `make_regression`, `make_blobs`, `make_moons` | 教学数据 + 合成数据生成 |
| `sklearn.random_projection` | 随机投影 | `GaussianRandomProjection`, `SparseRandomProjection` | Johnson-Lindenstrauss 降维 |
| `sklearn.utils` | 实用工具 | `shuffle`, `resample`, `compute_class_weight`, `Bunch`, `check_random_state`, `estimator_html_repr`, `all_estimators`, `validation.check_is_fitted`, `validation.check_X_y`, `validation.check_array` | 混洗、重采样、HTML 可视化、校验函数 |
| `sklearn.exceptions` | 异常类 | `NotFittedError`, `ConvergenceWarning`, `UndefinedMetricWarning`, `DataConversionWarning`, `FitFailedWarning`, `EfficiencyWarning` | 异常与警告处理 |
| `sklearn.set_config` | 全局配置 | `set_config(transform_output="pandas", display="diagram", working_memory=1024)` | 全局输出格式、显示样式、内存限制 |

---

## 6. 快速上手示例

下面是一个完整的端到端示例，展示 scikit-learn 的典型工作流：

```python
# ========== 0. 导入 ==========
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# ========== 1. 加载数据 ==========
# 使用加州房价数据集（回归任务）
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target  # 房价中位数（单位：$100k）

print(f"X shape: {X.shape}")  # (20640, 8)
print(f"y range: {y.min():.2f} ~ {y.max():.2f}")

# ========== 2. 特征工程 ==========
# 创建一个新特征：平均每房间数
X['RoomsPerHousehold'] = X['AveRooms'] / X['AveOccup']

# 将收入按分位数离散化为类别特征（演示混合类型处理）
X['IncomeCat'] = pd.qcut(X['MedInc'], q=4, labels=['low', 'medium', 'high', 'very_high'])

# ========== 3. 训练/测试划分 ==========
# 分层抽样不适用回归，这里用普通划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# ========== 4. 定义预处理 ==========
# 数值特征（缩放）
numeric_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup',
                    'Population', 'AveBedrms', 'RoomsPerHousehold']
numeric_transformer = StandardScaler()

# 类别特征（独热编码）
categorical_features = ['IncomeCat']
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

# ColumnTransformer：对不同列应用不同预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# ========== 5. 构建 Pipeline ==========
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge())
])

print(pipeline)  # 显示 Pipeline 结构

# ========== 6. 训练模型 ==========
pipeline.fit(X_train, y_train)

# ========== 7. 评估模型 ==========
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n默认模型结果:")
print(f"  MSE: {mse:.4f}")
print(f"  R²:  {r2:.4f}")

# ========== 8. 超参数调优 ==========
param_grid = {
    'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'model__fit_intercept': [True, False],
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳 CV MSE: {-grid_search.best_score_:.4f}")

# ========== 9. 最终评估 ==========
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print(f"\n调优后的结果:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_best):.4f}")
print(f"  R²:  {r2_score(y_test, y_pred_best):.4f}")

# ========== 10. 保存模型 ==========
import joblib
joblib.dump(best_model, 'california_housing_model.pkl')

# 部署时恢复
# model = joblib.load('california_housing_model.pkl')
# predictions = model.predict(new_data)

# ========== 11. Pipeline 可视化 (Jupyter 中运行) ==========
# from sklearn import set_config
# set_config(display='diagram')
# pipeline
```

**这个示例展示了 sklearn 工作流的标准步骤**：

| 步骤 | 做什么 | 用到的 API |
|:---:|--------|-----------|
| 1 | 加载数据 | `datasets.fetch_*` |
| 2 | 特征工程 | pandas + sklearn 结合 |
| 3 | 数据划分 | `train_test_split` |
| 4 | 预处理定义 | `ColumnTransformer` + `StandardScaler` + `OneHotEncoder` |
| 5 | 构建 Pipeline | `Pipeline` |
| 6 | 训练 | `pipeline.fit()` |
| 7 | 评估 | `metrics.mean_squared_error` + `r2_score` |
| 8 | 超参数调优 | `GridSearchCV` |
| 9 | 最终评估 | `predict` + metrics |
| 10 | 持久化 | `joblib.dump` / `joblib.load` |

**关键设计要点**：
- **Pipeline 确保数据不泄漏** — 每次 fit 只在训练 fold 上执行预处理
- **ColumnTransformer 处理混合类型** — 数值和类别特征各走各的管道
- **`model__` 前缀访问 Pipeline 内参数** — `'model__alpha'` 指 Pipeline 中名为 `model` 的步骤的 `alpha` 参数
- **`n_jobs=-1` 并行加速** — 使用所有 CPU 核心

---

## 相关概念

- [[Scikit-Learn 概述]] ← 当前文件
- [[数据预处理API]]
- [[模型选择与调参API]]
- [[线性模型API]]
- [[集成学习API]]
- [[聚类API]]
- [[Pipeline与ColumnTransformer]]
- [[评估指标API]]
- [[特征选择API]]
- [[降维API]]
- [[SVM API]]
- [[决策树与随机森林API]]
- [[朴素贝叶斯API]]
- [[近邻算法API]]
- [[神经网络API]]
- [[模型持久化与部署]]
- [[sklearn与pandas集成]]
- [[高斯过程与贝叶斯优化]]
- [[概率校准API]]
- [[半监督学习API]]
