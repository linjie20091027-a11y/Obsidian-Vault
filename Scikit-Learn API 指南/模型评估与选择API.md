# 模型评估与选择 API 详解（sklearn.model_selection & sklearn.metrics）

> 模型评估不是跑完 `.score()` 就结束了——你需要知道这道题怎么做、用什么评判标准、结果可信度有多高。本章覆盖 `sklearn.model_selection` 和 `sklearn.metrics` 的完整 API，从数据划分到交叉验证，从超参数搜索到 30+ 评估指标，每一个都给出"什么时候用"和"踩坑点"。

---

## 1. 数据集划分

### 1.1 train_test_split

**功能说明**：将数据集随机划分为训练集和测试集。这是建模的**第一步**，几乎所有 sklearn 项目都会用到。

> `from sklearn.model_selection import train_test_split`

**什么时候用**：任何时候你要在固定数据上评估模型，而不是用交叉验证。

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `*arrays` | array-like | - | 要划分的数据（支持多组，如 X, y, sample_weight） |
| `test_size` | float / int | 0.25 | float=比例（0.2=20%测试集），int=绝对值样本数 |
| `train_size` | float / int | None | 训练集比例，与 test_size 互斥 |
| `random_state` | int | None | **必须设**，否则每次结果不同，复现不了 |
| `shuffle` | bool | True | 是否先打乱再划分 |
| `stratify` | array-like | None | **分类问题必传 y**，保持训练/测试集中各类别比例一致 |

**`stratify` 参数的重要性（分类问题必用）**：

如果你有一个极度不平衡的数据集（比如正样本只占 1%），不用 `stratify=y` 的话，测试集可能 0 个正样本——那你连评价都没法评价。

```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.random.randn(1000, 10)
y = np.array([0] * 990 + [1] * 10)  # 1% 正样本

# 危险！测试集可能没有正样本
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 正确做法
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 验证分层效果
print(np.bincount(y))           # [990, 10]
print(np.bincount(y_train))     # [792, 8]  -> 仍为 99:1
print(np.bincount(y_test))      # [198, 2]  -> 仍为 99:1
```

**支持同时划分多组数据**：

```python
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weight, test_size=0.2, random_state=42
)
```

**注意事项 / 踩坑点**：

| 坑 | 说明 |
|---|---|
| **时间序列不能随机划分** | 用 `shuffle=False` 或用 `TimeSeriesSplit`，否则未来信息会泄漏到训练集 |
| **不用 `stratify` 导致灾难** | 不平衡分类 + 随机划分 = 测试集类别分布随机，评估无意义 |
| **不设 `random_state`** | 每次运行结果不同，同事复现不了你的结果 |
| **特征与标签数量不匹配** | 传入的数组第一维长度必须相同 |
| **测试集太小** | `test_size=0.1` 对 100 样本只有 10 个测试样本，评估方差极大 |

---

## 2. 交叉验证

> 交叉验证的核心哲学：你用全部数据做 k 次"轮流实验"，取平均分，杜绝"刚好分到好运测试集"的侥幸。

### 2.1 cross_val_score

**功能说明**：最简单的交叉验证包装器——自动划分、训练、评估，返回 k 个得分。

> `from sklearn.model_selection import cross_val_score`

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | estimator | 必填 | 已实例化的模型对象（未 fit 的） |
| `X` | array-like | 必填 | 特征矩阵 |
| `y` | array-like | 必填 | 标签 |
| `cv` | int / splitter | 5 | 折数；可传 KFold / StratifiedKFold 实例 |
| `scoring` | str / callable | None（用 estimator 默认） | 评分函数名 |
| `n_jobs` | int | None | 并行数，-1 全开 |
| `verbose` | int | 0 | 日志级别 |
| `error_score` | str / float | np.nan | 某折报错时返回的值，可设 'raise' 看具体错误 |

**`scoring` 常用取值**：

| 取值 | 适用任务 | 含义 | 方向 |
|---|---|---|---|
| `'accuracy'` | 分类 | 准确率 | 越大越好 |
| `'precision'` | 分类 | 精确率 | 越大越好 |
| `'recall'` | 分类 | 召回率 | 越大越好 |
| `'f1'` | 分类 | F1 分数 | 越大越好 |
| `'roc_auc'` | 二分类 | ROC 曲线下面积 | 越大越好 |
| `'neg_mean_squared_error'` | 回归 | 负均方误差（注意是负值！） | 越大越好 |
| `'neg_mean_absolute_error'` | 回归 | 负MAE | 越大越好 |
| `'r2'` | 回归 | R² | 越大越好 |
| `'neg_root_mean_squared_error'` | 回归 | 负RMSE | 越大越好 |

**代码示例**：

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(max_iter=10000)

# 默认 5 折，用 estimator 自带 scoring
scores = cross_val_score(clf, X, y, cv=5)
print(f"Accuracy per fold: {scores}")
print(f"Mean ± Std: {scores.mean():.4f} ± {scores.std():.4f}")

# 指定 scoring，8 核并行
scores_auc = cross_val_score(
    clf, X, y, cv=5, scoring='roc_auc', n_jobs=-1
)
print(f"AUC per fold: {scores_auc}")
print(f"Mean AUC: {scores_auc.mean():.4f}")
```

```python
# 回归：指标是负值！
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
reg = LinearRegression()

scores = cross_val_score(reg, X, y, cv=5, scoring='neg_mean_squared_error')
print(scores)   # 输出全部是负值，如: [-120, -130, ...]
# 正确解读：取负号还原
mse_scores = -scores
print(f"MSE: {mse_scores.mean():.2f} ± {mse_scores.std():.2f}")
```

**注意事项 / 踩坑点**：

| 坑 | 说明 |
|---|---|
| **回归指标是负值** | `neg_mean_squared_error` 返回负值，越大越好意思是 -100 > -200，取反后 MSE=100 |
| **不同 cv 对象影响分层** | 分类问题一定要用 `StratifiedKFold`，否则某些折可能全是一个类别 |
| **`cv=5` 对分类自动分层** | 0.22+ 版本对分类 estimator，`cv=5` 会自动用 `StratifiedKFold` |
| **不能传已 fit 的模型** | `cross_val_score` 内部会 clone estimator，传入已 fit 的也可以但没必要 |
| **得分波动大** | 数据 < 1000 条时，5 折标准差可能很大，考虑 RepeatedKFold |

---

### 2.2 cross_validate

**功能说明**：`cross_val_score` 的**加强版**。除了评分还能返回训练时间、测试时间、训练集分数。

> `from sklearn.model_selection import cross_validate`

**为什么比 cross_val_score 更强**：

1. 同时返回多个 scoring
2. 记录每折训练/测试时间
3. 通过 `return_train_score=True` 获得训练集分数——**检测过拟合的利器**

**关键参数（新增的）**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `scoring` | str / list / dict | None | 支持多个评分，如 `['accuracy', 'roc_auc']` |
| `return_train_score` | bool | False | **设为 True**，对比训练分和测试分看是否过拟合 |
| `return_estimator` | bool | False | 是否返回每折训练好的 estimator |
| `return_indices` | bool | False | 是否返回每折训练/测试的索引 |

**代码示例**：

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                           random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 多个指标 + 训练分数
results = cross_validate(
    clf, X, y, cv=10,
    scoring=['accuracy', 'roc_auc'],
    return_train_score=True,
    n_jobs=-1
)

print("返回的键:", results.keys())
# dict_keys(['fit_time', 'score_time', 'test_accuracy', 'train_accuracy',
#            'test_roc_auc', 'train_roc_auc'])

print(f"Test accuracy:   {results['test_accuracy'].mean():.4f} ± {results['test_accuracy'].std():.4f}")
print(f"Train accuracy:  {results['train_accuracy'].mean():.4f}")

# 过拟合检测
gap = results['train_accuracy'].mean() - results['test_accuracy'].mean()
if gap > 0.05:
    print(f"⚠ 过拟合警告！训练-测试差距: {gap:.4f}")
```

**注意事项 / 踩坑点**：

| 坑 | 说明 |
|---|---|
| `return_train_score` 默认 False | 不加就看不到训练集分数，没法诊断过拟合 |
| 多 scoring 时键名规则 | `f'test_{metric_name}'` 和 `f'train_{metric_name}'` |
| 耗时 | 多 scoring 比单 scoring 慢，`return_estimator=True` 更吃内存 |

---

### 2.3 cross_val_predict

**功能说明**：获取交叉验证的**OOF（Out-Of-Fold）预测**——每一行都是由"未见过该行"的模型预测的。

> `from sklearn.model_selection import cross_val_predict`

**什么时候用**：

1. **Stacking 的元特征（meta-feature）生成**：OOF 预测作为第二层模型的输入
2. 想画出"真实 vs 预测"散点图（回归）或混淆矩阵（分类）——基于 OOF 而不是测试集
3. 做更可靠的误差分析

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | estimator | 必填 | 模型 |
| `X`, `y` | array | 必填 | 数据 |
| `cv` | int / splitter | 5 | 折数 |
| `method` | str | 'predict' | **重要**：`'predict'` / `'predict_proba'` / `'predict_log_proba'` / `'decision_function'` |
| `n_jobs` | int | None | 并行 |

**代码示例**：

```python
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(max_iter=10000)

# OOF 预测
y_pred_oof = cross_val_predict(clf, X, y, cv=10, n_jobs=-1)

# 基于 OOF 画混淆矩阵（比用 test 集更可靠）
cm = confusion_matrix(y, y_pred_oof)
ConfusionMatrixDisplay(cm).plot()
plt.title("OOF Confusion Matrix (10-fold CV)")
plt.show()

# 获取概率预测（Stacking 常用）
y_proba_oof = cross_val_predict(clf, X, y, cv=10, method='predict_proba', n_jobs=-1)
print(f"OOF probability shape: {y_proba_oof.shape}")  # (569, 2)
```

**注意事项 / 踩坑点**：

| 坑 | 说明 |
|---|---|
| **method 默认是 'predict'** | 需要概率时别忘了设 `method='predict_proba'` |
| **结果顺序** | 与输入 y 的顺序一致，不会打乱 |
| **不是独立的测试结果** | OOF 预测不能直接当作"最终测试集"报告——它们来自不同模型 |
| **数据泄漏风险** | 不要在 `cross_val_predict` 之后跑 `GridSearchCV`——没有数据泄漏，但逻辑上不合理 |
| **method='decision_function'** | 用于 SVM 的距离输出 |

---

### 2.4 交叉验证分割器

#### 2.4.1 KFold

**功能说明**：基本的 K 折交叉验证——把数据切成 k 份，轮流取 1 份当测试，其余当训练。

> `from sklearn.model_selection import KFold`

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `n_splits` | int | 5 | 折数 |
| `shuffle` | bool | False | 是否先打乱再切分 |
| `random_state` | int | None | shuffle=True 时必须设 |

**代码示例**：

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold}: train {len(train_idx)}, test {len(test_idx)}")
```

**注意事项**：
- **分类问题不建议用纯 KFold**——某折可能缺失某个类别
- `shuffle=True` 能打破数据的原始顺序

#### 2.4.2 StratifiedKFold ⭐

**功能说明**：分层 K 折——保证每一折中各类别比例与整体一致。**分类问题的标配分割器。**

> `from sklearn.model_selection import StratifiedKFold`

**什么时候用**：任何分类问题——除非你有特殊理由不用。

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `n_splits` | int | 5 | 折数 |
| `shuffle` | bool | False | 是否打乱 |
| `random_state` | int | None | shuffle=True 时必须设 |

**代码示例**：

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold}:")
    print(f"  Train class dist: {np.bincount(y[train_idx])}")
    print(f"  Test  class dist: {np.bincount(y[test_idx])}")
    # 两行比例几乎一致
```

**为什么重要**：假设 1000 条数据，正样本 10 个。如果用 `KFold(n_splits=10)`，某个 fold 的测试集可能 0 个正样本——这一折的训练会失败（或者用 accuracy 评分完全没意义）。

**注意事项**：

| 坑 | 说明 |
|---|---|
| 需要同时传 `X` 和 `y` | `.split(X, y)` 两个都传，只用 y 做分层 |
| 分类 estimator + `cv=5` 自动用这个 | sklearn 0.22+ 的便利特性 |
| 多标签分类不支持 | 要用 `StratifiedKFold` 的替代方案（如 `MultilabelStratifiedKFold` 第三方库） |

#### 2.4.3 GroupKFold

**功能说明**：分组 K 折——同一个 group 的数据**不会**同时出现在训练集和测试集中。

> `from sklearn.model_selection import GroupKFold`

**什么时候用**：数据中存在"分组"概念，同一组的数据相互依赖、不应泄漏：

- 医疗数据：同一患者的多条记录
- 时间序列：同一用户的多天行为日志
- 图像数据：同一视频的多帧
- 推荐系统：同一用户的多条交互

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `n_splits` | int | 5 | 折数 |

**代码示例**：

```python
from sklearn.model_selection import GroupKFold

# 模拟：20 条记录来自 5 个患者（groups）
X = np.random.randn(20, 5)
y = np.array([0]*10 + [1]*10)
groups = np.array([0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4])

gkf = GroupKFold(n_splits=3)
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    print(f"Fold {fold}: Train groups {train_groups}, Test groups {test_groups}, Overlap: {overlap}")
    # Overlap 永远为空集
```

**注意事项**：

| 坑 | 说明 |
|---|---|
| `split(X, y, groups)` | 第三个参数是 groups |
| groups 值是整数 | 是分组编号，不是类别标签 |
| 每组的样本数不同 | 可能导致各折大小不均衡 |

#### 2.4.4 StratifiedGroupKFold

**功能说明**：`StratifiedKFold` + `GroupKFold` 的二合一——同时保证组不泄漏和类别比例一致。

> `from sklearn.model_selection import StratifiedGroupKFold`

**什么时候用**：有分组需求且类别不平衡的分类问题——比如每位患者数据作为一个 group，需要分层保证每折都有正样本。

**代码示例**：

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert train_groups & test_groups == set(), f"Fold {fold}: 组泄漏！"
    print(f"Fold {fold}: Train groups {train_groups}, Test groups {test_groups}")
    print(f"  Train y dist: {np.bincount(y[train_idx])}")
    print(f"  Test  y dist: {np.bincount(y[test_idx])}")
```

**注意事项**：
- 约束比 GroupKFold 更严，有时无法完全满足——会抛异常
- sklearn 0.24+ 才引入

#### 2.4.5 TimeSeriesSplit ⭐

**功能说明**：时间序列交叉验证——训练集永远在测试集**之前**，模拟实际的时序预测场景。

> `from sklearn.model_selection import TimeSeriesSplit`

**什么时候用**：股票预测、销售预测、天气预测等——**绝不能用随机 KFold**，因为未来信息会泄漏到过去。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `n_splits` | int | 5 | 折数 |
| `gap` | int | 0 | 训练集和测试集之间的间隔，防止边界泄漏 |
| `max_train_size` | int | None | 限制训练集最大样本数（像滑动窗口） |
| `test_size` | int | None | 每折测试集大小（等长） |

**代码示例**：

```python
from sklearn.model_selection import TimeSeriesSplit

X = np.arange(100).reshape(-1, 1)
y = np.arange(100)

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}:")
    print(f"  Train indices: {train_idx[0]}..{train_idx[-1]} (len={len(train_idx)})")
    print(f"  Test  indices: {test_idx[0]}..{test_idx[-1]} (len={len(test_idx)})")
    print(f"  Test after train? {test_idx[0] > train_idx[-1]}")
    # 永远为 True

# 输出示例：
# Fold 0: Train[0..15]  Test[16..29]  — 训练集不断扩展
# Fold 1: Train[0..29]  Test[30..43]
# Fold 2: Train[0..43]  Test[44..57]
# Fold 3: Train[0..57]  Test[58..71]
# Fold 4: Train[0..71]  Test[72..85]
```

**带 gap 和固定大小训练集的用法**：

```python
# 每天更新模型，只保留最近 60 天训练数据，预测未来 7 天
tscv = TimeSeriesSplit(n_splits=10, gap=3, max_train_size=60, test_size=7)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}: Train last {len(train_idx)} days, Test next {len(test_idx)} days")
    # 训练集最新索引 + gap + 1 == 测试集最早索引
```

**注意事项**：

| 坑 | 说明 |
|---|---|
| **不能 shuffle** | 时间顺序是核心假设 |
| **训练集逐渐变大** | 默认不设 `max_train_size` 时训练集每次都增加 |
| **gap 防边界泄漏** | 比如用前 3 天预测第 4 天，gap=1 表示跳过第 4 天直接预测第 5 天 |
| **折数受数据量限制** | n_splits 太大导致早期训练集太小 |

#### 2.4.6 RepeatedKFold / RepeatedStratifiedKFold

**功能说明**：K 折的"重复实验"版本——交叉验证做 N 次（每次随机打乱），取所有折的平均。

> `from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold`

**什么时候用**：
- 数据量不大（< 1000），单次 5 折方差大
- 想要置信区间而不是单点估计
- 论文/报告需要更稳健的评估

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `n_splits` | int | 5 | 每轮折数 |
| `n_repeats` | int | 10 | 重复轮数 |
| `random_state` | int | None | 必须设 |

**代码示例**：

```python
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC

X, y = make_classification(n_samples=200, n_features=10, random_state=42)
svc = SVC(kernel='rbf')

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(svc, X, y, cv=rskf, scoring='accuracy')
print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"Total evaluations: {len(scores)}")  # 50 = 5 * 10
```

#### 2.4.7 LeaveOneOut / LeavePOut

**功能说明**：

- `LeaveOneOut`：每次留 1 条测试，其余训练
- `LeavePOut`：每次留 p 条测试

> `from sklearn.model_selection import LeaveOneOut, LeavePOut`

**什么时候用**：**数据量极小（< 100 条）**，舍不得留出测试集。

**代码示例**：

```python
from sklearn.model_selection import LeaveOneOut

X = np.random.randn(30, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

loo = LeaveOneOut()
print(f"Splits: {loo.get_n_splits(X)}")  # 30 — 等于样本数

scores = cross_val_score(
    LogisticRegression(), X, y, cv=loo, scoring='accuracy'
)
print(f"LOO accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

**注意事项**：

| 坑 | 说明 |
|---|---|
| **极慢** | 100 条数据 = 100 次训练 |
| **方差大** | 每次只用 1 条数据评估，标准差不可靠 |
| **LeavePOut 组合爆炸** | p=2, n=100 → 4950 折 |
| **几乎不用于生产** | 学术研究中偶有出现 |

#### 2.4.8 交叉验证分割器选择指南

| 分割器 | 适用场景 | 保持分层 | 防止组泄漏 | 防时间泄漏 |
|---|---|---|---|---|
| `KFold` | 回归、基本分类 | ❌ | ❌ | ❌ |
| `StratifiedKFold` | **分类标配** | ✅ | ❌ | ❌ |
| `GroupKFold` | 有分组的数据 | ❌ | ✅ | ❌ |
| `StratifiedGroupKFold` | 分组 + 不平衡分类 | ✅ | ✅ | ❌ |
| `TimeSeriesSplit` | **时间序列** | ❌ | ❌ | ✅ |
| `RepeatedKFold` | 需要稳定评估 | ❌ | ❌ | ❌ |
| `LeaveOneOut` | 极小数据集 | ❌ | ❌ | ❌ |

---

## 3. 超参数搜索

### 3.1 GridSearchCV

**功能说明**：对给定的参数组合穷举搜索，结合交叉验证选出最优参数。**最常用的调参工具。**

> `from sklearn.model_selection import GridSearchCV`

**什么时候用**：参数空间不太大（总组合数 < 1000），需要精确搜索。

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | estimator | 必填 | 模型对象 |
| `param_grid` | dict / list[dict] | 必填 | 参数搜索空间，如 `{'C': [0.1, 1, 10]}` |
| `scoring` | str / callable | None | 评分函数；多指标可用 `{'f1': 'f1', 'auc': 'roc_auc'}` |
| `cv` | int / splitter | 5 | 交叉验证配置 |
| `refit` | bool / str | True | 搜索完用最优参数在全量数据上重训练；多 scoring 时指定用哪个 |
| `n_jobs` | int | None | 并行数，-1 全开 |
| `verbose` | int | 0 | 日志详细度，设 2+ 看进度 |
| `return_train_score` | bool | False | 开 True 看是否过拟合 |
| `error_score` | str | np.nan | 某组合报错时的行为，'raise' 查看错误 |

**代码示例**：

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=500, n_features=20, random_state=42)

# 定义搜索空间
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # l1 只能用 saga 或 liblinear
}

clf = LogisticRegression(max_iter=10000, random_state=42)
grid = GridSearchCV(
    clf, param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid.fit(X, y)

print(f"Best params: {grid.best_params_}")     # {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}
print(f"Best score:  {grid.best_score_:.4f}")  # 交叉验证最优分
print(f"Best estimator: {grid.best_estimator_}") # 已用全量数据 re-fit 的模型

# 直接预测
y_pred = grid.predict(X_test)   # 用 best_estimator_ 预测
y_proba = grid.predict_proba(X_test)

# 全量结果
import pandas as pd
cv_results = pd.DataFrame(grid.cv_results_)
print(cv_results[['param_C', 'param_penalty', 'mean_test_score', 'std_test_score']].head(10))
```

**多评分搜索**：

```python
grid = GridSearchCV(
    clf, param_grid,
    cv=5,
    scoring={'accuracy': 'accuracy', 'f1': 'f1', 'auc': 'roc_auc'},
    refit='auc',  # 按 AUC 选最优
    n_jobs=-1
)
grid.fit(X, y)

print(f"Best AUC: {grid.best_score_:.4f}")
print(f"Correspond accuracy: {grid.cv_results_['mean_test_accuracy'][grid.best_index_]:.4f}")
print(f"Correspond F1: {grid.cv_results_['mean_test_f1'][grid.best_index_]:.4f}")
```

**自定义 cv 对象（分类必须分层）**：

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid = GridSearchCV(clf, param_grid, cv=skf, scoring='f1', n_jobs=-1)
grid.fit(X, y)
```

**注意事项 / 踩坑点**：

| 坑 | 说明 |
|---|---|
| **参数组合爆炸** | 3 个超参各 10 个值 = 1000 组合 × 5 折 = 5000 次训练 |
| **`param_grid` 写错键名不报错** | `{'C': [1, 10], 'panalty': ['l2']}` 拼错 penalty——不报错，这个参数不会被调，悄无声息 |
| **dict 里值是 list，不是 tuple** | `{'C': [0.1, 1, 10]}` 不是 `{'C': (0.1, 1, 10)}` |
| **`refit=True`（默认）用全量数据重训练** | 如果你的数据有 1000 万行，最后 all-data refit 会很慢 |
| **并行 + 多 scoring 可能变慢** | scoring 计算也需要时间 |
| **`best_score_` 是 cv 平均分，不是测试集评分** | 报告结果时不要直接说"模型准确率 95%"——这是 cv score |

### 3.2 RandomizedSearchCV

**功能说明**：从参数分布中随机采样 N 组参数组合，评估后选最优。**参数空间大时的首选。**

> `from sklearn.model_selection import RandomizedSearchCV`

**什么时候用**：
- 参数空间太大——比如 3 个参数各有 100 个候选，总组合 1,000,000
- 想先快速找到大致范围，再用 GridSearch 细搜

**关键参数（不同于 GridSearchCV）**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `param_distributions` | dict | 必填 | 参数分布；值可以是 list（等概采样）或 scipy distribution |
| `n_iter` | int | 10 | **采样次数**，越大越可能找到最优，也是速度控制阀 |
| `random_state` | int | None | 必须设 |

**分布对象的使用（scipy.stats）**：

```python
from scipy.stats import uniform, randint, loguniform

randint.rvs(10, 50)        # 10~49 的整数，均匀采样
uniform.rvs(0, 1)           # 0~1 的连续值，均匀采样
loguniform.rvs(1e-3, 100)   # 对数均匀——适合 C、alpha、gamma 等
```

用 scipy distribution 的优势：可以指定无限范围或非均匀分布（像 `loguniform` 更关注小值），而 list 是等概率采样。

**代码示例**：

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=1000, n_features=30, random_state=42)

param_dist = {
    'n_estimators': randint(50, 500),           # 整数，均匀
    'max_depth': randint(3, 30),                 # 整数，均匀
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9),           # 0.1~1.0 连续值（sqrt 的替代）
    'bootstrap': [True, False],
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_dist,
    n_iter=100,          # 100 次采样 > 全组合 1000 万 +
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X, y)

print(f"Best params: {random_search.best_params_}")
print(f"Best score:  {random_search.best_score_:.4f}")
```

**与 GridSearchCV 的对比**：

| 维度 | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| 搜索方式 | 穷举所有组合 | 随机采样 n_iter 组 |
| 参数空间大时 | 爆炸 | ✅ 可控 |
| 找到全局最优 | 是（在离散空间内） | 大概率找到接近最优 |
| 速度 | 慢 | 快 |
| 典型场景 | 小空间精确搜索 | 大空间快速搜索 |
| 论文图 | 适合画 heatmap | 不太能画规律图 |

**注意事项**：

| 坑 | 说明 |
|---|---|
| `param_distributions` 不是 `param_grid` | 名字不同，别写错 |
| `n_iter` 太小 | 默认 10 可能不够，建议 50~200 |
| 连续分布需转换 | `max_features='sqrt'` 要改用 `uniform(0.1, 0.9)` 或保持为 category |
| 结果不能保证全局最优 | 有别于 GridSearch，存在随机性 |

### 3.3 HalvingGridSearchCV / HalvingRandomSearchCV

**功能说明**：逐步减半搜索（Successive Halving）。用少量资源（如数据子集）快速淘汰差组合，再用更多资源深入评估好组合。**比传统搜索快很多。**

> `from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV`

**原理简述**：
1. 第 1 轮：所有候选参数组合用小数据集评估
2. 淘汰一半（或 best 1/n）
3. 第 2 轮：幸存者用稍大数据集评估
4. 重复直到只剩 1 个或数据集满

**核心参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `resource` | str | `'n_samples'` | 控制资源增长维度，默认用样本量 |
| `max_resources` | int | 'auto' | 最大资源量（=样本数），auto 时用全部数据 |
| `min_resources` | int | 'exhaust' | 首轮资源量，auto 时自动算 |
| `factor` | int | 3 | 每轮淘汰比例（留 1/factor） |
| `aggressive_elimination` | bool | False | True 时会利用最后一轮额外资源 |

**代码示例**：

```python
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint

X, y = make_classification(n_samples=5000, n_features=50, random_state=42)

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
}

rf = RandomForestClassifier(random_state=42)
halving = HalvingRandomSearchCV(
    rf, param_dist,
    factor=3,                  # 每轮保留 1/3
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
halving.fit(X, y)

print(f"Best params: {halving.best_params_}")
print(f"Best score:  {halving.best_score_:.4f}")
print(f"Iterations:  {halving.n_iterations_}")
print(f"Candidates per round: {halving.n_candidates_}")
```

**注意事项**：
- sklearn 0.24+ 的实验特性
- 适合大样本（> 1000），小样本优势不明显
- `factor=3` 表示每轮留 1/3，不要设太大

### 3.4 超参数搜索对比表

| 方法 | 速度 | 覆盖度 | 适用场景 | 是否穷举 |
|---|---|---|---|---|
| `GridSearchCV` | 慢 | 100%（离散内） | 小空间精确搜索 | ✅ |
| `RandomizedSearchCV` | 快 | 采样覆盖率 | 大空间快速搜索 | ❌ |
| `HalvingGridSearchCV` | 较快 | 100%（离散内） | 大数据量的网格搜索 | ✅ |
| `HalvingRandomSearchCV` | 最快 | 采样覆盖率 | 大数据量快速筛选 | ❌ |

### 3.5 第三方工具简述

**Optuna**（推荐）：

```python
# pip install optuna
import optuna
from optuna.integration import OptunaSearchCV

optuna_search = OptunaSearchCV(
    RandomForestClassifier(),
    {'n_estimators': (50, 300), 'max_depth': (3, 20)},
    n_trials=100,
    cv=5,
    random_state=42
)
optuna_search.fit(X_train, y_train)
```

Optuna 使用贝叶斯优化（TPE），比随机搜索更快逼近最优。

**Hyperopt**：

```python
from hyperopt import fmin, tpe, hp, Trials
# 使用 Tree-structured Parzen Estimator
# 需要自己写目标函数，比 Optuna 麻烦一些
```

**建议**：日常用 `RandomizedSearchCV` + `GridSearchCV` 组合，需要极致效果时上 Optuna。

---

## 4. 分类评估指标（sklearn.metrics）

> 评分之前先问自己：我的问题到底在乎什么？把所有假阳关在门外（precision），还是把所有真阳找出来（recall）？

### 4.1 accuracy_score

**功能说明**：预测正确的样本比例。

> `from sklearn.metrics import accuracy_score`

**公式**：`accuracy = (TP + TN) / (TP + TN + FP + FN)`

**什么时候用**：**类别平衡**的分类问题。比如猫狗识别 50:50。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `y_true` | 1d array | 必填 | 真实标签 |
| `y_pred` | 1d array | 必填 | 预测标签 |
| `normalize` | bool | True | True=比例，False=正确个数 |
| `sample_weight` | array | None | 样本权重 |

**代码示例**：

```python
from sklearn.metrics import accuracy_score

y_true = [0, 0, 1, 1, 0, 1, 0, 1]
y_pred = [0, 1, 0, 1, 0, 1, 1, 1]

print(accuracy_score(y_true, y_pred))        # 0.625
print(accuracy_score(y_true, y_pred, normalize=False))  # 5 (正确个数)
```

**注意事项**：

| 坑 | 说明 |
|---|---|
| **类别不平衡时有误导性** | 99% 负样本，全预测负 = 99% 准确率——但模型毫无价值 |
| **多分类的正常率** | 没问题，所有类别一视同仁 |
| **不是概率评估** | 预测对了 51% 置信度和 99% 置信度的正样本算同样正确 |

### 4.2 confusion_matrix

**功能说明**：生成混淆矩阵——展示每个类别的真实 vs 预测分布。

> `from sklearn.metrics import confusion_matrix`

**二分类混淆矩阵详解**：

```
              预测负类    预测正类
真实负类       TN          FP
真实正类       FN          TP
```

- **TP（True Positive）**：真阳——预测阳性，确实是阳性 ✅
- **TN（True Negative）**：真阴——预测阴性，确实是阴性 ✅
- **FP（False Positive）**：假阳——预测阳性，实际阴性（第一类错误/误报）
- **FN（False Negative）**：假阴——预测阴性，实际阳性（第二类错误/漏报）

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `y_true` | 1d array | 必填 | 真实标签 |
| `y_pred` | 1d array | 必填 | 预测标签 |
| `labels` | array | None | 指定标签顺序 |
| `normalize` | str | None | `'true'`/`'pred'`/`'all'` |

**代码示例**：

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[3 2]   <- 真实 0 中 3 个预测正确，2 个错判为 1
#  [1 4]]  <- 真实 1 中 1 个错判为 0，4 个预测正确

# 按行归一化（真实分布）= 召回率
cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
print(cm_norm)
# [[0.6 0.4]
#  [0.2 0.8]]

# 可视化
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
plt.title("混淆矩阵")
plt.show()
```

**注意事项**：
- 默认 `labels=None` 时按 `np.unique(y)` 排序
- `normalize='true'` = 每行和为 1，对角线是 recall
- `normalize='pred'` = 每列和为 1，对角线是 precision

### 4.3 classification_report ⭐

**功能说明**：一站式报告——precision、recall、f1-score、support（样本数）一张表全出来。**日常工作最常用的分类评估。**

> `from sklearn.metrics import classification_report`

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `y_true` | 1d array | 必填 | 真实标签 |
| `y_pred` | 1d array | 必填 | 预测标签 |
| `labels` | array | None | 要列出的标签 |
| `target_names` | list | None | 标签对应的名称 |
| `sample_weight` | array | None | 样本权重 |
| `digits` | int | 2 | 小数位数 |
| `output_dict` | bool | False | True 返回 dict，可转 DataFrame |
| `zero_division` | str / int | 'warn' | 分母为 0 时的行为（'warn'/0/1） |

**代码示例**：

```python
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 2, 1, 0, 0, 2, 1, 2]
y_pred = [0, 2, 1, 2, 1, 0, 1, 2, 1, 2]

print(classification_report(
    y_true, y_pred,
    target_names=['cat', 'dog', 'bird']
))

#               precision    recall  f1-score   support
#         cat       0.67      0.67      0.67         3
#         dog       1.00      0.67      0.80         3
#        bird       0.60      1.00      0.75         3
#     accuracy                           0.80        10
#    macro avg       0.76      0.78      0.74        10
# weighted avg       0.76      0.80      0.75        10

# 转 DataFrame（写报告用）
report = classification_report(y_true, y_pred, output_dict=True)
import pandas as pd
pd.DataFrame(report).T
```

**理解 report 中的关键词**：

- **precision**：预测为该类的样本中，有多少真的属于该类
- **recall**：该类全部真实样本中，有多少被成功识别
- **f1-score**：precision 和 recall 的调和平均
- **support**：该类在真实数据中的样本数
- **macro avg**：各类别指标的平均值（每个类权重相同）
- **weighted avg**：按 support 加权的平均值（大类的指标权重更大）
- **accuracy**：整体准确率（只在 report 底部，不是各类别平均）

### 4.4 precision_score / recall_score / f1_score

**功能说明**：单独计算精确率、召回率、F1 分数。

> `from sklearn.metrics import precision_score, recall_score, f1_score`

**定义**：

| 指标 | 公式 | 含义 |
|---|---|---|
| Precision | TP / (TP + FP) | 预测为正的样本中，多少是真正正 |
| Recall | TP / (TP + FN) | 真实为正的样本中，多少被预测出来 |
| F1 | 2 × P × R / (P + R) | 精确率和召回率的调和平均 |

**关键参数（三者共享）**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `average` | str | 'binary' | **核心参数**：`'binary'` / `'micro'` / `'macro'` / `'weighted'` |
| `pos_label` | int / str | 1 | 正类标签（二分类时用） |
| `sample_weight` | array | None | 样本权重 |
| `zero_division` | str | 'warn' | 分母为 0 时的行为 |

**`average` 参数详解**：

| 取值 | 计算方式 | 适用场景 |
|---|---|---|
| `'binary'` | 只计算 pos_label 指定的类的指标 | 二分类 |
| `'micro'` | 把各类的 TP/FP/FN 先合并再算 | **多分类整体评估**（等价于 accuracy） |
| `'macro'` | 每个类独立算，取算术平均 | 各类平等对待（小类不受大类的压制） |
| `'weighted'` | 每个类独立算，按 support 加权平均 | 各类按样本量权重（大类影响更大） |

**代码示例**：

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [0, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 1, 0, 1, 0, 1, 1, 0]

print(f"Precision (binary): {precision_score(y_true, y_pred):.3f}")  # 0.667
print(f"Recall    (binary): {recall_score(y_true, y_pred):.3f}")     # 0.500
print(f"F1        (binary): {f1_score(y_true, y_pred):.3f}")         # 0.571

# 多分类
y_true_multi = [0, 1, 2, 2, 1, 0]
y_pred_multi = [0, 2, 1, 2, 1, 0]

print(f"Macro    F1: {f1_score(y_true_multi, y_pred_multi, average='macro'):.3f}")
print(f"Weighted F1: {f1_score(y_true_multi, y_pred_multi, average='weighted'):.3f}")
print(f"Micro    F1: {f1_score(y_true_multi, y_pred_multi, average='micro'):.3f}")
# Micro F1 == accuracy_score
```

**什么时候侧重 precision vs recall**：

| 场景 | 侧重指标 | 原因 |
|---|---|---|
| 疾病诊断 / 癌症筛查 | **Recall** | 漏诊代价极高（宁可鸡毛蒜皮假阳也要全检出） |
| 垃圾邮件检测 | **Precision** | 假阳 = 重要邮件被扔进垃圾箱，不可接受 |
| 信用卡欺诈检测 | **Recall** | 漏掉一笔欺诈损失大；假阳（误判欺诈）客服可处理 |
| 搜索引擎推荐 | **Precision** | 用户只看 Top 5，展示错了就跑 |
| 法律判决 | **Recall** | 宁可错杀不放（假阳多）不如全放（漏掉真凶） |

### 4.5 roc_auc_score ⭐

**功能说明**：计算 ROC 曲线下面积（Area Under the ROC Curve）。**二分类的"金标准"评估指标。**

> `from sklearn.metrics import roc_auc_score`

**什么是 AUC**：
- AUC = 1.0：完美分类器
- AUC = 0.5：和随机猜测一样
- AUC < 0.5：比随机还差（反向指标，取反即可）
- 通俗理解：随机抽一个正样本和一个负样本，正样本得分 > 负样本得分的概率

**什么时候用**：
- 二分类问题的主要评估指标（尤其是需要比 accuracy 更全面的评判）
- 需要比较不同分类器的排序能力
- 类别不平衡时不依赖 threshold

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `y_true` | 1d array | 必填 | 真实标签 |
| `y_score` | 1d array | 必填 | 预测概率（正类概率）或决策函数值 |
| `average` | str | 'macro' | 多分类用：`'macro'`/'weighted' |
| `multi_class` | str | 'ovr' | 多分类策略：`'ovr'`(一对多) / `'ovo'`(一对一) |
| `max_fpr` | float | None | 只关心低 FPR 区域的 AUC |
| `labels` | array | None | 多分类指定哪些类 |

**代码示例**：

```python
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression()
clf.fit(X_train, y_train)

# 传概率（不是标签！）
y_proba = clf.predict_proba(X_test)[:, 1]  # 正类概率
auc = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc:.4f}")

# ROC 曲线
RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier (AUC=0.5)')
plt.title(f'ROC Curve (AUC = {auc:.3f})')
plt.legend()
plt.show()

# 手动绘制
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
print(f"FPR  shape: {fpr.shape}")
print(f"TPR  shape: {tpr.shape}")
print(f"Thresholds: {thresholds[:5]}")
```

**多分类 ROC AUC**：

```python
X, y = make_classification(n_samples=500, n_classes=3, n_informative=5,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)

print(f"AUC (ovr):     {roc_auc_score(y_test, y_proba, multi_class='ovr'):.4f}")
print(f"AUC (ovo):     {roc_auc_score(y_test, y_proba, multi_class='ovo'):.4f}")
print(f"AUC (weighted):{roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'):.4f}")
```

**注意事项**：

| 坑 | 说明 |
|---|---|
| **传入概率，不是标签** | `y_score` 要传 `predict_proba()[:, 1]` 或 `decision_function()`——传 `predict()` 返回标签就没有 ROC 曲线了 |
| **AUC 只看排序，不看校准** | 概率全偏移 0.5 但排序不变 AUC 不变——需要 log_loss 校准 |
| **不平衡时 AUC 可能高估** | 正样本极少时 ROC 曲线仍可能好看——考虑 Precision-Recall AUC |
| **多分类 ovr vs ovo** | ovr 更快（N 次），ovo 更费时（N×(N-1)/2 次）但理论上更准 |

### 4.6 log_loss

**功能说明**：对数损失（Log Loss / Cross-entropy Loss）——衡量预测概率与真实标签的差异。

> `from sklearn.metrics import log_loss`

**什么时候用**：
- 需要概率校准的评估（如 Kaggle 比赛的评分标准）
- 比较两个 AUC 相同的模型时
- 多分类问题有"信心惩罚"

**公式**：`log_loss = -(1/N) * Σ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]`

**代码示例**：

```python
from sklearn.metrics import log_loss

# 真实标签
y_true = [0, 0, 1, 1]
# 预测概率（模型 A：自信正确）
y_proba_good = np.array([[0.9, 0.1],
                         [0.8, 0.2],
                         [0.3, 0.7],   # 标签 1，概率 0.7
                         [0.1, 0.9]])  # 标签 1，概率 0.9
# 预测概率（模型 B：犹豫但方向对）
y_proba_ok = np.array([[0.6, 0.4],
                       [0.7, 0.3],
                       [0.6, 0.4],    # 错误！标签 1 但概率 0.4
                       [0.4, 0.6]])

print(f"Log loss (good): {log_loss(y_true, y_proba_good):.4f}")  # 0.2703
print(f"Log loss (ok):   {log_loss(y_true, y_proba_ok):.4f}")    # 0.5108
# 值越小越好！ok 模型因为第三个样本的概率错方向被狠狠惩罚
```

**注意事项**：
- 必须传概率 `y_proba`，不能传标签
- `labels` 参数在多分类时指定类别映射
- 对极端预测（概率 0 或 1）敏感——`log(0)` = -inf

### 4.7 cohen_kappa_score

**功能说明**：Cohen's Kappa——衡量分类一致性的指标，**排除了随机一致的影响**。

> `from sklearn.metrics import cohen_kappa_score`

**公式**：`κ = (p₀ - pₑ) / (1 - pₑ)`，其中 p₀ 是 observed agreement，pₑ 是 expected agreement by chance。

**什么时候用**：
- 多分类任务需要排除随机猜测
- 医疗诊断的一致性评估
- 评分员一致性

**Kappa 值解读**：

| Kappa 范围 | 一致性强度 |
|---|---|
| < 0 | 比随机还差 |
| 0.0 - 0.2 | 微小 (slight) |
| 0.21 - 0.4 | 一般 (fair) |
| 0.41 - 0.6 | 中等 (moderate) |
| 0.61 - 0.8 | 高度 (substantial) |
| 0.81 - 1.0 | 几乎完美 (almost perfect) |

**代码示例**：

```python
from sklearn.metrics import cohen_kappa_score, accuracy_score

y_true = [0, 1, 2, 2, 1, 0, 0, 2, 1, 2]
y_pred = [0, 2, 1, 2, 1, 0, 1, 2, 1, 2]

print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"Kappa:    {cohen_kappa_score(y_true, y_pred):.3f}")
# Kappa < Accuracy，因为 Kappa 排除了随机猜对的部分
```

### 4.8 matthews_corrcoef (MCC) ⭐

**功能说明**：Matthews 相关系数——**类别不平衡分类的最佳单一指标**。

> `from sklearn.metrics import matthews_corrcoef`

**公式**：利用混淆矩阵的 4 个值计算和皮尔逊相关系数相同的指标。

**为什么 MCC 比 F1 更好**：
- F1 不考虑 TN：全预测正样本也能 F1 > 0
- MCC 同时利用 TP/TN/FP/FN：范围 [-1, 1]，0 = 随机，1 = 完美
- 任何不平衡比例下都是公平的度量

**代码示例**：

```python
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

# 极度不平衡数据集
y_true = [0] * 95 + [1] * 5   # 5% 正样本
y_pred_bad = [0] * 100          # 全预测负类

print(f"Accuracy: {accuracy_score(y_true, y_pred_bad):.3f}")  # 0.95 ✅ 误以为好
print(f"F1:       {f1_score(y_true, y_pred_bad):.3f}")         # 0.00
print(f"MCC:      {matthews_corrcoef(y_true, y_pred_bad):.3f}") # 0.00

# 好模型
y_true = [0, 0, 0, 1, 1]
y_pred =  [0, 0, 0, 1, 1]
print(f"\nMCC (best): {matthews_corrcoef(y_true, y_pred):.3f}")   # 1.0

# 坏模型
y_pred_bad2 = [1, 1, 1, 0, 0]
print(f"MCC (worst): {matthews_corrcoef(y_true, y_pred_bad2):.3f}")  # -1.0
```

### 4.9 分类指标选择指南

| 场景 | 推荐指标 | 备选 |
|---|---|---|
| 类别平衡 | Accuracy + F1 | — |
| 类别不平衡 | **MCC** / F1-macro | AUC |
| 关心排序能力 | **AUC** | AP (Average Precision) |
| 关心概率校准 | **Log Loss** | Brier Score |
| 关心全部假阳 | **Precision** | 加入 Top-1 error |
| 关心全部假阴 | **Recall** | Sensitivity |
| 多分类 | F1-macro / weighted | — |
| 评审员一致性 | Cohen's Kappa | — |

---

## 5. 回归评估指标（sklearn.metrics）

### 5.1 mean_squared_error / mean_absolute_error

**功能说明**：MSE 和 MAE 是最基本的回归误差指标。

> `from sklearn.metrics import mean_squared_error, mean_absolute_error`

**MSE vs MAE 对比**：

| 维度 | MSE | MAE |
|---|---|---|
| 公式 | (1/n) Σ (yᵢ - ŷᵢ)² | (1/n) Σ |yᵢ - ŷᵢ| |
| 对异常值的敏感度 | **高**（平方放大） | 低（线性） |
| 最优预测 | 条件均值 | 条件中位数 |
| 梯度 | 连续可导（适合优化） | 有尖点（不可导） |
| 单位 | 原始单位的平方 | 原始单位 |

**代码示例**：

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")  # 0.375
print(f"MAE: {mae:.4f}")  # 0.500

# 异常值的影响
y_pred_outlier = [2.5, 0.0, 2, 20]  # 第 4 个预测错大了
print(f"MAE (with outlier): {mean_absolute_error(y_true, y_pred_outlier):.4f}")
print(f"MSE (with outlier): {mean_squared_error(y_true, y_pred_outlier):.4f}")
# MSE 急剧增大 >>> MAE 受影响相对小
```

**选择建议**：
- 异常值多 → MAE
- 大误差不可接受 → MSE（平方惩罚放大）
- 需要和 ML 损失函数一致 → 看是否用 L1/L2 损失

### 5.2 root_mean_squared_error (RMSE)

**功能说明**：MSE 的平方根——把误差单位恢复到原始单位。

> `from sklearn.metrics import root_mean_squared_error`（sklearn 1.4+）
>
> 旧版本可用：`np.sqrt(mean_squared_error(y_true, y_pred))`

**为什么比 MSE 更常用**：
- 单位与 target 一致（钱、人、销量等）
- 直观解释："RMSE=200" 意思是"平均预测误差大约 200"

**代码示例**：

```python
from sklearn.metrics import root_mean_squared_error

y_true = [100, 200, 300, 400]
y_pred = [110, 190, 310, 380]

rmse = root_mean_squared_error(y_true, y_pred)
print(f"RMSE: {rmse:.2f}")  # 14.14
# 解释：预测值和真实值的平均差距约 14.14
```

### 5.3 r2_score (R² 决定系数)

**功能说明**：衡量回归模型对因变量变异性的解释比例。

> `from sklearn.metrics import r2_score`

**公式**：`R² = 1 - (SS_res / SS_tot)`

- R² = 1：完美拟合
- R² = 0：和瞎猜均值一样
- **R² < 0**：比瞎猜均值还差——模型出错

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `y_true` | array | 必填 | 真实值 |
| `y_pred` | array | 必填 | 预测值 |
| `sample_weight` | array | None | 样本权重 |
| `force_finite` | bool | True | sklearn 1.4+，是否给无穷值替换为常数 |

**代码示例**：

```python
from sklearn.metrics import r2_score
import numpy as np

y_true = [3, 5, 7, 9]
y_pred = [2.5, 5.1, 6.8, 9.2]
print(f"R²: {r2_score(y_true, y_pred):.4f}")  # 0.97+

# 常数预测 = R² = 0
y_mean = np.mean(y_true)
y_dumb = [y_mean] * len(y_true)
print(f"R² (dumb): {r2_score(y_true, y_dumb):.4f}")  # 0.0

# 比瞎猜还差 = R² < 0
y_worse = [100, 100, 100, 100]
print(f"R² (worse): {r2_score(y_true, y_worse):.4f}")  # 负值
```

**注意事项**：

| 坑 | 说明 |
|---|---|
| **R² 可以为负** | 不是 bug——确实是比预测目标均值还差 |
| **R² 高 ≠ 模型好** | 加了无穷多特征后 R² 逼近 1（过拟合）——用 Adjusted R² |
| **对异常值敏感** | R² 分母用了平方和，一个离谱的异常值就能把 R² 打到负值 |
| **测试集 R² ≠ 训练集 R²** | 训练集 R² 永远高于测试集——永远用测试集 R² 评估 |

### 5.4 mean_absolute_percentage_error (MAPE)

**功能说明**：百分比误差的平均值——跨领域沟通的最好指标。

> `from sklearn.metrics import mean_absolute_percentage_error`

**何时用**：
- 需要"百分之几"的直观解释（向非技术人员汇报）
- 目标值不能为 0（分母为 0 时除零问题）

**代码示例**：

```python
from sklearn.metrics import mean_absolute_percentage_error

y_true = [100, 200, 300, 400]
y_pred = [110, 180, 330, 380]

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")  # 0.0667 (6.67%)
# 解释：预测值和真实值平均偏差约 6.67%
```

**注意事项**：
- 分母有 0 时 → `np.inf`（新版 sklearn 会警告）
- 通常乘以 100 报告
- 对小值和大值权重不同（100 块差 10 块 = 10%，10000 块差 10 块 = 0.1%）

### 5.5 mean_squared_log_error (MSLE / RMSLE)

**功能说明**：对数均方误差——先取 log 再算 MSE，对偏态分布友好。

> `from sklearn.metrics import mean_squared_log_error`

**什么时候用**：
- Target 偏态分布（房价、销量、收入）
- Kaggle 比赛常用 RMSLE
- 相对误差比绝对误差更重要

**代码示例**：

```python
from sklearn.metrics import mean_squared_log_error
import numpy as np

y_true = [100, 1000, 10000, 100000]
y_pred = [120, 1100, 9000, 120000]

msle = mean_squared_log_error(y_true, y_pred)
rmsle = np.sqrt(msle)
print(f"MSLE:   {msle:.6f}")
print(f"RMSLE:  {rmsle:.6f}")

# 对比 MSE —— 大值的支配效应
mse = np.mean((np.log1p(y_true) - np.log1p(y_pred))**2)  # 等同于 MSLE
```

**注意事项**：
- 负值不能处理 → 确保 y_true 和 y_pred 都是正值
- `mean_squared_log_error` 计算 `log(1+y)` vs `log(1+ŷ)` 的 MSE

### 5.6 回归指标选择指南

| 场景 | 推荐指标 | 原因 |
|---|---|---|
| 一般回归 | RMSE | 和 target 同单位 |
| 异常值多 | MAE | 不受平方放大 |
| 偏态分布 target | RMSLE | 关注相对误差 |
| 可解释性（% 报告） | MAPE | 业务报告首选 |
| 模型比较 | R² | 标准化（范围明确） |
| 优化目标 | MSE | RMSE 可导 |

---

## 6. 自定义评估指标

### 6.1 make_scorer

**功能说明**：将你自己的 Python 函数变成 sklearn 可用的 scorer 对象，可在 `cross_val_score`、`GridSearchCV` 等工具中使用。

> `from sklearn.metrics import make_scorer`

**什么时候用**：
- 不存在 sklearn 内置的指标
- 业务自定义评分（加权 F1、成本敏感的分数）
- Kaggle 特殊指标

**关键参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `score_func` | callable | 必填 | 自定义评分函数：`func(y_true, y_pred, **kwargs)` |
| `greater_is_better` | bool | True | True=分数越大越好，False=越小越好（如误差） |
| `needs_proba` | bool | False | True=需要概率输入（不是标签） |
| `needs_threshold` | bool | False | True=需要阈值（比如自定义的二分类评分） |
| `**kwargs` | dict | - | 传给 score_func 的额外参数 |

**代码示例**：

```python
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import cross_val_score, GridSearchCV

# 自定义：偏重召回率的 F-beta
def f2_score_func(y_true, y_pred):
    """F-beta with beta=2 — recall 比 precision 重要 2 倍"""
    from sklearn.metrics import fbeta_score
    return fbeta_score(y_true, y_pred, beta=2)

f2_scorer = make_scorer(f2_score_func, greater_is_better=True)

# 在 cross_val_score 中使用
scores = cross_val_score(clf, X, y, cv=5, scoring=f2_scorer)
print(f"F2 score: {scores.mean():.4f}")

# 在 GridSearchCV 中使用
grid = GridSearchCV(clf, param_grid, cv=5, scoring=f2_scorer)
grid.fit(X, y)
```

**自定义误差指标（`greater_is_better=False`）**：

```python
def root_mean_squared_error_custom(y_true, y_pred):
    """自定义 RMSE（越小越好）"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

rmse_scorer = make_scorer(
    root_mean_squared_error_custom,
    greater_is_better=False  # 越小越好
)

# cross_val_score 返回负值（越大越好）
scores = cross_val_score(reg, X, y, cv=5, scoring=rmse_scorer)
print(scores)  # 都是负值——符合"越大越好"约定
```

**自定义 scorer（需要概率的）**：

```python
from sklearn.metrics import roc_auc_score, make_scorer

# 方式 1：包装现有 sklearn 函数
def auc_func(y_true, y_proba):
    """需要 predict_proba 的概率"""
    return roc_auc_score(y_true, y_proba[:, 1])  # 取正类概率

auc_scorer = make_scorer(
    auc_func,
    greater_is_better=True,
    needs_proba=True  # ← 关键
)

# GridSearchCV 会自动调用 predict_proba 而不是 predict
grid = GridSearchCV(clf, param_grid, cv=5, scoring=auc_scorer)
grid.fit(X, y)
# 注意：estimator 必须支持 predict_proba
```

**注意事项**：

| 坑 | 说明 |
|---|---|
| `needs_proba=True` | 自动调用 `predict_proba()` 而不是 `predict()`——确保模型有此方法 |
| `needs_threshold=True` | 用于需要阈值调参的 scorer |
| `greater_is_better` | cross_val_score 内部会乘 -1 做转换 |
| 异常处理 | 自定义函数内加 try/except，否则某折失败整个 cv 报错 |

---

## 7. 学习曲线与验证曲线

### 7.1 learning_curve

**功能说明**：绘制"训练样本数量 vs 模型性能"曲线——**诊断过拟合和欠拟合的核心工具**。

> `from sklearn.model_selection import learning_curve`

**怎么读图**：
- 训练分高 + 验证分低 + 差距大 → **过拟合**（方差大）
- 训练分低 + 验证分低 + 差距小 → **欠拟合**（偏差大）
- 训练分和验证分都高 + 差距小 → 刚刚好

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | estimator | 必填 | 模型 |
| `X`, `y` | array | 必填 | 数据 |
| `train_sizes` | array | `np.linspace(0.1, 1.0, 5)` | 训练比例 |
| `cv` | int / splitter | None | 交叉验证配置 |
| `scoring` | str | None | 评分函数 |
| `n_jobs` | int | None | 并行数 |
| `shuffle` | bool | False | 是否打乱 |
| `random_state` | int | None | 随机种子 |

**代码示例**：

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

X, y = make_classification(n_samples=500, n_features=20,
                           n_informative=5, random_state=42)

# 故意过拟合（复杂模型 + 少数据）
from sklearn.svm import SVC
svc = SVC(kernel='rbf', gamma=10, C=1000)

train_sizes, train_scores, test_scores = learning_curve(
    svc, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1, random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.plot(train_sizes, train_mean, 'o-', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.title('学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 诊断建议
gap = train_mean[-1] - test_mean[-1]
print(f"Training-CV gap at full data: {gap:.4f}")
if train_mean[-1] > 0.95 and test_mean[-1] < 0.85:
    print("⚠ 过拟合：训练分高验证分低，差距大")
elif train_mean[-1] < 0.75 and test_mean[-1] < 0.75:
    print("⚠ 欠拟合：训练分和验证分都低")
else:
    print("✅ 偏差-方差基本平衡")
```

### 7.2 validation_curve

**功能说明**：绘制"单个超参数 vs 模型性能"曲线——**帮你确定最佳超参数范围**。

> `from sklearn.model_selection import validation_curve`

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | estimator | 必填 | 模型 |
| `X`, `y` | array | 必填 | 数据 |
| `param_name` | str | 必填 | 要调查的参数名（如 `'C'`、`'max_depth'`） |
| `param_range` | array | 必填 | 参数值列表 |
| `cv` | int/splitter | None | cv 配置 |
| `scoring` | str | None | 评分函数 |

**代码示例**：

```python
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=500, n_features=20, random_state=42)

param_range = [1, 3, 5, 10, 20, 50, 100]
train_scores, test_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X, y,
    param_name='max_depth',
    param_range=param_range,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, 'o-', label='Training score')
plt.plot(param_range, test_mean, 'o-', label='Validation score')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('验证曲线 — max_depth 调参')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 找出最优值
best_depth = param_range[np.argmax(test_mean)]
print(f"Best max_depth: {best_depth} (score: {test_mean.max():.4f})")
```

**如何解读**：
- 训练分高 + 验证分低 → 该参数值范围导致了过拟合（如 max_depth 太大）
- 训练分低 + 验证分低 → 该参数值范围导致了欠拟合（如 C 太小）
- 两线接近 + 都高 → 最优范围

---

## 相关概念

- [[Pipeline与实用工具API]]
- [[监督学习模型API]]
- [[无监督学习API]]
- [[特征工程API]]
- [[scikit-learn 核心原理与最佳实践]]
