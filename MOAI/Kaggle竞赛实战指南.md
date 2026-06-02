# Kaggle 竞赛实战指南：从入门到抢分

---

## 1. 竞赛全流程 SOP（标准操作流程）

> 每个竞赛周期约 3-4 周，时间就是排名。以下是经过大量竞赛验证的标准操作流程，按天拆解，直接照做即可。

### 1.1 赛前准备（比赛开始第 1 天）

**第 1 步：精读竞赛描述（30 分钟）**

不要跳过任何细节。逐字阅读以下内容：
- **任务类型**：分类（二分类/多分类/多标签）、回归、排序、分割还是生成
- **评估指标**：AUC、RMSE、LogLoss、F1、mAP 还是自定义指标 — 这决定了你后续的损失函数和后处理策略
- **数据说明**：train.csv、test.csv、sample_submission.csv 的列名、含义、数据类型
- **提交格式**：sample_submission.csv 的格式必须严格遵循，header 和 id 列名一个都不能错
- **时间线**：比赛截止日期、组队截止日期、每日提交次数限制

```python
# 第一步永远是检查提交格式
import pandas as pd

sub = pd.read_csv("sample_submission.csv")
print(sub.head())
print(sub.columns)
print(sub.dtypes)
```

**第 2 步：快速 EDA（1-2 小时）**

目的不是做深入分析，而是建立直觉。快速检查以下问题：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. 基础信息
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {train.columns.tolist()}")

# 3. 缺失值分布
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"Missing columns:\n{missing}")

# 4. 目标变量分布
target_col = "target"  # 替换为实际目标列名
print(train[target_col].describe())
if train[target_col].nunique() < 50:
    print(train[target_col].value_counts())

# 5. 数据类型分类
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()
print(f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

# 6. 检查 train 和 test 列是否一致
train_only = set(train.columns) - set(test.columns)
test_only = set(test.columns) - set(train.columns)
print(f"Train only: {train_only}")  # 通常包含 target
print(f"Test only: {test_only}")   # 应该是空或 id
```

**第 3 步：提交最简单的 Baseline（10 分钟）**

此步的目的是：验证提交流程通畅、拿到初始 LB 分数、建立后续对比基准。

```python
# 回归 baseline：均值
sub = pd.read_csv("sample_submission.csv")
sub["target"] = train["target"].mean()
sub.to_csv("sub_mean.csv", index=False)
# 此时立即提交！

# 分类 baseline：众数
sub["target"] = train["target"].mode()[0]
sub.to_csv("sub_mode.csv", index=False)

# 稍复杂 baseline：用单个重要特征的简单规则
sub["target"] = test["feature_x"].apply(lambda x: 1 if x > threshold else 0)
sub.to_csv("sub_rule.csv", index=False)
```

**第 4 步：阅读 Discussion 区高质量帖子（1 小时）**

按 "Most Votes" 排序，找到以下类型并精读：
- **赛事元信息帖**：Host 发的 FAQ、数据说明补充
- **EDA 帖**：高质量的数据探索，揭示数据特性
- **Shared Solution 帖**：以往类似比赛的金牌方案（去 Kaggle Blog 搜 "Winning Solution"）

**第 5 步：建立本地验证框架（最重要的一步）**

这是区分高手和菜鸟的关键。CV 策略必须尽量模拟 LB 的计算方式。

```python
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import roc_auc_score, mean_squared_error

# 设置全局固定 seed
SEED = 42
N_FOLDS = 5

def seed_everything(seed):
    import random
    import os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

# 分类：StratifiedKFold
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 回归：KFold
folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 分组数据：GroupKFold
folds = GroupKFold(n_splits=N_FOLDS)
for fold, (train_idx, val_idx) in enumerate(folds.split(X, y, groups=g)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # 训练 + 评估
```

---

### 1.2 第一阶段：Baseline 构建（第 1-3 天）

**目标**：搭建一条从数据到提交的完整 pipeline，保证代码可复现。

**最小可行 Pipeline**：

```python
# ========== minimal_pipeline.py ==========
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import joblib

SEED = 42
N_FOLDS = 5

def seed_everything(seed):
    import random, os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

# 1. 读数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")

# 2. 简单处理
target = "target"
train = train.fillna(-999)
test = test.fillna(-999)

# 识别特征列
feat_cols = [c for c in train.columns if c not in [target, "id"]]
feat_cols = [c for c in feat_cols if c in test.columns]

X = train[feat_cols]
y = train[target]
X_test = test[feat_cols]

# 3. 类别特征编码
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

# 4. CV 训练
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.05,
        random_state=SEED,
        verbosity=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgbm.early_stopping(50), lgbm.log_evaluation(100)]
    )
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    joblib.dump(model, f"model_fold{fold}.pkl")

# 5. 评估
print(f"CV AUC: {roc_auc_score(y, oof_preds):.5f}")

# 6. 保存预测
sub["target"] = test_preds
sub.to_csv("sub_baseline.csv", index=False)
```

**可复现性检查清单**：
- [ ] `seed_everything(SEED)` 在代码最前面
- [ ] 所有模型传入 `random_state=SEED`
- [ ] 数据划分用 `random_state=SEED`
- [ ] 使用配置文件管理超参数（不用硬编码）
- [ ] 每次实验的 CV 分数和 LB 分数记录到 csv

```python
# 实验记录模板
import csv
import os
from datetime import datetime

def log_experiment(exp_name, cv_score, lb_score, note=""):
    log_file = "experiment_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "exp_name", "cv_score", "lb_score", "note"])
        writer.writerow([datetime.now().isoformat(), exp_name, cv_score, lb_score, note])
    print(f"Logged: {exp_name} | CV: {cv_score:.5f} | LB: {lb_score}")

log_experiment("baseline_lgb", 0.8523, 0.8489, "simple LGBM, no feature engineering")
```

**确认 CV 与 LB 的相关性**：
- 至少跑 3 个有明显差异的模型（如 LGB vs XGB vs 简单平均），对比 CV 和 LB 的排序是否一致
- 如果 LB 排名和 CV 排名完全相反，说明 CV 策略有问题 — 重新设计 CV
- 理想情况：CV 和 LB 高度正相关（Pearson r > 0.7）

---

### 1.3 第二阶段：特征工程（第 3-14 天）

> 这是竞赛中最值得投入时间的阶段。两个方向：**领域知识驱动**和**统计驱动**。

#### 1.3.1 数据清洗与预处理

```python
# --- 缺失值处理 ---
# 策略 1：用特殊值填充（让模型学习"缺失"这个信号）
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)

# 策略 2：用统计量填充
for col in numeric_cols:
    median_val = train[col].median()
    train[col].fillna(median_val, inplace=True)
    test[col].fillna(median_val, inplace=True)

# 策略 3：创建缺失值指示列（强烈推荐）
for col in numeric_cols:
    if train[col].isnull().sum() > 0:
        train[f"{col}_is_missing"] = train[col].isnull().astype(int)
        test[f"{col}_is_missing"] = test[col].isnull().astype(int)

# --- 异常值处理 ---
# clip 到 1st 和 99th 百分位（比直接删除更安全）
for col in numeric_cols:
    lower = train[col].quantile(0.01)
    upper = train[col].quantile(0.99)
    train[col] = train[col].clip(lower, upper)
    test[col] = test[col].clip(lower, upper)

# --- 类别特征处理 ---
# 低频类别归为 "Other"
for col in cat_cols:
    freq = train[col].value_counts(normalize=True)
    rare = freq[freq < 0.01].index.tolist()
    train[col] = train[col].replace(rare, "Other")
    test[col] = test[col].replace(rare, "Other")
```

#### 1.3.2 特征创造

**领域知识驱动的特征**（需要理解业务逻辑）：
```python
# 示例：房价预测
train["total_area"] = train["ground_living_area"] + train["basement_area"]
train["age"] = train["year_sold"] - train["year_built"]
train["age_at_remodel"] = train["year_remodeled"] - train["year_built"]
train["has_pool"] = (train["pool_area"] > 0).astype(int)
train["room_density"] = train["bedrooms"] / train["total_area"]

# 示例：金融风控
train["debt_to_income"] = train["total_debt"] / (train["income"] + 1)
train["credit_utilization"] = train["balance"] / (train["credit_limit"] + 1)
train["delinquency_accounts_ratio"] = train["delinquent_accounts"] / (train["total_accounts"] + 1)
```

**统计驱动的特征**：

```python
# 1. 聚合特征（groupby + agg）—— 最有效的特征类型之一
def add_aggregation_features(train, test, group_col, agg_col):
    """对 agg_col 按 group_col 分组，计算多种聚合统计量"""
    aggs = train.groupby(group_col)[agg_col].agg([
        "mean", "std", "min", "max", "median",
        "skew", "sum", "count"
    ]).reset_index()
    aggs.columns = [group_col] + [f"{group_col}_{agg_col}_{s}" for s in ["mean","std","min","max","median","skew","sum","count"]]

    train = train.merge(aggs, on=group_col, how="left")
    test = test.merge(aggs, on=group_col, how="left")
    return train, test

# 使用示例
for group in ["city", "category", "user_id"]:
    for agg in ["amount", "price", "count"]:
        if group in train.columns and agg in train.columns:
            train, test = add_aggregation_features(train, test, group, agg)

# 2. 交叉特征（组合两个类别特征，生成高基数新特征）
def add_combination_feature(train, test, col1, col2):
    comb = train[col1].astype(str) + "_" + train[col2].astype(str)
    # 对组合后的特征做目标编码（见下节）
    # 此处先只生成基础组合
    train[f"{col1}_{col2}_combo"] = train[col1].astype(str) + "_" + train[col2].astype(str)
    test[f"{col1}_{col2}_combo"] = test[col1].astype(str) + "_" + test[col2].astype(str)
    return train, test

# 3. 频率编码（类别出现次数，替代或补充 One-Hot）
for col in cat_cols:
    freq_map = train[col].value_counts(normalize=True).to_dict()
    train[f"{col}_freq"] = train[col].map(freq_map)
    test[f"{col}_freq"] = test[col].map(freq_map).fillna(0)

# 4. 日期/时间特征拆解
date_col = "transaction_date"
train[date_col] = pd.to_datetime(train[date_col])
test[date_col] = pd.to_datetime(test[date_col])

for df in [train, test]:
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["quarter"] = df[date_col].dt.quarter
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    df["days_since_epoch"] = (df[date_col] - pd.Timestamp("1970-01-01")).dt.days
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)

# 5. 数值分箱（将连续值离散化以捕捉非线性关系）
for col in numeric_cols:
    train[f"{col}_binned"] = pd.cut(train[col], bins=10, labels=False)
    test[f"{col}_binned"] = pd.cut(test[col], bins=10, labels=False)
    # 然后可对分箱做 target encoding
```

**目标编码（Target Encoding）**：

> 高分竞赛中几乎必用的技巧。核心思想：用目标变量的统计量替代类别值。但必须做 cross-fold 编码，防止数据泄漏。

```python
from sklearn.model_selection import StratifiedKFold

def target_encode_cv(train, test, cat_cols, target_col, n_folds=5, seed=42):
    """
    Cross-fold Target Encoding（防泄漏版本）
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for col in cat_cols:
        train[f"{col}_te"] = np.nan
        test_enc = np.zeros(len(test))

        global_mean = train[target_col].mean()

        for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[target_col])):
            X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]

            # 用 train fold 计算 target mean
            enc_map = X_tr.groupby(col)[target_col].mean().to_dict()
            train.loc[train.index[val_idx], f"{col}_te"] = X_val[col].map(enc_map)

            # test 用每个 fold 的编码取平均
            test_enc += test[col].map(enc_map).fillna(global_mean).values / n_folds

        # 填充 train 中未匹配的值（冷启动用全局均值）
        train[f"{col}_te"].fillna(global_mean, inplace=True)

        # 将 test 编码写入 test
        test[f"{col}_te"] = test_enc

    return train, test

# 调用
cat_cols_to_encode = ["city", "category", "user_id"]
train, test = target_encode_cv(train, test, cat_cols_to_encode, target_col)

# 高级技巧：加入平滑参数（减少低频类别的过拟合）
def smooth_target_encode(train_series, target_series, min_samples=10, smoothing=20):
    mean = target_series.mean()
    agg = train_series.groupby(train_series).agg(count="count", sum="sum")
    agg["smoothed"] = (agg["sum"] + smoothing * mean) / (agg["count"] + smoothing)
    return agg["smoothed"].to_dict()
```

#### 1.3.3 特征选择

```python
# 方法 1：基于 LightGBM 的 feature importance
import lightgbm as lgbm
from sklearn.model_selection import train_test_split

X_tr, X_hold, y_tr, y_hold = train_test_split(X, y, test_size=0.2, random_state=SEED)

model = lgbm.LGBMClassifier(n_estimators=1000, random_state=SEED, verbosity=-1)
model.fit(X_tr, y_tr)
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

# 只保留 importance > 0 的特征
useful_features = importance[importance["importance"] > 0]["feature"].tolist()
print(f"Selected {len(useful_features)} / {len(X.columns)} features")

# 方法 2：消融实验（逐个移除特征看 CV 变化）
def ablation_study(X, y, features, baseline_cv):
    results = []
    for feat in features:
        reduced = [f for f in features if f != feat]
        cv_score = quick_cv(X[reduced], y)  # 快速跑一折 CV
        delta = cv_score - baseline_cv
        results.append({"feature": feat, "delta_cv": delta})
    return pd.DataFrame(results).sort_values("delta_cv", ascending=False)

# 方法 3：相关性过滤（移除高度相关的冗余特征）
corr_matrix = X[numeric_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
X.drop(columns=to_drop, inplace=True)
test.drop(columns=to_drop, inplace=True)
print(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")
```

---

### 1.4 第三阶段：模型优化（第 14-21 天）

#### 1.4.1 多模型训练

```python
# LightGBM（通常是最强的 baseline）
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 10000,
    "learning_rate": 0.05,
    "num_leaves": 256,
    "min_child_samples": 20,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "verbosity": -1,
}

lgb_model = LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(100), log_evaluation(500)]
)

# XGBoost
from xgboost import XGBClassifier

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 10000,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbosity": 0,
}
xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    verbose=500
)

# CatBoost
from catboost import CatBoostClassifier

cb_model = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.05,
    depth=7,
    random_seed=SEED,
    verbose=500
)
cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
```

#### 1.4.2 超参数优化（Optuna）

> 在特征工程基本稳定之后再做调参。Optuna 比 GridSearch 快 10 倍以上。

```python
import optuna
from optuna.samplers import TPESampler

def optimize_lgb(X, y, n_trials=50):
    """
    Optuna 自动搜索 LightGBM 超参数
    """
    def objective(trial, X, y):
        # 交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_estimators": 10000,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 512),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": SEED,
            }
            model = LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(50), log_evaluation(0)]
            )
            # 使用 best_iteration 的 AUC
            from sklearn.metrics import roc_auc_score
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        study_name="lgb_optimization"
    )
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best CV score: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")
    return study.best_params

# 运行（在所有特征上）
best_lgb_params = optimize_lgb(X, y, n_trials=100)
```

**调参顺序建议（从最重要到最不重要）**：

1. `num_leaves` + `min_child_samples`（控制树复杂度）
2. `learning_rate` + `n_estimators`（学习率与迭代数配合）
3. `subsample` + `colsample_bytree`（行/列采样防过拟合）
4. `reg_alpha` + `reg_lambda`（正则化）

---

### 1.5 第四阶段：冲刺（最后一周）

#### 1.5.1 Stacking

```python
# ======== 完整的 Stacking Pipeline ========
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

class StackingEnsemble:
    """
    两层 Stacking：
    - Level 1：多个基模型（不同算法/参数/特征集）
    - Level 2：元模型（用 Level 1 的 OOF 预测做特征）
    """
    def __init__(self, base_models, meta_model, n_folds=5, seed=42):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.seed = seed

    def fit_predict(self, X, y, X_test):
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        # 存储 OOF 预测和 Test 预测
        n_train = len(X)
        n_test = len(X_test)
        n_models = len(self.base_models)

        oof_predictions = np.zeros((n_train, n_models))  # Level 1 的特征
        test_predictions = np.zeros((n_test, n_models))

        for model_idx, (model_name, model_class) in enumerate(self.base_models):
            print(f"\n===== Training Base Model {model_idx + 1}: {model_name} =====")
            test_fold_preds = np.zeros(n_test)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                print(f"  Fold {fold + 1}/{self.n_folds}")
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = model_class  # clone or instantiate
                model.fit(X_tr, y_tr)
                oof_predictions[val_idx, model_idx] = model.predict_proba(X_val)[:, 1]
                test_fold_preds += model.predict_proba(X_test)[:, 1] / self.n_folds

            test_predictions[:, model_idx] = test_fold_preds
            print(f"  OOF AUC: {roc_auc_score(y, oof_predictions[:, model_idx]):.5f}")

        # Level 2：训练元模型
        print("\n===== Training Meta Model =====")
        self.meta_model.fit(oof_predictions, y)

        # 最终预测
        final_oof = self.meta_model.predict(oof_predictions)
        final_test = self.meta_model.predict(test_predictions)

        print(f"\nFinal OOF AUC: {roc_auc_score(y, final_oof):.5f}")
        return final_oof, final_test, oof_predictions, test_predictions

# 使用示例
from sklearn.linear_model import LogisticRegression

base_models = [
    ("LGB_1", LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=128, random_state=SEED, verbosity=-1)),
    ("LGB_2", LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=256, random_state=SEED+1, verbosity=-1)),
    ("XGB_1", XGBClassifier(n_estimators=5000, learning_rate=0.05, max_depth=7, random_state=SEED, verbosity=0)),
    ("CatB_1", CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=7, random_seed=SEED, verbose=0)),
    ("LGB_3", LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=64, subsample=0.7, random_state=SEED+2, verbosity=-1)),
]

meta_model = LogisticRegression(C=1.0, random_state=SEED)
stacking = StackingEnsemble(base_models, meta_model, n_folds=5, seed=SEED)

final_oof, final_test, oof_preds, test_preds = stacking.fit_predict(X, y, X_test)

sub["target"] = final_test
sub.to_csv("sub_stacking.csv", index=False)
```

#### 1.5.2 Blending

> Stacking 用 OOF（全数据交叉），Blending 用 Hold-Out（固定的验证集）。Blending 更简单，但数据利用率更低。

```python
from sklearn.model_selection import train_test_split

# 划分 hold-out 集
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

blend_train = np.zeros((len(X_hold), 5))
blend_test = np.zeros((len(X_test), 5))

for i, model in enumerate(base_models):
    model.fit(X_train, y_train)
    blend_train[:, i] = model.predict_proba(X_hold)[:, 1]
    blend_test[:, i] = model.predict_proba(X_test)[:, 1]

# 用 hold-out 集训练元模型
meta = LogisticRegression()
meta.fit(blend_train, y_hold)
final_test = meta.predict(blend_test)

sub["target"] = final_test
sub.to_csv("sub_blending.csv", index=False)
```

#### 1.5.3 后处理技巧

```python
# 技巧 1：阈值优化（分类问题）
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, y_prob):
    """搜索使 F1 最大的阈值"""
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob > thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    print(f"Optimal threshold: {best_thr:.3f}, Best F1: {best_f1:.5f}")
    return best_thr

optimal_threshold = find_optimal_threshold(y_val, oof_preds)

# 应用到测试集
final_preds = (test_preds > optimal_threshold).astype(int)

# 技巧 2：排序优化（利用已知的先验分布修正排序）
# 如果 test 和 train 的分布有偏移，对预测做 rank 归一化
from scipy.stats import rankdata

def normalize_rank(preds, reference_min, reference_max):
    ranks = rankdata(preds) / len(preds)
    return reference_min + ranks * (reference_max - reference_min)

# 技巧 3：规则修正（利用领域知识的硬修正）
# 例如：年龄不能为负、概率必须在 [0,1]
test_preds = np.clip(test_preds, 0, 1)

# 已知某些样本必定为正/负
known_positives_idx = test[test["rule_flag"] == 1].index
test_preds[known_positives_idx] = 1.0

# 技巧 4：伪标签（Pseudo-labeling，高风险高回报）
# 用当前模型对 test 预测，选高置信度的样本加入训练
test_preds_all = model.predict_proba(X_test)[:, 1]
high_conf_idx = np.where((test_preds_all > 0.95) | (test_preds_all < 0.05))[0]

if len(high_conf_idx) > 0:
    pseudo_labels = (test_preds_all[high_conf_idx] > 0.5).astype(int)
    X_pseudo = X_test.iloc[high_conf_idx]
    y_pseudo = pd.Series(pseudo_labels, index=X_pseudo.index)

    # 合并回训练集重新训练
    X_augmented = pd.concat([X, X_pseudo])
    y_augmented = pd.concat([y, y_pseudo])
    model.fit(X_augmented, y_augmented)
    print(f"Added {len(high_conf_idx)} pseudo-labeled samples")
```

#### 1.5.4 最终提交策略

```markdown
**策略原则：2 次提交，一稳一险**

| 提交 | 策略 | 说明 |
|------|------|------|
| 提交 1 | 安全提交 | 集成最稳定的 3 个模型（LGB+XGB+CatB 简单平均） |
| 提交 2 | 激进提交 | 完整 Stacking + 伪标签 + 后处理 |

**为什么选两个？**
- 安全提交：大概率在 Private LB 不掉队，保底
- 激进提交：如果 Shake Up 对你有利，可能冲金

**什么是 Shake Up？**
Public LB 排名和 Private LB 排名剧烈变动。主因：
- 对手在 Public LB 上过拟合了
- Public/Private 数据分布不完全一致
- 策略：CV > LB，永远不要为了 Public LB 调参
```

---

## 2. Kaggle 抢分核心技巧（重中之重）

### 2.1 CV（交叉验证）策略——一切的基石

#### 为什么 CV 比 LB 更重要？

- **LB 只有你提交时才更新**：一天最多 3-5 次，你无法通过 LB 快速迭代
- **LB 只看 test set 的部分子集**（Public LB 约 20-30% 的 test），有方差
- **LB 会背叛你**：Shake Up 时 Public LB 金牌可能变成 Private LB 铜牌
- **CV 是你的诊断工具**：告诉你模型是否学到了真实模式还是噪声

**黄金法则：Trust your CV, not your LB.**

#### 不同场景的 CV 选择

```python
# ========== 场景 1：普通分类/回归 ==========
from sklearn.model_selection import StratifiedKFold, KFold

# 分类 -> StratifiedKFold（保持每折类别分布一致）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# 回归 -> KFold（简单随机分割）
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 回归但想分层 -> 将 y 离散化后用 StratifiedKFold
y_binned = pd.qcut(y, q=10, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for train_idx, val_idx in skf.split(X, y_binned):
    ...

# ========== 场景 2：时间序列 ==========
from sklearn.model_selection import TimeSeriesSplit

# 标准 TimeSeriesSplit（不用 shuffle，时间顺序展开）
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # train_idx 总是早于 val_idx
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    # 注意：不能 shuffle，否则时间泄漏

# Purged K-Fold（更严格的时序交叉验证，防止信息泄漏）
# 核心思想：在 train/val 之间留一个 gap（purge 窗），避免相邻时间点的信息渗漏
def purged_kfold(X, y, dates, n_splits=5, embargo_pct=0.01):
    """自实现 Purged K-Fold"""
    sorted_idx = np.argsort(dates)
    splits = np.array_split(sorted_idx, n_splits)

    for i in range(n_splits):
        val_idx = splits[i]
        # train 不包括 val 和紧邻 val 的 embargo 区间
        train_idx = np.concatenate([s for j, s in enumerate(splits) if j != i])

        # 去掉紧邻 val 边界的数据（purge）
        embargo_size = int(len(sorted_idx) * embargo_pct)
        purge_after = splits[i-1][-embargo_size:] if i > 0 else np.array([])
        purge_before = splits[(i+1) % n_splits][:embargo_size] if i < n_splits - 1 else np.array([])
        purge_idx = np.concatenate([purge_after, purge_before])
        train_idx = np.setdiff1d(train_idx, purge_idx)

        yield train_idx, val_idx

# ========== 场景 3：分组数据 ==========
from sklearn.model_selection import GroupKFold

# 同一 group 的所有样本必须在同一折（训练或验证），不能跨折
# 典型场景：用户多次行为记录、同一医院的患者数据
groups = train["user_id"]  # 按用户分组
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    # 确保：train 中的 user_id 不会出现在 val 中

# ========== 场景 4：多标签分类 ==========
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# pip install iterative-stratification

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# y 现在是 n_samples x n_labels 的矩阵
for train_idx, val_idx in mskf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
```

#### CV 与 LB 不一致怎么办？

```markdown
**诊断步骤：**

1. **检查 CV 的稳定性**
   - 跑 5 折 CV，看各折分数的标准差。如果 std 很大（如 0.03+），说明数据量不够或分布不均
   - 加大折数（10 折）减小方差

2. **检查特征泄漏**
   - 是否存在 Test data 的统计信息泄漏到 Train（如全局 Target Encoding）
   - 是否存在时间泄漏（未来的数据拿来训练）

3. **检查 LB 的 Public/Private 分布**
   - 如果 Host 没说 Public/Private 如何划分，可能存在分布差异
   - 对抗性验证（Adversarial Validation）检测 train/test 分布偏移

4. **仍然不一致时**
   - **Trust your CV.** 这是 Kaggle Master 的共识
   - CV 可以看到全量数据，LB 只看部分
   - 历史上无数个 Public LB 金牌在 Private LB 暴跌的案例
```

**对抗性验证（Adversarial Validation）检测分布偏移**：

```python
# 检测 train 和 test 的分布是否一致
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

# 创建伪标签：train = 0, test = 1
X_all = pd.concat([X, X_test], ignore_index=True)
y_adv = np.array([0] * len(X) + [1] * len(X_test))

# 训练一个分类器区分 train/test
adv_model = LGBMClassifier(n_estimators=100, random_state=SEED, verbosity=-1)
cv_scores = cross_val_score(adv_model, X_all, y_adv, cv=5, scoring="roc_auc")

print(f"Adversarial Validation AUC: {cv_scores.mean():.4f}")
# AUC 接近 0.5 → 分布一致，LB 可信
# AUC > 0.6   → 分布偏移，LB 可能不可信
# AUC > 0.7   → 严重偏移，需要仔细处理
```

#### 标准 5-Fold CV Pipeline 代码模板

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

def run_cv(X, y, X_test, params, n_folds=5, seed=42):
    """
    标准 5-Fold CV pipeline
    返回：OOF预测、Test预测、各折分数
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*40}")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=params.get("n_estimators", 10000),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 128),
            random_state=seed,
            verbosity=-1,
            **{k: v for k, v in params.items()
               if k not in ["n_estimators", "learning_rate", "num_leaves"]}
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50), log_evaluation(500)]
        )

        preds_val = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds_val
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds

        score = roc_auc_score(y_val, preds_val)
        fold_scores.append(score)
        print(f"Fold {fold + 1} AUC: {score:.5f}")

    # 总体评估
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    oof_score = roc_auc_score(y, oof_preds)

    print(f"\n{'='*40}")
    print(f"Mean CV AUC: {mean_score:.5f} (+/- {std_score:.5f})")
    print(f"Overall OOF AUC: {oof_score:.5f}")
    print(f"{'='*40}")

    return oof_preds, test_preds, fold_scores, oof_score
```

---

### 2.2 集成学习——Kaggle 的终极武器

#### 为什么集成有效？

单个模型的误差 = 偏差² + 方差 + 噪声。
- **多个相同模型不同 seed** → 降低方差
- **多个不同算法模型** → 降低偏差
- **不同特征子集训练** → 增加多样性，降低相关性

**实际案例**：
- 单 LGB (seed=42): CV 0.8512
- 5 LGB (seed=42, 99, 123, 256, 777) 简单平均: CV 0.8538 → +26 个基点
- LGB + XGB + CatB 加权平均: CV 0.8561 → +49 个基点
- 5 模型 Stacking: CV 0.8583 → +71 个基点

#### 集成方法详解

```python
# ========== 1. 简单平均 ==========
def simple_average(predictions_list):
    """predictions_list: 每个模型的 test 预测 (n_samples,)"""
    return np.mean(predictions_list, axis=0)

# 多 seed 模型简单平均
seeds = [42, 99, 123, 256, 777]
test_preds_seeds = []

for s in seeds:
    model = LGBMClassifier(random_state=s, **best_params)
    model.fit(X, y)
    test_preds_seeds.append(model.predict_proba(X_test)[:, 1])

test_preds_avg = np.mean(test_preds_seeds, axis=0)

# ========== 2. 加权平均 ==========
def weighted_average(predictions, weights):
    """
    predictions: list of (n_samples,) arrays
    weights: 每个模型的权重，按 CV 分数分配
    """
    weights = np.array(weights) / np.sum(weights)
    return np.sum([p * w for p, w in zip(predictions, weights)], axis=0)

# 按 CV 分数加权
cv_scores = [0.8512, 0.8493, 0.8478, 0.8505, 0.8489]
weights = np.array(cv_scores) / np.sum(cv_scores)
test_preds_weighted = weighted_average(test_preds_seeds, weights)

# 优化权重（用 OOF 预测在元模型上学习权重）
from scipy.optimize import minimize

def optimize_weights(oof_preds_list, y_true):
    """用 Nelder-Mead 搜索最优权重（OOF 上优化）"""
    def objective(w):
        w = np.abs(w)
        w = w / w.sum()
        ensemble = np.sum([p * wi for p, wi in zip(oof_preds_list, w)], axis=0)
        return -roc_auc_score(y_true, ensemble)

    n_models = len(oof_preds_list)
    init_w = np.ones(n_models) / n_models
    result = minimize(objective, init_w, method="Nelder-Mead")
    optimal_w = np.abs(result.x)
    optimal_w = optimal_w / optimal_w.sum()
    print(f"Optimal weights: {optimal_w.round(4)}")
    return optimal_w

# ========== 3. Rank 平均（对异常值鲁棒） ==========
from scipy.stats import rankdata

def rank_average(predictions):
    """将每个预测向量转为排位，然后取平均排位"""
    ranks = np.array([rankdata(p) for p in predictions])
    mean_ranks = np.mean(ranks, axis=0)
    return mean_ranks / len(mean_ranks)  # 归一化到 [0, 1]

# ========== 4. 多策略集成的完整示例 ==========
class EnsembleMethods:
    @staticmethod
    def simple_mean(preds): return np.mean(preds, axis=0)
    @staticmethod
    def weighted_mean(preds, weights): return np.average(preds, axis=0, weights=weights)
    @staticmethod
    def geometric_mean(preds): return np.exp(np.mean(np.log(preds + 1e-10), axis=0))
    @staticmethod
    def rank_mean(preds):
        ranks = np.array([rankdata(p) for p in preds])
        return np.mean(ranks, axis=0) / preds.shape[1]
```

#### 多样性的来源（集成效果的核心）

| 多样性来源 | 具体操作 | 提分效果 |
|-----------|---------|---------|
| 不同算法 | LGB + XGB + CatB + NN | +0.003 ~ +0.008 |
| 不同特征子集 | 随机选 80% 特征训练 5 个模型 | +0.002 ~ +0.005 |
| 不同超参数 | 不同 num_leaves / learning_rate | +0.001 ~ +0.003 |
| 不同 seed | 同模型同参数不同 seed | +0.001 ~ +0.003 |
| Bagging | Bootstrap 采样训练 | +0.001 ~ +0.002 |
| 不同预处理 | 有/无标准化、不同编码方式 | +0.001 ~ +0.002 |

**组合策略**：算法多样性 × 特征多样性 × 参数多样性 = 最大提分效果

---

### 2.3 后处理技巧

```python
# ========== 1. 概率校准 ==========
# 如果你的模型输出的概率分布与实际标签分布不一致
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
calibrated.fit(X, y)
cal_preds = calibrated.predict_proba(X_test)[:, 1]

# ========== 2. 目标分布匹配 ==========
# 如果已知 test 的正样本比例，调整预测使均值匹配
def match_target_distribution(preds, target_mean):
    """缩放预测使均值 = target_mean"""
    current_mean = preds.mean()
    if current_mean > 0 and current_mean < 1:
        # 用简单线性缩放
        adjusted = preds * (target_mean / current_mean)
        return np.clip(adjusted, 0, 1)
    return preds

# ========== 3. 极端预测的惩罚处理 ==========
# 对极低/极高预测做平滑，防止单个样本的极端预测拉偏集成
def smooth_extreme_predictions(preds, lower=0.001, upper=0.999):
    return np.clip(preds, lower, upper)

# ========== 4. 排序后处理（竞赛中一种安全技巧） ==========
# 如果你的 CV 显示模型排序能力强但校准差，用排序做最终预测
def rank_based_prediction(oof_preds, test_preds, y_train):
    """将预测映射回 train 目标分布"""
    sorted_train = np.sort(y_train)
    sorted_oof_idx = np.argsort(oof_preds)
    n = len(sorted_train)

    # 将 test 预测按 OOF 分布的对应分位映射回 y 值
    mapped_test = np.zeros_like(test_preds)
    for i, p in enumerate(test_preds):
        # 找到 p 在 OOF 中的分位
        rank = np.searchsorted(np.sort(oof_preds), p) / n
        idx = int(rank * (n - 1))
        mapped_test[i] = sorted_train[min(idx, n - 1)]
    return mapped_test
```

---

### 2.4 特征工程的抢分思路（补充强化）

```python
# ========== 时间窗口特征（时序竞赛核心） ==========
def add_rolling_features(df, date_col, value_col, group_col, windows=[7, 14, 30, 90]):
    """为时序数据添加滑动窗口特征"""
    df = df.sort_values([group_col, date_col])

    for window in windows:
        # 滑动均值
        df[f"{value_col}_rolling_mean_{window}d"] = df.groupby(group_col)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # 滑动标准差
        df[f"{value_col}_rolling_std_{window}d"] = df.groupby(group_col)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        # 滑动最大/最小
        df[f"{value_col}_rolling_max_{window}d"] = df.groupby(group_col)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        df[f"{value_col}_rolling_min_{window}d"] = df.groupby(group_col)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
    return df

# ========== Lag Features（时序） ==========
def add_lag_features(df, value_col, group_col, lags=[1, 2, 3, 7, 14, 30]):
    """添加滞后特征"""
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df.groupby(group_col)[value_col].shift(lag)
    return df

# ========== Diff Features（时序） ==========
def add_diff_features(df, value_col, group_col, diffs=[1, 7, 30]):
    """添加差分特征"""
    for d in diffs:
        df[f"{value_col}_diff_{d}"] = df.groupby(group_col)[value_col].diff(d)
    return df

# ========== NLP 特征 ==========
from sklearn.feature_extraction.text import TfidfVectorizer

def add_tfidf_features(train_text, test_text, max_features=500):
    """提取 TF-IDF 特征"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        strip_accents="unicode",
        analyzer="char_wb"
    )
    train_tfidf = vectorizer.fit_transform(train_text)
    test_tfidf = vectorizer.transform(test_text)
    return train_tfidf, test_tfidf, vectorizer

# 基于预训练模型的 Embedding 特征
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("all-MiniLM-L6-v2")
# train_embeddings = model.encode(train_texts, show_progress_bar=True)
# test_embeddings = model.encode(test_texts, show_progress_bar=True)
```

---

### 2.5 模型调参的高效策略

**黄金原则**：
1. 先特征工程，再调参 — 一个好特征比 10 小时调参更值
2. 先用默认参数跑通，再优化
3. 用 Optuna 自动搜索，不要手动调
4. 调参的目标是稳健提升 CV，不是追 Public LB

```python
# Optuna 高级用法：Pruning（提前终止差 trial）
import optuna
from optuna.pruners import MedianPruner

def objective_with_pruning(trial, X, y):
    params = {
        "n_estimators": 10000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 512),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": SEED,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []

    for step, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(50), log_evaluation(0)])
        score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        scores.append(score)

        # Pruning: 跑了 3 折后如果中间分数太差就提前终止
        trial.report(np.mean(scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)

study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3)
)
study.optimize(lambda trial: objective_with_pruning(trial, X, y), n_trials=100)
```

---

## 3. 比赛中写代码的实战方法论

### 3.1 代码组织

> Notebook 适合 EDA，Script 适合生产。推荐 Script 为主，Notebook 为辅。

**推荐项目目录结构**：

```
competition-name/
├── input/                  # 原始数据（不要修改，使用软链接或复制）
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── output/                 # 所有预测结果和中间文件
│   ├── sub_baseline.csv
│   ├── sub_lgb_fold0.csv
│   ├── oof_lgb.npy
│   ├── test_lgb.npy
│   ├── experiment_log.csv
│   └── feature_importance.csv
├── models/                 # 保存的模型文件（可选，方便后续集成）
│   ├── lgb_fold0.pkl
│   ├── lgb_fold1.pkl
│   └── stacking_meta.pkl
├── notebooks/              # Jupyter notebooks（仅用于 EDA）
│   ├── 01_eda.ipynb
│   └── 02_feature_analysis.ipynb
├── src/                    # 核心代码
│   ├── __init__.py
│   ├── config.py           # 所有超参数和路径配置
│   ├── data.py             # 数据加载、清洗、预处理
│   ├── features.py         # 特征工程（按功能拆分为多个函数）
│   ├── model.py            # 模型定义、训练、预测
│   ├── ensemble.py         # 集成方法（Stacking/Blending/Averaging）
│   ├── postprocess.py      # 后处理逻辑
│   └── utils.py            # seed设置、日志、指标计算等工具函数
├── run.py                  # 主入口脚本（一键执行）
├── run_feature_selection.py
├── run_hyperparameter_optimization.py
└── requirements.txt        # Python 依赖
```

**config.py 模板**：

```python
# ====== config.py ======
import os

class CFG:
    # 路径
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    MODEL_DIR = "models"

    # 数据
    TARGET_COL = "target"
    ID_COL = "id"

    # 训练
    SEED = 42
    N_FOLDS = 5
    DEBUG = False  # True 时只用 1000 行，快速调试

    # 模型
    LGB_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 10000,
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": SEED,
        "verbosity": -1,
    }

    # 特征工程开关
    USE_TARGET_ENCODING = True
    USE_AGGREGATION_FEATURES = True
    USE_FREQUENCY_ENCODING = True
    USE_COMBINATION_FEATURES = False  # 高基数特征，谨慎开启
    TARGET_ENCODE_COLS = ["col1", "col2"]

    @staticmethod
    def ensure_dirs():
        for d in [CFG.INPUT_DIR, CFG.OUTPUT_DIR, CFG.MODEL_DIR]:
            os.makedirs(d, exist_ok=True)

CFG.ensure_dirs()
```

### 3.2 代码编写原则

```python
# ====== utils.py ======
import random
import os
import numpy as np
import pandas as pd

def seed_everything(seed: int = 42):
    """固定所有随机种子，保证可复现"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def setup_logging(exp_name: str):
    """创建实验日志"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{exp_name}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_oof_and_test(oof_preds, test_preds, exp_name, oof_score=None):
    """保存 OOF 和 Test 预测，方便后续集成加载"""
    np.save(f"output/oof_{exp_name}.npy", oof_preds)
    np.save(f"output/test_{exp_name}.npy", test_preds)
    if oof_score is not None:
        # 记录到汇总文件
        record_file = "output/ensemble_records.csv"
        record = pd.DataFrame([
            {"name": exp_name, "oof_score": oof_score, "n_folds": len(oof_preds)}
        ])
        if os.path.exists(record_file):
            record.to_csv(record_file, mode="a", header=False, index=False)
        else:
            record.to_csv(record_file, index=False)
    print(f"Saved OOF & Test for: {exp_name}")

def load_oof_and_test(exp_name):
    """加载之前保存的预测"""
    oof = np.load(f"output/oof_{exp_name}.npy")
    test = np.load(f"output/test_{exp_name}.npy")
    return oof, test
```

### 3.3 Debug 与迭代策略

```markdown
**迭代流程（每轮不超过 1 小时）：**

1. **假设**："添加用户级别的聚合特征能提升 CV"
2. **实现**：在 features.py 中实现聚合特征函数
3. **实验**：跑小规模实验（取 10% 数据，2 折 CV）
4. **评估**：比较添加前后的 CV 分数差
5. **决策**：
   - CV +0.002 → 保留，展开到全数据
   - CV 持平 → 放弃或组合其他特征再试
   - CV 下降 → 立即回滚

**消融实验（Ablation Study）模板：**
- 全特征 baseline CV: 0.8512
- 移除聚合特征: 0.8481 → 聚合特征贡献 +0.0031 ✓
- 移除目标编码: 0.8495 → 目标编码贡献 +0.0017 ✓
- 移除交叉特征: 0.8504 → 交叉特征贡献 +0.0008 ~
```

### 3.4 时间管理

| 阶段 | 时间占比 | 核心活动 | 产出 |
|------|---------|---------|------|
| 赛前准备 | 5% | 读题、EDA、Baseline | 首次提交 |
| Baseline | 10% | 搭 pipeline、确认 CV/LB 相关性 | 可复现框架 |
| 特征工程 | 50% | 造特征、选特征、迭代实验 | 最强单模 CV |
| 模型优化 | 20% | 多模型训练、Optuna 调参 | 多个强模型 |
| 冲刺集成 | 15% | Stacking、后处理、提交策略 | 最终排名 |

### 3.5 标准代码模板（完整版 ~80 行）

```python
# ========== run.py ==========
"""
主脚本：一键执行完整 pipeline
用法: python run.py
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import CFG
from utils import seed_everything, save_oof_and_test, setup_logging

# ========== 1. 初始化 ==========
seed_everything(CFG.SEED)
logger = setup_logging("main_pipeline")

# ========== 2. 加载数据 ==========
logger.info("Loading data...")
train = pd.read_csv(f"{CFG.INPUT_DIR}/train.csv")
test = pd.read_csv(f"{CFG.INPUT_DIR}/test.csv")
sub = pd.read_csv(f"{CFG.INPUT_DIR}/sample_submission.csv")

if CFG.DEBUG:
    train = train.sample(n=1000, random_state=CFG.SEED).reset_index(drop=True)
    logger.warning("DEBUG MODE: using 1000 samples")

# ========== 3. 特征工程 ==========
logger.info("Feature engineering...")
# ... 在此处调用 features.py 中的特征工程函数 ...
# from features import add_aggregation_features, add_target_encoding
# train, test = add_aggregation_features(train, test)
# train, test = add_target_encoding(train, test)

# 识别特征列
feat_cols = [c for c in train.columns if c not in [CFG.TARGET_COL, CFG.ID_COL]]
feat_cols = [c for c in feat_cols if c in test.columns]

# 处理类别列
cat_cols = train[feat_cols].select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

X = train[feat_cols]
y = train[CFG.TARGET_COL]
X_test = test[feat_cols]

logger.info(f"Features: {len(feat_cols)} | Samples: {len(X)}")

# ========== 4. 5-Fold CV 训练 ==========
logger.info("Starting 5-Fold CV training...")
skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    logger.info(f"Fold {fold + 1}/{CFG.N_FOLDS}")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(**CFG.LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50), log_evaluation(500)]
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

# ========== 5. 评估 ==========
oof_score = roc_auc_score(y, oof_preds)
logger.info(f"Overall OOF AUC: {oof_score:.5f}")

# ========== 6. 保存 ==========
save_oof_and_test(oof_preds, test_preds, "lgb_baseline", oof_score)

sub[CFG.TARGET_COL] = test_preds
sub.to_csv(f"{CFG.OUTPUT_DIR}/sub_lgb_baseline.csv", index=False)
logger.info("Done! Submit output/sub_lgb_baseline.csv")
```

---

## 4. 常见竞赛类型的策略

### 4.1 表格数据竞赛

> Kaggle 最经典的竞赛类型，占所有比赛 60% 以上。

**核心策略**：

```markdown
1. **GBDT 为主力**（LightGBM / XGBoost / CatBoost）
   - 天然支持缺失值、类别特征、非线性
   - 特征工程收益远大于换模型

2. **NN 做辅助**
   - TabNet / FT-Transformer / MLP
   - NN 和 GBDT 的误差模式不同，集成收益大

3. **完整技术栈**
   ├── LightGBM (feature importance + training)
   ├── XGBoost   (different inductive bias)
   ├── CatBoost  (best for many categorical features)
   ├── TabNet    (NN for tabular, sacrebleu implementation)
   └── Stacking  (LogisticRegression/Ridge as meta-model)

4. **典型 CV 提升路径**
   0.850 (baseline LGB) → 0.860 (+特征工程) → 0.865 (+多模型) → 0.868 (+Stacking)
```

### 4.2 NLP 竞赛

**核心策略**：

```python
# ========== NLP Pipeline ==========
# 1. 微调预训练 Transformer
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
import torch
from datasets import Dataset

# 加载模型（根据任务选 base 还是 large）
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 分词
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

dataset = Dataset.from_pandas(train_df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()

# ========== 2. 多模型集成 ==========
# 典型组合：DeBERTa-v3-large + RoBERTa-large + ELECTRA-large
models_to_ensemble = [
    "microsoft/deberta-v3-large",
    "roberta-large",
    "google/electra-large-discriminator",
]
# 每个模型 5-Fold CV 微调，然后 OOF + Test 预测做平均

# ========== 3. AWP / FGM 对抗训练 ==========
# 提升模型的鲁棒性和泛化能力
# AWP (Adversarial Weight Perturbation)
class AWP:
    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.2):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch):
        """一次对抗扰动 + 梯度回传"""
        self._save()
        self._attack_step()  # 添加扰动
        _, adv_loss = self.model(**batch)[:2]
        adv_loss.backward()
        self._restore()

    def _attack_step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.adv_lr * param.grad / (norm + 1e-8)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                self.backup[name] = param.data.clone()
                self.backup_eps[name] = (
                    self.backup[name] - self.adv_eps,
                    self.backup[name] + self.adv_eps
                )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# ========== 4. R-Drop（正则化 Dropout） ==========
# 对同一输入两次前向，用 KL 散度使两次输出一致
# 在 Transformers 中，这不需额外代码 — 用 Trainer 的 compute_loss 包装即可

# ========== 5. TTA（Test Time Augmentation） ==========
# 对测试文本做多种数据增强，取平均预测
# 如：原始文本、回译文本、替换同义词文本
```

### 4.3 CV（计算机视觉）竞赛

**核心策略**：

```python
# ========== CV Pipeline ==========
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. 数据增强
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 2. 模型选择
# ImageNet-22k 预训练 > ImageNet-1k 预训练
model = models.convnext_large(pretrained=True)  # or vit_large_patch16_224
model.head = torch.nn.Linear(model.head.in_features, num_classes)

# 3. 混合精度训练
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# ========== TTA（Test Time Augmentation） ==========
def predict_with_tta(model, image, num_tta=5):
    """测试时增强：多次增强推理取平均"""
    model.eval()
    preds = []
    tta_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),   # 50% 概率水平翻转
        A.VerticalFlip(p=0.5),     # 50% 概率垂直翻转
        A.RandomRotate90(p=0.5),   # 50% 概率旋转 90°
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    with torch.no_grad():
        for _ in range(num_tta):
            aug_image = tta_transforms(image=image)["image"].unsqueeze(0).cuda()
            preds.append(model(aug_image).softmax(1).cpu().numpy())

    return np.mean(preds, axis=0)

# ========== 通用 CV 提分技巧 ==========
# - 更大的输入分辨率（224 → 384 → 512）
# - EMA（指数移动平均）权重
# - 多模型集成（CNN + ViT + ConvNext）
# - 伪标签：用当前最佳模型对 unlabeled/test 预测，加入训练
# - MixUp / CutMix 数据增强
```

### 4.4 时间序列竞赛

**核心策略**：

```python
# 时间序列竞赛 = GBDT + 时序特征 + 时间感知 CV

# 1. 时间感知 CV（严格使用 Purged K-Fold 或 TimeSeriesSplit）
# 2. 特征工程是核心（见 2.4 节 Rolling/Lag/Diff 特征）
# 3. LightGBM 通常优于 LSTM/Transformer（表格数据量小的情况下）

# ========== 时序特有特征模板 ==========
def create_time_series_features(df, date_col, target_col, group_col):
    """完整的时间序列特征创建"""
    df = df.sort_values([group_col, date_col]).copy()

    # Lag 特征
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)

    # 滚动窗口特征
    for window in [7, 14, 30, 90]:
        df[f"rolling_mean_{window}"] = df.groupby(group_col)[target_col] \
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"rolling_std_{window}"] = df.groupby(group_col)[target_col] \
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        df[f"rolling_skew_{window}"] = df.groupby(group_col)[target_col] \
            .transform(lambda x: x.rolling(window, min_periods=3).skew())

    # 指数加权移动平均（给近期值更高权重）
    for span in [7, 14, 30]:
        df[f"ewm_{span}"] = df.groupby(group_col)[target_col] \
            .transform(lambda x: x.ewm(span=span, adjust=False).mean())

    # 趋势特征（简单线性回归斜率）
    def calc_trend(series):
        if len(series) < 3:
            return 0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0] if not np.isnan(series).any() else 0

    for window in [30, 90]:
        df[f"trend_{window}"] = df.groupby(group_col)[target_col] \
            .transform(lambda x: x.rolling(window, min_periods=3).apply(calc_trend, raw=True))

    return df

# ========== 模型选择 ==========
# GBDT（主力） + N-BEATS / N-HiTS（NN辅助） + 集成
```

---

## 5. 比赛心态与策略

### 5.1 不要被 Public LB 迷惑

```markdown
**Shake Up 的本质：**

Public LB 约占 test 数据的 20-30%。你的模型可能：
- 在 Public LB 上高分，但在 Private LB 上暴跌（过拟合 Public）
- 在 Public LB 上低分，但在 Private LB 上暴涨（你的 CV 工作做得好）

**典型案例：**
- Kaggle 某比赛中，Public LB 第 1 名在 Private LB 跌至 1500+ 名
- 某比赛中，Public LB 500 名在 Private LB 升至第 3 名

**应对方法：**
- 永远用 CV 指导决策，LB 仅用于验证提交格式
- 不要抄 Public LB 高分方案的参数——他们可能过拟合了
- 多个不同 CV 策略表现一致的模型更可信
```

### 5.2 提交策略

```markdown
| 策略类型 | 描述 | 何时使用 |
|---------|------|---------|
| **安全提交** | 稳健的集成方案，CV 稳定且方差小 | 始终提交，作为保底 |
| **激进提交** | 高风险高回报：伪标签、极端后处理 | 最后 1-2 天冲刺 |
| **差异提交** | 与安全提交完全不同的方案 | 期望一个人跌、一个人涨 |

**关键原则：**
- 不要用完每日提交限额（留到关键时刻用）
- 最后一次提交在比赛截止前 2-3 小时（不要等到最后 5 分钟，平台可能挂）
- 两个最终提交最好用不同策略生成
```

### 5.3 学会从失败中学习

```markdown
**赛后复盘清单：**

1. 我的 CV 和 Private LB 的差距是多少？
   - 差距 < 0.003：CV 策略有效，继续保持
   - 差距 > 0.005：CV 策略需要调整

2. 我的特征工程哪些有效、哪些无效？
   - 记录每个特征的 CV 增量

3. 金牌方案和我差在哪里？
   - 读 Winning Solution Discussion
   - 重点关注：他们用了什么我没想到的特征/技巧

4. 时间分配是否合理？
   - 是否在某个阶段花了太多时间？

5. 将学到的技巧沉淀到自己的代码库中
```

### 5.4 组队策略

```markdown
**单人参赛的利弊：**
✓ 完全掌控代码和决策
✓ 学到的东西最多
✗ 试错成本高（一个人扛所有实验）
✗ 冲刺阶段精力有限

**组队参赛（推荐 2-3 人）：**

| 角色 | 职责 |
|------|------|
| 特征工程师 | 数据清洗、特征创造、EDA 分析 |
| 模型工程师 | 模型训练、调参、集成 |
| 全栈选手 | 代码框架、自动化、提交管理 |

**组队最佳实践：**
- 用 GitHub + branch 管理代码
- 用 MLflow / wandb 共享实验记录
- 各自跑不同的特征方向，每周合并一次
- 最后冲刺阶段分工：一个人做集成，一个人做后处理

**什么时候合队？**
- 通常在比赛进行到一半时（第 2 周左右）
- 先独立探索 2 周，各自有了方向再合并
- 避免比赛一开始就组大团队（容易方向混乱）
```

### 5.5 长期成长路径

```markdown
**入门 → Competitor → Contributor → Master → Grandmaster**

1. **入门阶段（0-3 个月）**
   - 完成 Kaggle 官方 "Getting Started" 竞赛
   - Titanic, House Prices, Digit Recognizer
   - 重点学：Pandas、EDA、简单的模型训练

2. **Competitor 阶段（3-6 个月）**
   - 参加活跃的 Featured 竞赛
   - 学习复制 Discussion 区的 Kernel
   - 重点学：CV 策略、特征工程、LightGBM

3. **Contributor 阶段（6-12 个月）**
   - 发布高质量的 Notebook / Discussion
   - 至少获得 1 枚银牌
   - 重点学：Stacking、TPU/GPU 训练、竞赛策略

4. **Master 阶段（1-2 年）**
   - 持续获得金牌
   - 有自己的一套竞赛代码库
   - 重点学：新赛题的快速适应能力

5. **Grandmaster 阶段（2 年+）**
   - 成为领域专家
   - 竞赛 + 写作 + 比赛组织
   - 综合影响力
```

> **最重要的建议**：不要以拿奖牌为目标，以"每次比赛学到 3 个新技巧"为目标。奖牌自然会来。

---

## 相关概念

- [[数据预处理与特征工程]]
- [[模型选择指南]]
- [[Kaggle学习与成长]]
- [[机器学习评估指标]]
- [[集成学习从理论到实践]]
- [[深度学习调参经验汇总]]
