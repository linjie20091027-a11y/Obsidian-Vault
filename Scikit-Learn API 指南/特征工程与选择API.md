# 特征工程与选择 API 详解

> 特征工程决定了模型的上限，算法只是在逼近这个上限。本文涵盖 sklearn 中特征提取、特征选择、降维、流形学习四大模块的完整 API。

---

## 1. 特征提取（sklearn.feature_extraction）

### 1.1 文本特征提取

#### CountVectorizer

**功能**：将文本集合转换为词频矩阵（词袋模型，Bag of Words）。统计每个文档中每个词汇出现的次数。

**什么时候用**：
- 文本分类/聚类任务的 baseline 特征
- 需要词频信息且不关心词序的场景
- 配合 `TfidfTransformer` 使用可实现与 `TfidfVectorizer` 等价的效果
- 需要保留原始计数信息（如多项式朴素贝叶斯）

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_features` | int/None | None | 按词频排序保留前 N 个词，预防维度爆炸 |
| `ngram_range` | tuple | (1, 1) | n-gram 范围，如 (1, 2) 包含 unigram + bigram |
| `stop_words` | str/list/None | None | 停用词列表，'english' 为内置英文停用词 |
| `min_df` | int/float | 1 | 最低文档频率，int 为绝对次数，float 为比例 (0.0~1.0) |
| `max_df` | int/float | 1.0 | 最高文档频率，超过的视为语料库词（corpus-specific stop words） |
| `analyzer` | str | 'word' | 'word' / 'char' / 'char_wb'（字符级 n-gram） |
| `binary` | bool | False | True 时所有非零计数置为 1，仅关注是否出现 |
| `token_pattern` | str | r'(?u)\b\w\w+\b' | 分词正则，仅 analyzer='word' 时有效 |
| `lowercase` | bool | True | 是否转小写 |

**代码示例**：

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "我爱机器学习，机器学习很有趣",
    "深度学习是机器学习的一个分支",
    "自然语言处理是人工智能的重要领域",
    "人工智能和机器学习密不可分"
]

# 基础用法
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("特征词列表:", vectorizer.get_feature_names_out())
print("词频矩阵形状:", X.shape)
print("词频矩阵:\n", X.toarray())

# 进阶用法：限制特征数 + bigram + 停用词
vectorizer_adv = CountVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,          # 至少在 2 个文档中出现
    max_df=0.8,        # 在超过 80% 文档中出现的词视为停用词
    binary=True        # 只记录出现与否
)
X_adv = vectorizer_adv.fit_transform(corpus)
```

**注意事项/踩坑点**：
- **维度爆炸**：中文分词后词汇量通常很大，务必设置 `max_features` 或 `min_df` 限制。
- **中文字符间无空格**：`CountVectorizer` 默认按空格分词，中文需要先用 jieba 分词并用空格连接，或自定义 `tokenizer` 参数。
- **`min_df` 和 `max_df` 的类型陷阱**：传入 float 时按比例（如 0.01 = 1%），传入 int 时按绝对次数。混淆会导致特征维度异常。
- **`binary=True` 的妙用**：对于短文本（如短信、标题），`binary=True` 往往比原始计数效果更好，因为短文本中同一词很少重复出现。
- **稀疏矩阵**：`fit_transform` 返回的是 `scipy.sparse.csr_matrix`，占用内存小，但某些操作（如 `X.mean()`）需要先转密集矩阵或使用稀疏矩阵专属方法。
- **训练集和测试集的词汇表必须一致**：对训练集用 `fit_transform`，对测试集必须用 `transform`，否则维度不同。
- **OOV 处理**：测试集中出现训练集未见的词会被静默忽略，可能导致信息丢失。可通过 `min_df=1` 和足够大的 `max_features` 缓解。

#### TfidfVectorizer

**功能**：将文本转换为 TF-IDF (Term Frequency-Inverse Document Frequency) 加权矩阵。词频 (TF) × 逆文档频率 (IDF)，高权重赋予在少数文档中出现但对当前文档重要的词。

**什么时候用**：
- 文本分类的标准首选方法（配合 SVM、逻辑回归、XGBoost 等）
- 需要抑制常见词（如"的""是""了"）权重的场景
- 信息检索、文档相似度计算、搜索引擎排序

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_features` | int/None | None | 同 CountVectorizer |
| `ngram_range` | tuple | (1, 1) | 同 CountVectorizer |
| `sublinear_tf` | bool | False | True 时使用 1+log(tf)，抑制高频词的过度影响 |
| `min_df` | int/float | 1 | 同 CountVectorizer |
| `max_df` | int/float | 1.0 | 同 CountVectorizer |
| `norm` | str/None | 'l2' | 归一化方式：'l2' / 'l1' / None |
| `smooth_idf` | bool | True | 加平滑（分母+1），防止除零 |
| `use_idf` | bool | True | 是否启用 IDF 加权 |
| `stop_words` | str/list/None | None | 同 CountVectorizer |
| `token_pattern` | str | r'(?u)\b\w\w+\b' | 同 CountVectorizer |

**代码示例**：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "苹果手机很好用",
    "苹果电脑性能强大",
    "安卓手机的性价比很高",
    "手机是日常必备工具"
]

# 标准用法
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(corpus)

print("特征词:", tfidf.get_feature_names_out())
print("TF-IDF 矩阵:\n", X_tfidf.toarray())

# sublinear_tf 的妙用
tfidf_linear = TfidfVectorizer(sublinear_tf=False)
tfidf_sublinear = TfidfVectorizer(sublinear_tf=True)
X_linear = tfidf_linear.fit_transform(corpus)
X_sublinear = tfidf_sublinear.fit_transform(corpus)
# sublinear_tf=True 时，词频 100 和词频 10 的差距被压缩，
# 防止某个词因重复出现而获得过高权重

# 查看 IDF 值
import numpy as np
idf_values = tfidf.idf_
for word, idf in zip(tfidf.get_feature_names_out(), idf_values):
    print(f"  {word}: IDF = {idf:.4f}")
```

**注意事项/踩坑点**：
- **`sublinear_tf=True`**：当文档长度差异很大时（如短评论文 vs 长文章），强烈推荐开启。它可以避免长文档中重复词获得异常高的权重。
- **归一化 (`norm`)**：默认 L2 归一化，使每条文本向量长度为 1。如果后续用余弦相似度比较，必须开启归一化。`norm=None` 时保留原始 TF-IDF 值。
- **`smooth_idf` 语义**：`smooth_idf=True` 时 IDF = log((N+1)/(df+1)) + 1，`smooth_idf=False` 时 IDF = log(N/df) + 1。平滑版更稳健，通常保持默认。
- **`use_idf=False`**：退化为纯 TF 矩阵，等价于 `CountVectorizer` 的输出，但多了归一化步骤。
- **与 CountVectorizer 的区别**：CountVectorizer 输出原始词频，TfidfVectorizer 输出加权词频。对于多项式朴素贝叶斯（假设特征为计数），应使用 CountVectorizer；对于 SVM/逻辑回归，应使用 TfidfVectorizer。
- **内存问题**：TF-IDF 矩阵是浮点型稀疏矩阵，比 CountVectorizer 的整型矩阵占用更多内存。大数据集建议先用 `CountVectorizer` + `TfidfTransformer` 分步处理。

#### HashingVectorizer

**功能**：使用哈希技巧将文本映射为固定维度的特征向量。不维护词汇表字典，内存占用极低。

**什么时候用**：
- 超大规模文本数据（无法将词汇表装入内存）
- 流式处理或增量学习场景
- 不需要反向映射（不关心特征对应的原始词）
- 分布式系统中需要无状态转换

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_features` | int | 2**20 | 哈希空间的维度（输出特征数） |
| `alternate_sign` | bool | True | 是否使用符号哈希（减少哈希碰撞偏差） |
| `norm` | str/None | 'l2' | 归一化方式 |
| `ngram_range` | tuple | (1, 1) | n-gram 范围 |
| `stop_words` | str/list/None | None | 停用词 |
| `lowercase` | bool | True | 是否转小写 |

**代码示例**：

```python
from sklearn.feature_extraction.text import HashingVectorizer

corpus = [
    "机器学习是人工智能的核心",
    "深度学习推动了人工智能的发展",
    "强化学习在游戏领域取得突破"
]

# 无状态向量化
hv = HashingVectorizer(n_features=128, alternate_sign=True, norm='l2')
X_hashed = hv.fit_transform(corpus)

print("输出形状:", X_hashed.shape)
# 注意：无法调用 get_feature_names_out()，因为哈希是不可逆的

# 流式增量处理
hv_stream = HashingVectorizer(n_features=1024)
batch1 = hv_stream.transform(["第一篇文档"])
batch2 = hv_stream.transform(["第二篇文档"])
batch3 = hv_stream.transform(["第三篇文档"])
# 三批结果的特征维度完全一致，可以直接拼接
import scipy.sparse as sp
all_data = sp.vstack([batch1, batch2, batch3])
print("合并后形状:", all_data.shape)
```

**注意事项/踩坑点**：
- **哈希碰撞**：不同词可能被映射到同一索引。`n_features` 越大，碰撞概率越低。推荐设为 2^18 ~ 2^22（262144 ~ 4194304）。
- **不能反向映射**：无法从特征索引反推出原始词汇。调试和可解释性极差，不适合需要特征分析的场景。
- **`alternate_sign=True`**（默认）：对哈希值进行符号修正，可将哈希碰撞导致的偏差期望降至零，强烈建议保持开启。
- **不支持 IDF 加权**：HashingVectorizer 没有内置 IDF。如需 IDF，可配合 `TfidfTransformer` 使用，但 IDF 的统计意义在大数据流式场景下会打折扣。
- **`n_features` 选择经验**：一般设为预估词汇表的 2~4 倍。假设预估有 50000 个不同词，设 n_features=131072 (2^17) 较为安全。
- **与 TfidfVectorizer 的性能对比**：在内存受限场景下，HashingVectorizer 是不可替代的选择。在常规场景下，TfidfVectorizer 因可解释性更好而更受青睐。

#### TfidfTransformer

**功能**：将词频矩阵（CountVectorizer 输出）转换为 TF-IDF 矩阵。相当于 TfidfVectorizer 中除去分词和计数之外的部分。

**什么时候用**：
- 已用 CountVectorizer 生成了词频矩阵，想复用计数结果
- 需要分别控制词频统计和 TF-IDF 变换的参数
- 流式处理：先用 CountVectorizer 累积计数，再统一做 TF-IDF

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `norm` | str/None | 'l2' | 归一化方式 |
| `use_idf` | bool | True | 是否使用 IDF |
| `smooth_idf` | bool | True | IDF 平滑 |
| `sublinear_tf` | bool | False | 是否使用对数缩放词频 |

**代码示例**：

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

# 等效于 TfidfVectorizer 的分步实现
pipeline = Pipeline([
    ('count', CountVectorizer(max_features=500)),
    ('tfidf', TfidfTransformer(sublinear_tf=True, norm='l2'))
])

corpus = ["文档一", "文档二", "文档三"]
X_tfidf = pipeline.fit_transform(corpus)

# 查看中间步骤的输出
count_vec = pipeline.named_steps['count']
X_counts = count_vec.fit_transform(corpus)
print("词频矩阵形状:", X_counts.shape)

tfidf_transformer = pipeline.named_steps['tfidf']
X_tfidf_manual = tfidf_transformer.fit_transform(X_counts)
print("TF-IDF 矩阵形状:", X_tfidf_manual.shape)
```

**注意事项/踩坑点**：
- **`fit` 在 CountVectorizer 还是在 TfidfTransformer**：CountVectorizer 的 `fit` 构建词汇表（需要看数据），TfidfTransformer 的 `fit` 计算 IDF 统计量（也需要看数据）。两者必须使用同一份训练数据。
- **Pipeline 中的注意点**：`Pipeline.fit_transform(X)` 内部先 `count.fit_transform(X)` 再 `tfidf.fit_transform(X_counts)`，自动保证数据一致性。
- **与 `TfidfVectorizer` 的等价性**：`TfidfVectorizer` 实际上是 `CountVectorizer` + `TfidfTransformer` 的组合，参数完全等价。

---

### 1.2 字典特征提取

#### DictVectorizer

**功能**：将 "特征名→值" 字典列表转换为数值矩阵（one-hot 编码或直接取值）。

**什么时候用**：
- 数据格式为 JSON/字典列表（如 MongoDB 导出、API 返回结构）
- 结构化数据中同时包含类别和数值特征
- 快速将半结构化数据转为 sklearn 兼容格式

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sparse` | bool | True | 是否返回稀疏矩阵 |
| `sort` | bool | True | 是否按字母顺序排列特征名 |
| `dtype` | dtype | np.float64 | 输出矩阵数据类型 |

**代码示例**：

```python
from sklearn.feature_extraction import DictVectorizer

data = [
    {'city': '北京', 'temperature': 25, 'humidity': 60},
    {'city': '上海', 'temperature': 28, 'humidity': 75},
    {'city': '北京', 'temperature': 22, 'humidity': 55},
    {'city': '广州', 'temperature': 30, 'humidity': 80},
]

vec = DictVectorizer(sparse=False)
X = vec.fit_transform(data)

print("特征名:", vec.get_feature_names_out())
print("特征矩阵:\n", X)
# 输出: city=上海, city=北京, city=广州, humidity, temperature
# 类别特征自动 one-hot，数值特征原样保留

# 反向转换：矩阵 → 字典
data_reconstructed = vec.inverse_transform(X)
print("还原:", data_reconstructed[0])
```

**注意事项/踩坑点**：
- **类别高基数问题**：如果某个类别特征有大量不同取值（如 "用户ID"），one-hot 后维度爆炸。此时应先用 `OrdinalEncoder` 或目标编码替代。
- **缺失值处理**：字典中不存在的 key 对应的特征值会被置为 0。务必确保所有字典的 key 集合一致。建议预先用 `collections.defaultdict` 或显式填充缺失 key。
- **`sparse` 参数**：类别特征多时务必保持 `sparse=True`（默认），否则密集矩阵内存可能溢出。
- **与 Pandas get_dummies 的区别**：DictVectorizer 可处理混合类型数据（类别+数值），get_dummies 仅处理类别数据。DictVectorizer 会记住训练时见过的所有特征名，transform 时对新出现的类别值静默忽略。
- **数值特征不会缩放**：需要后续配合 `StandardScaler` 或 `MinMaxScaler`。

---

### 1.3 图像特征提取

#### extract_patches_2d

**功能**：从二维图像中提取滑动窗口小块（patch），用于构建卷积网络的输入或图像预处理。

**什么时候用**：
- 图像分类前的数据增强
- 需要从大图中切分小块做 patch-based 分析
- 构建自定义的图像特征提取 pipeline

**代码示例**：

```python
from sklearn.feature_extraction import image
import numpy as np

img = np.arange(100).reshape(10, 10)
patches = image.extract_patches_2d(img, patch_size=(3, 3))
print("提取的小块数量:", patches.shape[0])
print("小块形状:", patches.shape)  # (64, 3, 3) — 64 个 3x3 小块

# 重建图像
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
reconstructed = reconstruct_from_patches_2d(patches, image_size=(10, 10))
```

**注意事项/踩坑点**：
- 返回顺序为 C-order（先行后列），且从左上角开始。
- `max_patches` 参数可限制提取数量，配合 `random_state` 保证可重复性。
- 在 sklearn 0.24+ 中此函数的活跃度降低，实际图像处理建议用 Pillow/OpenCV。

---

## 2. 特征选择（sklearn.feature_selection）

### 2.1 过滤法（Filter）

过滤法独立于建模过程，基于统计指标快速筛选特征。速度最快，但不考虑特征交互。

#### SelectKBest

**功能**：根据评分函数计算每个特征与标签的相关性得分，保留得分最高的 K 个特征。

**什么时候用**：
- 快速粗筛，特征数成千上万的初始降维
- 基准方案：先用它筛出前 N 个特征，再尝试更精细的方法
- 特征与标签关系近似线性时的快速筛选

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `score_func` | callable | 必填 | 评分函数，输入 (X, y)，输出 (scores, pvalues) |
| `k` | int/'all' | 10 | 保留的特征数 |

**`score_func` 选择指南**：

| 评分函数 | 适用场景 | 标签类型 | 说明 |
|----------|----------|----------|------|
| `f_classif` | 分类 | 离散 | ANOVA F 值，假设特征与标签线性相关 |
| `mutual_info_classif` | 分类 | 离散 | 互信息，能捕捉非线性关系 |
| `chi2` | 分类 | 离散 | 卡方检验，要求特征值为非负 |
| `f_regression` | 回归 | 连续 | 单变量线性回归的 F 值 |
| `mutual_info_regression` | 回归 | 连续 | 互信息回归版，捕捉非线性关系 |

**代码示例**：

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# ANOVA F 值选择
selector_f = SelectKBest(score_func=f_classif, k=2)
X_f = selector_f.fit_transform(X, y)
print("F 值:", selector_f.scores_)
print("P 值:", selector_f.pvalues_)

# 互信息选择（可捕捉非线性关系）
selector_mi = SelectKBest(score_func=mutual_info_classif, k=2)
X_mi = selector_mi.fit_transform(X, y)
print("互信息得分:", selector_mi.scores_)

# 卡方检验（必须非负数据 → 先 MinMaxScaler）
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(X)
selector_chi = SelectKBest(score_func=chi2, k=2)
X_chi = selector_chi.fit_transform(X_scaled, y)
print("卡方值:", selector_chi.scores_)

# 获取被选中特征的索引和名称
feature_indices = selector_f.get_support(indices=True)
print("选中特征索引:", feature_indices)
```

**注意事项/踩坑点**：
- **`chi2` 要求输入非负**：卡方检验假设特征是频率/计数。如果特征有负值，`chi2` 会报错或结果无意义。务必先用 `MinMaxScaler` 将数据缩放到 [0, 1]。
- **`f_classif` 假设线性关系**：如果特征与标签是非线性关系（如 X^2 与 y 相关），F 值可能很低，导致重要特征被误删。此时应改用 `mutual_info_classif`。
- **`mutual_info_*` 的随机性**：互信息估计涉及 KNN 近邻搜索，结果可能有随机性。设置 `random_state` 保证可重复。
- **`k` 的选择**：没有金标准。建议先用 `SelectKBest(k='all')` 查看所有特征的得分分布，再参考肘部法或业务理解设定 K。
- **忽略了特征冗余**：两个高得分但高度相关的特征会被同时保留，导致多重共线性。可通过相关矩阵或 VIF 进一步处理。

#### SelectPercentile

**功能**：按得分从高到低排序，保留前 `percentile` 百分比的特征。

**什么时候用**：
- 不知道具体保留多少个特征（比 SelectKBest 更灵活）
- 特征总数差异大的不同数据集上使用统一百分比策略
- 配合 `score_func=f_classif` 做快速粗筛

**代码示例**：

```python
from sklearn.feature_selection import SelectPercentile, f_classif

selector = SelectPercentile(score_func=f_classif, percentile=50)  # 保留前 50%
X_selected = selector.fit_transform(X, y)
print(f"原始特征数: {X.shape[1]}, 保留特征数: {X_selected.shape[1]}")
```

#### VarianceThreshold

**功能**：移除方差低于阈值的特征。方差为 0 → 常量特征（所有样本该特征值相同），方差极低 → 近常量特征。

**什么时候用**：
- 特征工程流水线的第一步（最轻量级）
- 移除 ID 列、常量列、几乎不变的列
- 防止后续建模时出现除零错误或数值问题

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 0.0 | 方差阈值，低于此值的特征被移除 |

**代码示例**：

```python
from sklearn.feature_selection import VarianceThreshold

X = [[1, 0, 2, 3],
     [1, 0, 2, 4],
     [1, 0, 2, 5],
     [1, 0, 2, 6]]
# 第 0 列全为 1（方差=0），第 1 列全为 0（方差=0），第 2 列全为 2（方差=0）

selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)
print("保留的特征方差:", selector.variances_)
print("选中的特征:", X_selected)
# 只保留第 3 列（有变化）
```

**注意事项/踩坑点**：
- **方差与尺度相关**：方差受特征量纲影响。数值大的特征天然方差大，数值小的特征方差小。在做 `VarianceThreshold` 之前不应标准化（会破坏方差信息），但做完之后务必标准化。
- **Bernoulli 分布的方差阈值**：对于二值特征（0/1），方差 = p(1-p)。若设 threshold=0.16，则 p<0.2 或 p>0.8 的特征被移除。
- **稀疏矩阵**：VarianceThreshold 支持稀疏矩阵输入，但方差计算方式不同，结果可能略有差异。
- **实战建议**：threshold 一般不设太大，0.0 ~ 0.01 即可，主要用于移除纯常量和近常量特征。

#### GenericUnivariateSelect

**功能**：提供模式参数 `mode` 的通用单变量特征选择器。

**什么时候用**：需要灵活切换选择策略（取前 K 个 / 百分比 / FPR / FDR / FWE）的场景。

**代码示例**：

```python
from sklearn.feature_selection import GenericUnivariateSelect, f_classif

selector = GenericUnivariateSelect(
    score_func=f_classif,
    mode='fpr',       # 基于假阳性率
    param=0.01        # 显著性水平 0.01
)
X_selected = selector.fit_transform(X, y)
```

---

### 2.2 包装法（Wrapper）

包装法将模型训练嵌入特征选择过程，考虑特征交互，但计算开销大。

#### RFE（Recursive Feature Elimination）

**功能**：反复训练模型，每次移除最不重要的特征（系数最小或重要性最低），直到达到目标特征数。

**什么时候用**：
- 需要精确的特征重要性排名
- 特征数不太多（< 1000）且需要精确选择
- 配合线性模型或树模型使用

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `estimator` | estimator | 必填 | 带 `coef_` 或 `feature_importances_` 的基础模型 |
| `n_features_to_select` | int/float/None | None | 保留特征数（或比例） |
| `step` | int/float | 1 | 每次迭代移除的特征数（或比例） |
| `verbose` | int | 0 | 日志级别 |

**代码示例**：

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=50, n_informative=10,
                           n_redundant=10, random_state=42)

estimator = LogisticRegression(max_iter=500)
selector = RFE(estimator=estimator, n_features_to_select=15, step=3)
selector.fit(X, y)

print("选中特征:", selector.get_support(indices=True))
print("特征排名 (1 = 最优):", selector.ranking_)

# RFECV：自动选择最优特征数
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

rfecv = RFECV(
    estimator=LogisticRegression(max_iter=500),
    step=2,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    min_features_to_select=5
)
rfecv.fit(X, y)

print("最优特征数:", rfecv.n_features_)
print("交叉验证得分:", rfecv.cv_results_['mean_test_score'])

import matplotlib.pyplot as plt
plt.plot(range(5, 51), rfecv.cv_results_['mean_test_score'][5-1:])
plt.xlabel("特征数")
plt.ylabel("交叉验证准确率")
plt.title("RFECV 最优特征数搜索")
plt.show()
```

**注意事项/踩坑点**：
- **计算开销大**：每轮迭代训练一次模型，特征数多时极其耗时。`step` 设大一些（如 5~10）可以加速，但精度略降。
- **`estimator` 必须有 `coef_` 或 `feature_importances_`**：SVM 的线性核有 `coef_`，RBF 核没有。树模型有 `feature_importances_`。KNN 没有特征重要性，不能用 RFE。
- **特征的 `coef_` 可正可负**：RFE 按绝对值排序，系数为 -3 的特征比 +1 的特征排名更靠前。
- **共线性的影响**：两个高度相关的特征，每次迭代可能只移除其中一个，导致另一个排名虚高。RFE 无法自动处理共线性。
- **`n_features_to_select=0.5`**：传入 float 时表示比例（如 0.5 保留 50%），而不是特征数的值。
- **RFECV 的 CV 策略**：分类任务用 `StratifiedKFold`，回归任务用 `KFold`，避免数据划分引入偏差。

---

### 2.3 嵌入法（Embedded）

嵌入法在模型训练过程中自动完成特征选择，兼具效率和交互考虑。

#### SelectFromModel

**功能**：训练一个带有 `coef_` 或 `feature_importances_` 的基础模型，保留重要性超过阈值的特征。

**什么时候用**：
- 配合 L1 正则化模型（Lasso、LogisticRegression(penalty='l1')）
- 配合树模型（RandomForest、XGBoost、LightGBM）
- 需要快速、自动化地筛选特征

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `estimator` | estimator | 必填 | 基础模型 |
| `threshold` | str/float | None | 'mean' / 'median' / 数值倍数 / None |
| `max_features` | int/None | None | 最多保留的特征数 |
| `prefit` | bool | False | 是否使用已训练的模型（避免重复 fit） |
| `norm_order` | int | 1 | 系数的范数（一般在多标签中影响）|

**代码示例**：

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression

# 方式 1：配合随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector_rf = SelectFromModel(estimator=rf, threshold='mean')
selector_rf.fit(X, y)
print("随机森林选中的特征:", selector_rf.get_support(indices=True))

# 方式 2：配合 L1 逻辑回归
lr_l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, C=0.1)
selector_l1 = SelectFromModel(estimator=lr_l1, threshold=1e-5)
selector_l1.fit(X, y)
print("L1 逻辑回归选中的特征:", selector_l1.get_support(indices=True))

# 方式 3：配合 Lasso 回归
lasso = LassoCV(cv=5, random_state=42)
selector_lasso = SelectFromModel(estimator=lasso)
selector_lasso.fit(X, y)
print("Lasso 选中的特征:", selector_lasso.get_support(indices=True))

# 方式 4：prefit 模式（模型已训练好，跳过 fit 阶段）
rf.fit(X, y)
selector_prefit = SelectFromModel(estimator=rf, threshold='mean', prefit=True)
X_selected = selector_prefit.transform(X)
print("prefit 模式选中特征数:", X_selected.shape[1])

# 获取特征重要性阈值
print("特征重要性:", rf.feature_importances_)
print("阈值 (mean):", rf.feature_importances_.mean())
```

**注意事项/踩坑点**：
- **`threshold` 的取值**：
  - `None`：使用 `estimator` 内置的阈值（仅 L1 正则化模型有效，系数为 0 的特征被移除）
  - `'mean'`：使用平均重要性作为阈值
  - `'median'`：使用中位数重要性作为阈值
  - 数值（如 0.01）：保留重要性 > 0.01 的特征
  - 字符串数值（如 `'1.25*mean'`）：保留重要性 > 1.25 倍均值的特征
- **L1 正则化的 C 值**：C 越小，正则化越强，越多的特征系数被压缩到 0。建议用 `LassoCV` / `LogisticRegressionCV` 自动选择 C。
- **`max_features` 的限制**：即使阈值很低，也不会超过 max_features。如果同时指定 threshold 和 max_features，取两者的交集。
- **树模型的重要性偏差**：树模型倾向于给高基数类别特征和连续特征更高的重要性。`threshold='mean'` 可能保留过多低质量特征，建议设 `'1.5*mean'` 或 `'2*mean'`。
- **与 RFE 的比较**：SelectFromModel 一次性筛选，速度快但精度较低；RFE 迭代筛选，速度慢但精度高。特征数 < 100 用 RFE，> 1000 用 SelectFromModel。

---

### 2.4 序列特征选择

#### SequentialFeatureSelector

**功能**：贪心地逐步添加（前向）或移除（后向）特征，每次基于交叉验证得分选择最佳特征。

**什么时候用**：
- 特征数不多（< 50），预算充足
- 需要精确的特征子集
- 无法使用基于系数的特征选择（如使用 KNN 作为评估器）

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `estimator` | estimator | 必填 | 评估模型 |
| `n_features_to_select` | int/float/'auto' | 'auto' | 目标特征数 |
| `direction` | str | 'forward' | 'forward'（前向） / 'backward'（后向） |
| `scoring` | str/callable | None | 评估指标 |
| `cv` | int/cv splitter | 5 | 交叉验证折数 |
| `tol` | float | None | 得分提升低于此值则提前停止 |

**代码示例**：

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
sfs = SequentialFeatureSelector(
    knn,
    n_features_to_select=5,
    direction='forward',  # 前向选择
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
sfs.fit(X, y)
print("选中特征:", sfs.get_support(indices=True))

# 后向选择
sfs_backward = SequentialFeatureSelector(
    knn,
    n_features_to_select=5,
    direction='backward',
    scoring='accuracy',
    cv=5
)
sfs_backward.fit(X, y)
print("后向选择结果:", sfs_backward.get_support(indices=True))
```

**注意事项/踩坑点**：
- **计算复杂度**：前向选择复杂度 O(K × N × C)，其中 K 为目标特征数，N 为总特征数，C 为交叉验证成本。特征数 > 100 时极其缓慢。
- **`direction='backward'` 通常优于 `'forward'`**：后向选择可以评估特征组合的完整效果，而前向选择首轮选入的特征可能后续变得冗余。但后向选择的初始模型需用全部 N 个特征，N 很大时不可行。
- **贪心算法，不保证全局最优**：SequentialFeatureSelector 找到的是局部最优，不是全局最优子集。
- **与其他方法的对比**：如果要选的特征配合树模型/线性模型，优先用 SelectFromModel 或 RFE。只有在评估器是 KNN、SVM(RBF) 等无法输出特征重要性的模型时，才考虑 SequentialFeatureSelector。

---

### 2.5 特征选择方法对比总表

| 方法 | 速度 | 特征交互 | 适用评估器 | 需指定 K | 适合数据规模 |
|------|------|----------|-----------|----------|-------------|
| VarianceThreshold | 极快 | 不考虑 | 无 | 否 | 任意 |
| SelectKBest | 快 | 不考虑 | 无 | 是 | 大规模 |
| SelectPercentile | 快 | 不考虑 | 无 | 否 | 大规模 |
| SelectFromModel(L1) | 快 | 考虑 | 线性模型 | 否 | 中大规模 |
| SelectFromModel(Tree) | 中 | 考虑 | 树模型 | 否 | 中大规模 |
| RFE | 慢 | 考虑 | coef_/feature_importances_ | 是 | 中小规模 |
| RFECV | 很慢 | 考虑 | coef_/feature_importances_ | 否 | 小规模 |
| SequentialFeatureSelector | 极慢 | 考虑 | 任意 | 是 | 小规模 |

---

## 3. 降维（sklearn.decomposition）

### 3.1 PCA（主成分分析）

**功能**：线性降维。通过正交变换将原始特征投影到方差最大的方向（主成分），实现降维和去相关。适用于密集矩阵。

**什么时候用**：
- 高维数据可视化（降到 2D/3D），如探索性数据分析
- 去除特征间的多重共线性
- 数据压缩和去噪（小成分通常代表噪声）
- 加速下游模型训练（特征数从几百降到几十）

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_components` | int/float/str/None | None | int 为成分数，float(0~1) 为保留方差比例，'mle' 自动选择 |
| `svd_solver` | str | 'auto' | SVD 求解器：'auto' / 'full' / 'arpack' / 'randomized' |
| `whiten` | bool | False | 是否白化（各成分方差均为 1） |
| `random_state` | int/None | None | `svd_solver='randomized'` 或 `'arpack'` 时需要 |
| `tol` | float | 0.0 | `svd_solver='arpack'` 的收敛容差 |

**代码示例**：

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)

# ★ 关键：PCA 前必须标准化
X_scaled = StandardScaler().fit_transform(X)

# 方式A：指定成分数
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("各成分方差比例:", pca.explained_variance_ratio_)
print("累计方差比例:", np.cumsum(pca.explained_variance_ratio_))

# 可视化
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('PCA 降维可视化')
plt.colorbar()
plt.show()

# 方式B：n_components=0.95 → 保留 95% 方差
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)
print(f"保留 95% 方差 → 成分数: {pca_95.n_components_}")

# 方式C：n_components='mle' → Minka's MLE 自动选择
pca_mle = PCA(n_components='mle')
X_pca_mle = pca_mle.fit_transform(X_scaled)
print(f"MLE 自动选择的成分数: {pca_mle.n_components_}")

# 方差解释曲线（确定最优成分数）
pca_full = PCA().fit(X_scaled)
evr = pca_full.explained_variance_ratio_
cumsum = np.cumsum(evr)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(evr)+1), evr)
plt.xlabel('主成分'); plt.ylabel('方差比例')
plt.title('Scree Plot')
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum)+1), cumsum, 'o-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
plt.xlabel('主成分数'); plt.ylabel('累计方差比例')
plt.title('累计方差曲线')
plt.legend()
plt.show()

# 白化（各成分独立且方差=1）
pca_whiten = PCA(n_components=10, whiten=True)
X_whiten = pca_whiten.fit_transform(X_scaled)
print(f"白化后各成分标准差: {X_whiten.std(axis=0)}")
```

**注意事项/踩坑点**：
- **必须先标准化**：PCA 对方差敏感。原始特征中量纲大的特征会主导主成分方向，导致 PCA 结果被量纲绑架。务必先用 `StandardScaler`。
- **`n_components=0.95`**：保留 95% 方差的自动选择，比手动指定更稳健。配合 `svd_solver='auto'` 自动选择最优算法。
- **`svd_solver` 选择**：
  - `'auto'`：自动选择（推荐）
  - `'full'`：精确 SVD，小数据用
  - `'arpack'`：适用于只需要少量成分（<< 样本数）的场景
  - `'randomized'`：大数据 + 少量成分，速度最快但有随机性
- **PCA 不能用于稀疏矩阵**：对稀疏文本数据（如 TF-IDF 矩阵），使用 `TruncatedSVD` 代替。
- **`whiten=True` 的影响**：白化后各成分独立且方差为 1，对某些下游模型（如 SVM）可能有帮助，但会放大噪声成分的影响。
- **解释性差**：PCA 成分是原始特征的线性组合，难以解释业务含义。如果需要可解释性，用特征选择而非降维。
- **PCA 假设线性**：如果数据有强烈的非线性结构，PCA 效果差，应考虑 KernelPCA 或 t-SNE。

---

### 3.2 TruncatedSVD

**功能**：截断奇异值分解，适用于稀疏矩阵。相当于不对数据进行中心化的 PCA。

**什么时候用**：
- 文本数据的降维（TF-IDF 矩阵是稀疏的，PCA 无法处理）
- 潜在语义分析（LSA / LSI）
- 大规模稀疏矩阵的压缩和去噪
- 推荐系统中的矩阵分解

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_components` | int | 2 | 目标成分数 |
| `algorithm` | str | 'randomized' | 'randomized' / 'arpack' |
| `n_iter` | int | 5 | randomized 算法的迭代次数 |
| `random_state` | int/None | None | 随机种子 |
| `tol` | float | 0.0 | arpack 算法的收敛容差 |

**代码示例**：

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "机器学习是人工智能的核心分支",
    "深度学习使用神经网络进行训练",
    "强化学习通过奖励信号学习策略",
    "自然语言处理让计算机理解文本",
    "计算机视觉让机器看懂图像",
    "自动驾驶结合了计算机视觉和强化学习"
]

# 文本 → TF-IDF → TruncatedSVD
tfidf = TfidfVectorizer(max_features=100)
X_tfidf = tfidf.fit_transform(corpus)

svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

print("各成分方差比例:", svd.explained_variance_ratio_)
print("累计方差:", svd.explained_variance_ratio_.sum())

# 查看每个成分对应的 top 词（语义分析）
terms = tfidf.get_feature_names_out()
for i, comp in enumerate(svd.components_):
    top_idx = comp.argsort()[-5:][::-1]  # 绝对值最大的 5 个词
    top_terms = [f"{terms[j]}({comp[j]:.3f})" for j in top_idx]
    print(f"成分 {i+1} Top 词: {', '.join(top_terms)}")
```

**注意事项/踩坑点**：
- **不进行中心化**：PCA 会先将数据减去均值（中心化），但稀疏矩阵减去均值后会变成稠密矩阵，丧失稀疏优势。TruncatedSVD 跳过中心化，因此保留了稀疏性。
- **方差解释比例**：由于没有中心化，`explained_variance_ratio_` 的和不是 100%（通常 > 1），这是正常现象。
- **与 PCA 的区别**：对稠密矩阵做 PCA（已中心化）和 TruncatedSVD（未中心化）结果不同。当数据均值接近 0 向量时，两者近似等价。
- **`algorithm='randomized'`** 适用于大数据集，速度快；`'arpack'` 适用于需要精确解的场景。
- **文本数据管线**：`CountVectorizer/TfidfVectorizer → TruncatedSVD` 是经典的 LSA pipeline。如果需要主题模型，NMF 通常效果更好且结果可解释。

---

### 3.3 KernelPCA

**功能**：通过核技巧实现非线性 PCA。将数据映射到高维核空间后做 PCA。

**什么时候用**：
- 数据存在明显的非线性结构，线性 PCA 无法捕捉
- 需要非线性降维但保留可逆性（与 t-SNE 不同）
- 作为分类前的特征工程（用 KPCA 提取非线性特征后送入线性分类器）

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_components` | int/None | None | 目标成分数 |
| `kernel` | str | 'linear' | 'linear' / 'poly' / 'rbf' / 'sigmoid' / 'cosine' / 'precomputed' |
| `gamma` | float/None | None | RBF/Poly/Sigmoid 核的系数（None → 1/n_features） |
| `degree` | int | 3 | Poly 核的次数 |
| `coef0` | float | 1 | Poly/Sigmoid 核的常数项 |
| `eigen_solver` | str | 'auto' | 'auto' / 'dense' / 'arpack' / 'randomized' |
| `random_state` | int/None | None | randomized 算法的随机种子 |

**代码示例**：

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

# 生成非线性数据
X, y = make_circles(n_samples=200, factor=0.3, noise=0.05, random_state=42)

# 线性 PCA vs 核 PCA
pca = PCA(n_components=2)
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=5, random_state=42)

X_pca = pca.fit_transform(X)
X_kpca = kpca_rbf.fit_transform(X)

# kpca_rbf 可以将同心圆在核空间中变成线性可分的

# 对比可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
axes[0].set_title('原始数据')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
axes[1].set_title('PCA')
axes[2].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis')
axes[2].set_title('KernelPCA (RBF)')
plt.show()
```

**注意事项/踩坑点**：
- **核函数选择**：`kernel='rbf'` + `gamma` 调参是最常用的组合。`gamma` 控制核的宽度——gamma 越大，核函数的局部影响范围越小，越容易过拟合。
- **`gamma` 默认值陷阱**：默认 `gamma=1/n_features`，对于特征数多的数据可能过小，导致核 PCA 退化为线性 PCA。建议尝试 gamma ∈ [0.001, 0.01, 0.1, 1, 10]。
- **`eigen_solver='randomized'`**：大数据时首选的加速方案，但结果有随机性。
- **计算复杂度**：KernelPCA 需要计算 n×n 的核矩阵，内存和计算复杂度均为 O(n²)，样本数 > 10000 时慎用。
- **不能做 inverse_transform（部分核除外）**：RBF 核无法逆向映射。如果需要还原，考虑用 AutoEncoder。

---

### 3.4 NMF（非负矩阵分解）

**功能**：将非负矩阵 X 分解为两个非负矩阵 W 和 H 的乘积 (X ≈ WH)。所有元素 ≥ 0，结果天然具有"部分构成整体"的解释性。

**什么时候用**：
- 主题模型（文本 → 文档-主题矩阵 + 主题-词矩阵）
- 图像特征分解（面部特征、部件拆分）
- 推荐系统（用户-物品交互矩阵分解）
- 需要可解释性分解的场景

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_components` | int | 必填 | 分解后的主题数/成分数 |
| `init` | str | None | 'random' / 'nndsvd' / 'nndsvda' / 'nndsvdar' / 'custom' |
| `solver` | str | 'cd' | 'cd'（坐标下降）/ 'mu'（乘法更新） |
| `max_iter` | int | 200 | 最大迭代次数 |
| `alpha` | float | 0.0 | 正则化强度（0 为无正则化） |
| `l1_ratio` | float | 0.0 | L1 正则化比例（0=纯 L2, 1=纯 L1） |
| `random_state` | int/None | None | 随机种子 |

**代码示例**：

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=200)
X_tfidf = tfidf.fit_transform(corpus)
feature_names = tfidf.get_feature_names_out()

nmf = NMF(n_components=3, init='nndsvda', random_state=42, max_iter=500)
W = nmf.fit_transform(X_tfidf)   # 文档-主题矩阵
H = nmf.components_               # 主题-词矩阵

# 打印每个主题的 top 词
for topic_idx, topic in enumerate(H):
    top_idx = topic.argsort()[-8:][::-1]
    top_words = [f"{feature_names[i]}({topic[i]:.3f})" for i in top_idx]
    print(f"主题 {topic_idx+1}: {', '.join(top_words)}")

# 文档的主题分布
for doc_idx, doc_topics in enumerate(W):
    dominant_topic = doc_topics.argmax()
    print(f"文档 {doc_idx}: 主题{doc_topics} → 主导主题 {dominant_topic}")
```

**注意事项/踩坑点**：
- **输入必须非负**：NMF 严格假设 X ≥ 0。TF-IDF 天然满足，但标准化后的数据可能有负值。如果用原始数据，务必确认非负。
- **`init='nndsvda'`**：推荐使用，比 `'random'` 收敛更快更稳定。`'nndsvd'` 系列初始化基于 SVD 的确定性分解。
- **`n_components` 的选择**：NMF 没有 PCA 的"方差解释比例"供参考。需要通过业务含义（主题是否连贯、是否可解释）或重建误差来判断。
- **与 LDA（主题模型）的区别**：NMF 是矩阵分解，LDA 是概率生成模型。NMF 速度快、实现简单、结果确定性好（用 nndsvd 初始化时）。LDA 统计学基础更扎实，能给出主题分布的概率。
- **正则化**：`alpha` 和 `l1_ratio` 可以生成更稀疏、更易解释的主题矩阵，但设置不当会降低重建质量。
- **收敛问题**：如果 `reconstruction_err_` 不下降，尝试增大 `max_iter`、更换 `solver`、或用 `nndsvd` 初始化。

---

### 3.5 LDA（LinearDiscriminantAnalysis）

**功能**：线性判别分析，是**有监督**的降维方法。寻找投影方向，使得同类样本投影后尽可能近、不同类样本尽可能远。

**什么时候用**：
- 分类任务的降维（保留类别信息）
- 作为分类器的前置步骤
- 需要同时降维和分类

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `solver` | str | 'svd' | 'svd' / 'lsqr' / 'eigen' |
| `n_components` | int/None | None | 降维后的维度（最多为类别数-1） |
| `shrinkage` | str/float/None | None | 协方差矩阵的收缩估计（'auto' 使用 Ledoit-Wolf） |
| `priors` | array/None | None | 类先验概率 |
| `store_covariance` | bool | False | 是否存储协方差矩阵 |

**代码示例**：

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# LDA 作为降维
lda_dim = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda_dim.fit_transform(X_train, y_train)
print(f"降维后形状: {X_lda.shape}")
print(f"各类别中心:\n{lda_dim.means_}")

# LDA 作为分类器
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)
y_pred = lda_clf.predict(X_test)
print(f"分类准确率: {(y_pred == y_test).mean():.2%}")

# 对比：LDA（有监督）vs PCA（无监督）的可视化效果
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, X_trans, title in zip(axes, [X_pca, X_lda], ['PCA (无监督)', 'LDA (有监督)']):
    for cls in range(3):
        ax.scatter(X_trans[y_train == cls, 0], X_trans[y_train == cls, 1], label=f'类{cls}')
    ax.set_title(title); ax.legend()
plt.show()
```

**注意事项/踩坑点**：
- **所在模块**：`sklearn.discriminant_analysis`，不是 `sklearn.decomposition`。
- **`n_components` 上限**：最多为 `min(n_features, n_classes - 1)`。如三分类问题最多降到 2 维，二分类问题最多 1 维。
- **特征数 > 样本数**：在高维小样本情况下，LDA 的协方差矩阵不可逆。解决方法：用 `solver='lsqr'` 或 `solver='eigen'` 配合 `shrinkage='auto'`。
- **假设**：LDA 假设各类别数据服从多元正态分布且协方差矩阵相同。违反了此假设效果会下降，但实际应用中仍常有效。
- **标准化**：LDA 在做 SVD 求解时不太需要标准化（内部处理），但如果特征量纲差异很大，建议先 `StandardScaler` 以避免数值问题。

---

### 3.6 降维方法对比总表

| 方法 | 类型 | 线性/非线性 | 有监督 | 稀疏矩阵支持 | 可解释性 | 性能 |
|------|------|-----------|--------|-------------|---------|------|
| PCA | 降维 | 线性 | 无 | 否 | 差 | 快 |
| TruncatedSVD | 降维 | 线性 | 无 | 是 | 差 | 快 |
| KernelPCA | 降维 | 非线性 | 无 | 否 | 差 | 中 |
| NMF | 分解 | 线性 | 无 | 是 | 好 | 中 |
| LDA | 降维 | 线性 | 有 | 否 | 中 | 快 |
| t-SNE | 流形 | 非线性 | 无 | 否 | 差 | 慢 |

---

## 4. 流形学习（sklearn.manifold）

### 4.1 TSNE

**功能**：t-SNE（t-distributed Stochastic Neighbor Embedding）将高维数据的局部相似性映射到低维空间，保持近邻关系。主要用于高维数据可视化。

**什么时候用**：
- 高维数据（如图像、文本向量）的 2D/3D 可视化
- 聚类效果的直观检查
- 数据分布探索

**关键参数表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_components` | int | 2 | 输出维度（通常 2 或 3） |
| `perplexity` | float | 30 | 困惑度，控制有效近邻数（建议 5~50） |
| `learning_rate` | float/str | 'auto' | 学习率（'auto' 设为 max(N/12, 200)） |
| `n_iter` | int | 1000 | 优化迭代次数 |
| `init` | str | 'pca' | 初始化方式：'pca' / 'random' |
| `random_state` | int | None | 随机种子 |
| `method` | str | 'barnes_hut' | 'barnes_hut'（快速近似）/ 'exact' |

**代码示例**：

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# t-SNE 降维到 2D
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    random_state=42,
    init='pca'           # PCA 初始化更稳定
)
X_tsne = tsne.fit_transform(X)

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, ticks=range(10))
plt.title('t-SNE 可视化 MNIST (Digits)')
plt.xlabel('t-SNE 维度 1'); plt.ylabel('t-SNE 维度 2')
plt.show()

# perplexity 调参对比
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, perp in zip(axes.ravel(), [5, 15, 30, 50, 80, 100]):
    X_tsne_perp = TSNE(n_components=2, perplexity=perp,
                       n_iter=1000, random_state=42).fit_transform(X)
    ax.scatter(X_tsne_perp[:, 0], X_tsne_perp[:, 1], c=y, cmap='tab10', s=5)
    ax.set_title(f'perplexity={perp}')
plt.tight_layout()
plt.show()
```

**注意事项/踩坑点**：
- **只能用于可视化，不能用于预测**：t-SNE 无法 `transform` 新数据。不能作为 feature engineering 的步骤加入 Pipeline。如果需要可用于新数据的非线性降维，用 KernelPCA 或 AutoEncoder。
- **`perplexity` 的选择**：
  - 太小（< 5）：只关注极局部结构，远距离样本随机排列
  - 太大（> 100）：试图保留全局结构，但可能失真
  - 经验值：样本数的 1%~20% 之间，通常 5~50 内尝试
- **`n_iter` 不足**：如果 `n_iter` 太小（< 250），t-SNE 未收敛，结果不稳定。至少设 1000，复杂数据可设 5000。
- **每次运行结果不同**：t-SNE 是非凸优化，不同随机种子结果可能差异很大。设置 `random_state` 保证可重复。如果多次运行结果不稳定，增大 `n_iter`。
- **簇的大小和距离无意义**：t-SNE 只保留局部近邻关系，簇之间的距离和簇的大小不代表原始空间中的关系。
- **`learning_rate` 的影响**：过高 → 样本点分散成均匀网格；过低 → 样本点聚集成一个密集球。`'auto'` 通常没问题。
- **预处理**：建议先 PCA 降维到 50 维再输入 t-SNE，可以减少噪声和加速计算。

---

### 4.2 其他流形方法简述

#### Isomap

**功能**：等距映射（Isometric Mapping），通过测地距离（最短路径）保持全局几何结构。

**参数要点**：`n_neighbors`（邻接图的近邻数，关键参数，太小导致图断开，太大导致近路效应），`n_components`。

**适用场景**：数据位于低维流形上（如 Swiss Roll），需要保持全局几何结构。比 t-SNE 计算量大但保留更多全局信息。

```python
from sklearn.manifold import Isomap
isomap = Isomap(n_neighbors=10, n_components=2)
X_iso = isomap.fit_transform(X)
```

#### MDS（Multidimensional Scaling）

**功能**：多维缩放，保持样本间的距离（相异度）关系。包括 metric MDS（保持欧氏距离）和 nonmetric MDS（仅保持排序）。

**适用场景**：已知样本间的相异度矩阵（如心理量表、排序数据），希望可视化。

```python
from sklearn.manifold import MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)
```

#### LocallyLinearEmbedding（LLE）

**功能**：局部线性嵌入，假设每个样本可以通过其近邻的线性组合重构，在降维后保持这种重构关系。

**参数要点**：`n_neighbors`（近邻数），`method`（'standard' / 'hessian' / 'modified' / 'ltsa'）。

**优势**：比 Isomap 快，适合展开卷曲流形。对缺失近邻不敏感。

```python
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_lle = lle.fit_transform(X)
```

---

## 5. 实战选择指南

拿到数据后，特征工程 API 选择流程图（伪代码 + 决策树）：

```
START
├─ 1. VarianceThreshold → 移除常量列
├─ 2. 数据类型?
│   │
│   ├─ 文本数据
│   │   ├─ 中文? → jieba分词 → ' '.join()
│   │   ├─ 大数据? → HashingVectorizer(n_features)
│   │   ├─ 常规?  → TfidfVectorizer(max_features, ngram_range, sublinear_tf=True)
│   │   └─ 需降维? → TruncatedSVD(n_components) 或 NMF(n_components)
│   │
│   ├─ 字典/JSON数据
│   │   ├─ 混合类型 → DictVectorizer(sparse=True)
│   │   └─ 大量文本 → 提取文本列 → 文本处理流程
│   │
│   └─ 数值数据
│       │
│       ├─ 特征数 < 100
│       │   ├─ 模型有 coef_/feature_importances_?
│       │   │   ├─ 是 → RFECV(cv=5)  ← 自动选择最优特征数
│       │   │   └─ 否 → SequentialFeatureSelector(cv=5)
│       │   └─ 可视化需要 → PCA/LDA → plt.scatter
│       │
│       ├─ 特征数 100~1000
│       │   ├─ 线性模型 → SelectFromModel(Lasso/L1-LR)
│       │   ├─ 树模型 → SelectFromModel(RF/XGB, threshold='mean')
│       │   └─ 快速粗筛 → SelectKBest(f_classif/mutual_info, k)
│       │
│       ├─ 特征数 > 1000
│       │   ├─ 第1轮: SelectKBest(mutual_info, k=500)
│       │   ├─ 第2轮: SelectFromModel 或 RFE
│       │   └─ 最终: 保留 20~50 个特征
│       │
│       └─ 降维替代特征选择
│           ├─ 线性 → PCA(n_components=0.95)
│           ├─ 稀疏 → TruncatedSVD(n_components)
│           ├─ 非线性 → KernelPCA
│           └─ 可视化 → TSNE(perplexity=30)
│
└─ 3. 标准化 StandardScaler → 建模
```

**核心原则**：
1. 先方差筛（VarianceThreshold），后重要性筛，最后降维。
2. 能用嵌入法（SelectFromModel），不用包装法（RFE）；能用过滤法，不直接用包装法。
3. 特征数 > 1000 时，采用多轮筛选策略（粗筛 + 精筛）。
4. 文本数据默认 `TfidfVectorizer(sublinear_tf=True, max_features=5000)` 作为 baseline。
5. 降维前务必标准化。稀疏数据用 TruncatedSVD，稠密数据用 PCA。

---

## 相关概念

- [[Scikit-Learn 核心API速查]]
- [[监督学习与模型选择API]]
- [[聚类与降维API]]
- [[数据预处理API详解]]
- [[Scikit-Learn Pipeline 实战指南]]
- [[交叉验证与超参调优API]]
