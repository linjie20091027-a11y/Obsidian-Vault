# 生成对抗网络（GAN）原理

## 1. 核心思想：二元博弈

生成对抗网络（Generative Adversarial Network, GAN）由 Ian Goodfellow 于 2014 年提出，其核心思想源于博弈论中的**二人零和博弈**。整个框架由两个神经网络组成：

- **生成器（Generator, G）**：以随机噪声 $z \sim P_z$ 为输入，试图生成与真实数据分布 $P_{data}$ 无法区分的样本 $G(z)$。生成器的目标是"以假乱真"。
- **判别器（Discriminator, D）**：以样本 $x$ 为输入，输出一个标量值 $D(x) \in [0,1]$，表示 $x$ 来自真实数据分布（而非生成器）的概率。判别器的目标是"明辨真伪"。

两者构成一个**极小极大（Minimax）博弈**：
- G 试图最小化判别器识别出生成样本的概率（即最大化 D 犯错的可能性）。
- D 试图最大化正确区分真实样本与生成样本的准确率。

最终理想状态下，生成器将学到真实数据分布 $P_{data}$，判别器对所有样本都输出 $1/2$（即无法分辨），系统达到**纳什均衡（Nash Equilibrium）**。

## 2. 目标函数

GAN 的目标函数可以形式化地表达为：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim P_{data}}[\log D(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]$$

其中：
- $D(x)$ 是判别器对真实样本的输出，理想情况下应趋近于 1。
- $D(G(z))$ 是判别器对生成样本的输出，理想情况下应趋近于 0。
- $\log(1 - D(G(z)))$ 在生成器训练早期会饱和，因此实践中常改用最大化 $\mathbb{E}_{z \sim P_z}[\log D(G(z))]$（Non-Saturating GAN）。

**判别器的优化**：固定 G，最大化 $V(D,G)$，即提高对真实样本打分、降低对生成样本打分。

**生成器的优化**：固定 D，最小化 $V(D,G)$，即让生成样本尽可能被判别器判为真。

## 3. 数学推导

### 3.1 最优判别器

给定固定生成器 G，我们需要找到使 $V(D, G)$ 最大的 $D$。

将期望展开为积分形式：

$$V(D, G) = \int_x P_{data}(x) \log D(x) \, dx + \int_z P_z(z) \log(1 - D(G(z))) \, dz$$

令 $P_g$ 为生成器诱导的数据分布，上式等价于：

$$V(D, G) = \int_x \left[ P_{data}(x) \log D(x) + P_g(x) \log(1 - D(x)) \right] dx$$

对任意 $x$，最大化被积函数 $f(D) = P_{data}(x) \log D + P_g(x) \log(1 - D)$，令导数为零：

$$\frac{\partial f}{\partial D} = \frac{P_{data}(x)}{D} - \frac{P_g(x)}{1 - D} = 0$$

解得最优判别器：

$$D^*(x) = \frac{P_{data}(x)}{P_{data}(x) + P_g(x)}$$

### 3.2 等价于 JS 散度最小化

将 $D^*$ 代入 $V(D, G)$：

$$C(G) = \max_D V(D, G) = \mathbb{E}_{x \sim P_{data}}\left[\log \frac{P_{data}}{P_{data} + P_g}\right] + \mathbb{E}_{x \sim P_g}\left[\log \frac{P_g}{P_{data} + P_g}\right]$$

整理可得：

$$C(G) = -\log 4 + 2 \cdot JSD(P_{data} \| P_g)$$

其中 **Jensen-Shannon 散度（JSD）** 定义为：

$$JSD(P \| Q) = \frac{1}{2} KL\!\left(P \,\|\, \frac{P+Q}{2}\right) + \frac{1}{2} KL\!\left(Q \,\|\, \frac{P+Q}{2}\right)$$

因此，训练 GAN 本质上是在**最小化 $P_{data}$ 与 $P_g$ 之间的 JS 散度**。当 $P_g = P_{data}$ 时，JSD 达到最小值 $0$，此时 $D^*(x) = 1/2$，且 $C(G) = -\log 4$。

## 4. 训练算法流程

标准的 GAN 训练采用交替优化的方式。以下是伪代码：

```
for 训练迭代次数:
    # 步骤 1: 训练判别器 (k 步，通常 k=1)
    for k 步:
        从噪声分布 P_z 采样 m 个噪声样本 {z^(1), ..., z^(m)}
        从真实数据分布 P_data 采样 m 个样本 {x^(1), ..., x^(m)}
        利用随机梯度下降更新判别器:
            ∇_{θ_d} [ (1/m) Σ log D(x^(i)) + (1/m) Σ log(1 - D(G(z^(i)))) ]
    结束

    # 步骤 2: 训练生成器 (1 步)
    从噪声分布 P_z 采样 m 个噪声样本 {z^(1), ..., z^(m)}
    利用随机梯度下降更新生成器:
            ∇_{θ_g} [ (1/m) Σ log(1 - D(G(z^(i)))) ]
            （或 Non-Saturating 版本: ∇_{θ_g} [ -(1/m) Σ log D(G(z^(i))) ]）
结束
```

**关键点**：
- 判别器通常比生成器训练更多步（`k > 1`），以保证其保持较强的判别能力，为生成器提供有效梯度。
- 实际实现中使用 Non-Saturating Loss，在训练初期提供更强的梯度信号。

## 5. 训练不稳定的原因

### 5.1 JS 散度的固有问题

如前所述，原始 GAN 等价于最小化 $JSD(P_{data} \| P_g)$。**当两个分布的支撑集（support）不相交或交集测度为零时，JS 散度恒为常数 $\log 2$**。

在高维空间中，$P_{data}$ 通常位于低维流形上，而 $P_g$ 在训练初期与其几乎无重叠。此时：

$$JSD(P_{data} \| P_g) = \log 2 \quad (\text{常数})$$

判别器可以完美区分两种分布（准确率接近 100%），但**其梯度趋近于零**，生成器无法得到有效的更新信号，训练陷入停滞。这就是所谓的 **"梯度消失"（Vanishing Gradient）** 问题。

### 5.2 训练动态的不稳定性

- GAN 的训练是一个动态博弈过程，而非简单的优化问题。交替梯度下降**不一定收敛**到纳什均衡点，可能表现为震荡甚至发散。
- 判别器与生成器的"军备竞赛"可能导致模式坍塌或训练崩溃。

## 6. Mode Collapse（模式坍塌）

**模式坍塌（Mode Collapse）** 是 GAN 训练中最常见且最严重的失败模式之一。

### 6.1 定义与表现

生成器将**多个不同的真实数据模式映射到同一个或少量的生成模式**上。例如训练一个手写数字 GAN，生成器可能只能生成"1"和"7"两种数字，而完全忽略了其他数字类别。

形式上，生成器 $G$ 生成的分布 $P_g$ 仅覆盖真实分布 $P_{data}$ 的少数几个模态。

### 6.2 成因分析

1. **生成器"投机取巧"**：当生成器发现某种特定的输出可以有效欺骗当前判别器时，它会倾向于总是生成那种输出，缺乏探索动力。
2. **交替优化的非收敛性**：判别器-生成器的交替梯度下降可以近似为在参数空间中的旋转向量场，容易陷入极限环（Limit Cycle），使生成器在少数模式间循环。
3. **目标函数的局部最优**：JS 散度下，生成器宁可只完美覆盖一部分模式（准确度高），也不愿覆盖全部模式（每个都不够好）。

## 7. 常用训练技巧

### 7.1 标签平滑（Label Smoothing）

将真实样本的标签从 $1$ 修改为 $1 - \alpha$（如 0.9），将生成样本标签从 $0$ 修改为 $\beta$（如 0.1）。这可以：
- 减轻判别器过度自信，使梯度更加平滑。
- 降低对抗攻击的脆弱性。

### 7.2 特征匹配（Feature Matching）

不直接最大化判别器的输出，而是让生成器生成的数据在**判别器中间层特征**的统计量上与真实数据一致：

$$\min_G \left\| \mathbb{E}_{x \sim P_{data}}[f(x)] - \mathbb{E}_{z \sim P_z}[f(G(z))] \right\|_2^2$$

其中 $f(\cdot)$ 是判别器某一中间层的激活值。这种方法鼓励生成器匹配真实数据的"内在表示"，而非仅追求判别器输出的标量分数。

### 7.3 历史平均（Historical Averaging）

在损失函数中加入参数与历史平均值的二范数惩罚项：

$$L_{ha} = \left\| \theta - \frac{1}{t} \sum_{i=1}^{t} \theta_i \right\|^2$$

这有助于抑制训练中的震荡和模式跳跃。

### 7.4 Minibatch Discrimination

允许判别器观察**整个小批量的统计信息**而非孤立地判别每个样本。具体做法是计算每个样本的 minibatch 特征（如与同批次其他样本的 $L_1$ 距离），将这些特征拼接后输入后续层。

这迫使生成器在生成一个小批量时产生多样化样本——如果批量内所有样本过于相似，判别器将轻易识别。

### 7.5 其他技巧

- **One-sided Label Smoothing**：仅平滑真实标签（如 0.9），不平滑生成标签（保持 0），避免生成器过度扩张。
- **批量归一化**（BatchNorm）：稳定训练，减少内部协变量偏移。
- **使用 Adam 优化器**：比 SGD 在 GAN 训练中表现更稳定。

## 8. WGAN：Wasserstein GAN

### 8.1 动机

如前所述，JS 散度在分布支撑不重叠时恒为常数，导致梯度消失。WGAN 引入 **Wasserstein 距离（Earth-Mover 距离）** 替代 JS 散度。

**Earth-Mover 距离**衡量将一个分布"搬运"成另一个分布所需的最小代价：

$$W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

其中 $\Pi(P_r, P_g)$ 是所有边际分布为 $P_r$ 和 $P_g$ 的联合分布的集合。

### 8.2 Kantorovich-Rubinstein 对偶

利用 KR 对偶性，Wasserstein 距离可表达为：

$$W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)]$$

其中 $\|f\|_L \leq 1$ 表示 $f$ 是 **1-Lipschitz 函数**（即 $|f(x) - f(y)| \leq \|x - y\|$）。

### 8.3 WGAN 训练

WGAN 将判别器重新定义为"评论家"（Critic），以 $f_w$ 逼近 $W(P_r, P_g)$：

$$\max_{w \in \mathcal{W}} \mathbb{E}_{x \sim P_r}[f_w(x)] - \mathbb{E}_{z \sim P_z}[f_w(G(z))]$$

Lipschitz 约束通过**权重裁剪（Weight Clipping）** 实现：将 $w$ 裁剪到某个紧致区间 $[-c, c]$。

**WGAN 的优势**：
- 损失值 $W(P_r, P_g)$ 与生成质量呈正相关，可有效监控训练进度。
- 在分布不重叠时仍有平滑梯度，彻底解决了梯度消失问题。
- 显著减少模式坍塌。

## 9. WGAN-GP：梯度惩罚

权重裁剪过于粗暴，可能导致容量利用不足（capacity underuse）和梯度爆炸/消失。WGAN-GP 以**梯度惩罚（Gradient Penalty）** 替代权重裁剪。

### 9.1 核心公式

1-Lipschitz 约束等价于 $\|\nabla_x f(x)\|_2 \leq 1$ 对所有 $x$ 成立。WGAN-GP 在损失中加入软约束：

$$\lambda \cdot \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}\left[ (\|\nabla_{\hat{x}} f(\hat{x})\|_2 - 1)^2 \right]$$

其中 $\hat{x}$ 是从真实样本与生成样本的连线上随机采样的插值点：

$$\hat{x} = \epsilon x_r + (1 - \epsilon) x_g, \quad \epsilon \sim U[0, 1]$$

### 9.2 完整目标函数

$$L_{WGAN-GP} = \mathbb{E}_{x \sim P_g}[f(x)] - \mathbb{E}_{x \sim P_r}[f(x)] + \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}\left[ (\|\nabla_{\hat{x}} f(\hat{x})\|_2 - 1)^2 \right]$$

其中 $\lambda$ 通常取 10。**判别器中不能使用 BatchNorm**（因为梯度惩罚是针对每个独立样本计算的），推荐使用 LayerNorm 或不使用归一化。

## 10. LSGAN：最小二乘 GAN

LSGAN 将交叉熵损失替换为**最小二乘损失（Least Squares Loss）**。

### 10.1 损失函数

$$\min_D \mathcal{L}(D) = \frac{1}{2} \mathbb{E}_{x \sim P_{data}}\left[ (D(x) - 1)^2 \right] + \frac{1}{2} \mathbb{E}_{z \sim P_z}\left[ (D(G(z)))^2 \right]$$

$$\min_G \mathcal{L}(G) = \frac{1}{2} \mathbb{E}_{z \sim P_z}\left[ (D(G(z)) - 1)^2 \right]$$

### 10.2 动机与优势

- 对处于判别边界正确一侧但距离边界较远的样本，交叉熵的梯度很小；而最小二乘损失会持续地向决策边界"拉拽"这些样本。
- LSGAN 等价于最小化 **Pearson $\chi^2$ 散度**，生成的样本质量更高，训练更稳定。
- 最小二乘损失的凸性使其优化特性更好（在固定 D 时对 G 而言）。

## 11. 总结

| 模型 | 核心改进 | 散度/距离 |
|------|---------|----------|
| Vanilla GAN | 二元博弈框架 | JS 散度 |
| WGAN | Wasserstein 距离 + 权重裁剪 | Earth-Mover 距离 |
| WGAN-GP | 梯度惩罚替代权重裁剪 | Earth-Mover 距离 (GP) |
| LSGAN | 最小二乘损失 | Pearson $\chi^2$ 散度 |

GAN 的训练稳定性从最初的梯度消失与模式坍塌，到 WGAN 的 EM 距离解决梯度消失，再到 WGAN-GP 的精细化 Lipschitz 约束实现稳定训练，经历了一条清晰的理论与实践演进路径。
