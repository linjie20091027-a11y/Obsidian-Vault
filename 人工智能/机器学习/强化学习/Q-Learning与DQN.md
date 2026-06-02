# Q-Learning 与 DQN（Deep Q-Network）

## 一、Q-Learning 基础

### 1.1 TD 学习（Temporal Difference Learning）

时间差分（TD）学习是强化学习的核心思想之一，融合了**蒙特卡洛方法**（从完整轨迹中学习）和**动态规划**（使用自举，bootstrap）的优点。

TD 学习的基本形式：

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

其中 $\alpha$ 为学习率，$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 称为**TD 误差（TD Error）**。

### 1.2 Q-Learning（Off-Policy TD 控制）

Q-Learning 是由 Watkins 于 1989 年提出的**无模型、Off-Policy** 的 TD 控制算法。

**核心更新公式：**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

**关键特征：**

- 使用 $\max_{a} Q(S_{t+1}, a)$ 而非实际选择的动作的 Q 值——即直接逼近最优 Q 函数 $Q^*$，与行为策略无关（Off-Policy）
- 当满足一定条件（如每个状态-动作对被无限次访问）时，Q-Learning 以概率 1 收敛到 $Q^*$
- 行为策略通常使用 $\varepsilon$-greedy 或其他探索策略

**Q-Learning 算法伪代码：**

```
初始化 Q(s, a) 为任意值（通常为 0）
对每个 episode:
    初始化状态 s
    对每个时间步:
        以 ε-greedy 选择 a
        执行 a，观察 r 和 s'
        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
        s ← s'
    直到 s 为终止状态
```

## 二、SARSA（On-Policy TD 控制）

SARSA 是 Q-Learning 的 On-Policy 变体，名称来自状态-动作-奖励-状态-动作的序列 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。

**核心更新公式：**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

**与 Q-Learning 的核心区别：**

| 特性 | Q-Learning | SARSA |
|------|-----------|-------|
| 类型 | Off-Policy | On-Policy |
| 备份目标 | $\max_a Q(s', a)$ | $Q(s', a')$（实际选择的动作）|
| 学习策略 | 最优策略 | 行为策略的价值 |
| 收敛性 | 更积极（乐观） | 更保守（考虑探索的代价）|

在悬崖行走（Cliff Walking）等任务中，SARSA 由于考虑了 $\varepsilon$-greedy 探索带来的风险，会选择更安全（远离悬崖）的路径，而 Q-Learning 会学习最短路径（但探索时容易掉下悬崖）。

## 三、DQN —— 从表格到深度

### 3.1 为何需要 DQN

传统 Q-Learning 使用表格存储 Q 值，当状态空间很大或连续时（如 Atari 游戏的像素输入），表格方法不再可行。DQN（Deep Q-Network, Mnih et al., 2015）使用**深度神经网络**来近似 Q 函数：

$$Q(s, a; \theta) \approx Q^*(s, a)$$

### 3.2 DQN 的三大核心创新

#### （1）经验回放（Experience Replay）

解决深度学习面临的**样本相关性与非平稳分布**问题。

**机制：**
- 维护一个**经验回放缓冲区（Replay Buffer）** $\mathcal{D}$，存储 transition $(s, a, r, s')$
- 每次更新时，从 $\mathcal{D}$ 中**随机采样**一个 mini-batch 进行训练
- 打破了时序相关性，提高了数据利用率

**优点：**
- 每个样本可被重复使用，提高样本效率
- 打破样本间的时序相关性，满足 SGD 的 i.i.d. 假设
- 平滑训练数据的分布变化

#### （2）目标网络（Target Network）

解决 Q-Learning 中的**非平稳目标**问题——由于同一个网络同时用于估计当前 Q 值和计算目标 Q 值，导致训练不稳定。

**机制：**
- 维护两个网络：当前 Q 网络（在线网络）$Q(s, a; \theta)$ 和目标网络 $Q(s, a; \theta^-)$
- 目标网络的参数 $\theta^-$ 定期（如每 $C$ 步）从在线网络复制，而非每步更新
- 目标值计算：$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$

**损失函数：**

$$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

#### （3）奖励裁剪（Reward Clipping）

将所有奖励裁剪到 $[-1, +1]$ 范围，使得同一组超参数可以跨不同游戏使用。

### 3.3 DQN 算法流程

```
初始化回放缓冲区 D 容量 N
初始化 Q 网络参数 θ，随机初始化
初始化目标网络参数 θ⁻ = θ
对每个 episode = 1, 2, ..., M:
    初始化状态 s₁
    对 t = 1, 2, ..., T:
        以 ε-greedy 选择动作 a_t
        执行 a_t，观察 r_t 和 s_{t+1}
        将 (s_t, a_t, r_t, s_{t+1}) 存入 D
        从 D 中采样 mini-batch (s_j, a_j, r_j, s'_{j})
        计算目标:
            y_j = r_j, 如果 s'_{j} 为终止状态
            y_j = r_j + γ max_a' Q(s'_j, a'; θ⁻), 否则
        对 (y_j - Q(s_j, a_j; θ))² 执行梯度下降
        每 C 步: θ⁻ ← θ
```

## 四、DQN 的改进变体

### 4.1 Double DQN

Q-Learning 存在**最大化偏差（maximization bias）**——用同一个网络选择和评估动作导致对 Q 值的系统性高估。

Double DQN（van Hasselt et al., 2016）**解耦动作选择与动作评估**：

**目标计算：**

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

- 使用**在线网络** $\theta$ 选择最优动作
- 使用**目标网络** $\theta^-$ 评估该动作的价值

改进仅需修改一行代码，效果显著。

### 4.2 Dueling DQN

Dueling DQN（Wang et al., 2016）将 Q 函数分解为两个部分：

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)$$

其中：
- $V(s)$ 是**状态价值**——状态本身的好坏
- $A(s, a)$ 是**优势函数（Advantage）**——某个动作相对于平均的好坏

减去均值是为了解决可辨识性（identifiability）问题——给定 $Q$，$V$ 和 $A$ 不是唯一确定的。

**优势：** 当不同动作的价值差异很小时（如在很多状态下选择哪个动作都差不多），Dueling 结构可以更有效地学习状态价值，提高策略评估的效率。

### 4.3 Prioritized Experience Replay（优先经验回放）

标准经验回放**均匀随机采样**所有 transition。但不同 transition 的学习价值不同：**TD 误差大的 transition 应该被更频繁地回放**。

**采样概率：**

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

其中 $p_i = |\delta_i| + \epsilon$（$\delta_i$ 为 TD 误差，$\epsilon$ 防止零概率），$\alpha$ 控制优先级程度。

**重要性采样权重：** 优先回放改变了采样分布，需要重要性采样来修正偏差：

$$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta$$

通常 $\beta$ 从初始值逐渐增加到 1（完全修正）。

### 4.4 其他改进

- **Multi-step Returns**：使用 $n$ 步回报 $r_t + \gamma r_{t+1} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n \max Q(s_{t+n}, a)$ 替代单步 TD 目标
- **Distributional RL (C51)**：学习回报的**完整分布**而非仅期望值
- **Noisy Nets**：在权重中加入可学习的噪声参数，以促进探索
- **Rainbow**：将上述所有改进（DDQN + Dueling + PER + Multi-step + Distributional + Noisy Nets）整合在一起，达到 SOTA 性能

## 五、DQN 系列总结

| 算法 | 核心创新 | 发表年份 |
|------|----------|----------|
| DQN | 经验回放 + 目标网络 | 2015 |
| Double DQN | 解耦选择与评估 | 2016 |
| Dueling DQN | $Q = V + A$ 分解 | 2016 |
| PER | TD-error 加权采样 | 2016 |
| C51 | 学习回报分布 | 2017 |
| Noisy DQN | 参数噪声探索 | 2018 |
| Rainbow | 整合上述所有改进 | 2018 |

## 六、关键实现细节

### 6.1 $\varepsilon$ 衰减策略

通常使用线性衰减或指数衰减：

$$\varepsilon_t = \varepsilon_{end} + (\varepsilon_{start} - \varepsilon_{end}) \cdot e^{-t / \tau_{decay}}$$

### 6.2 梯度裁剪

为防止梯度爆炸，常对梯度进行裁剪：

$$g \leftarrow \min(g, \text{clip\_value})$$

### 6.3 Huber Loss

替代 MSE 以提高对异常值的鲁棒性：

$$L_\delta(y, f(x)) = \begin{cases} \frac{1}{2}(y - f(x))^2, & \text{if } |y - f(x)| \leq \delta \\ \delta |y - f(x)| - \frac{1}{2}\delta^2, & \text{otherwise} \end{cases}$$

## 参考文献

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
- van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-Learning. *AAAI 2016*.
- Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *ICML 2016*.
- Schaul, T., et al. (2016). Prioritized Experience Replay. *ICLR 2016*.
- Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. *AAAI 2018*.
