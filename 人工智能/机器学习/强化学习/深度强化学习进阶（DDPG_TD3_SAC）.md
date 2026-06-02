# 深度强化学习进阶：DDPG / TD3 / SAC

## 一、连续动作空间的挑战

基于价值的 DQN 系列算法在离散动作空间中表现出色，但现实世界中的许多问题（如机器人控制、自动驾驶）涉及**连续动作空间**：转向角度、油门大小、关节力矩等。这些问题无法使用 $\arg\max_a Q(s, a)$ 直接选择动作。

解决连续动作空间的强化学习主要有两条路径：

1. **基于策略的方法**（如 PPO）—— 直接输出动作分布参数
2. **基于 Actor-Critic 的 Off-Policy 方法**（DDPG、TD3、SAC）—— Actor 输出确定性或随机动作，Critic 评估价值

本章重点介绍第二类：面向连续动作空间的 Off-Policy Actor-Critic 算法。

## 二、DDPG（Deep Deterministic Policy Gradient）

### 2.1 核心思想

DDPG（Lillicrap et al., 2016）是 DPG（Deterministic Policy Gradient）的深度扩展，本质上是 DQN 在连续动作空间的**Off-Policy Actor-Critic** 适配。

- **Actor** $\mu(s; \theta^\mu)$：输出**确定性**动作 $a = \mu(s)$（直接映射，无概率分布）
- **Critic** $Q(s, a; \theta^Q)$：评估状态-动作对的价值

### 2.2 策略梯度

对于确定性策略，Silver et al. (2014) 证明了确定性策略梯度定理：

$$\nabla_{\theta^\mu} J \approx \mathbb{E}_{s \sim \rho^\beta} \left[ \nabla_a Q(s, a; \theta^Q) \big|_{a=\mu(s)} \cdot \nabla_{\theta^\mu} \mu(s; \theta^\mu) \right]$$

**直觉：** Critic 的梯度告诉 Actor 应该朝哪个方向调整动作以提高 Q 值（即最大化期望回报）。

### 2.3 从 DQN 继承的技术

DDPG 从 DQN 借用了两项关键技术：

#### （1）经验回放（Experience Replay）
维护缓冲区 $\mathcal{D}$，从中随机采样 mini-batch 进行训练。

#### （2）目标网络（Target Networks）
Actor 和 Critic 各有目标网络，通过**软更新（Soft Update / Polyak Averaging）**缓慢更新：

$$\theta^{Q^-} \leftarrow \tau \theta^Q + (1 - \tau) \theta^{Q^-}$$
$$\theta^{\mu^-} \leftarrow \tau \theta^\mu + (1 - \tau) \theta^{\mu^-}$$

其中 $\tau \ll 1$（通常 $\tau = 0.001$），确保目标网络平滑变化，提高训练稳定性。

> 与 DQN 的硬更新（定期完全复制）不同，软更新使目标网络**持续缓慢**地向在线网络移动。

### 2.4 Critic 更新

最小化 TD 误差：

$$L_Q = \frac{1}{N} \sum_i \left( y_i - Q(s_i, a_i; \theta^Q) \right)^2$$

其中目标：

$$y_i = r_i + \gamma Q^- (s'_i, \mu^-(s'_i; \theta^{\mu^-}); \theta^{Q^-})$$

### 2.5 探索策略

由于确定性策略本身不产生探索，DDPG 在训练时向 Actor 的输出加入探索噪声：

$$a_t = \mu(s_t; \theta^\mu) + \mathcal{N}_t$$

通常使用 Ornstein-Uhlenbeck 过程（时间相关噪声）或简单的高斯噪声。

### 2.6 DDPG 的局限

- **对超参数敏感**，训练不稳定
- **Q 值过估计（Overestimation Bias）**— 与 DQN 类似，Critic 容易高估 Q 值
- 探索效率有限

## 三、TD3（Twin Delayed DDPG）

TD3（Fujimoto et al., 2018）针对 DDPG 的三个关键问题提出了改进，是目前最常用的确定性策略算法之一。

### 3.1 改进一：Clipped Double Q-Learning

DDPG 的 Critic 容易过估计 Q 值。借鉴 Double DQN 的思想，TD3 使用**两个 Critic** $Q_{\phi_1}$ 和 $Q_{\phi_2}$，在计算目标时取两者的**最小值**：

$$y = r + \gamma \min_{i=1,2} Q_{\phi_i^-}(s', a')$$

取 $\min$ 而非 $\arg\max$ + 另一网络的评估，进一步加强了对过估计的抑制。

> **关键直觉：** 如果有两个略有偏高的估计，取最小值更安全——宁可稍微低估，也不冒着高估的风险。这被称为 **Clipped Double Q-Learning**。

**实际实现：**
- 两个 Critic 使用相同的目标 $y$（都取 $\min$），各自独立更新
- Actor 只使用 $Q_{\phi_1}$ 指导更新（也可用 $\min$）

### 3.2 改进二：Delayed Policy Updates

策略网络应比价值网络**更新得更慢**，以允许 Critic 充分收敛，减少策略更新引入的方差。

**具体做法：** 每 $d$ 次 Critic 更新后才做一次 Actor 和目标网络的更新（通常 $d = 2$）。

```
对每次环境交互:
    更新两个 Critic
    如果 t % d == 0:
        更新 Actor
        软更新目标网络
```

### 3.3 改进三：Target Policy Smoothing

在目标动作上加入微小噪声（正则化），防止策略在 Q 值的窄峰上过拟合：

$$a^- = \mu^-(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma)$$

$$y = r + \gamma \min_{i=1,2} Q_{\phi_i^-}(s', a^-)$$

这类似于**平滑正则化**：告诉 Q 函数，相似的动作应有相似的 Q 值，防止 Q 值在某些动作上的尖锐峰值。

### 3.4 TD3 总览

```
初始化 Actor π_θ, Critic Q_{φ₁}, Q_{φ₂}
初始化目标网络 θ⁻←θ, φ₁⁻←φ₁, φ₂⁻←φ₂
初始化经验回放 D

对每个时间步:
    选择动作 a = π_θ(s) + ε, ε~N(0,σ)
    执行 a, 存入 D
    从 D 采样 mini-batch
    
    计算目标动作: a⁻ = π_θ⁻(s') + clip(N(0,σ̃), -c, c)
    目标值: y = r + γ min_i Q_{φ_i⁻}(s', a⁻)
    更新 Critic: L = (Q_{φ_i} - y)²  for i=1,2
    
    如果 t % d == 0:
        更新 Actor: ∇_θ J = ∇_a Q_{φ₁} ⋅ ∇_θ π_θ
        软更新目标网络: θ⁻←τθ+(1-τ)θ⁻, φ_i⁻←τφ_i+(1-τ)φ_i⁻
```

### 3.5 TD3 vs DDPG 对比

| 特性 | DDPG | TD3 |
|------|------|-----|
| Critic 数量 | 1 | 2（Clipped Double Q）|
| 更新频率 | Actor 与 Critic 同步 | Critic 更新频率 > Actor |
| 目标动作 | $\mu^-(s')$ 无噪声 | $\mu^-(s')$ + 裁剪噪声 |
| Q 值估计 | 易过估计 | 更准确 |

## 四、SAC（Soft Actor-Critic）

SAC（Haarnoja et al., 2018）是目前最先进的 Off-Policy 算法之一，核心创新是将**最大熵（Maximum Entropy）**原则引入强化学习。

### 4.1 最大熵强化学习

传统 RL 目标：最大化期望累积奖励 $\mathbb{E}[\sum \gamma^t r_t]$

SAC 目标：在最大化奖励的同时，也最大化策略的熵：

$$J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r_t + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right) \right]$$

其中 $\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ 是策略的香农熵，$\alpha$ 是温度参数（权衡奖励与熵）。

**熵正则化的好处：**
- 鼓励探索——策略不会过早地变得确定
- 策略具有随机性，对模型误差更鲁棒
- 自动进行探索-利用权衡
- 倾向于捕获多种接近最优的行为模式

### 4.2 软价值函数

**软状态价值函数：**

$$V_\pi(s) = \mathbb{E}_{a \sim \pi} \left[ Q_\pi(s, a) - \alpha \log \pi(a|s) \right]$$

**软动作价值函数（软 Bellman 方程）：**

$$Q_\pi(s, a) = r + \gamma \mathbb{E}_{s' \sim \mathcal{P}} \left[ V_\pi(s') \right]$$

代入后得到软 Q 函数的 TD 目标：

$$Q_{soft}(s, a) \leftarrow r + \gamma \mathbb{E}_{s'} \left[ \mathbb{E}_{a' \sim \pi} [ Q(s', a') - \alpha \log \pi(a'|s') ] \right]$$

### 4.3 SAC 架构

SAC 维护五个网络：

1. **Actor / 策略网络** $\pi_\phi(a|s)$：输出动作的均值和方差（高斯分布）
2. **两个 Soft Q 网络** $Q_{\theta_1}, Q_{\theta_2}$：类似 TD3（Clipped Double Q）
3. **两个目标 Q 网络** $Q_{\theta_1^-}, Q_{\theta_2^-}$：软更新

（可选）**温度参数 $\alpha$** 可自动调节。

### 4.4 Critic（Q 网络）更新

**目标值（与 TD3 类似，但包含熵项）：**

$$y = r + \gamma \left( \min_{i=1,2} Q_{\theta_i^-}(s', a') - \alpha \log \pi_\phi(a'|s') \right), \quad a' \sim \pi_\phi(\cdot|s')$$

> 注意：目标中 $a'$ 是从**当前策略** $\pi_\phi$ 采样的（而非目标策略），这是 Off-Policy 特征。

**损失函数：**

$$L_Q(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( Q_{\theta_i}(s, a) - y \right)^2 \right]$$

### 4.5 Actor（策略网络）更新

SAC 使用**重参数化技巧（Reparameterization Trick）**来对随机动作进行反向传播：

$$a = f_\phi(s, \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

对于高斯策略：$a = \mu_\phi(s) + \sigma_\phi(s) \odot \epsilon$

**Actor 损失：**

$$L_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \alpha \log \pi_\phi(f_\phi(s, \epsilon) | s) - \min_{i=1,2} Q_{\theta_i}(s, f_\phi(s, \epsilon)) \right]$$

目标是最小化 KL 散度到最优策略的指数 Q 分布（在熵约束下）。

### 4.6 自动温度调节

温度参数 $\alpha$ 可以在训练中自动调整，以维持策略的**目标熵水平**：

$$L(\alpha) = \mathbb{E}_{a \sim \pi} \left[ -\alpha \log \pi(a|s) - \alpha \bar{\mathcal{H}} \right]$$

其中 $\bar{\mathcal{H}} = -|\mathcal{A}|$ 为目标熵（通常设为动作维度的负值）。

当策略熵高于目标时减小 $\alpha$（降低探索），反之增大 $\alpha$。

### 4.7 SAC 算法伪代码

```
初始化 Actor π_φ, Critic Q_{θ₁}, Q_{θ₂}
初始化目标网络 θ₁⁻←θ₁, θ₂⁻←θ₂
初始化温度参数 α（可选自动调节）
初始化经验回放 D

对每次环境交互:
    从 π_φ(·|s) 采样动作 a
    执行 a，观察 r, s'
    将 (s,a,r,s') 存入 D
    
    从 D 采样 mini-batch
    计算 Q 目标:
        a' ~ π_φ(·|s')  ← 注意：从当前策略采样
        y = r + γ(min_i Q_{θ_i⁻}(s',a') - α log π_φ(a'|s'))
    更新 Q 网络: min (Q_{θ_i}(s,a) - y)²
    
    更新 Actor:
        a_reparam = f_φ(s,ε)  ← 重参数化
        min α log π_φ - min_i Q_{θ_i}(s, a_reparam)
    
    更新 α（可选）
    软更新目标网络
```

### 4.8 SAC vs TD3 vs DDPG 对比

| 特性 | DDPG | TD3 | SAC |
|------|------|-----|-----|
| 策略类型 | 确定性 | 确定性 | **随机（高斯）** |
| Off-Policy | ✓ | ✓ | ✓ |
| Double Q | ✗ | ✓（Clipped）| ✓（Clipped）|
| 熵正则 | ✗ | ✗ | ✓（核心创新）|
| Target Smoothing | ✗ | ✓ | ✗（由随机性自然实现）|
| Delayed Update | ✗ | ✓ | ✗（通常不需要）|
| 探索机制 | 外部噪声 | 外部噪声 | **策略内置（熵）** |
| 样本效率 | 较高 | 高 | 最高 |

## 五、实用建议

### 5.1 算法选择指南

- **离散动作 + 高样本效率** → DQN 系列（Rainbow）
- **连续动作 + 简单任务** → PPO（稳定，易于调参）
- **连续动作 + 需要样本效率** → SAC（首选）
- **连续动作 + 确定性策略合适** → TD3
- **多智能体 / 对抗** → PPO, SAC

### 5.2 常见调参技巧

1. **学习率**：Actor 通常用比 Critic 更小的学习率
2. **$\tau$（软更新）**：$0.005$ 是常用值，越大更新越快（可能不稳定），越小越平滑（可能过慢）
3. **经验回放大小**：通常 $10^5 \sim 10^6$ 个 transition
4. **初始随机探索步数**：在训练初期进行纯随机探索以收集多样化的经验
5. **Batch Normalization**：在高维观测空间中能显著提升性能

## 参考文献

- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning. *ICLR*.
- Fujimoto, S., et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML*.
- Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *ICML*.
- Haarnoja, T., et al. (2019). Soft Actor-Critic Algorithms and Applications. *arXiv*.
- Silver, D., et al. (2014). Deterministic Policy Gradient Algorithms. *ICML*.
