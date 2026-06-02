# 策略梯度与 Actor-Critic

## 一、引言

基于价值的方法（如 Q-Learning、DQN）学习动作价值函数 $Q(s, a)$，然后通过 $\arg\max$ 选择动作。虽然在高维状态空间中取得了成功，但基于价值的方法有几个局限：

1. **难以处理连续动作空间**：$\arg\max$ 在连续空间中不可行
2. **确定性策略**：无法表示随机策略，某些问题（如石头剪刀布）最优策略本身就是随机的
3. **对价值函数的微小变化敏感**：微小的估值误差可能导致完全不同的策略

基于策略的方法直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升优化期望累积奖励。

### 1.1 策略目标函数

定义策略 $\pi_\theta$ 的期望累积奖励：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

或在无穷 horizon 下（稳态分布）：

$$J(\theta) = \sum_{s} d^{\pi_\theta}(s) \sum_{a} \pi_\theta(a|s) Q^{\pi_\theta}(s, a)$$

其中 $d^{\pi_\theta}(s)$ 是策略 $\pi_\theta$ 下的状态稳态分布。

## 二、策略梯度定理（Policy Gradient Theorem）

策略梯度定理（Sutton et al., 1999）给出了目标函数梯度的简洁形式：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]$$

**直觉理解：**
- $\nabla_\theta \log \pi_\theta(a|s)$ 指示了如何调整参数来增加动作 $a$ 在状态 $s$ 下被选中的概率
- $Q^{\pi_\theta}(s, a)$ 作为"得分"——好的动作（高 $Q$ 值）被增强，差的动作被抑制

### 2.1 对数概率技巧（Log-Derivative Trick）

$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$$

由此：

$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \sum_{t} \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$

## 三、REINFORCE（蒙特卡洛策略梯度）

REINFORCE 是最基础的策略梯度算法，使用**蒙特卡洛采样**来估计 $Q^{\pi_\theta}(s, a)$。

### 3.1 算法

用实际的折扣回报 $G_t$ 替代 $Q^{\pi_\theta}(s, a)$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$。

**更新规则：**

$$\theta \leftarrow \theta + \alpha \sum_{t} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t$$

### 3.2 REINFORCE 伪代码

```
初始化策略参数 θ
对每个 episode:
    使用 π_θ 生成完整轨迹 τ = (s₀,a₀,r₀, s₁,a₁,r₁, ..., s_T)
    对 t = 0,1,...,T:
        G_t ← 计算从 t 开始的折扣回报
        θ ← θ + α ∇_θ log π_θ(a_t|s_t) ⋅ G_t
```

### 3.3 REINFORCE 的缺点

- **高方差**：$G_t$ 是随机变量，方差很大，导致收敛缓慢
- **样本效率低**：每次更新需要完整轨迹
- **On-Policy**：必须使用当前策略的数据

## 四、引入 Baseline —— 减少方差

### 4.1 Baseline 技术

从 $Q^{\pi_\theta}(s, a)$ 中减去一个**不依赖于动作**的基线 $b(s)$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (Q^{\pi_\theta}(s_t, a_t) - b(s_t)) \right]$$

减去 baseline 不引入偏差（偏导数为零），但可以显著降低方差。

**证明 baseline 无偏：**

$$\mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) = b(s) \nabla_\theta \sum_a \pi_\theta(a|s) = 0$$

### 4.2 常用 Baseline

- **状态价值函数**：$b(s) = V^{\pi_\theta}(s)$
  - 此时 $(Q - V)$ 正是**优势函数** $A(s, a)$
- **常数 Baseline**：所有样本的平均回报
- **学习 Baseline**：用神经网络拟合 $V^{\pi_\theta}(s)$

## 五、Actor-Critic 架构

Actor-Critic 方法结合了基于策略的方法（Actor）和基于价值的方法（Critic）：

- **Actor（演员）**：参数化策略 $\pi_\theta(a|s)$，决定"做什么"
- **Critic（评论家）**：参数化价值函数 $V_\phi(s)$ 或 $Q_\phi(s, a)$，评价"做得好不好"

### 5.1 标准 Actor-Critic

**Actor 更新（策略梯度 + Baseline）：**

$$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot (Q_\phi(s, a) - V_\phi(s))]$$

**Critic 更新（TD 学习）：**

$$\phi \leftarrow \phi - \alpha_c \nabla_\phi \left( r + \gamma V_\phi(s') - V_\phi(s) \right)^2$$

### 5.2 优势函数（Advantage Function）

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- $A(s, a) > 0$：该动作比平均水平好
- $A(s, a) < 0$：该动作比平均水平差
- 使用优势函数可以在不损失信号的同时降低方差

**TD 误差作为优势估计：**

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\delta_t$ 是 $A(s_t, a_t)$ 的一个无偏估计（尽管有偏但方差低）。

## 六、A2C / A3C（Advantage Actor-Critic）

### 6.1 A2C（Advantage Actor-Critic）

同步版本。在多个并行环境中运行，收集所有环境的梯度后进行平均，然后统一更新：

**Actor 损失：**

$$L_{actor} = -\frac{1}{N} \sum_{i} \sum_{t} \log \pi_\theta(a_{i,t} | s_{i,t}) \cdot A(s_{i,t}, a_{i,t})$$

**Critic 损失：**

$$L_{critic} = \frac{1}{N} \sum_{i} \sum_{t} \left( r_{i,t} + \gamma V_\phi(s_{i,t+1}) - V_\phi(s_{i,t}) \right)^2$$

通常加上**熵正则项（Entropy Bonus）** 以鼓励探索：

$$L_{total} = L_{actor} + \beta_v L_{critic} - \beta_e H(\pi_\theta(\cdot|s))$$

### 6.2 A3C（Asynchronous Advantage Actor-Critic）

异步版本。多个 worker 各自在独立的环境副本中运行，异步更新全局参数，无需等待其他 worker。异步更新本身引入了天然的多样性，起到了正则化作用。

## 七、GAE（Generalized Advantage Estimation）

GAE（Schulman et al., 2016）通过 $\lambda$ 加权平均不同步长的 TD 误差，在偏差和方差之间取得平衡。

### 7.1 定义

$n$ 步 TD 误差：

$$\delta_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n}) - V(s_t)$$

**GAE 是 $\delta_t^{(n)}$ 的指数加权平均：**

$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

### 7.2 参数含义

- $\gamma$：控制回报的时间范围（标准折扣因子）
- $\lambda$：控制偏差-方差权衡
  - $\lambda = 0$：仅用单步 TD 误差（低方差，高偏差）
  - $\lambda = 1$：等价于蒙特卡洛回报（高方差，低偏差）
  - 通常 $\lambda \in [0.9, 0.99]$

## 八、PPO（Proximal Policy Optimization）

PPO（Schulman et al., 2017）是当前最广泛使用的策略梯度算法之一，它通过**裁剪（clipping）** 来限制策略更新的幅度，保持训练的稳定性。

### 8.1 重要性采样比率

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

当 $\theta = \theta_{old}$ 时，$r_t = 1$。如果新策略与旧策略相差太大，$r_t$ 会偏离 1。

### 8.2 Clipped 目标

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon) \hat{A}_t \right) \right]$$

其中 $\varepsilon$ 为裁剪范围（通常 $\varepsilon = 0.2$）。

**工作机制：**

- 当 $\hat{A}_t > 0$（好动作）：限制 $r_t$ 不超过 $1 + \varepsilon$，防止策略变化太大
- 当 $\hat{A}_t < 0$（坏动作）：限制 $r_t$ 不低于 $1 - \varepsilon$，防止过度惩罚

**裁剪操作** 将 $r_t$ 限制在 $[1 - \varepsilon, 1 + \varepsilon]$ 范围内，使用 $\min$ 确保目标函数是裁剪版本的下界（悲观估计）。

### 8.3 完整 PPO 目标

$$\mathcal{L}(\theta) = \mathbb{E}_t \left[ \mathcal{L}_t^{CLIP}(\theta) - c_1 \mathcal{L}_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

其中：
- $\mathcal{L}_t^{VF} = (V_\theta(s_t) - V_t^{target})^2$ 为值函数损失
- $S[\pi_\theta](s_t)$ 为策略的熵，$c_2$ 为熵系数

### 8.4 PPO 伪代码

```
对每次迭代:
    对每个 actor = 1,...,N:
        使用 π_θold 运行策略 T 个时间步
        计算优势估计 Â₁,...,Â_T（使用 GAE）
    对 K 个 epoch:
        使用小批量数据优化 L(θ)（SGD / Adam）
    θ_old ← θ
```

### 8.5 PPO 变体：PPO-Penalty

另一个变体不是裁剪，而是使用 **KL 散度惩罚**：

$$\mathcal{L}^{KLPEN}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \hat{A}_t - \beta \text{KL}[\pi_{\theta_{old}}, \pi_\theta] \right]$$

动态调整 $\beta$ 以保持 KL 散度在目标范围内。

## 九、方法对比总结

| 方法 | 类型 | 样本效率 | 稳定性 | 连续动作 | 离线数据 |
|------|------|----------|--------|----------|----------|
| REINFORCE | PG | 低 | 低（高方差）| ✓ | ✗ |
| A2C/A3C | Actor-Critic | 中等 | 中等 | ✓ | ✗ |
| TRPO | PG + 约束 | 中等 | 高 | ✓ | ✗ |
| PPO | PG + Clipping | 中等 | 高 | ✓ | ✗ |
| DDPG | Off-Policy AC | 较高 | 中等 | ✓(确定性) | ✓ |
| TD3 | Off-Policy AC | 较高 | 高 | ✓(确定性) | ✓ |
| SAC | Off-Policy AC | 高 | 高 | ✓(随机) | ✓ |

## 参考文献

- Sutton, R. S., et al. (1999). Policy Gradient Methods for Reinforcement Learning with Function Approximation. *NeurIPS*.
- Schulman, J., et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv*.
- Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML*.
