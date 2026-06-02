# 生成式 AI 概述

## 定义

**生成式 AI**（Generative AI, GenAI）是人工智能的一个分支，专注于**生成新的内容**——包括文本、图像、音频、视频、代码等。与判别式模型（学习 $P(y|x)$）不同，生成式模型学习数据的**联合分布** $P(x, y)$ 或**数据分布** $P(x)$，从而能够从分布中采样生成新样本。

## 判别模型 vs 生成模型

| 维度 | 判别模型 | 生成模型 |
|------|----------|----------|
| **数学目标** | 学习 $P(y\|x)$ | 学习 $P(x, y)$ 或 $P(x)$ |
| **核心任务** | 分类、回归 | 采样、生成、补全 |
| **决策边界** | 直接建模 | 间接推导 |
| **数据需求** | 相对较少 | 通常需要大量数据 |
| **典型模型** | 逻辑回归、SVM、CNN 分类器 | GAN、VAE、扩散模型、GPT |

## 生成模型的数学框架

### 最大似然估计

生成模型的核心目标：最大化训练数据的似然

$$\theta^* = \arg\max_\theta \prod_{i=1}^{N} P_\theta(x_i) = \arg\max_\theta \sum_{i=1}^{N} \log P_\theta(x_i)$$

### 不同生成范式的对比

| 范式 | 核心思想 | 公式 | 优点 | 缺点 |
|------|----------|------|------|------|
| **自回归模型** | 链式法则分解 | $P(x)=\prod_{t=1}^{T}P(x_t\|x_{<t})$ | 似然可精确计算 | 生成慢（逐 token） |
| **变分自编码器 (VAE)** | 变分推断 | $\log P(x) \geq \mathbb{E}_{q(z\|x)}[\log P(x\|z)] - D_{KL}(q(z\|x)\|P(z))$ | 有隐空间 | 生成质量一般 |
| **生成对抗网络 (GAN)** | 博弈论 | $\min_G \max_D V(D,G)$ | 生成质量高 | 训练不稳定 |
| **扩散模型** | 逐步去噪 | $P_\theta(x_0) = \int P(x_T)\prod_{t=1}^{T}P_\theta(x_{t-1}\|x_t)dx_{1:T}$ | 生成质量最高 | 推理慢 |
| **流模型 (Flow)** | 可逆变换 | $\log P(x) = \log P(z) + \log\|\det\frac{\partial f^{-1}}{\partial x}\|$ | 精确似然 | 架构受限 |

## 自回归生成原理

大语言模型和部分图像生成模型使用的核心范式。将联合分布分解为条件概率的乘积：

$$P(x_1, x_2, ..., x_T) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1, x_2) \cdot ... \cdot P(x_T|x_1, ..., x_{T-1})$$

**自回归生成过程**（逐 token 采样）：

1. 输入当前序列
2. 模型输出下一个 token 的概率分布（通常经 softmax）
3. 从分布中采样（或贪心选择）
4. 将采样结果拼接到序列
5. 重复直到生成了结束标记

**采样策略**：

| 策略 | 公式 | 特点 |
|------|------|------|
| **贪心** | $\arg\max P(x_t\|x_{<t})$ | 确定性，可能重复 |
| **Temperature** | $P(x_t) \propto \exp(z_t / \tau)$ | $\tau<1$ 更确定，$\tau>1$ 更多样 |
| **Top-k** | 仅从概率最高的 k 个 token 中采样 | 截断低概率 |
| **Top-p (Nucleus)** | 从累积概率 ≥ p 的最小集合中采样 | 动态截断 |
| **Beam Search** | 保留 top-b 个候选序列 | 近似最优但多样性差 |

## 扩散模型原理

扩散模型通过**逐步加噪 → 逐步去噪**的过程生成数据。

### 前向过程（加噪）

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

重参数化后直接计算任意 $t$ 步：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

其中 $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$。

### 反向过程（去噪）

$$P_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

### 训练目标（DDPM 简化）

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]$$

即：训练一个网络 $\epsilon_\theta$ 预测所加的噪声。

### 条件生成（Classifier-Free Guidance）

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

$w > 1$ 时增强条件控制，$w = 1$ 为标准条件生成。

## VAE 原理

变分自编码器通过编码器将数据映射到隐空间，再通过解码器重建：

**变分下界 (ELBO)**：

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

- 第一项：重建损失（希望解码精确）
- 第二项：KL 散度正则（希望隐分布接近先验）

**重参数化技巧**：

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

使梯度可以通过采样节点反向传播。

## GAN 原理速览

生成器 $G$ 与判别器 $D$ 的对抗博弈：

$$\min_G \max_D \mathbb{E}_{x \sim P_{data}}[\log D(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]$$

详见 [[生成对抗网络（GAN）]]

## 应用矩阵

| 模态 | 生成任务 | 代表模型 |
|------|----------|----------|
| **文本** | 对话、写作、代码、翻译 | GPT-4, Claude, Llama |
| **图像** | 文生图、图生图、编辑 | DALL-E, Stable Diffusion, Midjourney |
| **音频** | 语音合成、音乐生成 | ElevenLabs, MusicLM, Suno |
| **视频** | 文生视频、视频编辑 | Sora, Runway, Pika |
| **3D** | 文生3D、3D 重建 | DreamFusion, MeshGPT |
| **多模态** | 图文理解+生成 | GPT-4V, Gemini, Claude |

## 评估指标

| 指标 | 公式/含义 | 使用场景 |
|------|-----------|----------|
| **Perplexity** | $\exp(-\frac{1}{N}\sum\log P(x_i))$ | 语言模型 |
| **BLEU** | n-gram 精确匹配 | 机器翻译 |
| **ROUGE** | n-gram 召回 | 摘要生成 |
| **FID** | Fréchet Inception Distance | 图像生成质量 |
| **IS** | Inception Score | 图像多样性+质量 |
| **CLIP Score** | 图文匹配度 | 文生图 |

## 详细子领域

- [[大语言模型（LLM）]]
- [[扩散模型（Diffusion Model）]]
- [[多模态模型]]
