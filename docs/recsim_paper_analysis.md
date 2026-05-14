# RecSim 论文算法深度解读

> 论文: [RecSim: A Configurable Simulation Platform for Recommender Systems](https://arxiv.org/abs/1909.04847)
> 作者: Eugene Ie, Chih-wei Hsu, Martin Mladenov 等 (Google Research, 2019)
> 关联论文: [SlateQ](https://arxiv.org/abs/1905.12767) (IJCAI 2019), Mladenov et al. 2019

---

## 1. 论文定位

这是一篇**平台论文**而非方法论文。它的贡献不是提出新算法，而是构建了一个仿真平台，使得推荐系统的 RL 算法研究成为可能。但论文通过三个 Case Study 展示了平台上可以研究的关键算法问题，每个都对应一类推荐系统 RL 的核心挑战。

---

## 2. 数学建模：动态贝叶斯网络

### 2.1 轨迹概率分解

论文将推荐系统交互建模为 DBN，定义了轨迹的联合概率：

```
p(o₁..N, c₁..N, A₁..N) = Σ_{z₀..N} [ p(z₀) · p(A₀) · p(c₀|A₀,z₀) ·
  ∏ₜ p(oₜ|zₜ) · p(zₜ|zₜ₋₁,Aₜ,cₜ) · p(cₜ|Aₜ,zₜ₋₁) · p(Aₜ|Dₜ,Hₜ₋₁) · p(Dₜ) ]
```

各项含义：

| 符号 | 含义 | 对应 |
|:-----|:-----|:-----|
| zₜ | 用户隐状态 | satisfaction, interests |
| oₜ | 观测 | `create_observation()` 的返回值 |
| cₜ | 用户选择/响应 | click, watch_time |
| Aₜ | 推荐 slate | Agent 输出的文档索引数组 |
| Dₜ | 候选文档集 | `CandidateSet` |
| Hₜ₋₁ | 可观测历史 | Agent 可见的所有历史信息 |
| p(zₜ\|zₜ₋₁,Aₜ,cₜ) | 状态转移 | `update_state()` |
| p(cₜ\|Aₜ,zₜ₋₁) | 选择模型 | `ChoiceModel` |
| p(oₜ\|zₜ) | 观测模型 | 部分可观测性的核心 |
| p(Aₜ\|Dₜ,Hₜ₋₁) | 推荐策略 | Agent 的 policy |

### 2.2 核心公式

**用户-文档偏好**（所有环境共用）：

```
I(u, d) = u · d    （点积）
```

**用户满意度**（IE 环境）：

```
S(u, d) = (1 - α) · I(u, d) + α · L_d
```

其中 L_d 是文档固有质量，α 平衡兴趣驱动与质量驱动。

**时间预算衰减**：

```
B_u -= l(d) - b · S(u, d)
```

观看消耗 l(d) 时间，满意度 S(u,d) 越高补回越多。

**兴趣漂移**：

```
Δₜ(Iₜ) = (-y·|Iₜ| + y) · (-Iₜ)
```

y ∈ [0,1] 控制漂移幅度，|Iₜ| 越大（兴趣越强）漂移越小。

---

## 3. 算法一：隐状态 Bandit（Case Study 1）

### 3.1 问题形式化

- 单文档推荐（slate_size=1）
- 用户兴趣 **u** 是隐变量（Agent 看不到）
- 文档有主题向量 **d**（one-hot）和质量 L_d ~ lnN(μ_{T(d)}, σ²)
- 选择概率取决于 f(I(u,d) + L_d)
- 用户兴趣**不随时间变化**（静态 Bandit 设定）

### 3.2 核心挑战

不同主题的 reward 是**相关的**——都受文档制作质量 L_d 影响。高质量主题对所有用户的点击率都高，产生**流行度偏差**：短视策略只推高质量大众主题，忽略小众偏好。

这不是标准的独立臂赌博机，而是**相关臂赌博机** (Correlated Bandits)。

### 3.3 算法对比

| 方法 | 类型 | 策略 |
|:-----|:-----|:-----|
| Random | 基线 | 随机选择 |
| Greedy | 基线 | 全知选择（知道选择模型，跨用户平均） |
| TabularQ | RL | 离散化状态 + Q-learning |
| FullSlateQ | RL | 整个文档作为原子动作，DQN |
| UCB1 | Bandit | 置信上界，自动平衡探索/利用 |

所有非基线 Agent 都使用了 `ClusterClickStatsLayer`，为每个主题维护点击/曝光统计。

### 3.4 UCB1 算法

```
UCB(a) = x̄_a + √(2·ln(n) / n_a)

x̄_a = 主题 a 的平均 reward
n   = 总推荐次数
n_a = 主题 a 被推荐次数
```

被推荐次数少的主题 → 不确定性项 √(...) 大 → UCB 值高 → 优先探索。

### 3.5 实验结果

**论文 Table 1**：

| 策略 | 低主题亲和度 CTR | 高主题亲和度 CTR |
|:-----|:----------------|:----------------|
| Random | 7.86% | 14.97% |
| Greedy | 9.59% (+22%) | 17.56% (+17%) |
| TabularQ | 8.24% (+5%) | 20.16% (+35%) |
| FullSlateQ | 9.64% (+23%) | 23.28% (+56%) |
| **UCB1** | **9.76% (+24%)** | **25.17% (+68%)** |

### 3.6 关键发现

- **高亲和度场景**（用户兴趣主导）：UCB1 比随机策略提升 **68%**，探索价值巨大
- **低亲和度场景**（文档质量主导）：各方法差距缩小，因为主题偏好不太重要
- **FullSlateQ 也很强**：DQN 在高亲和度场景下提升 56%，说明深度 RL 能学到隐式探索策略
- 探索的价值随用户兴趣的重要性而增大

---

## 4. 算法二：SlateQ — Slate 分解（Case Study 2）

### 4.1 核心问题

推荐 slate 大小为 K，候选集大小为 N，动作空间 = C(N,K)。

例如 N=100, K=5 → 7500 万种 slate 组合。直接学习 Q(s, A) 不可行。

### 4.2 分解思想

**关键洞察**：如果用户的选择行为满足一定条件（如 MNL 或比例选择模型），slate 级别的 Q 值可以分解为单个文档 Q 值的加权和：

```
Q(s, A) = Σ_{d∈A} P(选择 d | A) · q(s, d)
```

其中：
- q(s, d) 是单个文档 d 在状态 s 下的 Q 值（由 DQN 学习）
- P(选择 d | A) 由选择模型给出

这样：
- **训练时**：只需学习 N 个文档级 Q 值 q(s, d)，而不是 C(N,K) 个 slate 级 Q 值
- **推理时**：用 q(s, d) 和选择模型概率组合出最优 slate

### 4.3 代码实现

`slate_decomp_q_agent.py` 中实现了三种 slate 构建策略和四种 target Q 计算方法的自由组合：

**Slate 构建策略**（推理时用）：

```python
# Top-K：直接选 score × q 最大的 K 个文档
select_slate_topk:    sorted(docs, key=s*q)[:K]

# Greedy：自适应贪心，逐个选文档，每次更新归一化
select_slate_greedy:  for k in K: pick argmax((num + s*q)/(den + s))

# Optimal：穷举所有 C(N,K) 种 slate，选 Q 值最大的（仅小规模可用）
select_slate_optimal: max over all slates Σ(s*q) / (Σs + s_no_click)
```

**Target Q 计算**（训练时用）：

```python
# SARSA：用实际采取的 slate 计算 target
compute_target_sarsa:    r + γ · Σ P(d|A_next) · q(s',d)

# Greedy Q：用 greedy 构建的最优 slate 计算 target
compute_target_greedy_q: r + γ · max_greedy Σ P(d|A*) · q(s',d)

# Top-K Q：用 top-k 构建的 slate 计算 target
compute_target_topk_q:   r + γ · max_topk Σ P(d|A*) · q(s',d)

# Optimal Q：穷举所有 slate 找最优 target
compute_target_optimal_q: r + γ · max_A Σ P(d|A) · q(s',d)
```

**预定义 Agent 组合**：

| Agent 名称 | Slate 构建 | Target 计算 | 特点 |
|:-----------|:-----------|:------------|:-----|
| slate_topk_sarsa | Top-K | SARSA | 快速，on-policy |
| slate_greedy_sarsa | Greedy | SARSA | 更精确的 slate，on-policy |
| slate_topk_topk_q | Top-K | Top-K Q | off-policy，快速 |
| slate_greedy_greedy_q | Greedy | Greedy Q | off-policy，更精确 |
| slate_greedy_optimal_q | Greedy | Optimal Q | 精确 target，适中推理 |
| slate_optimal_optimal_q | Optimal | Optimal Q | 最精确但最慢 |
| myopic_slate_topk_sarsa | Top-K | SARSA(γ=0) | 短视基线 |
| myopic_slate_greedy_sarsa | Greedy | SARSA(γ=0) | 短视基线 |

### 4.4 Q 值分解的数学推导

在 MNL 选择模型下，用户选择文档 d 的概率为：

```
P(d | A) = s(d) / (Σ_{d'∈A} s(d') + s_no_click)
```

其中 s(d) 是文档 d 的未归一化分数。Slate 的期望 Q 值为：

```
Q(s, A) = Σ_{d∈A} P(d|A) · q(s,d)
        = Σ_{d∈A} [s(d) · q(s,d)] / (Σ_{d'∈A} s(d') + s_no_click)
```

代码中的实现（`compute_target_topk_q`）：

```python
# scores: 每个文档的未归一化分数 s(d)
# q_values: 每个文档的 Q 值 q(s,d)
unnormalized_q = q_values * scores           # s(d) · q(s,d)
slate_q = Σ(unnormalized_q[slate])           # 分子
normalizer = Σ(scores[slate]) + s_no_click   # 分母
Q_slate = slate_q / normalizer               # Q(s, A)
```

### 4.5 Greedy Slate 构建算法

自适应贪心（`select_slate_greedy`）每次选一个文档加入 slate，选择标准是**边际价值最大化**：

```
初始: numerator = 0, denominator = s_no_click
for i = 1 to K:
    k = argmax_d [(numerator + s(d)·q(d)) / (denominator + s(d))]
    numerator += s(k)·q(k)
    denominator += s(k)
    slate.add(k)
```

每次选择使得 slate 的整体 Q 值 (numerator/denominator) 增加最多的文档。比 Top-K 更精确但更慢。

### 4.6 实验结论

- RL 规划（γ>0）比短视策略（γ=0）在长期 engagement 上有**显著提升**
- Greedy slate 构建在推理时接近 Optimal 的效果，但计算量小得多
- 分解方法对选择模型偏差有一定鲁棒性——即使假设的选择模型与真实不完全一致，仍能获得收益

---

## 5. 算法三：长期优势放大（Case Study 3）

### 5.1 核心问题

LTS 环境中，用户满意度**变化极慢**（memory_discount=0.7），导致：
- 不同动作的 Q 值差异（**advantage**）非常小
- 信噪比极低，标准 RL 难以学到有效策略
- 需要**非常长的 horizon** 才能看到策略差异

### 5.2 NPE 动态系统

```
NPE_{t+1} = 0.7 · NPE_t - 2·(clickbait - 0.5) + N(0, 0.05²)

satisfaction = σ(0.01 · NPE)

engagement ~ exp(N(μ, σ²))
μ = (clickbait · 5.0 + (1-clickbait) · 4.0) × satisfaction
```

关键数值：sensitivity=0.01 意味着 NPE 必须变化 100 个单位才能让 satisfaction 从 0.5 变到 0.73。而每步 NPE 最多变化约 1（当 clickbait=1 时减 1），所以需要 **上百步** 才能看到明显差异。

### 5.3 算法：时序聚合 (Temporal Aggregation)

**问题**：当 advantage 很小时，Q(s,a₁) ≈ Q(s,a₂)，Agent 随机选择，策略退化。

**解决方案**：降低控制频率。不是每步都重新选动作，而是每 K 步选一次，中间重复同一个动作。

```
原始: a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ ...   (每步独立决策)
聚合: a₁ a₁ a₁ a₁ a₅ a₅ a₅ a₅ ...   (K=4, 每 4 步决策一次)
```

**为什么有效**：

假设 Q(s, "推甘蓝") = 10.01，Q(s, "推巧克力") = 10.00，advantage = 0.01。

- 单步决策：几乎随机选择（差异 0.01 被噪声淹没）
- 聚合 K=4 步：累计 advantage = 4 × 0.01 = 0.04，信噪比提升 4 倍

代码中通过 `TemporalAggregationLayer` 实现：

```python
class TemporalAggregationLayer:
    # 每 K 步调用一次 base_agent.step()
    # 中间 K-1 步重复上次的 slate
    # 效果：放大 Q 值差异（advantage），提升信噪比
```

### 5.4 算法：时序正则化 (Temporal Regularization)

对频繁切换 slate 施加惩罚：

```
Reward'(t) = R(t) - λ · 𝟙[A_t ≠ A_{t-1}]
```

其中比较的是文档**特征**（不是 ID），λ 是惩罚系数。

**效果**：鼓励 Agent 坚持一种策略（比如持续推甘蓝），而不是每步都在巧克力和甘蓝之间摇摆。

### 5.5 实验结论

- 时序聚合将策略质量提升到**接近满意度完全可观测时的水平**
- 这说明部分可观测性带来的困难可以通过适当的 RL 技巧缓解
- 分层 Agent 设计的优势：`TemporalAggregationLayer` 可以包装任何基础 Agent，不需要修改算法本身

---

## 6. 分层 Agent 架构

### 6.1 设计理念

类似 Keras 的 layer 概念，将 Agent 拆为可堆叠的**特征工程层**和**算法层**：

```
原始观测 → [Layer 3] → [Layer 2] → [Layer 1: Base Agent] → Slate
              ↓             ↓              ↓
          时序聚合       历史窗口        Q-learning
```

### 6.2 三个具体 Layer

**ClusterClickStatsLayer**

```
输入: 原始观测 (user_obs, doc_obs, responses)
输出: 增强观测 + 每个主题聚类的 (曝光次数, 点击次数)

用途: 为 Bandit 算法提供统计信号
场景: Interest Exploration (Case Study 1)
```

**FixedLengthHistoryLayer**

```
输入: 当前步的观测
输出: 最近 L 步的观测拼接

用途: 给 Agent 提供时序上下文（类似 frame stacking）
场景: 需要短期记忆的任务
```

**TemporalAggregationLayer**

```
输入: 每步观测
输出: 每 K 步调用 base_agent，中间重复动作

用途: 降低控制频率，放大 advantage，提升信噪比
场景: Long-Term Satisfaction (Case Study 3)
```

### 6.3 组合示例

```python
# 用时序聚合层包装 TabularQ Agent
base_agent = TabularQAgent(...)
agent = TemporalAggregationLayer(base_agent, K=4)

# 用历史窗口 + 聚类统计包装 DQN Agent
base_agent = FullSlateQAgent(...)
agent = ClusterClickStatsLayer(
            FixedLengthHistoryLayer(base_agent, L=5))
```

---

## 7. 算法对比总览

### 7.1 三个 Case Study 对应的算法挑战

| Case Study | 挑战 | 最佳算法 | 核心思路 |
|:-----------|:-----|:---------|:---------|
| 1. Bandit | 探索隐藏的用户兴趣 | UCB1 | 置信上界平衡探索/利用 |
| 2. SlateQ | 组合动作空间爆炸 | SlateDecompQ | Q 值分解为文档级 |
| 3. LTS | 长期 advantage 信号微弱 | TemporalAggregation | 降频放大 advantage |

### 7.2 所有 Agent 能力矩阵

| Agent | 探索 | 长期规划 | 处理组合空间 | 处理部分可观测 |
|:------|:-----|:--------|:------------|:-------------|
| Random | ✓(但无效率) | ✗ | ✓(但随机) | — |
| Greedy pCTR | ✗ | ✗ | ✓ | ✗ |
| UCB1 | ✓ | ✗ | ✗(单文档) | 间接 |
| TabularQ | 通过ε-greedy | ✓ | ✗(离散化) | 通过 history |
| FullSlateQ | 通过ε-greedy | ✓ | ✗(组合爆炸) | 通过 DQN |
| **SlateDecompQ** | 通过ε-greedy | **✓** | **✓** | 通过 DQN |
| + TemporalAgg | 继承 | **增强** | 继承 | **增强** |

### 7.3 复杂度对比

| 方法 | 动作空间 | 训练复杂度 | 推理复杂度 |
|:-----|:---------|:-----------|:-----------|
| FullSlateQ | C(N,K) | O(C(N,K)) | O(C(N,K)) |
| SlateDecomp + Optimal | N (训练) | O(N) | O(C(N,K)) |
| SlateDecomp + Greedy | N (训练) | O(N) | O(N·K) |
| SlateDecomp + Top-K | N (训练) | O(N) | O(N·log K) |

---

## 8. 论文的算法贡献总结

1. **形式化框架**：将推荐系统建模为 DBN/POMDP，给出了严格的轨迹概率分解公式，为 RL 方法提供理论基础

2. **SlateQ 分解**：将组合 slate 空间分解为线性文档空间，是推荐系统 RL 可扩展性的关键突破

3. **时序聚合技巧**：通过降低控制频率放大 advantage，解决了长期满意度信号微弱的问题

4. **分层 Agent 架构**：将特征工程和算法解耦，使得不同的 RL 技巧可以正交组合

5. **仿真平台**：提供了可配置的环境来系统研究上述算法问题，填补了推荐系统 RL 缺乏标准化测试平台的空白

### 局限性

- SlateQ 的 Q 值分解依赖选择模型的特定假设（MNL 等），不适用于任意选择行为
- 时序聚合的 K 值需要手动调优，且假设最优策略是"持续推一类内容"
- 平台只支持风格化(stylized)的用户模型，不直接从真实数据拟合
- 论文未提供 sim-to-real 的迁移方法

### 后续方向

- 多用户并发仿真
- 从生产日志拟合风格化用户模型
- 支持更多交互模式（对话、搜索）
- Sim-to-real 迁移机制
