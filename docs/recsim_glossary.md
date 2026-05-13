# RecSim 核心概念词典

> 按类别整理 RecSim 论文和代码中的核心概念，帮助快速建立对推荐系统仿真的完整认知。
> 论文: [RecSim: A Configurable Simulation Platform for Recommender Systems](https://arxiv.org/abs/1909.04847)

---

## 目录

1. [强化学习基础概念](#1-强化学习基础概念)
2. [推荐系统特有概念](#2-推荐系统特有概念)
3. [RecSim 架构概念](#3-recsim-架构概念)
4. [用户建模概念](#4-用户建模概念)
5. [选择模型概念](#5-选择模型概念)
6. [LTS 环境专有概念](#6-lts-环境专有概念)
7. [Interest Evolution 环境专有概念](#7-interest-evolution-环境专有概念)
8. [Interest Exploration 环境专有概念](#8-interest-exploration-环境专有概念)
9. [Agent 与算法概念](#9-agent-与算法概念)
10. [外部基准与工具](#10-外部基准与工具)

---

## 1. 强化学习基础概念

### MDP (Markov Decision Process，马尔可夫决策过程)

强化学习的数学框架。系统在每个时间步有一个状态，Agent 选择动作，获得奖励，转移到下一个状态。关键假设：下一个状态只取决于当前状态和动作，与历史无关。

RecSim 中的对应：用户状态 = state，推荐 slate = action，用户反馈 = reward，用户状态变化 = transition。

### POMDP (Partially Observable MDP，部分可观测 MDP)

MDP 的扩展。Agent **看不到**完整的状态，只能看到一个观测值（observation）。需要通过历史观测来推断真实状态。

RecSim 中的对应：LTS 环境中 Agent 看不到 satisfaction，Interest Exploration 中 Agent 看不到用户兴趣细节。这使得问题从 MDP 变成了 POMDP。

### Episode（回合）

一次完整的交互序列，从用户进入系统到离开。在 RecSim 中，一个 episode 对应一个用户会话，终止条件通常是时间预算耗尽（`time_budget ≤ 0`）。

### Reward（奖励）

Agent 每步获得的即时反馈信号。RecSim 中不同环境的 reward 定义不同：
- LTS：engagement（参与度，如观看时长）
- Interest Evolution：clicked_watchtime（点击观看时长）
- Interest Exploration：total_clicks（是否点击，0 或 1）

### Discount Factor / Gamma（折扣因子）

衡量 Agent 对未来奖励的重视程度。γ=0 只看当下，γ=0.99 几乎同等重视未来。推荐系统的核心挑战之一就是要用高 γ 做长期规划，而不是贪心地最大化即时 reward。

### Exploration vs Exploitation（探索 vs 利用）

- **探索**：尝试不确定的选项以获取信息（如推荐用户没看过的主题）
- **利用**：选择已知最好的选项以最大化收益（如推荐用户一直点的主题）

Interest Exploration 环境专门考察这个权衡。

---

## 2. 推荐系统特有概念

### Slate（推荐列表）

一次推荐给用户的一组文档。在 RecSim 中，slate 是 Agent 的"动作"——一个整数数组，每个元素是候选文档的索引。

代码：`action = np.array([0, 3, 7])` 表示从候选集中选第 0、3、7 个文档组成 slate。

### Combinatorial Action Space（组合动作空间）

推荐系统的独特挑战。从 N 个候选文档中选 K 个组成 slate，动作空间大小为 C(N,K)。例如从 100 个候选选 5 个，有 7500 万种组合。传统 RL 方法无法处理这么大的动作空间，SlateDecompQ 通过分解方法将复杂度从 O(C(N,K)) 降为 O(N)。

### Dynamic Candidate Set（动态候选集）

现实中，推荐系统的候选内容池不断变化——新内容上线、旧内容下架。RecSim 通过 `resample_documents=True` 模拟这种动态性：每一步都重新采样候选文档集。

### Clickbait（标题党/低质诱导内容）

高即时吸引力但低实质价值的内容。用户点进去发现不值，长期降低对平台的信任。LTS 环境中用 `clickbait_score ∈ [0, 1]` 量化。

### CTR / pCTR（点击率 / 预测点击率）

Click-Through Rate。用户看到推荐后点击的概率。`GreedyPCTRAgent` 是一个短视的基线策略，只按预测点击率排序推荐，不考虑长期影响。

---

## 3. RecSim 架构概念

### Environment（环境）

仿真器的核心，编排每一步的交互。包含用户模型和文档采样器，接收 Agent 的 slate，返回观测和奖励。

代码：`simulator/environment.py` → `SingleUserEnvironment`

### User Model（用户模型）

封装用户的完整行为：状态管理、响应生成、状态转移、终止判断。

代码：`user.py` → `AbstractUserModel`

### Document Sampler（文档采样器）

生成候选文档集。每步可以重新采样以模拟动态内容池。

代码：`document.py` → `AbstractDocumentSampler`

### RecSimGymEnv（Gym 封装）

将 RecSim 环境包装为标准的 OpenAI Gym 接口（`reset()`、`step()`、`observation_space`、`action_space`），使其可以与任何标准 RL 库配合使用。

代码：`simulator/recsim_gym.py`

### Reward Aggregator（奖励聚合函数）

将用户的多维响应（点击、观看时长、是否喜欢等）压缩为单个标量 reward。不同环境定义不同的聚合方式：
- `clicked_watchtime_reward`：累计点击观看时长
- `clicked_engagement_reward`：累计参与度
- `total_clicks_reward`：累计点击次数

### Gin Config（配置系统）

Google 的超参数配置框架。通过 `@gin.configurable` 装饰器标记可配置的类，允许在不修改代码的情况下通过命令行调整参数。

```bash
--gin_bindings="LTSStaticUserSampler.memory_discount=0.9"
```

---

## 4. 用户建模概念

### User State（用户状态）

用户在某一时刻的完整内部状态。包含：
- **隐变量**（latent）：兴趣、满意度等，Agent 看不到
- **可观测变量**（observable）：人口统计、会话时长等，Agent 能看到

`create_observation()` 方法决定了哪些信息暴露给 Agent。

### Partial Observability（部分可观测性）

RecSim 的核心设计。Agent 只能看到用户状态的一个子集：

| 环境 | Agent 能看到什么 | Agent 看不到什么 |
|:-----|:---------------|:----------------|
| LTS | 空（什么都看不到） | satisfaction, NPE |
| Interest Evolution | 用户兴趣向量 | 无（兴趣可观测） |
| Interest Exploration | 用户类型 ID | 对每个主题的具体偏好 |

### User Sampler（用户采样器）

从分布中生成初始用户状态。每个 episode 开始时采样一个新用户，模拟"不同的人来使用系统"。

代码：`user.py` → `AbstractUserSampler`

### Response（用户响应）

用户对一个推荐文档的反馈。不同环境有不同的响应维度：
- `IEvResponse`：clicked, watch_time, liked, quality, cluster_id
- `LTSResponse`：clicked, engagement
- `IEResponse`：clicked

### Time Budget（时间预算）

用户在一个 session 中剩余的时间。每步消耗一些（观看视频花费时间），高质量推荐可能补回一些（用户觉得值得继续）。预算耗尽时 episode 结束。

### State Transition（状态转移）

用户状态随推荐内容而变化的动态过程。`UserModel.update_state(documents, responses)` 实现。不同环境有完全不同的转移逻辑。

---

## 5. 选择模型概念

### Choice Model（选择模型）

模拟用户面对一个 slate 时如何决定点哪个（或者不点）。

代码：`choice_model.py` → `AbstractChoiceModel`

### Multinomial Logit Model（多项式 Logit 模型）

最常用的选择模型。用户选择文档 x 的概率：

```
P(x) = exp(score(x)) / Σ_y exp(score(y))
```

score 越高，被选中概率越大，但低分文档仍有被选中的可能。本质就是对分数做 softmax。

Interest Exploration 环境使用此模型。

### Cascade Model（级联模型）

模拟用户**从上到下依次浏览** slate 的行为：

```
看第 1 个 → 以概率 p1 点击 → 点了就停
         → 没点 → 以概率 attention_prob 继续看第 2 个
                → 以概率 p2 点击 → 点了就停
                → 没点 → 继续看第 3 个 → ...
```

位置越靠后，被浏览到的概率越低。这模拟了真实推荐系统中的**位置偏差**。

Interest Evolution 环境使用此模型。

### No-Click Mass（不点击质量）

用户选择"什么都不点"的倾向。`no_click_mass` 越高，用户越容易划过整个 slate 不点任何东西。类比现实：用户看了推荐页面觉得都不感兴趣，直接关闭 app。

### Score Function（评分函数）

用户对单个文档的偏好分数。在 RecSim 中通常是用户兴趣向量和文档特征向量的**点积**：

```python
score = np.dot(user_interests, doc_features)    # I(u, d) = u · d
```

---

## 6. LTS 环境专有概念

### NPE (Net Positive Exposure，净正面曝光)

LTS 环境的核心隐变量。衡量用户历史上累计接收到的内容质量，类似一个**"被好内容/坏内容喂养的累计计分器"**。

转移公式：

```
NPE_{t+1} = γ × NPE_t − 2×(clickbait_score − 0.5) + ε
```

- `γ × NPE_t`：遗忘效应（γ=0.7），历史印象随时间衰减
- `−2×(clickbait_score − 0.5)`：推优质内容（score<0.5）NPE 上升，推 clickbait（score>0.5）NPE 下降
- `ε ~ N(0, σ²)`：用户情绪的随机波动

直觉类比：连续刷到标题党 → NPE 持续下降 → 用户越来越觉得"这 app 没意思"；穿插推荐有深度的内容 → NPE 回升 → 用户觉得"还挺有料"。

### Satisfaction（满意度）

从 NPE 推导的用户满意度，取值 (0, 1)：

```
satisfaction = sigmoid(sensitivity × NPE) = 1 / (1 + exp(−sensitivity × NPE))
```

NPE 高 → satisfaction 接近 1 → 用户活跃，所有内容 engagement 都高。
NPE 低 → satisfaction 接近 0 → 用户疲惫，推什么都没人看。

**关键**：satisfaction 是隐变量，Agent 完全看不到（`create_observation()` 返回空数组）。

### Engagement（参与度）

Agent 唯一能观测到的信号。从对数正态分布中采样：

```
engagement_mean = (clickbait × choc_mean + (1 − clickbait) × kale_mean) × satisfaction
engagement = exp(N(engagement_mean, scale))
```

Clickbait 的 `choc_mean`（默认 5.0）> `kale_mean`（默认 4.0），所以短期 engagement 更高。但它会降低 satisfaction，导致后续所有内容的 engagement 都缩水。

### Choc / Kale（巧克力 / 甘蓝）

论文用的比喻：
- **Choc（巧克力）**= clickbait，短期好吃但不健康。`clickbait_score` 接近 1。
- **Kale（甘蓝）**= 优质内容，短期没那么诱人但有长期价值。`clickbait_score` 接近 0。

核心困境：用户当下更想吃巧克力（高 engagement），但平台如果只推巧克力，用户会"吃腻"（satisfaction 下降）。好的推荐策略要像营养师一样搭配。

### Memory Discount（记忆衰减因子）

NPE 转移公式中的 γ（默认 0.7）。控制用户"遗忘"过去体验的速度。γ 越高，过去的好/坏体验影响越持久；γ 越低，用户越容易被最近的几次推荐左右。

### Sensitivity（敏感度）

`satisfaction = sigmoid(sensitivity × NPE)` 中的系数。sensitivity 越高，NPE 的微小变化就能引起 satisfaction 的剧烈波动；越低则用户越"钝感"。

---

## 7. Interest Evolution 环境专有概念

### User Interests（用户兴趣向量）

20 维向量，每个维度表示用户对一个主题的偏好程度，取值 [-1, 1]。+1 = 非常感兴趣，-1 = 厌恶。

在此环境中，兴趣是**可观测的**（`create_observation()` 直接返回兴趣向量），Agent 可以利用这个信息做精准推荐。

### Interest Drift（兴趣漂移）

用户兴趣会随推荐内容动态变化。观看一个文档后，用户兴趣以一定概率向该文档特征方向靠近或远离：

```
α(x) = (−y_intercept / x_intercept) × |x| + y_intercept
update = α × mask × (doc_features − user_interests)
```

关键特性：**越强的兴趣越不容易改变**（α 随 |interests| 减小）。类比现实：一个铁杆球迷不会因为看了一个烹饪视频就不喜欢足球了，但一个对烹饪态度中性的人可能因此产生兴趣。

### Cluster ID（聚类 ID）

每个文档属于一个主题聚类（共 20 个）。文档特征是 one-hot 编码，表示它属于哪个聚类。用于聚合分析各主题的推荐效果。

### Utility（效用）

用户从观看一个文档中获得的综合价值：

```
utility = user_quality_factor × expected_utility + document_quality_factor × doc_quality
```

高效用的推荐能**延长用户会话**（补回时间预算），低效用的推荐加速用户离开。

### Watch Time（观看时长）

用户观看视频的时间，等于 `min(time_budget, video_length)`。是 reward 的直接来源（`clicked_watchtime_reward`）。

---

## 8. Interest Exploration 环境专有概念

### User Type（用户类型）

用户被分为 N 种类型，每种类型对不同主题有不同的偏好模式。Agent 只能看到用户的类型 ID，看不到具体偏好值。

### Document Topic / Type（文档主题）

每个文档属于一个主题。不同主题有不同的"制作质量"分布——有些主题（如大众娱乐）的平均质量较高，有些主题（如小众学术）较低。

### Production Quality（制作质量）

文档的固有质量 `f_D(d)`，与用户无关。高制作质量的文档在所有用户中都有较高的点击概率。

### Affinity（亲和度）

用户对文档的最终偏好，由两部分组成：

```
affinity = g_U(u, d) + f_D(d)
         = 用户对该主题的个人偏好 + 文档制作质量
```

### Popularity Bias（流行度偏差）

短视策略（如 GreedyPCTR）会偏向推荐高制作质量的大众主题，因为这类内容对**所有**用户的点击率都不低。结果是小众但用户真正感兴趣的主题被忽略。

这是一个**相关臂赌博机问题**——不同主题的 reward 是相关的（都受制作质量影响），Agent 需要去除这种混淆因素才能发现用户的真实偏好。

### Correlated Bandits（相关臂赌博机）

经典多臂赌博机假设每条臂独立。但在推荐场景中，不同文档主题的 reward 是相关的（共享制作质量这个混淆因素）。解决这个问题需要比标准 UCB1 更精细的探索策略。

---

## 9. Agent 与算法概念

### Random Agent（随机 Agent）

最简单的基线，随机选择文档组成 slate。用于衡量"什么都不做"的效果。

### Greedy pCTR Agent（贪心 Agent）

按预测点击率从高到低排序，选 top-K 文档。是**短视策略**——只看当前这一步的点击率，不考虑长期影响。相当于只推巧克力，不管用户满意度。

### Tabular Q-Agent（表格 Q-learning）

经典 Q-learning 的表格实现。将连续的观测空间离散化为有限状态，学习每个状态-动作对的 Q 值。适合小规模问题。

### Full Slate Q-Agent（全 Slate DQN）

将整个 slate 作为一个原子动作，用 DQN 学习 Q 值。问题：slate 的组合数量太大，学习效率低。

### Slate Decomposition Q-Agent（Slate 分解 Q-learning）

**论文的重点贡献之一**。将 slate 级别的 Q 值分解为单个文档级别的 Q 值之和，独立评估每个文档的价值，再贪心选 top-K 组成 slate。将复杂度从 O(C(N,K)) 降为 O(N)。

### UCB1 (Upper Confidence Bound)

多臂赌博机算法。为每条臂维护一个置信上界：

```
UCB(a) = 平均 reward + √(2 × ln(总次数) / 该臂被拉次数)
```

被拉次数少的臂有更高的置信上界（不确定性大 → 值得探索）。自动平衡探索和利用。

### Hierarchical Agent Layers（分层 Agent）

类似 Keras 的 layer 概念，将 Agent 拆成可堆叠的层：
- **特征工程层**：如 `TemporalAggregationLayer`（时序聚合）、`FixedLengthHistoryLayer`（固定历史窗口）
- **算法层**：如 TabularQ、DQN

研究者可以独立替换任一层，实现特征工程和算法逻辑的解耦。

---

## 10. 外部基准与工具

### Atari

街机游戏模拟器，RL 领域最经典的基准之一。Agent 输入是游戏画面（像素），输出是手柄操作（上下左右、开火），目标是最大化游戏得分。DeepMind 2013 年用 DQN 在 Atari 上达到人类水平，是深度 RL 的里程碑。

代表游戏：Breakout（打砖块）、Pong（乒乓）、Space Invaders（太空侵略者）。

### MuJoCo (Multi-Joint dynamics with Contact)

物理仿真引擎，模拟机器人的关节运动。Agent 控制力矩让机器人完成走路、跑步、抓取等任务。

代表任务：HalfCheetah（猎豹跑步）、Humanoid（人形行走）、Ant（蚂蚁爬行）。

### 为什么推荐系统不能直接用 Atari / MuJoCo

| 特性 | Atari / MuJoCo | 推荐系统 |
|:-----|:--------------|:---------|
| 动作空间 | 固定的几个按键/力矩 | 从 N 个候选选 K 个，C(N,K) 组合爆炸 |
| 候选集 | 不变 | 每步都有新内容上线、旧内容下架 |
| 用户状态 | 完全可观测 | 兴趣、满意度是隐变量 |
| 多目标 | 单一得分 | 点击率、观看时长、满意度要同时权衡 |

这就是 RecSim 存在的意义——专门针对推荐系统的这些独特挑战提供仿真平台。

### OpenAI Gym

RL 环境的标准接口规范。定义了 `reset()`、`step(action)`、`observation_space`、`action_space` 等统一接口，使得任何 RL 算法可以接入任何环境。RecSim 通过 `RecSimGymEnv` 适配此接口。

### Dopamine

Google 开发的 RL 研究框架，提供 DQN 等深度 RL 算法的参考实现。RecSim 的 `FullSlateQAgent` 和 `SlateDecompQAgent` 基于 Dopamine 的 DQN 实现。

### TensorFlow

Google 的深度学习框架。RecSim 原始代码基于 TF 1.x（`tensorflow.compat.v1`），本项目已适配 TF 2.x。
