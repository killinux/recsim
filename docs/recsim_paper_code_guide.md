# RecSim 论文与代码对照解读

> 论文: [RecSim: A Configurable Simulation Platform for Recommender Systems](https://arxiv.org/abs/1909.04847)
> 作者: Eugene Ie, Chih-wei Hsu, Martin Mladenov, Vihan Jain, Sanmit Narvekar, Jing Wang, Rui Wu, Craig Boutilier (Google Research, 2019)

---

## 目录

1. [项目概述](#1-项目概述)
2. [代码目录结构](#2-代码目录结构)
3. [核心架构：论文到代码的映射](#3-核心架构论文到代码的映射)
4. [仿真交互循环](#4-仿真交互循环)
5. [四大核心组件详解](#5-四大核心组件详解)
   - 5.1 [用户模型 (User Model)](#51-用户模型)
   - 5.2 [文档模型 (Document Model)](#52-文档模型)
   - 5.3 [选择模型 (Choice Model)](#53-选择模型)
   - 5.4 [推荐 Agent](#54-推荐-agent)
6. [三个实验环境 (Case Studies)](#6-三个实验环境)
   - 6.1 [Interest Exploration — 兴趣探索](#61-interest-exploration--兴趣探索)
   - 6.2 [Interest Evolution — 兴趣演化](#62-interest-evolution--兴趣演化)
   - 6.3 [Long-Term Satisfaction — 长期满意度](#63-long-term-satisfaction--长期满意度)
7. [分层 Agent 设计](#7-分层-agent-设计)
8. [Gym 封装与实验运行](#8-gym-封装与实验运行)
9. [关键数学公式汇总](#9-关键数学公式汇总)

---

## 1. 项目概述

RecSim 是 Google Research 开发的**推荐系统仿真平台**。论文的核心动机是：

- 推荐系统本质上是一个**序列决策问题**，用户状态（兴趣、满意度）会随推荐内容动态演变
- 强化学习 (RL) 是解决该问题的自然方法，但在真实生产系统上做 RL 实验成本高、风险大
- 已有的 RL 基准环境（Atari、MuJoCo）无法捕捉推荐系统的独特挑战：组合动作空间、部分可观测性、动态候选集等

RecSim 通过提供可配置的仿真环境，让研究者能快速构建并测试不同的推荐场景，而不需要接入真实用户。

---

## 2. 代码目录结构

```
recsim/
├── __init__.py
├── agent.py                          # Agent 抽象基类
├── user.py                           # 用户状态与动态建模基类
├── document.py                       # 文档与候选集基类
├── choice_model.py                   # 用户选择模型（MNL, Cascade 等）
├── utils.py                          # 指标聚合工具
├── main.py                           # 实验入口脚本
│
├── simulator/                        # 仿真框架
│   ├── environment.py                # 环境核心（交互循环实现）
│   ├── recsim_gym.py                 # OpenAI Gym 接口封装
│   └── runner_lib.py                 # 训练/评估循环管理
│
├── agents/                           # Agent 具体实现
│   ├── random_agent.py               # 随机推荐（基线）
│   ├── tabular_q_agent.py            # 表格 Q-learning
│   ├── full_slate_q_agent.py         # 全 Slate DQN
│   ├── slate_decomp_q_agent.py       # Slate 分解 Q-learning
│   ├── greedy_pctr_agent.py          # 贪心 pCTR 策略
│   ├── cluster_bandit_agent.py       # 聚类 Bandit
│   ├── agent_utils.py                # Agent 辅助工具
│   ├── bandits/                      # Bandit 算法库
│   │   ├── algorithms.py             #   UCB1, Thompson Sampling
│   │   └── glm_algorithms.py         #   GLM Bandit
│   ├── dopamine/                     # Dopamine DQN 集成
│   │   └── dqn_agent.py
│   └── layers/                       # 分层 Agent 组件
│       ├── abstract_click_bandit.py
│       ├── sufficient_statistics.py
│       ├── cluster_click_statistics.py
│       ├── fixed_length_history.py
│       └── temporal_aggregation.py
│
├── environments/                     # 三个示例实验环境
│   ├── interest_exploration.py       # Case Study 1: 兴趣探索
│   ├── interest_evolution.py         # Case Study 2: 兴趣演化
│   └── long_term_satisfaction.py     # Case Study 3: 长期满意度
│
├── testing/                          # 测试工具
│   └── test_environment.py
└── colab/                            # Colab 教程 notebook
```

---

## 3. 核心架构：论文到代码的映射

论文 Figure 1 将系统建模为一个**动态贝叶斯网络 (DBN)**，包含以下随机变量在每个时间步 t 的交互：

```
时间步 t:

  D_t (候选文档集)  ──→  Agent 观测
  z_t (用户隐状态)  ──→  o_t (用户可观测状态) ──→  Agent 观测
                                                        │
                                                        ▼
                                                   A_t (Slate)
                                                        │
                                                        ▼
  z_t + D_t + A_t  ──→  c_t (用户选择/响应)
                                │
                                ▼
  z_t + c_t  ──→  z_{t+1} (用户状态转移)
```

### 代码映射表

| 论文符号/概念 | 代码位置 | 说明 |
|:------------|:---------|:-----|
| z_t (用户隐状态) | `user.AbstractUserState` | 包含兴趣、满意度等隐变量 |
| o_t (可观测状态) | `UserState.create_observation()` | 仅返回 Agent 可见的子集 |
| D_t (候选文档集) | `document.CandidateSet` | 每步可选择重新采样 |
| d (单个文档) | `document.AbstractDocument` | 也有可观测/隐含特征之分 |
| A_t (Slate) | Agent 输出的整数索引数组 | 从候选集中选 K 个文档 |
| c_t (用户选择) | `choice_model.AbstractChoiceModel` | 基于分数采样选中项 |
| r_t (用户响应) | `user.AbstractResponse` | 点击、观看时长、是否喜欢等 |
| 状态转移 P(z_{t+1}\|z_t,c_t) | `UserModel.update_state()` | 基于用户响应更新隐状态 |
| 终止判断 | `UserModel.is_terminal()` | 如时间预算耗尽 |

---

## 4. 仿真交互循环

### 论文描述

论文 Section 3 定义了轨迹概率的因式分解:

```
P(τ) = P(z_0) · ∏_t [ P(D_t) · P(o_t|z_t) · π(A_t|o_t,D_t) · P(c_t|z_t,A_t,D_t) · P(z_{t+1}|z_t,c_t) ]
```

### 代码实现: `simulator/environment.py` — `SingleUserEnvironment`

```python
# --- reset(): 初始化一个 episode ---
def reset(self):
    self._user_model.reset()                              # 采样初始用户 z_0
    user_obs = self._user_model.create_observation()      # 生成 o_0
    self._do_resample_documents()                         # 采样 D_0
    self._current_documents = self._candidate_set.create_observation()
    return (user_obs, self._current_documents)            # 返回给 Agent

# --- step(slate): 执行一步交互 ---
def step(self, slate):
    documents = self._candidate_set.get_documents(mapped_slate)  # 解析 A_t → 文档

    responses = self._user_model.simulate_response(documents)    # 模拟 c_t（含选择模型）

    self._user_model.update_state(documents, responses)          # 状态转移 z_t → z_{t+1}

    user_obs = self._user_model.create_observation()             # 新的 o_{t+1}
    done = self._user_model.is_terminal()                        # 是否结束

    if self._resample_documents:
        self._do_resample_documents()                            # 新的 D_{t+1}

    return (user_obs, self._current_documents, responses, done)
```

关键设计：**Agent 只能看到 `user_obs` 和 `doc_obs`**，它们是经过 `create_observation()` 过滤后的部分信息。环境内部使用完整状态来计算响应和状态转移。

---

## 5. 四大核心组件详解

### 5.1 用户模型

**文件**: `recsim/user.py`

用户模型由四个抽象类组成：

```
AbstractUserState          用户状态（兴趣向量、满意度、时间预算等）
  ├── create_observation()    仅暴露可观测部分
  └── score_document(doc)     计算用户对文档的偏好分数

AbstractResponse           用户响应（点击、观看时长、是否喜欢等）
  └── create_observation()    转为可观测张量

AbstractUserSampler        用户采样器（从分布中生成初始用户状态）
  └── sample_user()           采样一个新用户

AbstractUserModel          用户动态模型（封装状态+转移+响应）
  ├── simulate_response(docs)   通过 choice model 模拟用户选择
  ├── update_state(docs, resp)  状态转移：z_t → z_{t+1}
  ├── is_terminal()             判断 session 是否结束
  └── create_observation()      返回可观测的用户特征
```

**论文核心概念 — 部分可观测性**:

论文 Section 3.1 强调，Agent 只能观测到用户状态的一个子集。这在代码中通过 `create_observation()` 实现：

```python
# Interest Evolution: 用户兴趣是可观测的
class IEvUserState:
    def create_observation(self):
        return self.user_interests          # Agent 能看到兴趣向量

# Long-Term Satisfaction: 满意度是隐变量
class LTSUserState:
    def create_observation(self):
        return np.array([])                 # Agent 什么都看不到！
```

### 5.2 文档模型

**文件**: `recsim/document.py`

```
AbstractDocument           单个文档/内容
  ├── create_observation()    可观测特征（主题分布、标签等）
  └── doc_id()                唯一标识符

AbstractDocumentSampler    文档采样器
  ├── sample_document()       从生成过程中采样新文档
  └── update_state()          文档状态更新（如播放量增加）

CandidateSet               候选文档集合
  ├── add_document()          添加文档
  ├── get_documents(ids)      按 ID 获取文档
  └── create_observation()    返回所有文档的可观测特征
```

论文强调的**动态随机动作空间**：候选文档集可以在每一步重新采样（`resample_documents=True`），模拟新内容持续产出的真实场景。

### 5.3 选择模型

**文件**: `recsim/choice_model.py`

论文 Section 3.1 讨论了用户面对 Slate 时如何选择。代码实现了以下选择模型：

#### Multinomial Logit 模型

```python
class MultinomialLogitChoiceModel:
    def score_documents(self, user_state, doc_obs):
        logits = [user_state.score_document(doc) for doc in doc_obs]
        logits.append(self._no_click_mass)      # "不点击"选项
        all_scores = softmax(logits)             # p(x) = exp(x) / Σexp(y)
```

**公式**: p(x) = exp(score(x)) / Σ_y exp(score(y))

其中 score(x) = **u** · **d**（用户兴趣向量与文档特征向量的点积），对应论文中的 I(u, d)。

#### Cascade 模型

```python
class CascadeChoiceModel:
    def _positional_normalization(self, scores):
        score_no_click = 1.0
        for i in range(len(scores)):
            s = score_scaling * scores[i]
            scores[i] = score_no_click * attention_prob * s
            score_no_click *= (1.0 - attention_prob * s)
```

用户按 Slate 顺序从上到下浏览，每个位置以概率 `attention_prob * score_scaling * score(i)` 点击，一旦点击就停止浏览。这模拟了用户注意力的位置衰减效应。

#### 选择模型类层次

```
AbstractChoiceModel
└── NormalizableChoiceModel
    ├── MultinomialLogitChoiceModel          # softmax 选择
    ├── MultinomialProportionalChoiceModel   # 比例选择
    └── CascadeChoiceModel                   # 级联选择（位置感知）
        ├── ExponentialCascadeChoiceModel     #   exp(score) 级联
        └── ProportionalCascadeChoiceModel   #   线性 score 级联
```

### 5.4 推荐 Agent

**文件**: `recsim/agent.py` + `recsim/agents/`

```
AbstractRecommenderAgent
├── AbstractEpisodicRecommenderAgent        # 基于 episode 的 Agent
│   ├── RandomAgent                         #   随机推荐
│   ├── TabularQAgent                       #   表格 Q-learning
│   ├── GreedyPCTRAgent                     #   贪心 pCTR
│   ├── FullSlateQAgent                     #   全 Slate DQN
│   └── SlateDecompQAgent                   #   Slate 分解 Q-learning
└── AbstractHierarchicalAgentLayer          # 可堆叠的 Agent 层
    ├── AbstractClickBanditLayer
    └── SufficientStatisticsLayer
```

Agent 的核心接口：

```python
class AbstractRecommenderAgent:
    def begin_episode(self, observation):   # episode 开始，返回初始 slate
        ...
    def step(self, reward, observation):    # 收到反馈，返回下一个 slate
        ...
    def end_episode(self, reward, obs):     # episode 结束
        ...
```

**论文重点讨论的组合动作空间问题**：从 N 个候选中选 K 个组成 Slate，动作空间大小为 C(N,K)，随 N 增长迅速爆炸。`SlateDecompQAgent` 通过分解方法独立评估每个文档，将复杂度从 O(C(N,K)) 降为 O(N)。

---

## 6. 三个实验环境

### 6.1 Interest Exploration — 兴趣探索

**文件**: `recsim/environments/interest_exploration.py`

**论文 Section 5.1**: 隐状态 Bandit 问题。

- **场景**: 文档分为多个主题聚类，用户对不同主题有不同偏好
- **挑战**: 用户兴趣是**隐变量**，Agent 只能通过点击反馈间接推断
- **关键特性**: 用户兴趣**不随时间变化**（静态 Bandit），核心在于探索-利用权衡
- **实验结论**: UCB1 和 Q-learning 显著优于随机和贪心策略

### 6.2 Interest Evolution — 兴趣演化

**文件**: `recsim/environments/interest_evolution.py`

**论文 Section 5.2**: Slate RL 分解问题。

- **场景**: 用户兴趣**随推荐内容动态演化**
- **用户兴趣更新** (第505-556行):

```python
# 步长与当前兴趣强度反相关——越强的兴趣越难改变
alpha = (-y_intercept / x_intercept) * |interests| + y_intercept

# 更新方向随机（取决于兴趣的正面概率）
target = doc.features - user_interests
if random() < positive_update_prob:
    user_interests += alpha * mask * target     # 向文档特征靠拢
else:
    user_interests -= alpha * mask * target     # 远离文档特征

user_interests = clip(user_interests, -1.0, 1.0)
```

- **时间预算机制**:

```python
# 消耗 = 观看时间
time_budget -= watch_time
# 补偿 = 效用 × 观看时间 × 更新系数
time_budget += user_update_alpha * watch_time * received_utility

# 其中 received_utility 由用户偏好和文档质量共同决定:
received_utility = user_quality_factor * expected_utility
                 + document_quality_factor * doc.quality
```

这意味着推荐高质量、高匹配度的内容能延长用户会话，形成正反馈循环。

- **Slate 动作空间**: 使用 `MultinomialProportionalChoiceModel`，用 Slate 分解方法降低组合复杂度

### 6.3 Long-Term Satisfaction — 长期满意度

**文件**: `recsim/environments/long_term_satisfaction.py`

**论文 Section 5.3**: "巧克力 vs 甘蓝" (Choc/Kale) 困境。

这是论文最具洞察力的环境，实现了一个**受控连续隐马尔可夫模型 (HMM)**:

- **文档特征**: 每个文档只有一个特征 `clickbait_score ∈ [0, 1]`
  - 接近 1 = "巧克力"（clickbait，高即时参与度，损害长期满意度）
  - 接近 0 = "甘蓝"（优质内容，低即时参与度，提升长期满意度）

- **隐状态转移** — 净正面曝光 (NPE):

```python
NPE_{t+1} = memory_discount * NPE_t
           - 2 * (clickbait_score - 0.5)        # clickbait 降低 NPE
           + N(0, innovation_stddev)             # 噪声
```

当 `clickbait_score > 0.5` 时，`-2*(clickbait_score - 0.5)` 为负，NPE 下降。

- **满意度** (从 NPE 推导):

```python
satisfaction = sigmoid(sensitivity * NPE)       # ∈ (0, 1)
```

- **参与度** (Agent 唯一能观测到的信号):

```python
engagement_loc = clickbait_score * choc_mean
               + (1 - clickbait_score) * kale_mean
engagement_loc *= satisfaction                   # 满意度调制参与度

engagement = exp(N(engagement_loc, engagement_scale))
```

- **核心困境**: 推荐 clickbait 能获得高即时 engagement (因为 `choc_mean > kale_mean`)，但会降低 NPE → 降低 satisfaction → 最终**所有内容的 engagement 都下降**。Agent 必须在短期收益和长期健康之间权衡。

- **部分可观测性**: `create_observation()` 返回空数组，Agent 完全看不到 satisfaction，只能通过 engagement 的变化趋势间接推断。

---

## 7. 分层 Agent 设计

**文件**: `recsim/agents/layers/`

论文 Section 4.1 提出了类似 Keras 的分层 Agent 架构，将不同的关注点解耦:

```
原始观测  →  [特征工程层]  →  [算法层]  →  Slate 输出
```

### 可用的层

| 层 | 文件 | 功能 |
|:--|:-----|:-----|
| SufficientStatistics | `sufficient_statistics.py` | 跟踪每个聚类的点击计数 |
| ClusterClickStats | `cluster_click_statistics.py` | 聚类级别的点击统计 |
| FixedLengthHistory | `fixed_length_history.py` | 维护固定长度的交互历史窗口 |
| TemporalAggregation | `temporal_aggregation.py` | 对用户响应做时序聚合 |
| AbstractClickBandit | `abstract_click_bandit.py` | 用 Bandit 混合多个基础 Agent |

### 设计理念

- 每个层都继承自 `AbstractHierarchicalAgentLayer`
- 层可以任意组合堆叠（装饰器模式）
- 研究者可以独立替换特征工程层或算法层
- 例如：用 `TemporalAggregationLayer` 包装 `TabularQAgent`，让 Q-learning 能基于时序聚合特征做决策

---

## 8. Gym 封装与实验运行

### RecSimGymEnv

**文件**: `recsim/simulator/recsim_gym.py`

将 RecSim 环境包装为标准的 OpenAI Gym 接口:

```python
class RecSimGymEnv(gym.Env):
    def __init__(self, environment, reward_aggregator, ...):
        self.environment = environment
        self._reward_aggregator = reward_aggregator

    def step(self, action):
        user_obs, doc_obs, responses, done = self.environment.step(action)
        reward = self._reward_aggregator(responses)    # 将响应聚合为标量奖励
        obs = self._build_observation(user_obs, doc_obs, responses)
        return obs, reward, done, info
```

**奖励聚合函数** 是可配置的，每个环境定义自己的聚合方式:
- Interest Evolution: `clicked_watchtime_reward` — 总点击观看时长
- LTS: `clicked_engagement_reward` — 总参与度

### Runner

**文件**: `recsim/simulator/runner_lib.py`

管理训练/评估循环，支持:
- `TrainRunner`: 交替执行训练和评估
- `EvalRunner`: 仅评估已训练的 Agent
- TensorBoard 日志记录
- Checkpoint 保存和恢复
- Episode 数据序列化（TF SequenceExample，支持离线 Batch RL）

### 运行实验

```bash
python -m recsim.main \
  --base_dir=/tmp/recsim \
  --agent_name=full_slate_q \
  --environment_name=interest_evolution \
  --gin_bindings="runner_lib.Runner.max_steps_per_episode=100" \
  --gin_bindings="runner_lib.TrainRunner.num_iterations=10"
```

---

## 9. 关键数学公式汇总

### 用户-文档偏好 (所有环境通用)

```
I(u, d) = u · d     (用户兴趣向量与文档特征向量的点积)
```

代码: `IEvUserState.score_document()` → `np.dot(self.user_interests, doc_obs)`

### Multinomial Logit 选择概率

```
P(选择文档 x) = exp(score(x)) / Σ_y exp(score(y))
```

代码: `MultinomialLogitChoiceModel.score_documents()` → `softmax(logits)`

### Cascade 选择概率

```
P(点击位置 i) = P(未点击前 i-1 项) × attention_prob × score_scaling × score(i)
```

代码: `CascadeChoiceModel._positional_normalization()`

### Interest Evolution 兴趣更新

```
α(x) = (-y_intercept / x_intercept) × |x| + y_intercept
Δinterests = α × mask × (doc_features - user_interests)
```

更新方向以概率 `(interests + 1) / 2 · mask` 为正方向，否则为反方向。

### Interest Evolution 时间预算更新

```
budget_{t+1} = budget_t - watch_time + α_update × watch_time × utility
utility = quality_factor_u × expected_utility + quality_factor_d × doc_quality
```

### LTS 净正面曝光 (NPE) 转移

```
NPE_{t+1} = γ × NPE_t - 2 × (clickbait_score - 0.5) + ε,  ε ~ N(0, σ²)
```

### LTS 满意度

```
satisfaction = σ(sensitivity × NPE) = 1 / (1 + exp(-sensitivity × NPE))
```

### LTS 参与度生成

```
μ = (clickbait × choc_mean + (1 - clickbait) × kale_mean) × satisfaction
σ = clickbait × choc_stddev + (1 - clickbait) × kale_stddev
engagement = exp(N(μ, σ²))
```

---

## 附录: 依赖与配置

### 核心依赖

- `tensorflow 1.15` — 深度学习框架
- `dopamine-rl >= 2.0.5` — Google 的 RL 库（DQN 等）
- `gin-config` — 超参数配置框架
- `gym` — 环境标准接口
- `numpy`, `scipy` — 数值计算

### Gin 配置示例

Gin 允许在不修改代码的情况下调整参数:

```python
# 代码中标记可配置:
@gin.configurable
class UtilityModelUserSampler:
    def __init__(self, document_quality_factor=1.0, ...):
        ...

# 命令行覆盖:
--gin_bindings="UtilityModelUserSampler.document_quality_factor=2.0"
```
