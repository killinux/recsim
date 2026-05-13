# RecSim 实验报告

> 实验日期: 2026-05-13
> 运行平台: macOS (Apple Silicon, M系列芯片)
> Python: 3.9.6 (conda env: recsim)
> TensorFlow: 2.20.0 | Dopamine-RL: 4.1.2
> 论文: [RecSim: A Configurable Simulation Platform for Recommender Systems](https://arxiv.org/abs/1909.04847)

---

## 1. 实验概述

本实验在 macOS 上成功部署并运行了 RecSim 平台的三个实验环境，对应论文 Section 5 的三个 Case Study。每个环境使用随机策略 (Random Agent) 运行 20 步，验证仿真循环的正确性并观察不同环境的 reward 特征。

---

## 2. 环境部署

### 2.1 依赖安装

```bash
conda create -n recsim python=3.9
conda activate recsim
pip install absl-py gin-config gym numpy scipy tensorflow dopamine-rl gymnasium
```

### 2.2 代码适配

原始代码依赖 TensorFlow 1.x 的 `gin.tf` 模块，在 TF 2.x 中 `tf.estimator.SessionRunHook` 已被移除，导致 `import gin.tf` 失败。

**修复方案**: 将 8 个文件中的 `import gin.tf` 替换为 `import gin`：

| 文件 | 说明 |
|:-----|:-----|
| `recsim/environments/interest_evolution.py` | 兴趣演化环境 |
| `recsim/environments/interest_exploration.py` | 兴趣探索环境 |
| `recsim/environments/long_term_satisfaction.py` | 长期满意度环境 |
| `recsim/agents/slate_decomp_q_agent.py` | Slate 分解 Q-learning |
| `recsim/agents/full_slate_q_agent.py` | 全 Slate DQN |
| `recsim/agents/dopamine/dqn_agent.py` | Dopamine DQN 适配 |
| `recsim/simulator/runner_lib.py` | 训练/评估 Runner |
| `recsim/testing/test_environment.py` | 测试环境 |

**原因**: `gin.tf` 是 `gin-config` 对 TensorFlow 的扩展封装，提供 `GinConfigSaverHook` 等 TF 1.x 特有功能。核心的 `@gin.configurable` 装饰器在 `gin` 主模块中即可使用，无需 TF 扩展。

---

## 3. 三个实验环境原理与结果

### 3.1 Long-Term Satisfaction (长期满意度)

#### 原理

论文 Section 5.3 提出的 **"巧克力 vs 甘蓝" (Choc/Kale) 困境**。

该环境建模为一个**受控连续隐马尔可夫模型 (HMM)**:

- 每个文档有一个 `clickbait_score ∈ [0, 1]`
  - 接近 1 = "巧克力" (clickbait)：高即时参与度，损害长期满意度
  - 接近 0 = "甘蓝" (优质内容)：低即时参与度，提升长期满意度

- **隐状态转移** — 净正面曝光 (NPE):

```
NPE_{t+1} = γ · NPE_t - 2·(clickbait_score - 0.5) + ε
```

其中 γ 为遗忘因子（默认 0.7），ε ~ N(0, σ²) 为噪声。clickbait 内容使 NPE 下降。

- **满意度** (隐变量，Agent 不可观测):

```
satisfaction = σ(sensitivity · NPE) = 1 / (1 + exp(-sensitivity · NPE))
```

- **参与度** (Agent 唯一可观测的信号):

```
engagement_mean = (clickbait · choc_mean + (1 - clickbait) · kale_mean) × satisfaction
engagement = exp(N(engagement_mean, engagement_scale))
```

- **核心困境**: clickbait 的 `choc_mean`(默认 5.0) > `kale_mean`(默认 4.0)，所以短期 engagement 更高。但持续推荐 clickbait 会降低 NPE → 降低 satisfaction → 最终所有内容的 engagement 都下降。

#### 实验配置

```python
{
    'slate_size': 1,           # 每次推荐 1 个文档
    'num_candidates': 5,       # 候选集大小
    'resample_documents': True, # 每步重新采样候选集
    'seed': 42
}
```

#### 实验结果

```
Step  1: reward=    2.13
Step  2: reward=    8.04
Step  3: reward=   11.23
Step  4: reward=    2.84
Step  5: reward=    4.05
Step  6: reward=   88.27    ← engagement 突发峰值
Step  7: reward=   52.69
Step  8: reward=    3.03
Step  9: reward=    1.09
Step 10: reward=    9.56
Step 11: reward=    6.35
Step 12: reward=    1.38
Step 13: reward=    2.86
Step 14: reward=    6.98
Step 15: reward=   12.06
Step 16: reward=   11.17
Step 17: reward=    4.88
Step 18: reward=    5.42
Step 19: reward=    6.53
Step 20: reward=    4.08
─────────────────────────────
Total reward: 244.64 (20 steps)
```

#### 结果分析

- **Reward 波动极大** (1.09 ~ 88.27)，反映了 engagement 的对数正态分布特性 (`exp(N(μ, σ²))`)
- Step 6-7 出现高 reward 峰值，可能是随机推荐了高 clickbait 文档
- 随机策略无法感知隐藏的 satisfaction 变化，是一个**无差别的基线**
- 20 步未结束 (`done=False`)，因为 `time_budget=60`

---

### 3.2 Interest Evolution (兴趣演化)

#### 原理

论文 Section 5.2 的 **Slate RL 分解问题**。

- 用户有一个 20 维兴趣向量 `user_interests ∈ [-1, 1]^20`
- 文档有一个 20 维特征向量，属于某个聚类
- 用户偏好分数: `I(u, d) = u · d` (点积)

**兴趣动态演化**:

当用户观看一个文档后，兴趣向量会**随机向文档特征方向靠近或远离**:

```
α(x) = (-y_intercept / x_intercept) · |x| + y_intercept   # 越强的兴趣越不易改变
target = doc_features - user_interests
if random() < positive_update_prob:
    user_interests += α · mask · target     # 靠近
else:
    user_interests -= α · mask · target     # 远离
```

**时间预算机制**:

```
budget -= watch_time                                        # 消耗
budget += α_update × watch_time × utility                  # 高质量内容补偿
```

高质量、高匹配度的推荐能延长用户会话。

**选择模型**: 使用级联选择模型 (Cascade)，用户按 Slate 顺序浏览，以概率 `attention_prob × score_scaling × score` 点击。

#### 实验配置

```python
{
    'slate_size': 2,            # 每次推荐 2 个文档
    'num_candidates': 10,       # 候选集大小
    'resample_documents': True,
    'seed': 42
}
```

#### 实验结果

```
Step  1: reward=    4.00
Step  2: reward=    0.00
Step  3: reward=    0.00
Step  4: reward=    4.00
Step  5: reward=    4.00
Step  6: reward=    0.00
Step  7: reward=    4.00
Step  8: reward=    0.00
Step  9: reward=    4.00
Step 10: reward=    4.00
Step 11: reward=    0.00
Step 12: reward=    0.00
Step 13: reward=    0.00
Step 14: reward=    4.00
Step 15: reward=    0.00
Step 16: reward=    0.00
Step 17: reward=    0.00
Step 18: reward=    0.00
Step 19: reward=    4.00
Step 20: reward=    0.00
─────────────────────────────
Total reward: 32.00 (20 steps)
```

#### 结果分析

- **Reward 呈二值分布**: 要么 4.0 (用户点击并观看了完整视频)，要么 0.0 (用户未点击任何文档)
- Reward = `clicked_watchtime`，视频长度固定为 4.0 分钟，所以点击即为 4.0
- 点击率约 45% (9/20)，反映了随机策略下的低匹配度
- 使用级联选择模型，Slate 中的文档位置靠前更容易被浏览到
- 20 步未终止，因为 `time_budget=200` 远大于消耗速度

---

### 3.3 Interest Exploration (兴趣探索)

#### 原理

论文 Section 5.1 的**隐状态 Bandit 问题**。

- 文档分为 M 个主题聚类，每个聚类有不同的"制作质量" `f_D(d)`
- 用户分为 N 种类型，每种类型对不同主题有不同的亲和度 `g_U(u, d)`
- 最终偏好: `affinity = g_U(u, d) + f_D(d)`
- 用户兴趣**不随时间变化** (静态 Bandit 设定)

**核心挑战 — 流行度偏差**:

高制作质量的主题 (如大众娱乐) 在所有用户中点击率都较高，短视的贪心策略会偏向这类内容，忽略小众但用户真正感兴趣的主题。这是一个**相关臂赌博机问题**。

**选择模型**: Multinomial Logit，`P(选择 x) = exp(affinity_x) / Σ exp(affinity_y)`

#### 实验配置

```python
{
    'slate_size': 2,            # 每次推荐 2 个文档
    'num_candidates': 10,       # 候选集大小
    'resample_documents': True,
    'seed': 42
}
```

#### 实验结果

```
Step  1: reward=    0.00
Step  2: reward=    0.00
Step  3: reward=    0.00
Step  4: reward=    1.00
Step  5: reward=    0.00
Step  6: reward=    0.00
Step  7: reward=    0.00
Step  8: reward=    0.00
Step  9: reward=    1.00
Step 10: reward=    1.00
Step 11: reward=    0.00
Step 12: reward=    1.00
Step 13: reward=    1.00
Step 14: reward=    0.00
Step 15: reward=    0.00
Step 16: reward=    0.00
Step 17: reward=    0.00
Step 18: reward=    0.00
Step 19: reward=    1.00
Step 20: reward=    0.00
─────────────────────────────
Total reward: 6.00 (20 steps)
```

#### 结果分析

- **Reward 呈 0/1 二值**: 1.0 = 用户点击了推荐的文档，0.0 = 未点击
- 点击率仅 30% (6/20)，因为随机策略无法探索到用户真正感兴趣的主题
- 论文指出，UCB1 和 Q-learning 等探索策略能显著提升点击率
- 该环境重点测试的是**探索-利用权衡**，而非长期状态变化

---

## 4. 三个环境对比

| 特性 | LTS (长期满意度) | Interest Evolution | Interest Exploration |
|:-----|:----------------|:-------------------|:---------------------|
| **论文章节** | Section 5.3 | Section 5.2 | Section 5.1 |
| **核心问题** | 短期 vs 长期权衡 | 兴趣动态演化 + 组合动作空间 | 探索 vs 利用 |
| **用户状态** | satisfaction (隐变量) | interests (可观测) | interests (隐变量) |
| **部分可观测** | 完全不可观测 | 兴趣可观测 | 兴趣不可观测 |
| **状态变化** | NPE 随推荐变化 | 兴趣随观看内容漂移 | 不变 (静态 Bandit) |
| **Reward 类型** | engagement (连续) | clicked_watchtime (离散) | total_clicks (0/1) |
| **选择模型** | 固定选第一个 | 级联 (Cascade) | Multinomial Logit |
| **Reward 范围** | 1.09 ~ 88.27 | 0 或 4.0 | 0 或 1.0 |
| **随机策略表现** | 244.64 / 20 步 | 32.00 / 20 步 | 6.00 / 20 步 |
| **终止条件** | time_budget ≤ 0 | time_budget ≤ 0 | 固定步数 |

---

## 5. 仿真循环原理

### 5.1 整体架构

```
┌────────────────────────────────────────────────────────┐
│                  RecSimGymEnv                          │
│                                                        │
│  ┌──────────┐   slate    ┌───────────────────────┐    │
│  │  Agent   │ ─────────→ │    Environment        │    │
│  │          │ ←───────── │                       │    │
│  │ step()   │  obs+reward│  ┌─ UserModel         │    │
│  └──────────┘            │  │   ├─ UserState     │    │
│                          │  │   ├─ ChoiceModel   │    │
│                          │  │   └─ ResponseModel │    │
│                          │  └─ DocumentSampler   │    │
│                          └───────────────────────┘    │
└────────────────────────────────────────────────────────┘
```

### 5.2 每步交互流程

对应 `SingleUserEnvironment.step()`:

```
1. Agent 输出 slate (文档索引数组)
       │
       ▼
2. Environment 从候选集取出对应文档
       │
       ▼
3. UserModel.simulate_response(documents)
   ├─ ChoiceModel.score_documents()   # 计算每个文档的偏好分数
   ├─ ChoiceModel.choose_item()       # 按概率采样选中项
   └─ 生成 Response (click, watch_time, engagement...)
       │
       ▼
4. UserModel.update_state(documents, responses)
   └─ 根据用户响应更新隐状态 (interests, satisfaction, budget...)
       │
       ▼
5. reward = reward_aggregator(responses)  # 将响应聚合为标量
       │
       ▼
6. obs = (user_obs, doc_obs)              # 仅返回可观测部分
   done = UserModel.is_terminal()         # 检查是否结束
```

### 5.3 部分可观测性设计

论文的核心设计之一。通过 `create_observation()` 方法控制 Agent 能看到什么:

```python
# LTS: Agent 完全看不到用户状态
class LTSUserState:
    def create_observation(self):
        return np.array([])       # 空！satisfaction 是隐变量

# Interest Evolution: Agent 能看到用户兴趣
class IEvUserState:
    def create_observation(self):
        return self.user_interests  # 20 维兴趣向量可观测

# Interest Exploration: Agent 看不到兴趣细节
class IEUserState:
    def create_observation(self):
        return np.array([self.user_type])  # 只能看到用户类型 ID
```

### 5.4 选择模型公式

**Multinomial Logit** (Interest Exploration 使用):

```
P(选择文档 x) = exp(score(x)) / Σ_y exp(score(y))
score(x) = user_interests · doc_features   (点积)
```

**级联模型** (Interest Evolution 使用):

```
P(点击位置 i) = P(未点击 1..i-1) × attention_prob × score_scaling × score(i)
```

用户按 Slate 顺序浏览，每个位置以一定概率点击，点击后停止。位置越靠后被看到的概率越低。

---

## 6. 关键结论

### 6.1 随机策略是弱基线

三个环境中随机策略的表现都不理想:
- **LTS**: 无法感知 satisfaction 的变化，可能持续推荐 clickbait 导致长期 engagement 下降
- **Interest Evolution**: 仅 45% 的点击率，无法利用已知的用户兴趣信息
- **Interest Exploration**: 仅 30% 的点击率，无法发现用户的小众偏好

### 6.2 论文指出的改进方向

| 环境 | 推荐方法 | 论文验证的优势 |
|:-----|:---------|:---------------|
| LTS | RL (Q-learning) + TemporalAggregation | 学会推荐"甘蓝"维持长期满意度 |
| Interest Evolution | SlateDecompQ | 分解组合动作空间，规划长期 engagement |
| Interest Exploration | UCB1 / Thompson Sampling | 主动探索小众兴趣，发现高价值推荐 |

### 6.3 平台的研究价值

RecSim 提供的不是一个固定的 benchmark，而是一个**可配置的实验框架**。研究者可以:
- 调整用户动态（兴趣漂移速度、满意度敏感度）
- 替换选择模型（MNL → Cascade → 自定义）
- 控制可观测性（哪些特征对 Agent 可见）
- 组合不同的 Agent 层（特征工程 + 算法解耦）

---

## 7. 运行环境信息

```
平台:           macOS (Apple Silicon)
主机名:         KT0J3Q2VL2
用户:           bytedance
项目路径:       /Users/bytedance/Desktop/hehe/research/sim/recsim
Python:         3.9.6 (conda env: recsim)
TensorFlow:     2.20.0
Dopamine-RL:    4.1.2
Gin-Config:     0.5.0
Gym:            0.25.2
NumPy:          2.0.2
SciPy:          1.13.1
```
