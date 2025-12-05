# 数学原理详解：Context-Policy Gradient & Hierarchical Retrieval

## 目录
1. [Context-Policy Gradient (CPG) 数学理论](#1-context-policy-gradient-cpg-数学理论)
2. [Hierarchical Retrieval 数学基础](#2-hierarchical-retrieval-数学基础)
3. [实现细节与关键算法](#3-实现细节与关键算法)
4. [代码问题分析与修复](#4-代码问题分析与修复)

---

## 1. Context-Policy Gradient (CPG) 数学理论

### 1.1 问题形式化

#### 传统策略梯度（参数空间）

在标准强化学习中，策略 $\pi_\theta$ 由参数 $\theta \in \mathbb{R}^d$ 参数化：

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
$$

其中：
- $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$：期望回报
- $\nabla_\theta J(\theta)$：策略梯度
- $\alpha$：学习率

**策略梯度定理**（Sutton et al., 2000）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t\right]
$$

#### Context-Policy Gradient（语义空间）

在 Training-Free GRPO 中，"策略"不是参数化的神经网络，而是**经验集合** $E \in \mathcal{L}^m$（$\mathcal{L}$ 是自然语言空间）：

$$
E_{t+1} = E_t + \alpha \cdot \nabla_E J(E_t)
$$

**核心挑战**：自然语言空间 $\mathcal{L}$ 是离散的、非可微的，无法直接计算 $\nabla_E J(E)$。

### 1.2 语义梯度的定义

#### 1.2.1 离散空间的"梯度"

我们将语义梯度定义为一组**结构化的修改操作**：

$$
\nabla_E J(E) \triangleq \{(o_i, e_{\text{target}}^i, e_{\text{new}}^i, m_i)\}_{i=1}^k
$$

其中：
- $o_i \in \{\text{add}, \text{modify}, \text{delete}, \text{strengthen}, \text{weaken}\}$：操作类型
- $e_{\text{target}}^i$：目标经验（要修改的经验）
- $e_{\text{new}}^i$：新内容
- $m_i \in [0, 1]$：梯度幅度（类似于梯度的模）

#### 1.2.2 操作的语义等价

| 离散操作 | 连续梯度类比 | 数学含义 |
|---------|------------|---------|
| `add` | $\theta_{\text{new}} = \theta + \Delta\theta$ | 增加新维度 |
| `modify` | $\theta_i \leftarrow \theta_i + \delta$ | 调整现有参数 |
| `delete` | $\theta_i = 0$ | 置零参数 |
| `strengthen` | $\theta_i \leftarrow 2\theta_i$ | 放大参数 |
| `weaken` | $\theta_i \leftarrow 0.5\theta_i$ | 衰减参数 |

### 1.3 隐式梯度估计

#### 1.3.1 LLM 作为梯度估计器

我们用 LLM $f_{\text{LLM}}$ 来**隐式估计**语义梯度：

$$
\nabla_E J(E_t) \approx f_{\text{LLM}}(E_t, \{R_i\}_{i=1}^n, \text{context})
$$

**关键思想**：通过 in-context learning，LLM 可以：
1. 分析当前经验集合 $E_t$
2. 观察奖励轨迹 $\{R_i\}$
3. 推理出改进经验的方向

这等价于一个**黑盒优化器**：

$$
f_{\text{LLM}}: \mathcal{L}^m \times \mathbb{R}^n \rightarrow \mathcal{O}^k
$$

其中 $\mathcal{O}$ 是操作空间。

#### 1.3.2 梯度提示模板

实现中的提示（context_policy_gradient.py:239-282）：

```python
prompt = f"""
Current Performance:
- Mean Reward: {mean_reward:.3f}
- Reward Variance: {reward_variance:.3f}
- Trend: {reward_trend}

Current Experiences (ranked by effectiveness):
{ranked_experiences}

Generate semantic gradient updates to improve the experience set.
"""
```

这个提示相当于构造了一个函数：

$$
\text{Prompt}(E_t, R_t) \rightarrow \text{LLM} \rightarrow \nabla_E J(E_t)
$$

### 1.4 梯度应用与更新规则

#### 1.4.1 基本更新规则

给定语义梯度 $G = \{(o_i, e_{\text{target}}^i, e_{\text{new}}^i, m_i)\}$，更新经验集：

$$
E_{t+1} = \text{Apply}(E_t, G, \alpha)
$$

其中 $\alpha$ 是学习率，通过**梯度幅度阈值**实现：

$$
\text{Apply}(E_t, (o, e_{\text{target}}, e_{\text{new}}, m), \alpha) =
\begin{cases}
E_t \cup \{e_{\text{new}}\} & \text{if } o = \text{add} \land m \cdot \alpha > \tau \\
(E_t \setminus \{e_{\text{target}}\}) \cup \{e_{\text{new}}\} & \text{if } o = \text{modify} \land m \cdot \alpha > \tau \\
E_t \setminus \{e_{\text{target}}\} & \text{if } o = \text{delete} \land m \cdot \alpha > \tau \\
E_t & \text{otherwise}
\end{cases}
$$

代码实现（context_policy_gradient.py:115-170）：

```python
def apply_gradient(self, experiences, gradients):
    for grad in gradients:
        if grad.gradient_magnitude < 0.1:  # 阈值 τ = 0.1
            continue

        if grad.operation == "add":
            experiences.append(grad.new_content)
        elif grad.operation == "modify":
            # 找到相似度 > 0.8 的经验并替换
            ...
```

#### 1.4.2 动量机制

类比 SGD with Momentum：

$$
\begin{align}
v_t &= \beta v_{t-1} + (1-\beta) \nabla_E J(E_t) \\
E_{t+1} &= E_t + \alpha v_t
\end{align}
$$

在语义空间的实现（context_policy_gradient.py:311-335）：

```python
def _apply_momentum(self, updates):
    for update in updates:
        # 找到历史上相似的更新
        similar_historical = [h for h in recent_history
                             if similar(h, update)]

        if similar_historical:
            # 动量平滑
            avg_hist_mag = mean([h.magnitude for h in similar_historical])
            update.magnitude = (1 - β) * update.magnitude + β * avg_hist_mag
```

**数学形式**：

$$
m_t^i = (1-\beta) \cdot m_t^i + \beta \cdot \frac{1}{|H|} \sum_{h \in H} m_h
$$

其中 $H$ 是历史上相似操作的集合。

### 1.5 收敛性分析（理论）

#### 1.5.1 期望改进定理

**定理**：假设 LLM 梯度估计器是无偏的：

$$
\mathbb{E}[f_{\text{LLM}}(E_t, R_t)] = \nabla_E J(E_t)
$$

且满足 Lipschitz 连续性（在语义相似度度量下），则 CPG 更新保证：

$$
\mathbb{E}[J(E_{t+1})] \geq J(E_t) - O(\alpha^2)
$$

**证明思路**（略）：
1. 泰勒展开（在语义空间定义的度量下）
2. 利用无偏性
3. 控制二阶项

**实际情况**：
- LLM 估计是有偏的（取决于 prompt 和模型能力）
- 不满足严格的 Lipschitz 条件
- 但经验上观察到单调改进（实验验证）

#### 1.5.2 经验有效性分析

实现中通过**相关性分析**估计经验有效性（context_policy_gradient.py:193-219）：

$$
\text{Eff}(e) = \frac{1}{|R_e|} \sum_{r \in R_e} r
$$

其中 $R_e = \{r_i : \text{problem}_i \text{ 与 } e \text{ 相关}\}$

**相关性判断**（简化版）：

$$
\text{Relevant}(\text{problem}, e) = \mathbb{1}[\exists w \in e : w \in \text{problem}]
$$

---

## 2. Hierarchical Retrieval 数学基础

### 2.1 层级结构形式化

#### 2.1.1 经验空间分解

将经验空间 $\mathcal{E}$ 分解为三层：

$$
\mathcal{E} = \mathcal{E}_{\text{meta}} \oplus \bigoplus_{d \in D} \mathcal{E}_d^{\text{domain}} \oplus \bigoplus_{d \in D} \bigoplus_{t \in T_d} \mathcal{E}_{d,t}^{\text{task}}
$$

其中：
- $\mathcal{E}_{\text{meta}}$：元级经验（领域无关）
- $\mathcal{E}_d^{\text{domain}}$：领域 $d$ 的经验
- $\mathcal{E}_{d,t}^{\text{task}}$：领域 $d$ 中任务 $t$ 的经验
- $D$：领域集合（math, code, web, ...）
- $T_d$：领域 $d$ 中的任务类型集合

#### 2.1.2 检索候选集构造

给定问题 $p$，领域 $d$，任务类型 $t$，构造候选集：

$$
\mathcal{C}(p, d, t) = \mathcal{E}_{\text{meta}} \cup \mathcal{E}_d^{\text{domain}} \cup \mathcal{E}_{d,t}^{\text{task}}
$$

如果 $d$ 未知，使用全部经验：

$$
\mathcal{C}(p, \text{unknown}) = \mathcal{E}
$$

### 2.2 语义相似度与嵌入

#### 2.2.1 嵌入函数

定义嵌入函数 $\phi: \mathcal{L} \rightarrow \mathbb{R}^n$，将文本映射到向量空间。

**实现中的简化版本**（hierarchical_retrieval.py:366-398）：

使用 TF-IDF 风格的词袋嵌入：

$$
\phi(e) = \text{normalize}\left(\sum_{w \in e} \mathbf{1}_w\right)
$$

其中 $\mathbf{1}_w$ 是词 $w$ 的 one-hot 向量。

**更好的选择**（生产环境）：
- Sentence-BERT: `sentence-transformers` 库
- OpenAI Embeddings: `text-embedding-ada-002`

#### 2.2.2 余弦相似度

给定两个嵌入 $\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^n$：

$$
\text{sim}(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}
$$

代码实现（hierarchical_retrieval.py:352-364）：

```python
def _cosine_similarity(self, vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
```

**几何意义**：
- $\text{sim} = 1$：完全平行（语义完全相同）
- $\text{sim} = 0$：正交（语义无关）
- $\text{sim} = -1$：反向（语义相反）

#### 2.2.3 有效性加权相似度

实际使用的相似度结合了**语义相似度**和**经验有效性**：

$$
\text{score}(e, p) = \text{sim}(\phi(e), \phi(p)) \cdot \left(1 + 0.5 \cdot \text{Eff}(e)\right)
$$

代码（hierarchical_retrieval.py:156-162）：

```python
sim = self._cosine_similarity(problem_embedding, exp.embedding)
adjusted_sim = sim * (1 + 0.5 * exp.effectiveness_score)
```

**理论依据**：
- 高相似度但低有效性的经验：得分被压制
- 中等相似度但高有效性的经验：得分提升
- 平衡 exploration（相似度）和 exploitation（有效性）

### 2.3 Maximum Marginal Relevance (MMR) 算法

#### 2.3.1 动机：多样性-相关性权衡

如果只按相似度排序，可能选出的经验都很相似（信息冗余）。MMR 通过**边际效用**平衡多样性。

#### 2.3.2 MMR 公式

给定已选集合 $S$，候选集合 $C$，选择下一个经验：

$$
e^* = \arg\max_{e \in C \setminus S} \left[\lambda \cdot \text{sim}(e, p) - (1-\lambda) \cdot \max_{s \in S} \text{sim}(e, s)\right]
$$

其中：
- $\lambda \in [0, 1]$：权衡参数
  - $\lambda = 1$：纯相关性（可能冗余）
  - $\lambda = 0$：纯多样性（可能不相关）
  - 推荐：$\lambda = 0.7$（70% 相关性，30% 多样性）

**第二项的意义**：$\max_{s \in S} \text{sim}(e, s)$ 是 $e$ 与已选经验的**最大相似度**，作为**多样性惩罚**。

#### 2.3.3 MMR 算法伪代码

```
Algorithm: MMR Selection
Input: candidates C, problem p, top_k, λ
Output: selected experiences S

1. Compute relevance scores: {sim(e, p) for e in C}
2. S = [argmax_{e in C} sim(e, p)]  # 选择最相关的
3. C = C \ S
4. while |S| < top_k and C ≠ ∅:
5.     for each e in C:
6.         max_sim = max_{s in S} sim(e, s)
7.         mmr_score[e] = λ * sim(e, p) - (1-λ) * max_sim
8.     e* = argmax_{e in C} mmr_score[e]
9.     S = S ∪ {e*}
10.    C = C \ {e*}
11. return S
```

代码实现（hierarchical_retrieval.py:304-350）：

```python
def _mmr_selection(self, candidates, top_k, lambda_param=0.7):
    selected = []
    remaining = sorted(candidates, key=lambda x: x[1], reverse=True)

    selected.append(remaining[0][0])  # 最相关的
    remaining.pop(0)

    while len(selected) < top_k and remaining:
        best_score = -float('inf')
        best_idx = 0

        for idx, (exp, relevance) in enumerate(remaining):
            max_sim = max([self._cosine_similarity(exp.embedding, s.embedding)
                          for s in selected])
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(remaining[best_idx][0])
        remaining.pop(best_idx)

    return selected
```

#### 2.3.4 复杂度分析

**时间复杂度**：
- 初始排序：$O(|C| \log |C|)$
- MMR 迭代：$O(k \cdot |C| \cdot k) = O(k^2 |C|)$
  - 外层循环：$k$ 次
  - 每次遍历候选：$O(|C|)$
  - 计算与已选的相似度：$O(k)$
- 总计：$O(|C| \log |C| + k^2 |C|)$

对于典型情况（$|C| \sim 50, k \sim 5$）：
$$
50 \log 50 + 25 \times 50 = 283 + 1250 \approx 1500 \text{ 操作}
$$

**空间复杂度**：$O(|C| \cdot n)$，其中 $n$ 是嵌入维度。

### 2.4 难度自适应 Top-K

#### 2.4.1 动机

不同难度的问题需要不同数量的经验指导：
- 简单问题：3 条经验足够
- 中等问题：5 条经验
- 困难问题：8 条经验

#### 2.4.2 难度估计

**简化版本**（基于长度，hierarchical_retrieval.py:449-454）：

$$
\text{Difficulty}(p) =
\begin{cases}
\text{easy} & \text{if } |p| < 20 \text{ words} \\
\text{medium} & \text{if } 20 \leq |p| \leq 50 \text{ words} \\
\text{hard} & \text{if } |p| > 50 \text{ words}
\end{cases}
$$

**更好的方法**：
1. 使用 LLM 评估难度
2. 基于问题的数学复杂度（嵌套层级、操作数）
3. 历史数据（该类问题的平均通过率）

#### 2.4.3 自适应映射

$$
k(\text{difficulty}) =
\begin{cases}
3 & \text{if difficulty} = \text{easy} \\
5 & \text{if difficulty} = \text{medium} \\
8 & \text{if difficulty} = \text{hard}
\end{cases}
$$

代码（hierarchical_retrieval.py:183-196）：

```python
def retrieve_by_difficulty(self, problem, difficulty, domain=None):
    top_k_map = {"easy": 3, "medium": 5, "hard": 8}
    top_k = top_k_map.get(difficulty, 5)
    return self.retrieve_experiences(problem, domain, top_k=top_k)
```

### 2.5 经验有效性更新

#### 2.5.1 指数移动平均（EMA）

跟踪每个经验的成功率，使用 EMA 平滑：

$$
s_{t+1} = \alpha \cdot r_t + (1-\alpha) \cdot s_t
$$

其中：
- $s_t$：时刻 $t$ 的成功率估计
- $r_t \in \{0, 1\}$：当前使用是否成功
- $\alpha = 0.3$：学习率（权重）

代码（hierarchical_retrieval.py:198-220）：

```python
def update_effectiveness(self, experience, success):
    experience.usage_count += 1

    alpha = 0.3
    new_value = 1.0 if success else 0.0
    experience.success_rate = alpha * new_value + (1 - alpha) * experience.success_rate

    # 综合有效性 = 成功率 + 使用频率
    usage_factor = min(experience.usage_count / 10.0, 1.0)
    experience.effectiveness_score = (
        0.7 * experience.success_rate +
        0.3 * usage_factor
    )
```

#### 2.5.2 综合有效性得分

$$
\text{Eff}(e) = 0.7 \cdot s(e) + 0.3 \cdot \min\left(\frac{n(e)}{10}, 1\right)
$$

其中：
- $s(e)$：成功率（EMA）
- $n(e)$：使用次数
- 第二项：**使用频率奖励**（鼓励探索不常用的经验）

**理论依据**：
- 高成功率但很少使用：可能是好经验，但被低估
- 经常使用但成功率不高：需要修改或删除
- 平衡 exploitation（成功率）和 exploration（使用频率）

---

## 3. 实现细节与关键算法

### 3.1 文本相似度计算

#### 3.1.1 Jaccard 相似度（词袋）

代码中的 `_similarity` 函数（context_policy_gradient.py:337-351）：

$$
\text{Jaccard}(e_1, e_2) = \frac{|W_1 \cap W_2|}{|W_1 \cup W_2|}
$$

其中 $W_i$ 是经验 $e_i$ 的词集合。

**示例**：
```
e1 = "Break complex problems into steps"
e2 = "Divide complex tasks into smaller steps"

W1 = {break, complex, problems, into, steps}
W2 = {divide, complex, tasks, into, smaller, steps}

|W1 ∩ W2| = |{complex, into, steps}| = 3
|W1 ∪ W2| = |{break, divide, complex, problems, tasks, into, smaller, steps}| = 8

Jaccard(e1, e2) = 3/8 = 0.375
```

**优点**：
- 简单、快速
- 无需预训练模型

**缺点**：
- 忽略语义（"break" 和 "divide" 是同义但不匹配）
- 忽略词序

#### 3.1.2 改进：语义嵌入相似度

使用预训练模型（如 Sentence-BERT）：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

这会捕获语义相似性：
```
similarity("Break problems", "Divide problems") ≈ 0.85
similarity("Break problems", "Eat pizza") ≈ 0.12
```

### 3.2 奖励趋势分析

#### 3.2.1 时间窗口比较

代码（context_policy_gradient.py:172-191）：

$$
\text{Trend}(R) =
\begin{cases}
\text{improving} & \text{if } \bar{R}_{\text{recent}} - \bar{R}_{\text{early}} > 0.1 \\
\text{declining} & \text{if } \bar{R}_{\text{recent}} - \bar{R}_{\text{early}} < -0.1 \\
\text{stable} & \text{otherwise}
\end{cases}
$$

其中：
- $\bar{R}_{\text{recent}} = \frac{1}{5} \sum_{i=n-5}^n R_i$：最近 5 次的平均
- $\bar{R}_{\text{early}} = \frac{1}{n-5} \sum_{i=1}^{n-5} R_i$：早期的平均

**统计检验视角**：

这相当于一个简单的**两样本 t 检验**（但未计算显著性）。

更严格的版本：

$$
t = \frac{\bar{R}_{\text{recent}} - \bar{R}_{\text{early}}}{\sqrt{\frac{s_{\text{recent}}^2}{n_{\text{recent}}} + \frac{s_{\text{early}}^2}{n_{\text{early}}}}}
$$

如果 $|t| > t_{0.05, df}$，则趋势显著。

### 3.3 经验有效性的因果推断

#### 3.3.1 当前方法的局限

代码中的有效性分析（context_policy_gradient.py:193-219）：

```python
for exp in experiences:
    relevant_rewards = []
    for problem, reward in reward_trajectory:
        if any(keyword in problem.lower() for keyword in exp.lower().split()[:5]):
            relevant_rewards.append(reward)

    effectiveness[exp] = mean(relevant_rewards) if relevant_rewards else 0.0
```

**问题**：这只是**相关性**，不是**因果性**。

可能存在**混淆变量**：
- 某些经验恰好在简单问题上使用 → 高奖励但未必有用
- 某些经验在困难问题上使用 → 低奖励但实际有帮助

#### 3.3.2 改进：倾向得分匹配（Propensity Score Matching）

**步骤**：
1. 为每个 (问题, 经验) 对建模选择概率：
   $$
   P(\text{使用 } e | \text{问题特征}) = \text{logistic}(\beta^T x)
   $$

2. 匹配使用和未使用该经验但特征相似的问题

3. 比较奖励差异（因果效应）：
   $$
   \text{ATE}(e) = \mathbb{E}[R | \text{用 } e] - \mathbb{E}[R | \text{不用 } e]
   $$

**实际实施**：需要追踪每次 rollout 使用了哪些经验（当前未追踪）。

---

## 4. 代码问题分析与修复

### 4.1 关键问题：LLM 接口不兼容

#### 4.1.1 问题定位

**llm.py 接口**：

```python
class LLM:
    def chat(self, messages_or_prompt, max_tokens=16384, temperature=0, ...):
        ...
```

**context_policy_gradient.py 调用**（第 103 行）：

```python
response = self.llm.generate(gradient_prompt, temperature=0.7)
```

**问题**：`LLM` 类没有 `generate()` 方法！

#### 4.1.2 修复方案

**方案 1**：修改 CPG 代码，使用 `chat()` 方法

**方案 2**：给 `LLM` 类添加 `generate()` 包装方法（推荐）

### 4.2 其他潜在问题

#### 4.2.1 嵌入维度不一致

在 `_simple_embedding()` 中（hierarchical_retrieval.py:366-398）：

```python
vector = np.zeros(max(len(vocab_to_idx), 100))
```

**问题**：如果词汇表增长，向量维度会变化，导致后续嵌入不可比较。

**修复**：固定维度或使用哈希技巧。

#### 4.2.2 空候选集处理

在 `retrieve_experiences()` 中（hierarchical_retrieval.py:147-151）：

```python
if not candidates:
    candidates = self.all_experiences.copy()

if not candidates:
    return []
```

**边界情况**：如果 `all_experiences` 也是空的，返回空列表是合理的。

#### 4.2.3 MMR 的相似度计算开销

在 `_mmr_selection()` 中（hierarchical_retrieval.py:335-338）：

```python
max_sim = max([self._cosine_similarity(exp.embedding, s.embedding)
               for s in selected])
```

**复杂度**：$O(k)$ 每次迭代，总共 $O(k^2)$。

对于大 $k$（如 $k=20$），这可能很慢。

**优化**：预计算嵌入矩阵，使用矩阵乘法：

```python
# 预计算
selected_embeddings = np.vstack([s.embedding for s in selected])
candidate_embedding = exp.embedding

# 批量计算相似度
sims = np.dot(selected_embeddings, candidate_embedding) / (
    np.linalg.norm(selected_embeddings, axis=1) * np.linalg.norm(candidate_embedding)
)
max_sim = np.max(sims)
```

---

## 5. 理论扩展与未来工作

### 5.1 CPG 的理论保证

#### 5.1.1 Regret Bound

定义 regret：

$$
\text{Regret}_T = \sum_{t=1}^T \left[J(E^*) - J(E_t)\right]
$$

其中 $E^*$ 是最优经验集。

**目标**：证明 $\text{Regret}_T = O(\sqrt{T})$（sublinear）。

**挑战**：
1. 语义空间没有明确的度量
2. LLM 梯度估计的噪声难以刻画
3. 经验空间是组合的、离散的

#### 5.1.2 样本复杂度

**问题**：需要多少次 rollouts 才能学到好的经验集？

**上界**：如果经验空间大小为 $|\mathcal{E}|$，naive 枚举需要 $O(|\mathcal{E}|)$ 次评估。

**CPG 的优势**：通过梯度引导，可能降低到 $O(\log |\mathcal{E}|)$（类似于梯度方法 vs. 网格搜索）。

**开放问题**：严格证明这一点。

### 5.2 Hierarchical Retrieval 的信息论视角

#### 5.2.1 互信息最大化

检索的目标可以表述为最大化：

$$
I(E_{\text{selected}}; P) = H(P) - H(P | E_{\text{selected}})
$$

其中：
- $H(P)$：问题的熵（固定）
- $H(P | E_{\text{selected}})$：给定经验后问题的条件熵

**MMR 的信息论解释**：
- 相关性项 $\sim$ 降低条件熵
- 多样性项 $\sim$ 避免冗余信息

#### 5.2.2 子模性（Submodularity）

函数 $f(S)$ 是子模的，如果：

$$
f(S \cup \{e\}) - f(S) \geq f(T \cup \{e\}) - f(T), \quad \forall S \subseteq T
$$

**定理**：互信息 $I(E_{\text{selected}}; P)$ 是子模函数。

**推论**：贪心算法（每次选最大边际收益）有 $(1 - 1/e) \approx 0.63$ 近似保证。

**MMR 与子模性**：MMR 本质上是在近似求解子模最大化。

---

## 6. 总结与建议

### 6.1 数学原理总结

| 组件 | 核心数学工具 | 关键公式 |
|------|------------|---------|
| **CPG** | 策略梯度 + 离散优化 | $E_{t+1} = E_t + \alpha \nabla_E J(E_t)$ |
| **语义梯度** | 黑盒优化 + LLM | $\nabla_E J \approx f_{\text{LLM}}(E, R)$ |
| **动量** | 指数移动平均 | $v_t = \beta v_{t-1} + (1-\beta) g_t$ |
| **层级检索** | 集合论 + 嵌入空间 | $\mathcal{C} = \mathcal{E}_{\text{meta}} \cup \mathcal{E}_d \cup \mathcal{E}_{d,t}$ |
| **MMR** | 子模优化 | $\max \lambda \cdot \text{rel} - (1-\lambda) \cdot \text{div}$ |
| **有效性** | 贝叶斯更新 / EMA | $s_{t+1} = \alpha r_t + (1-\alpha) s_t$ |

### 6.2 实现建议

#### 6.2.1 必须修复的问题

1. **LLM 接口不兼容**：
   - 添加 `generate()` 方法到 `LLM` 类
   - 或修改所有 `generate()` 调用为 `chat()`

2. **嵌入维度固定**：
   - 使用固定大小的词汇表
   - 或迁移到预训练嵌入模型

#### 6.2.2 推荐改进

1. **使用 Sentence-BERT 嵌入**：
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   ```

2. **向量数据库**（大规模场景）：
   ```python
   import faiss
   index = faiss.IndexFlatIP(embedding_dim)  # Inner Product
   index.add(experience_embeddings)
   D, I = index.search(query_embedding, top_k)
   ```

3. **经验使用追踪**：
   - 记录每次 rollout 使用了哪些经验
   - 用于因果推断有效性

4. **A/B 测试框架**：
   - 对比不同超参数（$\lambda$, $\alpha$, $\beta$）
   - 科学验证改进效果

### 6.3 理论研究方向

1. **收敛性证明**：
   - 在适当的语义度量下证明 CPG 收敛
   - 刻画样本复杂度

2. **最优超参数**：
   - 理论推导最优学习率 $\alpha$
   - 自适应动量 $\beta$ 调度

3. **多任务迁移学习**：
   - 形式化跨域经验迁移
   - 元学习视角的理论分析

---

**参考文献**：

1. Sutton & Barto (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. Carbonell & Goldstein (1998). *The use of MMR, diversity-based reranking for reordering documents and producing summaries*. SIGIR.
3. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL.
4. Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
5. Krause & Golovin (2014). *Submodular Function Maximization*. Tractability (book chapter).

---

**版本**: 1.0
**日期**: 2024-11
**作者**: Claude Code Mathematical Analysis Team
