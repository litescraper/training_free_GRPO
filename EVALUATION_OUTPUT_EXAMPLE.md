# 评测输出示例与使用指南

## 快速开始

### 步骤 1: 运行 Baseline 评测

首先使用原始方法（main.py）建立基线：

```bash
python main.py \
    --mode agent \
    --domain math \
    --experiment_name AIME24_baseline \
    --dataset AIME24 \
    --experience_file data/math/train/DAPO100/step_3/experiences.json \
    --rollout_concurrency 5 \
    --pass_k 32
```

**注意**: 将 `--rollout_concurrency` 从 128 降低到 5 以避免 API 限流。

---

### 步骤 2: 运行增强版评测（自动对比）

```bash
python -m training_free_grpo.eval_cpg_hierarchical \
    --mode agent \
    --domain math \
    --experiment_name AIME24_enhanced \
    --dataset AIME24 \
    --library_file experiences_library.json \
    --use_difficulty_adaptive \
    --rollout_concurrency 5 \
    --pass_k 32
```

脚本会自动查找以下位置的 baseline 结果：
- `data/math/eval/AIME24_baseline_results.json`
- `data/math/eval/AIME24_baseline.json`
- `data/math/eval/baseline_results.json`

或者手动指定 baseline：

```bash
python -m training_free_grpo.eval_cpg_hierarchical \
    --mode agent \
    --domain math \
    --experiment_name AIME24_enhanced \
    --dataset AIME24 \
    --library_file experiences_library.json \
    --use_difficulty_adaptive \
    --baseline_file data/math/eval/AIME24_baseline_results.json \
    --rollout_concurrency 5 \
    --pass_k 32
```

---

## 输出示例

评测完成后，你会看到以下格式的输出：

```
======================================================================
EVALUATION SUMMARY - AIME24
======================================================================

📊 Main Results:
  • Dataset: AIME24
  • Pass@32: 63.25%
  • Average Reward: 0.680
  • Avg Tool Calls: 3.20
  • Avg Experiences Retrieved: 5.2

======================================================================
📈 COMPARISON WITH BASELINE
======================================================================

┌────────────────────────────────────────────────────────────────────┐
│ Metric                         │     Baseline │     Enhanced │   Change │
├────────────────────────────────────────────────────────────────────┤
│ 🟢 Pass@32                      │       58.00% │       63.25% │   +5.3% │
│ 🟢 Average Reward               │        0.612 │        0.680 │  +0.068 │
│ 🟢 Avg Experiences              │         50.0 │          5.2 │   -44.8 │
└────────────────────────────────────────────────────────────────────┘

✨ Key Improvements:
  • Accuracy improved by +9.1% (+5.3%)
  • Context reduced by 89.6% (44.8 fewer experiences)
  • Estimated token savings: ~1344 tokens per problem
  • Efficiency gain: 5.07% accuracy per unit context

======================================================================
📊 Performance by Difficulty Level
======================================================================

  Easy    : ████████████████████████████████░░░░░░░░  80.0% (8/10)
  Medium  : ████████████████████████░░░░░░░░░░░░░░░░  60.0% (12/20)
  Hard    : ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  30.0% (3/10)

======================================================================
💰 Token Efficiency Analysis
======================================================================

  • Baseline context: ~1500 tokens (50 experiences × 30)
  • Current context: ~156 tokens (5.2 experiences × 30)
  • Token savings: ~1344 tokens per problem (89.6% reduction)

  Estimated API cost (input only, GPT-4):
  • Baseline: $1.80 (40 problems × 1500 tokens)
  • Current: $0.19 (40 problems × 156 tokens)
  • Savings: $1.61 (89.6% reduction)

======================================================================
✅ Evaluation Complete!
======================================================================

📁 Output files:
  • Results: data/math/eval/AIME24_enhanced_results.json
  • Retrieval stats: data/math/eval/AIME24_enhanced_retrieval_stats.json
  • Detailed rollouts: data/math/eval/AIME24_enhanced.jsonl
```

---

## 关键指标解读

### 1. Pass@32 改进

```
Pass@32: 58.00% → 63.25% (+5.3%)
```

**含义**：
- Baseline: 32 次尝试中，58% 的问题至少成功一次
- Enhanced: 32 次尝试中，63.25% 的问题至少成功一次
- **提升**: +5.25 百分点（相对提升 9.1%）

**结论**: 使用层级经验库后，准确率提高了 **5.3%**

---

### 2. 上下文效率

```
Avg Experiences: 50.0 → 5.2 (-44.8)
Token savings: ~1344 tokens per problem (89.6% reduction)
```

**含义**：
- Baseline: 每个问题使用所有 50 条经验
- Enhanced: 平均只检索 5.2 条相关经验
- **减少**: 89.6% 的上下文长度

**结论**: 在提升准确率的同时，**减少了 90% 的 token 使用**

---

### 3. 成本节省

```
API Cost: $1.80 → $0.19 (savings: $1.61, 89.6% reduction)
```

**含义**：
- 对于 40 个问题的评测
- Baseline 成本: $1.80
- Enhanced 成本: $0.19
- **节省**: $1.61（基于 GPT-4 输入定价）

**结论**: **节省了 90% 的 API 成本**

---

### 4. 效率增益

```
Efficiency gain: 5.07% accuracy per unit context
```

**含义**：
- 这是一个综合指标，衡量"准确率提升"与"上下文减少"的比率
- 计算公式: `pass_k_improvement / (current_context / baseline_context)`
- 5.07% 表示每单位上下文获得了 5.07% 的准确率提升

**结论**: 层级检索不仅更准确，而且**更高效**

---

### 5. 难度级别分析

```
Easy:    80.0% (8/10)   ████████████████████████████████
Medium:  60.0% (12/20)  ████████████████████████
Hard:    30.0% (3/10)   ████████████
```

**含义**：
- Easy 问题: 80% 成功率（表现优秀）
- Medium 问题: 60% 成功率（中等表现）
- Hard 问题: 30% 成功率（需要改进）

**洞察**:
- 系统在简单问题上表现出色
- 复杂问题仍有较大改进空间
- 可以通过增加 hard 问题的检索数量来提升

---

## 实验场景示例

### 场景 1: 快速验证（小规模测试）

```bash
# Baseline
python main.py \
    --experiment_name quick_baseline \
    --dataset AIME24 \
    --experience_file experiences.json \
    --dataset_truncate 10 \
    --pass_k 4 \
    --rollout_concurrency 2

# Enhanced
python -m training_free_grpo.eval_cpg_hierarchical \
    --experiment_name quick_enhanced \
    --dataset AIME24 \
    --library_file library.json \
    --use_difficulty_adaptive \
    --dataset_truncate 10 \
    --pass_k 4 \
    --rollout_concurrency 2
```

**预计时间**: ~10 分钟
**用途**: 快速验证配置正确，初步对比结果

---

### 场景 2: 完整评测（生产级）

```bash
# Baseline
python main.py \
    --experiment_name AIME24_baseline \
    --dataset AIME24 \
    --experience_file experiences.json \
    --pass_k 32 \
    --rollout_concurrency 10

# Enhanced
python -m training_free_grpo.eval_cpg_hierarchical \
    --experiment_name AIME24_enhanced \
    --dataset AIME24 \
    --library_file library.json \
    --use_difficulty_adaptive \
    --baseline_file data/math/eval/AIME24_baseline_results.json \
    --pass_k 32 \
    --rollout_concurrency 10
```

**预计时间**: ~2-4 小时（取决于并发度和 API 限额）
**用途**: 获得准确的 Pass@32 指标，用于论文或报告

---

### 场景 3: 多数据集对比

```bash
# 对比 3 个数据集
for dataset in AIME24 MATH500 GSM8K; do
    # Baseline
    python main.py \
        --experiment_name ${dataset}_baseline \
        --dataset ${dataset} \
        --experience_file experiences.json \
        --pass_k 32 \
        --rollout_concurrency 10

    # Enhanced
    python -m training_free_grpo.eval_cpg_hierarchical \
        --experiment_name ${dataset}_enhanced \
        --dataset ${dataset} \
        --library_file library.json \
        --use_difficulty_adaptive \
        --pass_k 32 \
        --rollout_concurrency 10
done
```

**用途**: 评估跨数据集的泛化能力

---

## 结果文件说明

评测会生成 3 个文件：

### 1. `{experiment}_results.json` - 主要结果

```json
{
  "dataset": "AIME24",
  "pass_k": 32,
  "eval_stats": {
    "avg_reward": 0.680,
    "Pass@32": 0.6325,
    "avg_tool_call": 3.2
  },
  "difficulty_performance": {
    "easy": {"correct": 8, "total": 10, "pass_rate": 0.8},
    "medium": {"correct": 12, "total": 20, "pass_rate": 0.6},
    "hard": {"correct": 3, "total": 10, "pass_rate": 0.3}
  },
  "retrieval_stats": {
    "avg_experiences": 5.2
  }
}
```

**用途**:
- 用于自动化脚本读取结果
- 与其他实验对比
- 生成报告和图表

---

### 2. `{experiment}_retrieval_stats.json` - 检索统计

```json
{
  "retrieval_stats": {
    "avg_experiences": 5.2,
    "difficulty_distribution": {
      "easy": 10,
      "medium": 20,
      "hard": 10
    },
    "domain_classification": {
      "math": 38,
      "general": 2
    }
  },
  "config": {
    "use_difficulty_adaptive": true,
    "top_k": 5,
    "diversity_penalty": 0.3,
    "library_file": "library.json"
  }
}
```

**用途**:
- 分析检索行为
- 调试检索问题
- 了解难度分布

---

### 3. `{experiment}.jsonl` - 详细 rollouts

每行一个 rollout：

```jsonl
{"problem": "...", "difficulty": "medium", "num_experiences": 5, "response": "...", "reward": 1}
{"problem": "...", "difficulty": "hard", "num_experiences": 8, "response": "...", "reward": 0}
...
```

**用途**:
- 详细分析每个问题
- 调试失败案例
- 理解模型行为

---

## 常见问题

### Q1: 如何只看最终的对比结果？

**A**: 运行后直接看终端输出的 "COMPARISON WITH BASELINE" 部分，重点关注：
- Pass@32 的变化
- 关键改进总结（✨ Key Improvements）

---

### Q2: 如何导出对比报告？

**A**: 使用以下 Python 脚本：

```python
import json

# 加载结果
with open("data/math/eval/AIME24_baseline_results.json") as f:
    baseline = json.load(f)
with open("data/math/eval/AIME24_enhanced_results.json") as f:
    enhanced = json.load(f)

# 计算改进
pass_k = "Pass@32"
improvement = enhanced["eval_stats"][pass_k] - baseline["eval_stats"][pass_k]

print(f"Dataset: {enhanced['dataset']}")
print(f"Pass@32 Improvement: {improvement:+.2%}")
print(f"  Baseline: {baseline['eval_stats'][pass_k]:.2%}")
print(f"  Enhanced: {enhanced['eval_stats'][pass_k]:.2%}")
```

---

### Q3: 为什么没有显示对比？

**A**: 确认以下几点：
1. Baseline 结果文件存在
2. 文件命名正确（包含 `_results.json`）
3. 文件放在 `data/{domain}/eval/` 目录下
4. 或使用 `--baseline_file` 手动指定路径

---

### Q4: 如何调整 API 并发度避免限流？

**A**: 根据你的 API 限额设置 `--rollout_concurrency`:

| API Tier | 推荐并发 |
|----------|---------|
| Free | 2 |
| Tier 1 | 5-10 |
| Tier 2+ | 20-50 |

如果仍然限流，进一步降低并发度。

---

## 总结

使用 `eval_cpg_hierarchical.py` 评测，你可以：

✅ **自动对比** baseline 和增强版性能
✅ **清晰看到** 准确率提升百分比
✅ **量化分析** token 节省和成本降低
✅ **深入理解** 不同难度级别的表现
✅ **生成报告** 用于论文或展示

**关键结论模板**：
> "使用层级经验库后，在 {dataset} 数据集上 Pass@32 从 {baseline}% 提升到 {enhanced}%，提高了 {improvement}%，同时减少了 {reduction}% 的 token 使用。"

**示例**：
> "使用层级经验库后，在 AIME24 数据集上 Pass@32 从 58.0% 提升到 63.25%，提高了 5.3%，同时减少了 89.6% 的 token 使用。"

---

**文档版本**: 1.0
**日期**: 2024-11
**适用**: Training-Free GRPO Enhanced Evaluation
