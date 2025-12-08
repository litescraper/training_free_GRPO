"""
Evaluation script for CPG + Hierarchical Retrieval enhanced Training-Free GRPO.

This script evaluates a trained hierarchical experience library on test datasets.
It supports dynamic experience retrieval based on problem classification.

Usage:
    python -m training_free_grpo.eval_cpg_hierarchical \
        --mode agent \
        --domain math \
        --dataset AIME24 \
        --library_file experiences_library.json \
        --experiment_name AIME24_cpg_hierarchical \
        --pass_k 32 \
        --rollout_concurrency 128
"""

# Windows compatibility: must be first import
try:
    from training_free_grpo import _windows_compat
except ImportError:
    pass

import os
import json
import argparse
import asyncio
import multiprocessing
from typing import Dict, List
from collections import defaultdict

from training_free_grpo.main import rollout_dataset, load_rollouts, save_rollouts
from training_free_grpo.hierarchical_retrieval import (
    HierarchicalExperienceLibrary,
    ProblemClassifier
)
from utu.agents import SimpleAgent
from utu.config import ConfigLoader


def load_hierarchical_library(library_file: str) -> HierarchicalExperienceLibrary:
    """
    Load hierarchical experience library from JSON file.

    Args:
        library_file: Path to the library JSON file

    Returns:
        HierarchicalExperienceLibrary instance
    """
    library = HierarchicalExperienceLibrary(embedding_dim=256)

    if os.path.exists(library_file):
        library.import_from_json(library_file)
        print(f"Loaded hierarchical library from {library_file}")

        stats = library.get_statistics()
        print(f"  - Total experiences: {stats['total_experiences']}")
        print(f"  - Meta-level: {stats['meta_experiences']}")
        print(f"  - Domains: {list(stats['domain_breakdown'].keys())}")
        print(f"  - Average effectiveness: {stats['avg_effectiveness']:.3f}")
    else:
        raise FileNotFoundError(f"Library file not found: {library_file}")

    return library


def format_experiences_for_prompt(experiences: List) -> str:
    """
    Format retrieved experiences for prompt insertion.

    Args:
        experiences: List of Experience objects

    Returns:
        Formatted string for prompt
    """
    if not experiences:
        return "None"

    formatted_lines = []
    for i, exp in enumerate(experiences, 1):
        level_tag = f"[{exp.level.upper()}]"
        formatted_lines.append(f"{i}. {level_tag} {exp.content}")

    return "\n".join(formatted_lines)


async def main(args):
    # Set up domain-specific variables
    if args.domain == "math":
        from training_free_grpo.math.dataset import load_data
        from training_free_grpo.math.verify import verify_func
        from training_free_grpo.math.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        config_name = "simple/math_agent.yaml"
        domain_key = "math"
    elif args.domain == "web":
        from training_free_grpo.web.dataset import load_data
        from training_free_grpo.web.verify import verify_func
        from training_free_grpo.web.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        config_name = "simple/search_agent.yaml"
        domain_key = "web"
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")

    # Load hierarchical experience library
    library = load_hierarchical_library(args.library_file)
    classifier = ProblemClassifier()

    # Set up the agent
    if args.mode == "prompt":
        worker_agent = None
    elif args.mode == "agent":
        config = ConfigLoader.load_agent_config(config_name)
        worker_agent = SimpleAgent(config=config)
        await worker_agent.build()
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")

    # Load the dataset
    test_data = load_data(args.dataset)
    print(f"\nLoaded {len(test_data)} records from dataset: {args.dataset}")
    if args.dataset_truncate is not None:
        print(f"  - Truncated to {args.dataset_truncate} samples")
        test_data = test_data[:args.dataset_truncate]

    # Retrieve experiences for each problem
    print(f"\nRetrieving experiences for each problem...")
    formatted_test_data = []
    retrieval_stats = {
        "num_experiences_retrieved": [],
        "difficulty_distribution": defaultdict(int),
        "domain_classification": defaultdict(int),
    }

    for sample in test_data:
        problem = sample["problem"]

        # Classify problem
        detected_domain, task_type, difficulty = classifier.classify(problem)
        retrieval_stats["difficulty_distribution"][difficulty] += 1
        retrieval_stats["domain_classification"][detected_domain] += 1

        # Override domain if specified
        if args.use_detected_domain:
            retrieval_domain = detected_domain if detected_domain != "general" else domain_key
        else:
            retrieval_domain = domain_key

        # Retrieve relevant experiences
        if args.use_difficulty_adaptive:
            # Difficulty-adaptive retrieval
            relevant_experiences = library.retrieve_by_difficulty(
                problem=problem,
                difficulty=difficulty,
                domain=retrieval_domain
            )
        else:
            # Fixed top-k retrieval
            relevant_experiences = library.retrieve_experiences(
                problem=problem,
                domain=retrieval_domain,
                task_type=task_type if task_type != "general" else None,
                top_k=args.top_k,
                diversity_penalty=args.diversity_penalty
            )

        # Format experiences
        formatted_experiences = format_experiences_for_prompt(relevant_experiences)

        # Create prompt with experiences
        prompt = PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
            experiences=formatted_experiences,
            problem=problem
        )

        # Store formatted sample
        formatted_sample = {
            "prompt": prompt,
            "detected_domain": detected_domain,
            "task_type": task_type,
            "difficulty": difficulty,
            "num_experiences": len(relevant_experiences),
            "experience_levels": [exp.level for exp in relevant_experiences],
            **sample
        }
        formatted_test_data.append(formatted_sample)

        # Track stats
        retrieval_stats["num_experiences_retrieved"].append(len(relevant_experiences))

    # Print retrieval statistics
    print(f"\nRetrieval Statistics:")
    print(f"  - Avg experiences per problem: {sum(retrieval_stats['num_experiences_retrieved']) / len(retrieval_stats['num_experiences_retrieved']):.2f}")
    print(f"  - Difficulty distribution: {dict(retrieval_stats['difficulty_distribution'])}")
    print(f"  - Domain classification: {dict(retrieval_stats['domain_classification'])}")

    # Duplicate for Pass@k evaluation
    original_count = len(formatted_test_data)
    formatted_test_data = formatted_test_data * args.pass_k
    print(f"\nDuplicated to {len(formatted_test_data)} records for Pass@{args.pass_k} evaluation")

    # Load existing rollouts
    os.makedirs(f"data/{args.domain}/eval", exist_ok=True)
    rollout_filename = f"data/{args.domain}/eval/{args.experiment_name}.jsonl"
    rollouts = load_rollouts(rollout_filename)

    # Save retrieval stats
    stats_filename = f"data/{args.domain}/eval/{args.experiment_name}_retrieval_stats.json"
    with open(stats_filename, "w") as f:
        json.dump({
            "retrieval_stats": {
                "avg_experiences": sum(retrieval_stats["num_experiences_retrieved"]) / len(retrieval_stats["num_experiences_retrieved"]),
                "difficulty_distribution": dict(retrieval_stats["difficulty_distribution"]),
                "domain_classification": dict(retrieval_stats["domain_classification"]),
            },
            "config": {
                "use_difficulty_adaptive": args.use_difficulty_adaptive,
                "top_k": args.top_k,
                "diversity_penalty": args.diversity_penalty,
                "library_file": args.library_file,
            }
        }, f, indent=2)
    print(f"Saved retrieval statistics to {stats_filename}")

    # Rollout the dataset
    print(f"\n{'='*60}")
    print(f"Starting evaluation rollouts...")
    print(f"{'='*60}\n")

    rollouts, eval_stats = await rollout_dataset(
        worker_agent=worker_agent,
        data=formatted_test_data,
        rollouts=rollouts,
        verify_func=verify_func,
        rollout_filename=rollout_filename,
        rollout_concurrency=args.rollout_concurrency,
        task_timeout=args.task_timeout,
        max_tokens=args.rollout_max_tokens,
        temperature=args.temperature,
    )

    # Additional analysis by difficulty
    print(f"\n{'='*60}")
    print(f"Performance by Difficulty Level:")
    print(f"{'='*60}")

    difficulty_performance = defaultdict(lambda: {"correct": 0, "total": 0})
    for i in range(original_count):
        # Get all Pass@K attempts for this problem
        attempts = rollouts[i*args.pass_k:(i+1)*args.pass_k]
        difficulty = attempts[0].get("difficulty", "unknown")

        # Check if any attempt was correct (Pass@K)
        max_reward = max(attempt.get("reward", 0) for attempt in attempts)
        difficulty_performance[difficulty]["total"] += 1
        if max_reward > 0:
            difficulty_performance[difficulty]["correct"] += 1

    for difficulty in ["easy", "medium", "hard"]:
        if difficulty in difficulty_performance:
            stats = difficulty_performance[difficulty]
            pass_rate = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {difficulty.capitalize():8s}: {stats['correct']:3d}/{stats['total']:3d} = {pass_rate:.2%}")

    # Save final evaluation results
    results_data = {
        "dataset": args.dataset,
        "pass_k": args.pass_k,
        "eval_stats": eval_stats,
        "difficulty_performance": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "pass_rate": v["correct"] / v["total"] if v["total"] > 0 else 0
            }
            for k, v in difficulty_performance.items()
        },
        "retrieval_stats": {
            "avg_experiences": sum(retrieval_stats["num_experiences_retrieved"]) / len(retrieval_stats["num_experiences_retrieved"]),
            "difficulty_distribution": dict(retrieval_stats["difficulty_distribution"]),
        }
    }

    results_filename = f"data/{args.domain}/eval/{args.experiment_name}_results.json"
    with open(results_filename, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved evaluation results to {results_filename}")

    # Try to load baseline for comparison
    baseline_data = None
    if args.baseline_file:
        if os.path.exists(args.baseline_file):
            with open(args.baseline_file) as f:
                baseline_data = json.load(f)
            print(f"Loaded baseline from {args.baseline_file}")
        else:
            print(f"Warning: Baseline file not found: {args.baseline_file}")
    else:
        # Try to auto-detect baseline
        baseline_candidates = [
            f"data/{args.domain}/eval/{args.dataset}_baseline_results.json",
            f"data/{args.domain}/eval/{args.dataset}_baseline.json",
            f"data/{args.domain}/eval/baseline_results.json",
        ]
        for candidate in baseline_candidates:
            if os.path.exists(candidate):
                with open(candidate) as f:
                    baseline_data = json.load(f)
                print(f"Auto-detected baseline: {candidate}")
                break

    # Print final summary
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY - {args.dataset}")
    print(f"{'='*70}\n")

    # Main metrics
    pass_k_metric = f"Pass@{args.pass_k}"
    current_pass_k = eval_stats.get(pass_k_metric, 0)
    current_avg_reward = eval_stats.get("avg_reward", 0)
    avg_experiences = results_data["retrieval_stats"]["avg_experiences"]

    print(f"📊 Main Results:")
    print(f"  • Dataset: {args.dataset}")
    print(f"  • {pass_k_metric}: {current_pass_k:.2%}")
    print(f"  • Average Reward: {current_avg_reward:.3f}")
    print(f"  • Avg Tool Calls: {eval_stats.get('avg_tool_call', 0):.2f}")
    print(f"  • Avg Experiences Retrieved: {avg_experiences:.1f}")

    # Comparison with baseline
    if baseline_data:
        print(f"\n{'='*70}")
        print(f"📈 COMPARISON WITH BASELINE")
        print(f"{'='*70}\n")

        baseline_pass_k = baseline_data.get("eval_stats", {}).get(pass_k_metric, 0)
        baseline_avg_reward = baseline_data.get("eval_stats", {}).get("avg_reward", 0)

        # Calculate improvements
        pass_k_improvement = current_pass_k - baseline_pass_k
        pass_k_improvement_pct = (pass_k_improvement / baseline_pass_k * 100) if baseline_pass_k > 0 else 0
        reward_improvement = current_avg_reward - baseline_avg_reward
        reward_improvement_pct = (reward_improvement / baseline_avg_reward * 100) if baseline_avg_reward > 0 else 0

        # Token/Experience efficiency
        baseline_experiences = baseline_data.get("retrieval_stats", {}).get("avg_experiences", 50)  # Assume 50 if not specified
        experience_reduction = baseline_experiences - avg_experiences
        experience_reduction_pct = (experience_reduction / baseline_experiences * 100) if baseline_experiences > 0 else 0

        print(f"┌{'─'*68}┐")
        print(f"│ {'Metric':<30} │ {'Baseline':>12} │ {'Enhanced':>12} │ {'Change':>8} │")
        print(f"├{'─'*68}┤")

        # Pass@K comparison
        change_str = f"{pass_k_improvement:+.1%}" if pass_k_improvement != 0 else "  --  "
        baseline_str = f"{baseline_pass_k:.2%}"
        enhanced_str = f"{current_pass_k:.2%}"
        emoji = "🟢" if pass_k_improvement > 0 else "🔴" if pass_k_improvement < 0 else "⚪"
        print(f"│ {emoji} {pass_k_metric:<28} │ {baseline_str:>12} │ {enhanced_str:>12} │ {change_str:>8} │")

        # Avg Reward comparison
        change_str = f"{reward_improvement:+.3f}" if reward_improvement != 0 else "  --  "
        baseline_str = f"{baseline_avg_reward:.3f}"
        enhanced_str = f"{current_avg_reward:.3f}"
        emoji = "🟢" if reward_improvement > 0 else "🔴" if reward_improvement < 0 else "⚪"
        print(f"│ {emoji} Average Reward{'':>16} │ {baseline_str:>12} │ {enhanced_str:>12} │ {change_str:>8} │")

        # Experiences comparison
        change_str = f"-{experience_reduction:.1f}"
        baseline_str = f"{baseline_experiences:.1f}"
        enhanced_str = f"{avg_experiences:.1f}"
        emoji = "🟢"  # Always good to reduce
        print(f"│ {emoji} Avg Experiences{'':>15} │ {baseline_str:>12} │ {enhanced_str:>12} │ {change_str:>8} │")

        print(f"└{'─'*68}┘")

        # Key improvements summary
        print(f"\n✨ Key Improvements:")
        if pass_k_improvement > 0:
            print(f"  • Accuracy improved by {pass_k_improvement_pct:+.1f}% ({pass_k_improvement:+.1%})")
        if experience_reduction > 0:
            print(f"  • Context reduced by {experience_reduction_pct:.1f}% ({experience_reduction:.1f} fewer experiences)")
            token_savings = experience_reduction * 30  # Assume 30 tokens per experience
            print(f"  • Estimated token savings: ~{token_savings:.0f} tokens per problem")

        # Efficiency metric
        if pass_k_improvement > 0 and experience_reduction > 0:
            efficiency = pass_k_improvement / (avg_experiences / baseline_experiences)
            print(f"  • Efficiency gain: {efficiency:.2%} accuracy per unit context")

    else:
        print(f"\n💡 Tip: To compare with baseline, use --baseline_file or place baseline results at:")
        print(f"   data/{args.domain}/eval/{args.dataset}_baseline_results.json")

    # Performance by difficulty
    if difficulty_performance:
        print(f"\n{'='*70}")
        print(f"📊 Performance by Difficulty Level")
        print(f"{'='*70}\n")

        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in difficulty_performance:
                stats = difficulty_performance[difficulty]
                pass_rate = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                bar_length = int(pass_rate * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"  {difficulty.capitalize():8s}: {bar} {pass_rate:6.1%} ({stats['correct']}/{stats['total']})")

    # Token efficiency analysis
    print(f"\n{'='*70}")
    print(f"💰 Token Efficiency Analysis")
    print(f"{'='*70}\n")

    tokens_per_experience = 30  # Estimated average
    baseline_exp_count = baseline_data.get("retrieval_stats", {}).get("avg_experiences", 50) if baseline_data else 50
    baseline_tokens = baseline_exp_count * tokens_per_experience
    current_tokens = avg_experiences * tokens_per_experience
    token_savings = baseline_tokens - current_tokens
    token_savings_pct = (token_savings / baseline_tokens * 100) if baseline_tokens > 0 else 0

    print(f"  • Baseline context: ~{baseline_tokens:.0f} tokens ({baseline_exp_count:.0f} experiences × {tokens_per_experience})")
    print(f"  • Current context: ~{current_tokens:.0f} tokens ({avg_experiences:.1f} experiences × {tokens_per_experience})")
    print(f"  • Token savings: ~{token_savings:.0f} tokens per problem ({token_savings_pct:.1f}% reduction)")

    # Cost estimation (assuming GPT-4 pricing: $0.03/1K input tokens)
    input_cost_per_1k = 0.03
    num_problems = len(test_data)
    total_baseline_cost = (baseline_tokens * num_problems * input_cost_per_1k) / 1000
    total_current_cost = (current_tokens * num_problems * input_cost_per_1k) / 1000
    cost_savings = total_baseline_cost - total_current_cost

    print(f"\n  Estimated API cost (input only, GPT-4):")
    print(f"  • Baseline: ${total_baseline_cost:.2f} ({num_problems} problems × {baseline_tokens:.0f} tokens)")
    print(f"  • Current: ${total_current_cost:.2f} ({num_problems} problems × {current_tokens:.0f} tokens)")
    print(f"  • Savings: ${cost_savings:.2f} ({cost_savings/total_baseline_cost*100:.1f}% reduction)")

    print(f"\n{'='*70}")
    print(f"✅ Evaluation Complete!")
    print(f"{'='*70}")
    print(f"\n📁 Output files:")
    print(f"  • Results: {results_filename}")
    print(f"  • Retrieval stats: {stats_filename}")
    print(f"  • Detailed rollouts: {rollout_filename}")


if __name__ == "__main__":
    # Fix for Windows multiprocessing compatibility
    multiprocessing.freeze_support()
    if os.name == 'nt':  # Windows
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser(
        description="Evaluate CPG + Hierarchical Retrieval enhanced Training-Free GRPO"
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prompt", "agent"],
        help="Mode of inference: 'prompt' (direct LLM) or 'agent' (with tools)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["math", "web"],
        help="Domain of the experiment"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of evaluation dataset (e.g., AIME24, MATH500)"
    )
    parser.add_argument(
        "--library_file",
        type=str,
        required=True,
        help="Path to hierarchical experience library JSON file"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment (for output files)"
    )

    # Retrieval parameters
    parser.add_argument(
        "--use_difficulty_adaptive",
        action="store_true",
        help="Use difficulty-adaptive top-k retrieval (easy:3, medium:5, hard:8)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of experiences to retrieve (if not using difficulty-adaptive)"
    )
    parser.add_argument(
        "--diversity_penalty",
        type=float,
        default=0.3,
        help="MMR diversity penalty (0=pure relevance, 1=pure diversity)"
    )
    parser.add_argument(
        "--use_detected_domain",
        action="store_true",
        help="Use auto-detected domain instead of specified domain"
    )

    # Evaluation parameters
    parser.add_argument(
        "--pass_k",
        type=int,
        default=1,
        help="Pass@k metric: number of attempts per problem"
    )
    parser.add_argument(
        "--rollout_concurrency",
        type=int,
        default=5,
        help="Number of concurrent rollouts"
    )
    parser.add_argument(
        "--dataset_truncate",
        type=int,
        default=None,
        help="Truncate dataset to first N samples (for testing)"
    )
    parser.add_argument(
        "--baseline_file",
        type=str,
        default=None,
        help="Path to baseline results JSON for comparison (auto-detects if not specified)"
    )

    # Rollout parameters
    parser.add_argument(
        "--rollout_max_tokens",
        type=int,
        default=16384,
        help="Max tokens for each rollout"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for rollouts"
    )
    parser.add_argument(
        "--task_timeout",
        type=float,
        default=3600,
        help="Timeout for each individual task in seconds"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
