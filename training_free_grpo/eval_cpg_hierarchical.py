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
    results_filename = f"data/{args.domain}/eval/{args.experiment_name}_results.json"
    with open(results_filename, "w") as f:
        json.dump({
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
        }, f, indent=2)
    print(f"\nSaved evaluation results to {results_filename}")

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")


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
