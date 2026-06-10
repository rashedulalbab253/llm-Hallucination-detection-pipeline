"""
Benchmark evaluation engine for TruthfulQA using DeepEval.

This module orchestrates the evaluation of LLM models against the
TruthfulQA benchmark. It handles:
  - Running DeepEval's built-in TruthfulQA benchmark (MC1/MC2 modes)
  - Extracting per-category (task) scores
  - Serializing results to JSON for downstream analysis
  - Supporting multi-model comparison runs
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.tasks import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode
from deepeval.models import DeepEvalBaseLLM

from src.config import RESULTS_DIR, CATEGORY_DISPLAY_NAMES

console = Console()

# All 38 TruthfulQA categories
ALL_TASKS = [
    TruthfulQATask.LANGUAGE,
    TruthfulQATask.MISQUOTATIONS,
    TruthfulQATask.NUTRITION,
    TruthfulQATask.FICTION,
    TruthfulQATask.SCIENCE,
    TruthfulQATask.PROVERBS,
    TruthfulQATask.MANDELA_EFFECT,
    TruthfulQATask.INDEXICAL_ERROR_IDENTITY,
    TruthfulQATask.CONFUSION_PLACES,
    TruthfulQATask.ECONOMICS,
    TruthfulQATask.PSYCHOLOGY,
    TruthfulQATask.CONFUSION_PEOPLE,
    TruthfulQATask.EDUCATION,
    TruthfulQATask.CONSPIRACIES,
    TruthfulQATask.SUBJECTIVE,
    TruthfulQATask.MISCONCEPTIONS,
    TruthfulQATask.INDEXICAL_ERROR_OTHER,
    TruthfulQATask.MYTHS_AND_FAIRYTALES,
    TruthfulQATask.INDEXICAL_ERROR_TIME,
    TruthfulQATask.MISCONCEPTIONS_TOPICAL,
    TruthfulQATask.POLITICS,
    TruthfulQATask.FINANCE,
    TruthfulQATask.INDEXICAL_ERROR_LOCATION,
    TruthfulQATask.CONFUSION_OTHER,
    TruthfulQATask.LAW,
    TruthfulQATask.DISTRACTION,
    TruthfulQATask.HISTORY,
    TruthfulQATask.WEATHER,
    TruthfulQATask.STATISTICS,
    TruthfulQATask.MISINFORMATION,
    TruthfulQATask.SUPERSTITIONS,
    TruthfulQATask.LOGICAL_FALSEHOOD,
    TruthfulQATask.HEALTH,
    TruthfulQATask.STEREOTYPES,
    TruthfulQATask.RELIGION,
    TruthfulQATask.ADVERTISING,
    TruthfulQATask.SOCIOLOGY,
    TruthfulQATask.PARANORMAL,
]


def run_benchmark(
    model: DeepEvalBaseLLM,
    tasks: Optional[List[TruthfulQATask]] = None,
    mode: TruthfulQAMode = TruthfulQAMode.MC1,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Run the TruthfulQA benchmark on a given model.

    Parameters
    ----------
    model : DeepEvalBaseLLM
        The LLM model wrapper to evaluate.
    tasks : list of TruthfulQATask, optional
        Specific category tasks to evaluate. Defaults to ALL_TASKS.
    mode : TruthfulQAMode
        MC1 (single correct answer) or MC2 (multi-correct). Default: MC1.
    save_results : bool
        Whether to save the raw results to a JSON file.

    Returns
    -------
    dict
        Structured results including overall score, per-category scores,
        metadata, and evaluation timestamp.
    """
    tasks = tasks or ALL_TASKS
    model_name = model.get_model_name()

    console.print(f"\n[bold cyan]{'═' * 60}[/bold cyan]")
    console.print(f"[bold white]  🔬 TruthfulQA Benchmark Evaluation[/bold white]")
    console.print(f"[bold cyan]{'═' * 60}[/bold cyan]")
    console.print(f"  Model:      [bold yellow]{model_name}[/bold yellow]")
    console.print(f"  Mode:       [bold]{mode.value}[/bold]")
    console.print(f"  Categories: [bold]{len(tasks)}[/bold]")
    console.print(f"[bold cyan]{'─' * 60}[/bold cyan]\n")

    # Initialize benchmark
    benchmark = TruthfulQA(tasks=tasks, mode=mode)

    # Run evaluation
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(
            description=f"Evaluating {model_name} on TruthfulQA...",
            total=None,
        )
        benchmark.evaluate(model=model)

    elapsed = time.time() - start_time

    # Extract per-category scores
    # DeepEval stores task scores after evaluation
    category_scores = {}
    for task_score in benchmark.task_scores:
        task_name = task_score.task.value if hasattr(task_score.task, 'value') else str(task_score.task)
        category_scores[task_name] = task_score.score

    # Calculate hallucination rates (1 - accuracy = hallucination rate)
    category_hallucination_rates = {
        cat: round(1.0 - score, 4)
        for cat, score in category_scores.items()
    }

    # Build results structure
    results = {
        "metadata": {
            "model_name": model_name,
            "mode": mode.value if hasattr(mode, 'value') else str(mode),
            "num_categories": len(tasks),
            "total_questions": 817,
            "evaluation_timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
        },
        "overall_score": round(benchmark.overall_score, 4),
        "overall_hallucination_rate": round(1.0 - benchmark.overall_score, 4),
        "category_accuracy": category_scores,
        "category_hallucination_rates": category_hallucination_rates,
    }

    # Print summary table
    _print_results_table(results)

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.replace('/', '_')}_{mode.value}_{timestamp}.json"
        filepath = RESULTS_DIR / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n  📁 Results saved to: [bold green]{filepath}[/bold green]")

    return results


def run_comparison(
    model_a: DeepEvalBaseLLM,
    model_b: DeepEvalBaseLLM,
    tasks: Optional[List[TruthfulQATask]] = None,
    mode: TruthfulQAMode = TruthfulQAMode.MC1,
) -> Dict[str, Any]:
    """
    Run TruthfulQA on two models and produce a comparison report.

    Parameters
    ----------
    model_a : DeepEvalBaseLLM
        First model to evaluate.
    model_b : DeepEvalBaseLLM
        Second model to evaluate.
    tasks : list of TruthfulQATask, optional
        Specific tasks to evaluate. Defaults to all.
    mode : TruthfulQAMode
        MC1 or MC2 evaluation mode.

    Returns
    -------
    dict
        Combined results with per-category delta analysis.
    """
    console.print("\n[bold magenta]━━━ MODEL COMPARISON ━━━[/bold magenta]")
    console.print(f"  Model A: [bold cyan]{model_a.get_model_name()}[/bold cyan]")
    console.print(f"  Model B: [bold yellow]{model_b.get_model_name()}[/bold yellow]")
    console.print()

    # Evaluate both models
    results_a = run_benchmark(model_a, tasks=tasks, mode=mode, save_results=True)
    results_b = run_benchmark(model_b, tasks=tasks, mode=mode, save_results=True)

    # Calculate deltas per category
    category_deltas = {}
    all_categories = set(results_a["category_hallucination_rates"].keys()) | set(
        results_b["category_hallucination_rates"].keys()
    )

    for cat in sorted(all_categories):
        rate_a = results_a["category_hallucination_rates"].get(cat, 0)
        rate_b = results_b["category_hallucination_rates"].get(cat, 0)
        # Positive delta = model_b has HIGHER hallucination (model_a is better)
        # Negative delta = model_a has HIGHER hallucination (model_b is better)
        delta = round(rate_b - rate_a, 4)
        category_deltas[cat] = {
            "model_a_rate": rate_a,
            "model_b_rate": rate_b,
            "delta": delta,
            "better_model": model_a.get_model_name() if delta > 0 else model_b.get_model_name(),
        }

    comparison = {
        "model_a": results_a,
        "model_b": results_b,
        "category_deltas": category_deltas,
        "overall_delta": round(
            results_b["overall_hallucination_rate"]
            - results_a["overall_hallucination_rate"],
            4,
        ),
        "summary": {
            "model_a_name": model_a.get_model_name(),
            "model_b_name": model_b.get_model_name(),
            "model_a_overall_accuracy": results_a["overall_score"],
            "model_b_overall_accuracy": results_b["overall_score"],
            "accuracy_improvement": round(
                results_b["overall_score"] - results_a["overall_score"], 4
            ),
        },
    }

    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_{model_a.get_model_name()}_{model_b.get_model_name()}_{timestamp}.json".replace("/", "_")
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(comparison, f, indent=2)

    _print_comparison_table(comparison)

    console.print(f"\n  📁 Comparison saved to: [bold green]{filepath}[/bold green]")
    return comparison


def _print_results_table(results: Dict[str, Any]) -> None:
    """Print a formatted results table to the console."""
    model_name = results["metadata"]["model_name"]
    overall = results["overall_score"]
    hallucination_rate = results["overall_hallucination_rate"]

    console.print(f"\n[bold cyan]{'═' * 60}[/bold cyan]")
    console.print(f"  📊 Results: [bold]{model_name}[/bold]")
    console.print(f"  Overall Accuracy:        [bold green]{overall:.1%}[/bold green]")
    console.print(
        f"  Overall Hallucination:   [bold red]{hallucination_rate:.1%}[/bold red]"
    )
    console.print(f"  Time:                    {results['metadata']['elapsed_seconds']}s")
    console.print(f"[bold cyan]{'─' * 60}[/bold cyan]")

    table = Table(title="Per-Category Hallucination Rates", show_lines=True)
    table.add_column("Category", style="bold white", min_width=25)
    table.add_column("Accuracy", justify="center", min_width=10)
    table.add_column("Hallucination Rate", justify="center", min_width=18)
    table.add_column("Risk", justify="center", min_width=6)

    # Sort by hallucination rate (worst first)
    sorted_cats = sorted(
        results["category_hallucination_rates"].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    from src.config import HIGH_RISK_CATEGORIES, MEDIUM_RISK_CATEGORIES

    for cat, h_rate in sorted_cats:
        accuracy = results["category_accuracy"].get(cat, 0)
        display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat)

        # Color based on hallucination severity
        if h_rate >= 0.5:
            h_style = "bold red"
        elif h_rate >= 0.3:
            h_style = "bold yellow"
        else:
            h_style = "bold green"

        # Risk indicator
        if cat in HIGH_RISK_CATEGORIES:
            risk = "🔴 HIGH"
        elif cat in MEDIUM_RISK_CATEGORIES:
            risk = "🟡 MED"
        else:
            risk = "🟢 LOW"

        table.add_row(
            display_name,
            f"{accuracy:.1%}",
            f"[{h_style}]{h_rate:.1%}[/{h_style}]",
            risk,
        )

    console.print(table)


def _print_comparison_table(comparison: Dict[str, Any]) -> None:
    """Print a side-by-side model comparison table."""
    summary = comparison["summary"]

    console.print(f"\n[bold magenta]{'═' * 70}[/bold magenta]")
    console.print(f"  📊 Model Comparison Results")
    console.print(f"[bold magenta]{'─' * 70}[/bold magenta]")

    table = Table(title="Head-to-Head: Hallucination Rates by Category", show_lines=True)
    table.add_column("Category", style="bold white", min_width=22)
    table.add_column(summary["model_a_name"], justify="center", min_width=14)
    table.add_column(summary["model_b_name"], justify="center", min_width=14)
    table.add_column("Delta", justify="center", min_width=10)
    table.add_column("Winner", justify="center", min_width=14)

    sorted_deltas = sorted(
        comparison["category_deltas"].items(),
        key=lambda x: abs(x[1]["delta"]),
        reverse=True,
    )

    for cat, delta_info in sorted_deltas:
        display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        rate_a = delta_info["model_a_rate"]
        rate_b = delta_info["model_b_rate"]
        delta = delta_info["delta"]

        if delta > 0:
            delta_str = f"[bold green]↓ {abs(delta):.1%}[/bold green]"
            winner = f"[bold cyan]{summary['model_a_name']}[/bold cyan]"
        elif delta < 0:
            delta_str = f"[bold red]↑ {abs(delta):.1%}[/bold red]"
            winner = f"[bold yellow]{summary['model_b_name']}[/bold yellow]"
        else:
            delta_str = "—"
            winner = "Tie"

        table.add_row(
            display_name,
            f"{rate_a:.1%}",
            f"{rate_b:.1%}",
            delta_str,
            winner,
        )

    console.print(table)

    # Overall summary
    console.print(f"\n  Overall Accuracy Improvement: ", end="")
    improvement = summary["accuracy_improvement"]
    if improvement > 0:
        console.print(
            f"[bold green]+{improvement:.1%} ({summary['model_b_name']} is better)[/bold green]"
        )
    elif improvement < 0:
        console.print(
            f"[bold red]{improvement:.1%} ({summary['model_a_name']} is better)[/bold red]"
        )
    else:
        console.print("[bold]No difference[/bold]")
