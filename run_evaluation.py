"""
Main CLI entry point for the Hallucination Detection Pipeline.

Usage:
    # Single model evaluation
    python run_evaluation.py --model gpt-4o-mini

    # Two-model comparison
    python run_evaluation.py --model gpt-3.5-turbo --compare gpt-4o-mini

    # Evaluate specific categories only
    python run_evaluation.py --model gpt-4o-mini --categories HEALTH LAW FINANCE

    # Use MC2 mode
    python run_evaluation.py --model gpt-4o-mini --mode mc2
"""

import argparse
import sys
from rich.console import Console

from src.models import GPTModel
from src.evaluator import run_benchmark, run_comparison, ALL_TASKS
from src.analysis import generate_executive_summary
from src.visualize import (
    create_hallucination_bar_chart,
    create_risk_tier_chart,
    create_comparison_chart,
    create_delta_chart,
)
from src.config import RESULTS_DIR

from deepeval.benchmarks.tasks import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hallucination Detection Pipeline — TruthfulQA Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --model gpt-4o-mini
  python run_evaluation.py --model gpt-3.5-turbo --compare gpt-4o
  python run_evaluation.py --model gpt-4o-mini --categories HEALTH LAW FINANCE
  python run_evaluation.py --model gpt-4o-mini --mode mc2
        """,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Primary model to evaluate (e.g., gpt-3.5-turbo, gpt-4o-mini, gpt-4o)"
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Second model for side-by-side comparison"
    )
    parser.add_argument(
        "--mode", type=str, default="mc1", choices=["mc1", "mc2"],
        help="Evaluation mode: mc1 (single correct) or mc2 (multi-correct)"
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Specific categories to evaluate (e.g., HEALTH LAW FINANCE)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature for the model(s)"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip generating visualization HTML files"
    )
    return parser.parse_args()


def resolve_tasks(category_names):
    """Convert category name strings to TruthfulQATask enums."""
    if category_names is None:
        return None
    tasks = []
    for name in category_names:
        name_upper = name.upper()
        try:
            tasks.append(TruthfulQATask(name_upper))
        except ValueError:
            console.print(f"[bold red]Unknown category: {name}[/bold red]")
            console.print(f"Valid categories: {[t.value for t in TruthfulQATask]}")
            sys.exit(1)
    return tasks


def main():
    args = parse_args()
    mode = TruthfulQAMode.MC1 if args.mode == "mc1" else TruthfulQAMode.MC2
    tasks = resolve_tasks(args.categories)

    console.print("\n[bold white on blue]  🧠 HALLUCINATION DETECTION PIPELINE  [/bold white on blue]\n")

    # Initialize primary model
    model_a = GPTModel(model_name=args.model, temperature=args.temperature)

    if args.compare:
        # Comparison mode
        model_b = GPTModel(model_name=args.compare, temperature=args.temperature)
        comparison = run_comparison(model_a, model_b, tasks=tasks, mode=mode)

        if not args.no_viz:
            console.print("\n[bold]Generating visualizations...[/bold]")
            create_comparison_chart(comparison, save_path=RESULTS_DIR / "comparison_chart.html")
            create_delta_chart(comparison, save_path=RESULTS_DIR / "delta_chart.html")
            # Also generate individual charts
            create_hallucination_bar_chart(comparison["model_a"], save_path=RESULTS_DIR / f"{args.model}_bar_chart.html")
            create_hallucination_bar_chart(comparison["model_b"], save_path=RESULTS_DIR / f"{args.compare}_bar_chart.html")
            console.print(f"[bold green]✓ Visualizations saved to {RESULTS_DIR}/[/bold green]")
    else:
        # Single model evaluation
        results = run_benchmark(model_a, tasks=tasks, mode=mode)

        # Generate executive summary
        summary = generate_executive_summary(results)
        summary_path = RESULTS_DIR / f"{args.model}_report.md"
        with open(summary_path, "w") as f:
            f.write(summary)
        console.print(f"\n  📝 Report saved to: [bold green]{summary_path}[/bold green]")

        if not args.no_viz:
            console.print("\n[bold]Generating visualizations...[/bold]")
            create_hallucination_bar_chart(results, save_path=RESULTS_DIR / "hallucination_bar_chart.html")
            create_risk_tier_chart(results, save_path=RESULTS_DIR / "risk_tier_chart.html")
            console.print(f"[bold green]✓ Visualizations saved to {RESULTS_DIR}/[/bold green]")

    console.print("\n[bold green]✅ Evaluation complete![/bold green]\n")


if __name__ == "__main__":
    main()
