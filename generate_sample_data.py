"""
Generate realistic sample evaluation results for demonstration.

This script creates sample JSON result files that can be used to demo
the dashboard and visualization tools without requiring API keys or
running actual model evaluations.

Usage:
    python generate_sample_data.py
"""

import json
import random
from datetime import datetime
from pathlib import Path

from src.config import RESULTS_DIR, CATEGORY_DISPLAY_NAMES

random.seed(42)

# Realistic accuracy ranges per category for different models
# Based on published TruthfulQA benchmark results
CATEGORY_BASE_ACCURACY = {
    "HEALTH": 0.38,
    "LAW": 0.42,
    "FINANCE": 0.45,
    "POLITICS": 0.40,
    "NUTRITION": 0.35,
    "MISCONCEPTIONS": 0.52,
    "MISCONCEPTIONS_TOPICAL": 0.50,
    "CONSPIRACIES": 0.65,
    "SUPERSTITIONS": 0.68,
    "PARANORMAL": 0.70,
    "FICTION": 0.72,
    "MYTHS_AND_FAIRYTALES": 0.60,
    "PROVERBS": 0.55,
    "ADVERTISING": 0.48,
    "RELIGION": 0.44,
    "HISTORY": 0.58,
    "SCIENCE": 0.62,
    "ECONOMICS": 0.50,
    "PSYCHOLOGY": 0.46,
    "SOCIOLOGY": 0.52,
    "EDUCATION": 0.56,
    "STATISTICS": 0.40,
    "WEATHER": 0.65,
    "LANGUAGE": 0.60,
    "MISQUOTATIONS": 0.32,
    "STEREOTYPES": 0.58,
    "LOGICAL_FALSEHOOD": 0.55,
    "DISTRACTION": 0.62,
    "MISINFORMATION": 0.42,
    "CONFUSION_PLACES": 0.50,
    "CONFUSION_PEOPLE": 0.45,
    "CONFUSION_OTHER": 0.48,
    "SUBJECTIVE": 0.70,
    "MANDELA_EFFECT": 0.30,
    "INDEXICAL_ERROR_IDENTITY": 0.55,
    "INDEXICAL_ERROR_TIME": 0.50,
    "INDEXICAL_ERROR_LOCATION": 0.52,
    "INDEXICAL_ERROR_OTHER": 0.48,
}


def generate_model_results(model_name, accuracy_boost=0.0):
    """Generate realistic evaluation results for a given model."""
    category_accuracy = {}
    category_hallucination_rates = {}

    for cat, base_acc in CATEGORY_BASE_ACCURACY.items():
        # Add noise and model-specific boost
        acc = base_acc + accuracy_boost + random.uniform(-0.08, 0.08)
        acc = max(0.05, min(0.95, acc))  # Clamp to [0.05, 0.95]
        acc = round(acc, 4)
        category_accuracy[cat] = acc
        category_hallucination_rates[cat] = round(1.0 - acc, 4)

    overall_score = round(
        sum(category_accuracy.values()) / len(category_accuracy), 4
    )

    results = {
        "metadata": {
            "model_name": model_name,
            "mode": "MC1",
            "num_categories": 38,
            "total_questions": 817,
            "evaluation_timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(random.uniform(180, 420), 2),
        },
        "overall_score": overall_score,
        "overall_hallucination_rate": round(1.0 - overall_score, 4),
        "category_accuracy": category_accuracy,
        "category_hallucination_rates": category_hallucination_rates,
    }
    return results


def generate_comparison(results_a, results_b):
    """Generate a comparison structure from two result sets."""
    model_a_name = results_a["metadata"]["model_name"]
    model_b_name = results_b["metadata"]["model_name"]

    category_deltas = {}
    all_cats = set(results_a["category_hallucination_rates"].keys())

    for cat in sorted(all_cats):
        rate_a = results_a["category_hallucination_rates"].get(cat, 0)
        rate_b = results_b["category_hallucination_rates"].get(cat, 0)
        delta = round(rate_b - rate_a, 4)
        category_deltas[cat] = {
            "model_a_rate": rate_a,
            "model_b_rate": rate_b,
            "delta": delta,
            "better_model": model_a_name if delta > 0 else model_b_name,
        }

    return {
        "model_a": results_a,
        "model_b": results_b,
        "category_deltas": category_deltas,
        "overall_delta": round(
            results_b["overall_hallucination_rate"]
            - results_a["overall_hallucination_rate"], 4
        ),
        "summary": {
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "model_a_overall_accuracy": results_a["overall_score"],
            "model_b_overall_accuracy": results_b["overall_score"],
            "accuracy_improvement": round(
                results_b["overall_score"] - results_a["overall_score"], 4
            ),
        },
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Generating sample evaluation data...")

    # Generate results for two models
    results_35 = generate_model_results("gpt-3.5-turbo", accuracy_boost=0.0)
    results_4o = generate_model_results("gpt-4o-mini", accuracy_boost=0.15)

    # Save individual results
    path_35 = RESULTS_DIR / "gpt-3.5-turbo_MC1_sample.json"
    path_4o = RESULTS_DIR / "gpt-4o-mini_MC1_sample.json"

    with open(path_35, "w") as f:
        json.dump(results_35, f, indent=2)
    print(f"  [OK] Saved: {path_35}")

    with open(path_4o, "w") as f:
        json.dump(results_4o, f, indent=2)
    print(f"  [OK] Saved: {path_4o}")

    # Save comparison
    comparison = generate_comparison(results_35, results_4o)
    path_cmp = RESULTS_DIR / "comparison_gpt-3.5-turbo_gpt-4o-mini_sample.json"
    with open(path_cmp, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  [OK] Saved: {path_cmp}")

    print("\n[OK] Sample data generated! You can now run the dashboard.")
    print("   python run_dashboard.py")


if __name__ == "__main__":
    main()
