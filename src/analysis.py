"""
Analysis module for post-evaluation data processing.

Loads raw benchmark results from JSON files and produces:
  - Per-category hallucination rate rankings
  - Cross-model comparison analysis
  - Statistical summaries and exportable DataFrames
  - Risk-tiered category breakdowns
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

from src.config import (
    RESULTS_DIR,
    CATEGORY_DISPLAY_NAMES,
    HIGH_RISK_CATEGORIES,
    MEDIUM_RISK_CATEGORIES,
)


def load_results(filepath):
    """Load raw JSON evaluation results from disk."""
    with open(filepath, "r") as f:
        return json.load(f)


def list_available_results():
    """List all saved evaluation result files."""
    return sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def results_to_dataframe(results):
    """Convert raw evaluation results into a structured DataFrame."""
    rows = []
    for cat, h_rate in results["category_hallucination_rates"].items():
        accuracy = results["category_accuracy"].get(cat, 1 - h_rate)
        display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        if cat in HIGH_RISK_CATEGORIES:
            risk_tier = "High"
        elif cat in MEDIUM_RISK_CATEGORIES:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"
        rows.append({
            "category": cat,
            "display_name": display_name,
            "accuracy": round(accuracy, 4),
            "hallucination_rate": round(h_rate, 4),
            "risk_tier": risk_tier,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("hallucination_rate", ascending=False).reset_index(drop=True)
    return df


def comparison_to_dataframe(comparison):
    """Convert a model comparison result into a DataFrame."""
    rows = []
    model_a_name = comparison["summary"]["model_a_name"]
    model_b_name = comparison["summary"]["model_b_name"]
    for cat, delta_info in comparison["category_deltas"].items():
        display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        if cat in HIGH_RISK_CATEGORIES:
            risk_tier = "High"
        elif cat in MEDIUM_RISK_CATEGORIES:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"
        rows.append({
            "category": cat,
            "display_name": display_name,
            f"{model_a_name}_hallucination_rate": delta_info["model_a_rate"],
            f"{model_b_name}_hallucination_rate": delta_info["model_b_rate"],
            "delta": delta_info["delta"],
            "better_model": delta_info["better_model"],
            "risk_tier": risk_tier,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("delta", ascending=False, key=abs).reset_index(drop=True)
    return df


def get_worst_categories(results, n=10):
    """Get the N worst-performing categories by hallucination rate."""
    sorted_cats = sorted(
        results["category_hallucination_rates"].items(),
        key=lambda x: x[1], reverse=True,
    )
    return [(CATEGORY_DISPLAY_NAMES.get(c, c), r) for c, r in sorted_cats[:n]]


def get_best_categories(results, n=10):
    """Get the N best-performing categories (lowest hallucination rate)."""
    sorted_cats = sorted(
        results["category_hallucination_rates"].items(),
        key=lambda x: x[1],
    )
    return [(CATEGORY_DISPLAY_NAMES.get(c, c), r) for c, r in sorted_cats[:n]]


def compute_risk_tier_summary(results):
    """Compute average hallucination rates grouped by risk tier."""
    tiers = {"High": [], "Medium": [], "Low": []}
    for cat, h_rate in results["category_hallucination_rates"].items():
        if cat in HIGH_RISK_CATEGORIES:
            tiers["High"].append((cat, h_rate))
        elif cat in MEDIUM_RISK_CATEGORIES:
            tiers["Medium"].append((cat, h_rate))
        else:
            tiers["Low"].append((cat, h_rate))
    summary = {}
    for tier, cat_rates in tiers.items():
        if cat_rates:
            rates = [r for _, r in cat_rates]
            summary[tier] = {
                "avg_hallucination_rate": round(np.mean(rates), 4),
                "max_hallucination_rate": round(max(rates), 4),
                "min_hallucination_rate": round(min(rates), 4),
                "num_categories": len(cat_rates),
                "categories": [CATEGORY_DISPLAY_NAMES.get(c, c) for c, _ in cat_rates],
            }
        else:
            summary[tier] = {
                "avg_hallucination_rate": 0.0,
                "num_categories": 0,
                "categories": [],
            }
    return summary


def generate_executive_summary(results):
    """Generate a human-readable executive summary of evaluation results."""
    model_name = results["metadata"]["model_name"]
    overall_acc = results["overall_score"]
    overall_hr = results["overall_hallucination_rate"]
    worst = get_worst_categories(results, n=5)
    best = get_best_categories(results, n=5)
    risk_summary = compute_risk_tier_summary(results)
    lines = [
        f"# Hallucination Analysis Report: {model_name}",
        f"",
        f"**Evaluation Date:** {results['metadata']['evaluation_timestamp'][:10]}",
        f"**Mode:** {results['metadata']['mode']}",
        f"",
        f"## Overall Performance",
        f"- **Accuracy:** {overall_acc:.1%}",
        f"- **Hallucination Rate:** {overall_hr:.1%}",
        f"",
        f"## Risk Tier Analysis",
    ]
    for tier in ["High", "Medium", "Low"]:
        td = risk_summary[tier]
        lines.append(f"- **{tier} Risk** ({td['num_categories']} categories): avg hallucination = {td['avg_hallucination_rate']:.1%}")
    lines.extend(["", "## Top 5 Worst Categories"])
    for name, rate in worst:
        lines.append(f"1. **{name}** — {rate:.1%}")
    lines.extend(["", "## Top 5 Best Categories"])
    for name, rate in best:
        lines.append(f"1. **{name}** — {rate:.1%}")
    return "\n".join(lines)
