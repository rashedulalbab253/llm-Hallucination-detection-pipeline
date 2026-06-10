"""
Visualization module using Plotly for interactive hallucination analysis charts.

Generates:
  - Category-level hallucination rate bar charts
  - Risk tier radar/polar charts
  - Model comparison grouped bar charts
  - Heatmaps for multi-model analysis
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from src.config import COLORS, CATEGORY_DISPLAY_NAMES, HIGH_RISK_CATEGORIES, MEDIUM_RISK_CATEGORIES
from src.analysis import results_to_dataframe, comparison_to_dataframe, compute_risk_tier_summary


PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_card"],
        font=dict(family="Inter, sans-serif", color=COLORS["text_primary"], size=13),
        title=dict(font=dict(size=20, color=COLORS["text_primary"])),
        xaxis=dict(gridcolor=COLORS["bg_surface"], zerolinecolor=COLORS["bg_surface"]),
        yaxis=dict(gridcolor=COLORS["bg_surface"], zerolinecolor=COLORS["bg_surface"]),
        margin=dict(l=60, r=30, t=80, b=60),
    )
)


def create_hallucination_bar_chart(results, save_path=None):
    """Create a horizontal bar chart of hallucination rates by category."""
    df = results_to_dataframe(results)
    df = df.sort_values("hallucination_rate", ascending=True)
    colors = []
    for _, row in df.iterrows():
        if row["risk_tier"] == "High":
            colors.append(COLORS["danger"])
        elif row["risk_tier"] == "Medium":
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["accent"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["display_name"],
        x=df["hallucination_rate"],
        orientation="h",
        marker=dict(color=colors, line=dict(color=COLORS["bg_dark"], width=1)),
        text=[f"{r:.0%}" for r in df["hallucination_rate"]],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_secondary"]),
        hovertemplate="<b>%{y}</b><br>Hallucination Rate: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Hallucination Rate by Category — {results['metadata']['model_name']}"),
        xaxis=dict(title="Hallucination Rate", tickformat=".0%", range=[0, 1.1]),
        yaxis=dict(title=""),
        height=max(500, len(df) * 28),
        **PLOTLY_TEMPLATE["layout"],
    )
    if save_path:
        fig.write_html(str(save_path))
    return fig


def create_risk_tier_chart(results, save_path=None):
    """Create a grouped summary chart showing risk tier performance."""
    risk_summary = compute_risk_tier_summary(results)
    tiers = ["High", "Medium", "Low"]
    avg_rates = [risk_summary[t]["avg_hallucination_rate"] for t in tiers]
    num_cats = [risk_summary[t]["num_categories"] for t in tiers]
    tier_colors = [COLORS["danger"], COLORS["warning"], COLORS["success"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tiers,
        y=avg_rates,
        marker=dict(color=tier_colors, line=dict(color=COLORS["bg_dark"], width=2)),
        text=[f"{r:.0%}" for r in avg_rates],
        textposition="outside",
        textfont=dict(size=14, color=COLORS["text_primary"]),
        customdata=num_cats,
        hovertemplate="<b>%{x} Risk</b><br>Avg Hallucination: %{y:.1%}<br>Categories: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Hallucination by Risk Tier — {results['metadata']['model_name']}"),
        xaxis=dict(title="Risk Tier"),
        yaxis=dict(title="Avg Hallucination Rate", tickformat=".0%", range=[0, max(avg_rates) * 1.3 + 0.05]),
        height=420,
        **PLOTLY_TEMPLATE["layout"],
    )
    if save_path:
        fig.write_html(str(save_path))
    return fig


def create_comparison_chart(comparison, save_path=None):
    """Create a grouped bar chart comparing two models side by side."""
    model_a_name = comparison["summary"]["model_a_name"]
    model_b_name = comparison["summary"]["model_b_name"]
    deltas = comparison["category_deltas"]
    sorted_cats = sorted(deltas.items(), key=lambda x: max(x[1]["model_a_rate"], x[1]["model_b_rate"]), reverse=True)
    categories = [CATEGORY_DISPLAY_NAMES.get(c, c) for c, _ in sorted_cats]
    rates_a = [d["model_a_rate"] for _, d in sorted_cats]
    rates_b = [d["model_b_rate"] for _, d in sorted_cats]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=model_a_name,
        x=categories, y=rates_a,
        marker_color=COLORS["model_a"],
        hovertemplate="<b>%{x}</b><br>" + model_a_name + ": %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=model_b_name,
        x=categories, y=rates_b,
        marker_color=COLORS["model_b"],
        hovertemplate="<b>%{x}</b><br>" + model_b_name + ": %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Model Comparison: {model_a_name} vs {model_b_name}"),
        barmode="group",
        xaxis=dict(title="", tickangle=-45),
        yaxis=dict(title="Hallucination Rate", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=13)),
        height=600,
        **PLOTLY_TEMPLATE["layout"],
    )
    if save_path:
        fig.write_html(str(save_path))
    return fig


def create_delta_chart(comparison, save_path=None):
    """Create a diverging bar chart showing improvement/regression per category."""
    model_a_name = comparison["summary"]["model_a_name"]
    model_b_name = comparison["summary"]["model_b_name"]
    deltas = comparison["category_deltas"]
    sorted_cats = sorted(deltas.items(), key=lambda x: x[1]["delta"])
    categories = [CATEGORY_DISPLAY_NAMES.get(c, c) for c, _ in sorted_cats]
    delta_values = [d["delta"] for _, d in sorted_cats]
    colors = [COLORS["success"] if d < 0 else COLORS["danger"] for d in delta_values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories, x=delta_values,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{abs(d):.0%}" for d in delta_values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Delta: %{x:+.1%}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=2, line_color=COLORS["text_secondary"])
    fig.update_layout(
        title=dict(text=f"Hallucination Delta: {model_b_name} vs {model_a_name}"),
        xaxis=dict(title="← Better for " + model_b_name + "  |  Better for " + model_a_name + " →",
                   tickformat="+.0%"),
        yaxis=dict(title=""),
        height=max(500, len(categories) * 28),
        **PLOTLY_TEMPLATE["layout"],
    )
    if save_path:
        fig.write_html(str(save_path))
    return fig
