"""
Interactive Flask dashboard for hallucination analysis results.

Serves an interactive web UI with:
  - Overview metrics cards
  - Per-category hallucination rate charts
  - Risk tier breakdown
  - Model comparison views
  - Exportable reports

Usage:
    python run_dashboard.py
"""

import json
import sys
from pathlib import Path

from flask import Flask, render_template, jsonify, request
import plotly
import plotly.utils

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import RESULTS_DIR, CATEGORY_DISPLAY_NAMES, COLORS
from src.analysis import (
    load_results,
    list_available_results,
    results_to_dataframe,
    compute_risk_tier_summary,
    get_worst_categories,
    get_best_categories,
    generate_executive_summary,
)
from src.visualize import (
    create_hallucination_bar_chart,
    create_risk_tier_chart,
    create_comparison_chart,
    create_delta_chart,
)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "dashboard" / "templates"),
    static_folder=str(Path(__file__).parent / "dashboard" / "static"),
)


def _load_all_results():
    """Load all result files grouped by type."""
    files = list_available_results()
    single_results = []
    comparisons = []
    for f in files:
        data = load_results(f)
        if "category_deltas" in data:
            comparisons.append({"filename": f.name, "path": str(f), "data": data})
        else:
            single_results.append({"filename": f.name, "path": str(f), "data": data})
    return single_results, comparisons


@app.route("/")
def index():
    single_results, comparisons = _load_all_results()
    return render_template(
        "index.html",
        single_results=single_results,
        comparisons=comparisons,
        colors=COLORS,
    )


@app.route("/api/results")
def api_results():
    single_results, comparisons = _load_all_results()
    return jsonify({
        "single_results": [
            {"filename": r["filename"], "data": r["data"]}
            for r in single_results
        ],
        "comparisons": [
            {"filename": c["filename"], "data": c["data"]}
            for c in comparisons
        ],
    })


@app.route("/api/chart/bar/<filename>")
def api_bar_chart(filename):
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    data = load_results(filepath)
    fig = create_hallucination_bar_chart(data)
    return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))


@app.route("/api/chart/risk/<filename>")
def api_risk_chart(filename):
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    data = load_results(filepath)
    fig = create_risk_tier_chart(data)
    return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))


@app.route("/api/chart/comparison/<filename>")
def api_comparison_chart(filename):
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    data = load_results(filepath)
    fig = create_comparison_chart(data)
    return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))


@app.route("/api/chart/delta/<filename>")
def api_delta_chart(filename):
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    data = load_results(filepath)
    fig = create_delta_chart(data)
    return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))


@app.route("/api/summary/<filename>")
def api_summary(filename):
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    data = load_results(filepath)
    if "category_deltas" not in data:
        summary = generate_executive_summary(data)
        return jsonify({"summary": summary})
    return jsonify({"summary": "Comparison file — see comparison charts."})


if __name__ == "__main__":
    print("\n[INFO] Hallucination Detection Pipeline — Dashboard")
    print("=" * 50)
    print(f"Results directory: {RESULTS_DIR}")
    files = list_available_results()
    print(f"Found {len(files)} result file(s)")
    if not files:
        print("\n[WARNING] No results found! Generate sample data first:")
        print("   python generate_sample_data.py")
    print(f"\n[INFO] Dashboard: http://localhost:5000\n")
    app.run(debug=True, port=5000)
