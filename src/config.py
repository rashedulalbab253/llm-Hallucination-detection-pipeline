"""
Configuration module for the Hallucination Detection Pipeline.
Centralizes all settings, paths, and constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ──────────────────────────────────────────────────────────────
# Evaluation Settings
# ──────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "10"))
EVAL_MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "3"))
EVAL_RETRY_DELAY = float(os.getenv("EVAL_RETRY_DELAY", "2.0"))

# ──────────────────────────────────────────────────────────────
# TruthfulQA Category Mapping
# Maps DeepEval task enum names to human-readable labels
# ──────────────────────────────────────────────────────────────
CATEGORY_DISPLAY_NAMES = {
    "LANGUAGE": "Language",
    "MISQUOTATIONS": "Misquotations",
    "NUTRITION": "Nutrition",
    "FICTION": "Fiction",
    "SCIENCE": "Science",
    "PROVERBS": "Proverbs",
    "MANDELA_EFFECT": "Mandela Effect",
    "INDEXICAL_ERROR_IDENTITY": "Indexical Error: Identity",
    "CONFUSION_PLACES": "Confusion: Places",
    "ECONOMICS": "Economics",
    "PSYCHOLOGY": "Psychology",
    "CONFUSION_PEOPLE": "Confusion: People",
    "EDUCATION": "Education",
    "CONSPIRACIES": "Conspiracies",
    "SUBJECTIVE": "Subjective",
    "MISCONCEPTIONS": "Misconceptions",
    "INDEXICAL_ERROR_OTHER": "Indexical Error: Other",
    "MYTHS_AND_FAIRYTALES": "Myths & Fairytales",
    "INDEXICAL_ERROR_TIME": "Indexical Error: Time",
    "MISCONCEPTIONS_TOPICAL": "Misconceptions (Topical)",
    "POLITICS": "Politics",
    "FINANCE": "Finance",
    "INDEXICAL_ERROR_LOCATION": "Indexical Error: Location",
    "CONFUSION_OTHER": "Confusion: Other",
    "LAW": "Law",
    "DISTRACTION": "Distraction",
    "HISTORY": "History",
    "WEATHER": "Weather",
    "STATISTICS": "Statistics",
    "MISINFORMATION": "Misinformation",
    "SUPERSTITIONS": "Superstitions",
    "LOGICAL_FALSEHOOD": "Logical Falsehood",
    "HEALTH": "Health",
    "STEREOTYPES": "Stereotypes",
    "RELIGION": "Religion",
    "ADVERTISING": "Advertising",
    "SOCIOLOGY": "Sociology",
    "PARANORMAL": "Paranormal",
}

# ──────────────────────────────────────────────────────────────
# Risk tiers for categories — used in dashboard colour coding
# ──────────────────────────────────────────────────────────────
HIGH_RISK_CATEGORIES = {"HEALTH", "LAW", "FINANCE", "POLITICS", "NUTRITION"}
MEDIUM_RISK_CATEGORIES = {
    "SCIENCE", "ECONOMICS", "PSYCHOLOGY", "EDUCATION",
    "STATISTICS", "HISTORY", "RELIGION"
}
# Everything else is considered lower risk

# ──────────────────────────────────────────────────────────────
# Colour Palette (used across visualizations)
# ──────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366f1",       # Indigo-500
    "secondary": "#8b5cf6",     # Violet-500
    "accent": "#06b6d4",        # Cyan-500
    "success": "#10b981",       # Emerald-500
    "warning": "#f59e0b",       # Amber-500
    "danger": "#ef4444",        # Red-500
    "bg_dark": "#0f172a",       # Slate-900
    "bg_card": "#1e293b",       # Slate-800
    "bg_surface": "#334155",    # Slate-700
    "text_primary": "#f8fafc",  # Slate-50
    "text_secondary": "#94a3b8", # Slate-400
    "border": "#475569",        # Slate-600
    "model_a": "#6366f1",       # Indigo for model A
    "model_b": "#f97316",       # Orange for model B
}
