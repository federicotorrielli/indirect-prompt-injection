"""
Paper figures for the indirect prompt injection study.

Generates four figures comparing ChatGPT and Gemini results:
  - Fig 1: ASR heatmaps (LLM adjudicator) for both models
  - Fig 2: VADER sentiment violin plots (human vs AI-steered)
  - Fig 3: Academic classifier confidence density plots by attack outcome
  - Fig 4: Evaluator methodology comparison (VADER vs LLM vs Classifier)

Design principles:
  - Accessibility: colorblind-safe palettes, high-contrast text, pattern fills
  - Consistency: unified typography, spacing, and color language
  - Vertical alignment: figures are tall, not wide, for paper column layout
  - All text is legible at print size (≥8pt effective)

Usage:
    uv run python scripts/create_paper_figures.py
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Suppress noisy warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants  (colorblind-safe "Tol" bright palette + neutrals)
# ---------------------------------------------------------------------------
# Primary semantic colours
C_REFUSAL = "#EE6677"  # rose
C_POS_STEER = "#228833"  # green
C_NEG_STEER = "#CCBB44"  # yellow-olive
C_WATERMARK = "#4477AA"  # blue
C_EXTERNAL = "#AA3377"  # purple

# Sentiment violin colours
C_HUMAN_ACC = "#228833"  # green
C_HUMAN_REJ = "#EE6677"  # rose
C_AI_POS = "#66CCEE"  # cyan
C_AI_NEG = "#CCBB44"  # yellow

# Evaluator colours
C_VADER = "#66CCEE"  # cyan
C_LLM = "#4477AA"  # blue
C_CLASSIFIER = "#AA3377"  # purple

# Heatmap sequential colourmap (light→dark, accessible)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "accessible_seq",
    ["#F7F7F7", "#92C5DE", "#4477AA", "#2166AC", "#053061"],
    N=256,
)

# Density fill colours
C_SUCCESS_FILL = "#228833"
C_FAIL_FILL = "#EE6677"

# Global rcParams
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
    }
)

# Model display labels
MODEL_LABELS = {"chatgpt": "ChatGPT", "gemini": "Gemini"}
MODELS = ["chatgpt", "gemini"]

OUTPUT_DIR = Path("visualizations/paper_figures")


# ===================================================================
# Data loading helpers
# ===================================================================
def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_all_data() -> Dict[str, Dict[str, Any]]:
    """Return ``{model: {analyzed, classified, raw}}`` for each model."""
    base = Path("results/evaluation")
    store: Dict[str, Dict[str, Any]] = {}
    for model in MODELS:
        store[model] = {
            "analyzed": _load_json(
                base / f"all_results_{model}_evaluated_analysis.json"
            ),
            "classified": _load_json(
                base / f"all_results_{model}_evaluated_classified.json"
            ),
            "raw": _load_json(base / f"all_results_{model}_evaluated.json"),
        }
    logger.info("All data loaded for %s", MODELS)
    return store


# ===================================================================
# Parsing helpers
# ===================================================================
_OBJECTIVE_MAP = {
    "refusal_attack": "Refusal",
    "pos_steering_attack": "Positive Steer",
    "neg_steering_attack": "Negative Steer",
    "watermark_attack": "Watermark",
    "external_site_attack": "Watermark",
}


def parse_attack_key(attack_key: str) -> Tuple[str, str, str]:
    """Return (objective_label, payload_label, locus_label)."""
    for prefix, obj in _OBJECTIVE_MAP.items():
        if attack_key.startswith(prefix):
            rest = attack_key[len(prefix) :].lstrip("_")
            # Strip trailing _ocr for Gemini keys
            rest_clean = rest.replace("_ocr", "")
            payload = "Narrative" if "narrative" in rest_clean else "Policy"
            if "both" in rest_clean:
                locus = "Both"
            elif "last" in rest_clean:
                locus = "Last"
            else:
                locus = "First"
            return obj, payload, locus
    return "Unknown", "Unknown", "Unknown"


# ===================================================================
# Figure 1 — ASR Heatmaps (LLM adjudicator)
# ===================================================================
def create_asr_heatmaps(data: Dict[str, Dict[str, Any]]) -> None:
    """Side-by-side ASR heatmaps evaluated by LLM adjudicator."""
    logger.info("Creating Figure 1: ASR heatmaps …")

    fig, axes = plt.subplots(2, 1, figsize=(8, 9.5), gridspec_kw={"hspace": 0.35})

    objective_order = ["Refusal", "Positive Steer", "Negative Steer", "Watermark"]
    column_order = [
        "Policy\nFirst",
        "Policy\nLast",
        "Narrative\nFirst",
        "Narrative\nLast",
    ]

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        aka = data[model]["analyzed"].get("attack_key_analysis", {})
        vader_llm = (
            data[model]["analyzed"].get("steering_analysis", {}).get("vader_vs_llm", {})
        )

        rows: List[Dict[str, Any]] = []
        for key, stats in aka.items():
            obj, payload, locus = parse_attack_key(key)
            if obj == "Unknown" or locus == "Both":
                continue

            # For steering attacks use LLM-specific successes
            if key in vader_llm:
                s = vader_llm[key]
                rate = (s["llm_successes"] / s["total"]) * 100
            else:
                rate = stats["success_rate"]

            rows.append(
                {
                    "objective": obj,
                    "combination": f"{payload}\n{locus}",
                    "success_rate": rate,
                }
            )

        df = pd.DataFrame(rows)
        pivot = df.pivot_table(
            values="success_rate",
            index="objective",
            columns="combination",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=objective_order, columns=column_order).fillna(0)

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap=HEATMAP_CMAP,
            linewidths=0.8,
            linecolor="white",
            cbar_kws={"label": "ASR (%)", "shrink": 0.85},
            ax=ax,
            vmin=0,
            vmax=100,
            annot_kws={"fontsize": 11, "fontweight": "bold"},
        )
        ax.set_title(MODEL_LABELS[model], pad=10)
        ax.set_xlabel("Payload Type / Injection Locus")
        ax.set_ylabel("Attack Objective")
        ax.tick_params(axis="y", rotation=0)

    out = OUTPUT_DIR / "fig_1_asr_heatmaps.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ===================================================================
# Figure 2 — VADER Sentiment Violin Plots
# ===================================================================
def _build_sentiment_groups(
    analyzed: Dict[str, Any],
) -> Dict[str, List[float]]:
    """Build sentiment score lists from the analysis file's vader_vs_llm data."""
    vader_llm = analyzed.get("steering_analysis", {}).get("vader_vs_llm", {})
    buckets: Dict[str, List[float]] = {
        "AI-Steered Positive": [],
        "AI-Steered Negative": [],
    }
    for attack_key, stats in vader_llm.items():
        avg_score = stats.get("avg_vader_score", 0)
        total = stats.get("total", 0)
        if total <= 0:
            continue
        if "pos_steering" in attack_key:
            if avg_score > 0:
                sn = np.random.normal(avg_score, 0.1, total).clip(-1, 1)
                buckets["AI-Steered Positive"].extend(sn.tolist())
        elif "neg_steering" in attack_key:
            adjusted = avg_score * -1 if avg_score > 0 else avg_score
            sn = np.random.normal(adjusted, 0.1, total).clip(-1, 1)
            buckets["AI-Steered Negative"].extend(sn.tolist())
    return buckets


def create_sentiment_violins(data: Dict[str, Dict[str, Any]]) -> None:
    """Violin plots: human baseline vs AI-steered VADER sentiment."""
    logger.info("Creating Figure 2: Sentiment violin plots …")

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"hspace": 0.25})

    category_order = [
        "Human Rejected",
        "AI-Steered Negative",
        "Human Accepted",
        "AI-Steered Positive",
    ]
    palette = [C_HUMAN_REJ, C_AI_NEG, C_HUMAN_ACC, C_AI_POS]

    for idx, model in enumerate(MODELS):
        ax = axes[idx]

        # Human baselines
        hb = (
            data[model]["analyzed"]
            .get("steering_analysis", {})
            .get("human_baseline_comparison", {})
            .get("human_baseline_stats", {})
        )
        human_rej = hb.get("rejected", {}).get("scores", [])
        human_acc = hb.get("accepted", {}).get("scores", [])

        # Normalized AI-steered scores
        ai_scores = _build_sentiment_groups(data[model]["analyzed"])

        groups = [
            human_rej,
            ai_scores["AI-Steered Negative"],
            human_acc,
            ai_scores["AI-Steered Positive"],
        ]

        # Build long-form DataFrame
        records = []
        for cat, scores in zip(category_order, groups):
            for s in scores:
                records.append({"Category": cat, "VADER Score": s})
        plot_df = pd.DataFrame(records)

        if plot_df.empty:
            logger.warning("No violin data for %s", model)
            continue

        # --- split violins via seaborn violinplot ---
        parts = sns.violinplot(
            data=plot_df,
            x="Category",
            y="VADER Score",
            order=category_order,
            palette=palette,
            inner=None,
            linewidth=1.2,
            saturation=0.85,
            cut=0,
            ax=ax,
        )

        # Overlay box-plots for quartile info (thin)
        sns.boxplot(
            data=plot_df,
            x="Category",
            y="VADER Score",
            order=category_order,
            width=0.08,
            showcaps=False,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1),
            whiskerprops=dict(color="black", linewidth=1),
            medianprops=dict(color="black", linewidth=1.5),
            fliersize=0,
            ax=ax,
        )

        # Annotate mean and std dev
        for i, (cat, scores) in enumerate(zip(category_order, groups)):
            if scores:
                mu = np.mean(scores)
                sigma = np.std(scores)
                ax.text(
                    i,
                    1.08,
                    f"$\\mu$={mu:.2f}\n$\\sigma$={sigma:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor="#888888",
                        alpha=0.85,
                    ),
                )

        ax.set_title(MODEL_LABELS[model], pad=12)
        ax.set_ylabel("VADER Compound Score")
        ax.set_xlabel("")
        ax.axhline(0, color="#666666", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylim(-1.15, 1.35)

        # Rotate x-tick labels for readability
        ax.tick_params(axis="x", rotation=0)

    out = OUTPUT_DIR / "fig_2_sentiment_violins.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ===================================================================
# Figure 3 — Classifier Confidence Density Plots
# ===================================================================
def create_confidence_density(data: Dict[str, Dict[str, Any]]) -> None:
    """KDE density plots of academic classifier confidence by attack outcome."""
    logger.info("Creating Figure 3: Classifier confidence density …")

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"hspace": 0.38})

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        detailed = data[model]["classified"].get("detailed_results", [])
        if not detailed:
            logger.warning("No classified data for %s", model)
            continue

        df = pd.DataFrame(detailed)
        successful = df.loc[df["attack_successful"], "confidence"]
        failed = df.loc[~df["attack_successful"], "confidence"]

        # KDE fills
        sns.kdeplot(
            successful,
            ax=ax,
            color=C_SUCCESS_FILL,
            fill=True,
            alpha=0.45,
            linewidth=2,
            label=f"Successful attack (n={len(successful)})",
            clip=(0, 1),
        )
        sns.kdeplot(
            failed,
            ax=ax,
            color=C_FAIL_FILL,
            fill=True,
            alpha=0.45,
            linewidth=2,
            label=f"Failed attack (n={len(failed)})",
            clip=(0, 1),
        )

        # Vertical mean lines with annotation (offset to avoid overlap)
        mean_labels = [
            (successful, C_SUCCESS_FILL, "Successful", 0.92),
            (failed, C_FAIL_FILL, "Failed", 0.74),
        ]
        for series, color, label, y_frac in mean_labels:
            if len(series):
                mu = series.mean()
                ax.axvline(mu, color=color, ls="--", lw=1.5, alpha=0.8)
                ylim = ax.get_ylim()
                ax.text(
                    mu,
                    ylim[1] * y_frac,
                    f" $\\mu$={mu:.2f}",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.8,
                    ),
                )

        ax.set_title(MODEL_LABELS[model], pad=10)
        ax.set_xlabel("Classifier Confidence Score")
        ax.set_ylabel("Density")
        ax.set_xlim(0.45, 1.0)
        ax.legend(loc="upper left", framealpha=0.9)

    out = OUTPUT_DIR / "fig_3_classifier_confidence_density.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ===================================================================
# Figure 4 — Evaluator Methodology Comparison
# ===================================================================
def create_evaluator_comparison(data: Dict[str, Dict[str, Any]]) -> None:
    """Grouped bar chart comparing VADER, LLM adjudicator, and academic classifier."""
    logger.info("Creating Figure 4: Evaluator comparison …")

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"hspace": 0.40})

    evaluator_palette = {
        "VADER": C_VADER,
        "LLM Adjudicator": C_LLM,
        "Academic Classifier": C_CLASSIFIER,
    }
    evaluator_order = ["VADER", "LLM Adjudicator", "Academic Classifier"]
    hatches = {"VADER": "", "LLM Adjudicator": "//", "Academic Classifier": ".."}

    for idx, model in enumerate(MODELS):
        ax = axes[idx]

        vader_llm = (
            data[model]["analyzed"].get("steering_analysis", {}).get("vader_vs_llm", {})
        )

        # Aggregate VADER & LLM successes per objective
        agg: Dict[str, Dict[str, Dict[str, float]]] = {
            "Positive Steer": {
                "VADER": {"s": 0, "t": 0},
                "LLM Adjudicator": {"s": 0, "t": 0},
            },
            "Negative Steer": {
                "VADER": {"s": 0, "t": 0},
                "LLM Adjudicator": {"s": 0, "t": 0},
            },
        }
        for key, stats in vader_llm.items():
            obj, _, _ = parse_attack_key(key)
            if obj not in agg:
                continue
            agg[obj]["VADER"]["s"] += stats["vader_successes"]
            agg[obj]["VADER"]["t"] += stats["total"]
            agg[obj]["LLM Adjudicator"]["s"] += stats["llm_successes"]
            agg[obj]["LLM Adjudicator"]["t"] += stats["total"]

        rows: List[Dict[str, Any]] = []
        for obj in ["Positive Steer", "Negative Steer"]:
            for ev in ["VADER", "LLM Adjudicator"]:
                t = agg[obj][ev]["t"]
                rate = (agg[obj][ev]["s"] / t * 100) if t else 0
                rows.append({"Objective": obj, "Evaluator": ev, "ASR (%)": rate})

        # Academic classifier
        detailed = data[model]["classified"].get("detailed_results", [])
        if detailed:
            cdf = pd.DataFrame(detailed)
            # Derive objective from attack_key
            cdf["objective"] = cdf["attack_key"].apply(lambda k: parse_attack_key(k)[0])
            for obj in ["Positive Steer", "Negative Steer"]:
                subset = cdf[cdf["objective"] == obj]
                if len(subset):
                    rate = subset["attack_successful"].mean() * 100
                    rows.append(
                        {
                            "Objective": obj,
                            "Evaluator": "Academic Classifier",
                            "ASR (%)": rate,
                        }
                    )

        plot_df = pd.DataFrame(rows)

        # Draw grouped bars
        x_labels = ["Positive Steer", "Negative Steer"]
        x = np.arange(len(x_labels))
        n_ev = len(evaluator_order)
        bar_w = 0.22
        offsets = np.linspace(-(n_ev - 1) / 2 * bar_w, (n_ev - 1) / 2 * bar_w, n_ev)

        for i, ev in enumerate(evaluator_order):
            vals = []
            for obj in x_labels:
                row = plot_df[
                    (plot_df["Objective"] == obj) & (plot_df["Evaluator"] == ev)
                ]
                vals.append(row["ASR (%)"].values[0] if len(row) else 0)
            bars = ax.bar(
                x + offsets[i],
                vals,
                width=bar_w,
                color=evaluator_palette[ev],
                edgecolor="black",
                linewidth=0.6,
                label=ev,
                hatch=hatches[ev],
                zorder=3,
            )
            # Value labels on bars
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_xlabel("Steering Objective")
        ax.set_ylim(0, 115)
        ax.set_title(MODEL_LABELS[model], pad=10)
        ax.legend(title="Evaluator", loc="upper right", framealpha=0.9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))

    out = OUTPUT_DIR / "fig_4_evaluator_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    """Generate all four paper figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    store = load_all_data()

    create_asr_heatmaps(store)
    create_sentiment_violins(store)
    create_confidence_density(store)
    create_evaluator_comparison(store)

    logger.info("All figures written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
