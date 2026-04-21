"""
Paper figures for the indirect prompt injection study.

This script regenerates the paper figures from the current five-run results.
It avoids synthetic approximations and reads directly from the evaluated JSON
and summary artifacts used in the paper.

Generated figures:
  - Fig 1: ASR heatmaps by payload family, architecture, and locus
  - Fig 2: Pooled VADER sentiment violins for steering attacks
  - Fig 3: Academic-classifier confidence density by attack outcome
  - Fig 4: Steering ASR by evaluator family

Usage:
    python scripts/create_paper_figures.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


MODELS = ["chatgpt", "gemini"]
MODEL_LABELS = {"chatgpt": "ChatGPT", "gemini": "Gemini"}

SUMMARY_PATH = Path("results/evaluation/self_consistency_summary.json")
EVAL_DIR = Path("results/evaluation")
OUTPUT_DIRS = [
    Path("visualizations/paper_figures"),
    Path("paper_and_review/Prompt_Injection_Paper/images"),
]


# Color palette
C_EXTERNAL = "#AA3377"
C_NEG = "#CCBB44"
C_POS = "#228833"
C_REFUSAL = "#EE6677"
C_WATERMARK = "#4477AA"

C_HUMAN_ACC = "#55A868"
C_HUMAN_REJ = "#C44E52"
C_AI_POS = "#4C9ED9"
C_AI_NEG = "#DDCC77"

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_heatmap",
    ["#F7F7F7", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
    N=256,
)


plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, file_name: str) -> None:
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / file_name
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
        logger.info("Saved %s", out_path)
    plt.close(fig)


def parse_attack_key(attack_key: str) -> Tuple[str, str, str]:
    objective_map = {
        "external_site_attack": "External Site",
        "neg_steering_attack": "Negative Steering",
        "pos_steering_attack": "Positive Steering",
        "refusal_attack": "Refusal",
        "watermark_attack": "Watermark",
    }
    for prefix, label in objective_map.items():
        if attack_key.startswith(prefix):
            rest = attack_key[len(prefix) :].lstrip("_")
            architecture = "Narrative" if "narrative" in rest else "Policy Puppetry"
            if "both" in rest:
                locus = "L1+L2"
            elif "last" in rest:
                locus = "L2"
            else:
                locus = "L1"
            return label, architecture, locus
    raise ValueError(f"Unrecognized attack key: {attack_key}")


def load_summary() -> Dict[str, Any]:
    return _load_json(SUMMARY_PATH)["services"]


def analysis_paths(model: str) -> List[Path]:
    return sorted(EVAL_DIR.glob(f"all_results_{model}_run*_evaluated_analysis.json"))


def evaluated_paths(model: str) -> List[Path]:
    return sorted(EVAL_DIR.glob(f"all_results_{model}_run*_evaluated.json"))


def collect_steering_sentiments(model: str) -> Dict[str, List[float]]:
    positive_scores: List[float] = []
    negative_scores: List[float] = []
    human_accepted: List[float] = []
    human_rejected: List[float] = []

    for idx, path in enumerate(analysis_paths(model)):
        data = _load_json(path)
        comparison = data["steering_analysis"]["human_baseline_comparison"]
        human_stats = comparison["human_baseline_stats"]
        if idx == 0:
            human_accepted = human_stats["accepted"]["scores"]
            human_rejected = human_stats["rejected"]["scores"]

        ai_stats = comparison["ai_baseline_stats"]
        positive_scores.extend(ai_stats["positive_steering"]["scores"])
        negative_scores.extend(ai_stats["negative_steering"]["scores"])

    return {
        "human_accepted": human_accepted,
        "human_rejected": human_rejected,
        "positive": positive_scores,
        "negative": negative_scores,
    }


def iter_steering_records(model: str) -> Iterable[Dict[str, Any]]:
    for path in evaluated_paths(model):
        data = _load_json(path)
        for attack_key, request_map in data.items():
            if "steering" not in attack_key:
                continue
            direction = "positive" if "pos_steering" in attack_key else "negative"
            for request_type, items in request_map.items():
                for item in items:
                    yield {
                        "attack_key": attack_key,
                        "direction": direction,
                        "request_type": request_type,
                        "consensus_success": bool(item["llm_consensus_success"]),
                        "academic_success": bool(item["academic_classifier_success"]),
                        "academic_confidence": float(item["academic_classifier_confidence"]),
                    }


def create_asr_heatmaps(summary: Dict[str, Any]) -> None:
    logger.info("Creating Figure 1: ASR heatmaps")

    row_order = [
        ("External Site", "Narrative"),
        ("External Site", "Policy Puppetry"),
        ("Negative Steering", "Narrative"),
        ("Negative Steering", "Policy Puppetry"),
        ("Positive Steering", "Narrative"),
        ("Positive Steering", "Policy Puppetry"),
        ("Refusal", "Narrative"),
        ("Refusal", "Policy Puppetry"),
        ("Watermark", "Narrative"),
        ("Watermark", "Policy Puppetry"),
    ]
    col_order = ["L1", "L2", "L1+L2"]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 10.0), gridspec_kw={"hspace": 0.35})

    for ax, model in zip(axes, MODELS):
        matrix = np.zeros((len(row_order), len(col_order)))
        for attack_key, metrics in summary[model]["by_attack_key"].items():
            objective, architecture, locus = parse_attack_key(attack_key)
            r = row_order.index((objective, architecture))
            c = col_order.index(locus)
            matrix[r, c] = metrics["llm_primary"]["pooled_rate"] * 100

        im = ax.imshow(matrix, cmap=HEATMAP_CMAP, aspect="auto", vmin=50, vmax=100)
        ax.set_title(MODEL_LABELS[model], pad=10)
        ax.set_xticks(np.arange(len(col_order)))
        ax.set_xticklabels(col_order)
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels([f"{obj} / {arch}" for obj, arch in row_order])
        ax.set_xlabel("Injection Locus")
        ax.set_ylabel("Payload Family / Architecture")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val >= 82 else "black",
                    fontweight="bold",
                )

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("ASR (%)")
    save_figure(fig, "fig_1_asr_heatmaps.png")


def create_sentiment_violins() -> None:
    logger.info("Creating Figure 2: sentiment violins")

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.8), gridspec_kw={"hspace": 0.28})
    colors = [C_HUMAN_REJ, C_AI_NEG, C_HUMAN_ACC, C_AI_POS]
    labels = ["Human\nRejected", "AI\nNegative", "Human\nAccepted", "AI\nPositive"]

    for ax, model in zip(axes, MODELS):
        sentiment = collect_steering_sentiments(model)
        groups = [
            sentiment["human_rejected"],
            sentiment["negative"],
            sentiment["human_accepted"],
            sentiment["positive"],
        ]
        positions = np.arange(1, 5)

        violin = ax.violinplot(
            groups,
            positions=positions,
            widths=0.85,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#333333")
            body.set_linewidth(0.8)
            body.set_alpha(0.85)

        ax.boxplot(
            groups,
            positions=positions,
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "#333333", "linewidth": 1.0},
            capprops={"color": "#333333", "linewidth": 1.0},
            boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 1.0},
        )

        for x, scores in zip(positions, groups):
            mu = float(np.mean(scores))
            med = float(np.median(scores))
            ax.text(
                x,
                1.045,
                f"$\\mu$={mu:.3f}\nmed={med:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#999999",
                    "alpha": 0.9,
                },
            )

        ax.set_title(MODEL_LABELS[model], pad=10)
        ax.set_ylabel("VADER Compound Score")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylim(-1.08, 1.19)
        ax.axhline(0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    save_figure(fig, "fig_2_sentiment_violins.png")


def create_confidence_density() -> None:
    logger.info("Creating Figure 3: classifier confidence density")

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.8), gridspec_kw={"hspace": 0.34})

    for ax, model in zip(axes, MODELS):
        success = []
        failure = []
        for record in iter_steering_records(model):
            if record["consensus_success"]:
                success.append(record["academic_confidence"])
            else:
                failure.append(record["academic_confidence"])

        x = np.linspace(0.5, 1.0, 400)
        for values, color, label in [
            (success, C_POS, f"Consensus success (n={len(success)})"),
            (failure, C_REFUSAL, f"Consensus failure (n={len(failure)})"),
        ]:
            if not values:
                continue
            kde = gaussian_kde(values)
            y = kde(x)
            ax.plot(x, y, color=color, linewidth=2, label=label)
            ax.fill_between(x, y, color=color, alpha=0.28)
            mu = float(np.mean(values))
            ax.axvline(mu, color=color, linestyle="--", linewidth=1.2, alpha=0.8)

        ax.set_title(MODEL_LABELS[model], pad=10)
        ax.set_xlabel("Academic Classifier Confidence")
        ax.set_ylabel("Density")
        ax.set_xlim(0.5, 1.0)
        ax.legend(loc="upper left", framealpha=0.92)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_figure(fig, "fig_3_classifier_confidence_density.png")


def create_evaluator_comparison(summary: Dict[str, Any]) -> None:
    logger.info("Creating Figure 4: evaluator comparison")

    evaluators = [
        ("llm_judge_a", "Judge A", "#4477AA", ""),
        ("llm_judge_b", "Judge B", "#66A3D2", "//"),
        ("llm_consensus", "Consensus", "#228833", ""),
        ("academic_classifier", "Academic\nClassifier", "#AA3377", ".."),
        ("vader", "VADER", "#CCBB44", "xx"),
    ]
    attacks = [("pos_steering_attack", "Positive"), ("neg_steering_attack", "Negative")]

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 9.2), gridspec_kw={"hspace": 0.34})
    bar_w = 0.14
    x = np.arange(len(attacks))
    offsets = np.linspace(-2 * bar_w, 2 * bar_w, len(evaluators))

    for ax, model in zip(axes, MODELS):
        for offset, (ev_key, ev_label, color, hatch) in zip(offsets, evaluators):
            vals = [summary[model]["by_attack_type"][attack_key][ev_key]["pooled_rate"] * 100 for attack_key, _ in attacks]
            bars = ax.bar(
                x + offset,
                vals,
                width=bar_w,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch,
                label=ev_label,
                zorder=3,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.8,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_title(MODEL_LABELS[model], pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in attacks])
        ax.set_ylim(50, 104)
        ax.set_ylabel("ASR (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", framealpha=0.92, title="Evaluator")

    save_figure(fig, "fig_4_evaluator_comparison.png")


def main() -> None:
    summary = load_summary()
    create_asr_heatmaps(summary)
    create_sentiment_violins()
    create_confidence_density()
    create_evaluator_comparison(summary)
    logger.info("All paper figures regenerated")


if __name__ == "__main__":
    main()
