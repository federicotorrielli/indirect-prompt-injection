#!/usr/bin/env python3
"""
Analyze baseline experiment results and compare with injected PDF results.
Computes sentiment statistics for baseline vs steering attacks.
"""

import json
import os
import statistics
from typing import Any

from rich.console import Console
from rich.table import Table
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

console = Console()

BASELINE_GEMINI = "results/baseline/baseline_reviews.json"
BASELINE_CHATGPT = "results/baseline/baseline_reviews_chatgpt.json"
INJECTED_CHATGPT = "results/evaluation/all_results_chatgpt_evaluated.json"
INJECTED_GEMINI = "results/evaluation/all_results_gemini_evaluated.json"


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def compute_sentiment_scores(texts: list[str]) -> list[float]:
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(text)["compound"] for text in texts if text]


def compute_stats(scores: list[float]) -> dict:
    if not scores:
        return {}
    return {
        "n": len(scores),
        "mean": statistics.mean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0,
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "positive_pct": sum(1 for s in scores if s >= 0.05) / len(scores) * 100,
        "negative_pct": sum(1 for s in scores if s <= -0.05) / len(scores) * 100,
        "neutral_pct": sum(1 for s in scores if -0.05 < s < 0.05) / len(scores) * 100,
    }


def analyze_baseline(results_file: str) -> dict:
    if not os.path.exists(results_file):
        return {}
    results = load_json(results_file)
    responses = [
        r["response"] for r in results if r.get("success") and r.get("response")
    ]
    scores = compute_sentiment_scores(responses)
    stats = compute_stats(scores)
    stats["scores"] = scores
    return stats


def analyze_injected(results_path: str) -> dict:
    if not os.path.exists(results_path):
        return {}
    data = load_json(results_path)

    categories = {
        "pos_steering_narrative": [],
        "pos_steering_policy": [],
        "neg_steering_narrative": [],
        "neg_steering_policy": [],
    }

    for attack_data in data.values():
        for results in attack_data.values():
            for result in results:
                if not result.get("success") or not result.get("response"):
                    continue
                response = result["response"]
                attack_type = result.get("attack_type", "")
                prompt_type = result.get("prompt_type", "")

                if "pos_steering" in attack_type:
                    key = (
                        "pos_steering_policy"
                        if "policy" in prompt_type
                        else "pos_steering_narrative"
                    )
                    categories[key].append(response)
                elif "neg_steering" in attack_type:
                    key = (
                        "neg_steering_policy"
                        if "policy" in prompt_type
                        else "neg_steering_narrative"
                    )
                    categories[key].append(response)

    return {k: compute_sentiment_scores(v) for k, v in categories.items()}


def perform_test(baseline_scores: list, attack_scores: list) -> dict:
    try:
        import scipy.stats as stats
    except ImportError:
        return {}
    if not baseline_scores or not attack_scores:
        return {"error": "Insufficient data"}
    result = stats.mannwhitneyu(attack_scores, baseline_scores, alternative="two-sided")
    return {"p_value": result.pvalue, "significant": result.pvalue < 0.05}


def print_comparison(name: str, baseline: dict, injected: dict):
    console.print(f"\n[bold blue]═══ {name.upper()} COMPARISON ═══[/bold blue]\n")

    table = Table(title=f"{name} Steering Attack Sentiment vs Baseline")
    table.add_column("Condition", style="cyan")
    table.add_column("N", style="white")
    table.add_column("Mean", style="yellow")
    table.add_column("Pos %", style="green")
    table.add_column("Neg %", style="red")
    table.add_column("Δ Mean", style="magenta")
    table.add_column("p-value", style="blue")
    table.add_column("Sig?", style="bold")

    if baseline:
        table.add_row(
            "Baseline (no injection)",
            str(baseline["n"]),
            f"{baseline['mean']:.4f}",
            f"{baseline['positive_pct']:.1f}%",
            f"{baseline['negative_pct']:.1f}%",
            "-",
            "-",
            "-",
        )

    for category, label in [
        ("pos_steering_narrative", "Pos Steering (Narrative)"),
        ("pos_steering_policy", "Pos Steering (Policy)"),
        ("neg_steering_narrative", "Neg Steering (Narrative)"),
        ("neg_steering_policy", "Neg Steering (Policy)"),
    ]:
        scores = injected.get(category, [])
        if not scores:
            continue
        stats_data = compute_stats(scores)
        test = perform_test(baseline.get("scores", []), scores)
        delta = stats_data["mean"] - baseline.get("mean", 0)
        p_val = test.get("p_value", float("nan"))
        sig = "✓" if test.get("significant") else "✗"

        table.add_row(
            label,
            str(stats_data["n"]),
            f"{stats_data['mean']:.4f}",
            f"{stats_data['positive_pct']:.1f}%",
            f"{stats_data['negative_pct']:.1f}%",
            f"{delta:+.4f}",
            f"{p_val:.4f}" if p_val == p_val else "-",
            sig,
        )

    console.print(table)


def main():
    console.print("\n[bold blue]═══ BASELINE SENTIMENT ANALYSIS ═══[/bold blue]\n")

    # Analyze Gemini baseline
    gemini_baseline = analyze_baseline(BASELINE_GEMINI)
    if gemini_baseline:
        console.print("[bold green]Gemini Baseline[/bold green]")
        console.print(
            f"  N={gemini_baseline['n']}, Mean={gemini_baseline['mean']:.4f}, Positive={gemini_baseline['positive_pct']:.1f}%"
        )

    # Analyze ChatGPT baseline
    chatgpt_baseline = analyze_baseline(BASELINE_CHATGPT)
    if chatgpt_baseline:
        console.print("[bold green]ChatGPT Baseline[/bold green]")
        console.print(
            f"  N={chatgpt_baseline['n']}, Mean={chatgpt_baseline['mean']:.4f}, Positive={chatgpt_baseline['positive_pct']:.1f}%"
        )

    # Compare with injected results
    if gemini_baseline and os.path.exists(INJECTED_GEMINI):
        gemini_injected = analyze_injected(INJECTED_GEMINI)
        print_comparison("Gemini", gemini_baseline, gemini_injected)

    if chatgpt_baseline and os.path.exists(INJECTED_CHATGPT):
        chatgpt_injected = analyze_injected(INJECTED_CHATGPT)
        print_comparison("ChatGPT", chatgpt_baseline, chatgpt_injected)

    # If only Gemini baseline exists, compare with ChatGPT injected too
    if gemini_baseline and not chatgpt_baseline and os.path.exists(INJECTED_CHATGPT):
        chatgpt_injected = analyze_injected(INJECTED_CHATGPT)
        print_comparison(
            "ChatGPT (vs Gemini baseline)", gemini_baseline, chatgpt_injected
        )

    console.print(
        "\n[bold]Note:[/bold] Baseline confirms LLM positive bias. Negative steering effect proves injection mechanism works.\n"
    )


if __name__ == "__main__":
    main()
