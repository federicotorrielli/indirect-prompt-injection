#!/usr/bin/env python3
"""
Attack Effectiveness Analysis Script

Analyzes the success/failure patterns of different prompt injection attacks
and compares the effectiveness of narrative vs policy_puppetry approaches.
Provides statistical analysis and data insights.

Usage:
    uv run python scripts/analyze_attack_effectiveness.py [results_file]
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore
from rich.console import Console
from rich.table import Table
from scipy import stats  # type: ignore

console = Console()


def parse_attack_key(attack_key: str) -> Tuple[str, str, str]:
    """
    Parse attack key to extract attack type, prompt type, and injection locus.

    Args:
        attack_key: Key like "refusal_attack_narrative_first"

    Returns:
        Tuple of (attack_type, prompt_type, injection_locus)
    """
    parts = attack_key.split("_")

    # Handle different attack types
    if parts[0] == "pos" or parts[0] == "neg":
        attack_type = f"{parts[0]}_steering_attack"
        remaining = "_".join(parts[3:])
    elif parts[0] == "external":
        attack_type = "external_site_attack"
        remaining = "_".join(parts[3:])
    else:
        # refusal_attack, watermark_attack
        attack_type = f"{parts[0]}_attack"
        remaining = "_".join(parts[2:])

    # Extract prompt type and injection locus
    if "narrative" in remaining:
        prompt_type = "narrative"
        injection_locus = remaining.replace("narrative_", "")
    elif "policy_puppetry" in remaining:
        prompt_type = "policy_puppetry"
        injection_locus = remaining.replace("policy_puppetry_", "")
    else:
        prompt_type = "unknown"
        injection_locus = remaining

    return attack_type, prompt_type, injection_locus


def calculate_success_rate(results: List[Dict]) -> Tuple[float, int, int]:
    """Calculate success rate for a list of results."""
    if not results:
        return 0.0, 0, 0

    successes = sum(1 for result in results if result.get("evaluation_success", False))
    total = len(results)
    return successes / total, successes, total


def analyze_attack_data(data: Dict) -> pd.DataFrame:
    """
    Convert nested attack data into a flat DataFrame for analysis.

    Args:
        data: Nested dictionary with attack results

    Returns:
        DataFrame with columns: attack_type, prompt_type, injection_locus,
        request_type, success_rate, successes, total
    """
    rows = []

    for attack_key, attack_data in data.items():
        attack_type, prompt_type, injection_locus = parse_attack_key(attack_key)

        for request_type, results in attack_data.items():
            success_rate, successes, total = calculate_success_rate(results)

            rows.append(
                {
                    "attack_key": attack_key,
                    "attack_type": attack_type,
                    "prompt_type": prompt_type,
                    "injection_locus": injection_locus,
                    "request_type": request_type,
                    "success_rate": success_rate,
                    "successes": successes,
                    "total": total,
                }
            )

    return pd.DataFrame(rows)


def compare_prompt_types(df: pd.DataFrame) -> Dict:
    """
    Compare narrative vs policy_puppetry effectiveness for each attack type.

    Args:
        df: DataFrame with attack analysis results

    Returns:
        Dictionary with statistical comparisons
    """
    comparisons = {}

    # Group by attack type and compare prompt types
    attack_types = df["attack_type"].unique()

    for attack_type in attack_types:
        attack_df = df[df["attack_type"] == attack_type]

        # Get data for each prompt type
        narrative_data = attack_df[attack_df["prompt_type"] == "narrative"]
        policy_data = attack_df[attack_df["prompt_type"] == "policy_puppetry"]

        if len(narrative_data) == 0 or len(policy_data) == 0:
            continue

        # Calculate aggregate success rates
        narrative_successes = narrative_data["successes"].sum()
        narrative_total = narrative_data["total"].sum()
        narrative_rate = (
            narrative_successes / narrative_total if narrative_total > 0 else 0
        )

        policy_successes = policy_data["successes"].sum()
        policy_total = policy_data["total"].sum()
        policy_rate = policy_successes / policy_total if policy_total > 0 else 0

        # Statistical test (Fisher's exact test for small samples, chi-square for large)
        contingency_table = [
            [narrative_successes, narrative_total - narrative_successes],
            [policy_successes, policy_total - policy_successes],
        ]

        if min(contingency_table[0] + contingency_table[1]) < 5:
            # Use Fisher's exact test for small samples
            _, p_value = stats.fisher_exact(contingency_table)
            test_name = "Fisher's exact test"
        else:
            # Use chi-square test for larger samples
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            test_name = "Chi-square test"

        # Effect size (odds ratio)
        odds_ratio = None
        if narrative_total > 0 and policy_total > 0:
            try:
                odds_ratio = stats.contingency.odds_ratio(contingency_table).statistic
            except Exception:
                odds_ratio = None

        comparisons[attack_type] = {
            "narrative": {
                "success_rate": narrative_rate,
                "successes": narrative_successes,
                "total": narrative_total,
            },
            "policy_puppetry": {
                "success_rate": policy_rate,
                "successes": policy_successes,
                "total": policy_total,
            },
            "statistical_test": test_name,
            "p_value": p_value,
            "odds_ratio": odds_ratio,
            "significant": p_value < 0.05 if p_value is not None else False,
            "better_approach": "narrative"
            if narrative_rate > policy_rate
            else "policy_puppetry",
            "rate_difference": abs(narrative_rate - policy_rate),
        }

    return comparisons


def print_summary_statistics(df: pd.DataFrame):
    """Print overall summary statistics."""
    console.print("\n[bold blue]📊 Overall Attack Success Summary[/bold blue]")

    # Overall success rates by attack type
    attack_summary = (
        df.groupby("attack_type")
        .agg({"successes": "sum", "total": "sum"})
        .reset_index()
    )
    attack_summary["success_rate"] = (
        attack_summary["successes"] / attack_summary["total"]
    )
    attack_summary = attack_summary.sort_values("success_rate", ascending=False)

    table = Table(title="Success Rates by Attack Type")
    table.add_column("Attack Type", style="cyan")
    table.add_column("Success Rate", style="green")
    table.add_column("Successes", style="yellow")
    table.add_column("Total", style="blue")

    for _, row in attack_summary.iterrows():
        table.add_row(
            row["attack_type"],
            f"{row['success_rate']:.1%}",
            str(row["successes"]),
            str(row["total"]),
        )

    console.print(table)


def print_prompt_type_comparison(comparisons: Dict):
    """Print detailed comparison between narrative and policy_puppetry approaches."""
    console.print(
        "\n[bold green]🎯 Narrative vs Policy Puppetry Comparison[/bold green]"
    )

    table = Table(title="Attack Effectiveness: Narrative vs Policy Puppetry")
    table.add_column("Attack Type", style="cyan")
    table.add_column("Narrative Success", style="blue")
    table.add_column("Policy Success", style="magenta")
    table.add_column("Better Approach", style="green")
    table.add_column("Difference", style="yellow")
    table.add_column("P-value", style="red")
    table.add_column("Significant", style="bold")

    # Sort by rate difference (most dramatic differences first)
    sorted_attacks = sorted(
        comparisons.items(), key=lambda x: x[1]["rate_difference"], reverse=True
    )

    for attack_type, comp in sorted_attacks:
        narrative_rate = comp["narrative"]["success_rate"]
        policy_rate = comp["policy_puppetry"]["success_rate"]

        # Format success rates with counts
        narrative_str = f"{narrative_rate:.1%} ({comp['narrative']['successes']}/{comp['narrative']['total']})"
        policy_str = f"{policy_rate:.1%} ({comp['policy_puppetry']['successes']}/{comp['policy_puppetry']['total']})"

        # Determine better approach with emoji
        better = comp["better_approach"]
        if comp["rate_difference"] < 0.01:  # Less than 1% difference
            better_display = "🤝 Tie"
        elif better == "narrative":
            better_display = "📖 Narrative"
        else:
            better_display = "⚙️ Policy"

        # Format significance
        significance = "✅ Yes" if comp["significant"] else "❌ No"

        table.add_row(
            attack_type.replace("_", " ").title(),
            narrative_str,
            policy_str,
            better_display,
            f"{comp['rate_difference']:.1%}",
            f"{comp['p_value']:.3f}" if comp["p_value"] is not None else "N/A",
            significance,
        )

    console.print(table)


def print_failed_attacks_analysis(df: pd.DataFrame):
    """Analyze and print details about failed attacks."""
    console.print("\n[bold red]❌ Failed Attacks Analysis[/bold red]")

    # Find attacks with 0% success rate
    failed_attacks = df[df["success_rate"] == 0.0]

    if len(failed_attacks) == 0:
        console.print("[green]🎉 No completely failed attacks found![/green]")
        return

    table = Table(title="Completely Failed Attack Configurations")
    table.add_column("Attack Type", style="cyan")
    table.add_column("Prompt Type", style="blue")
    table.add_column("Request Type", style="magenta")
    table.add_column("Total Attempts", style="yellow")

    for _, row in failed_attacks.iterrows():
        table.add_row(
            row["attack_type"],
            row["prompt_type"],
            row["request_type"],
            str(row["total"]),
        )

    console.print(table)

    # Low success rate attacks (< 10%)
    low_success = df[df["success_rate"] < 0.1]
    low_success = low_success[
        low_success["success_rate"] > 0
    ]  # Exclude 0% (already shown)

    if len(low_success) > 0:
        console.print("\n[bold yellow]⚠️ Low Success Rate Attacks (< 10%)[/bold yellow]")

        table = Table(title="Low Success Rate Attack Configurations")
        table.add_column("Attack Type", style="cyan")
        table.add_column("Prompt Type", style="blue")
        table.add_column("Request Type", style="magenta")
        table.add_column("Success Rate", style="red")
        table.add_column("Successes/Total", style="yellow")

        for _, row in low_success.iterrows():
            table.add_row(
                row["attack_type"],
                row["prompt_type"],
                row["request_type"],
                f"{row['success_rate']:.1%}",
                f"{row['successes']}/{row['total']}",
            )

        console.print(table)


def print_statistical_insights(comparisons: Dict):
    """Print key statistical insights."""
    console.print("\n[bold purple]🔬 Statistical Insights[/bold purple]")

    significant_differences = [
        (attack, comp) for attack, comp in comparisons.items() if comp["significant"]
    ]

    if significant_differences:
        console.print(
            f"[green]✅ Found {len(significant_differences)} statistically significant differences:[/green]"
        )
        for attack, comp in significant_differences:
            better = comp["better_approach"].replace("_", " ").title()
            diff = comp["rate_difference"]
            console.print(
                f"  • {attack}: {better} performs {diff:.1%} better (p={comp['p_value']:.3f})"
            )
    else:
        console.print(
            "[yellow]⚠️ No statistically significant differences found between approaches[/yellow]"
        )

    # Overall trends
    narrative_wins = sum(
        1 for comp in comparisons.values() if comp["better_approach"] == "narrative"
    )
    policy_wins = sum(
        1
        for comp in comparisons.values()
        if comp["better_approach"] == "policy_puppetry"
    )

    console.print("\n[bold]📈 Overall Trends:[/bold]")
    console.print(
        f"  • Narrative approach wins: {narrative_wins}/{len(comparisons)} attack types"
    )
    console.print(
        f"  • Policy puppetry approach wins: {policy_wins}/{len(comparisons)} attack types"
    )

    if narrative_wins > policy_wins:
        console.print(
            "  • [green]Narrative approach appears more effective overall[/green]"
        )
    elif policy_wins > narrative_wins:
        console.print(
            "  • [green]Policy puppetry approach appears more effective overall[/green]"
        )
    else:
        console.print("  • [yellow]Both approaches perform similarly overall[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attack effectiveness and compare narrative vs policy_puppetry approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        default="results/all_results_chatgpt_evaluated.json",
        help="Path to the evaluated results JSON file",
    )

    args = parser.parse_args()

    console.print("[bold cyan]🔍 Attack Effectiveness Analysis[/bold cyan]\n")

    # Load data
    if not os.path.exists(args.results_file):
        console.print(f"[red]❌ Error: File {args.results_file} not found[/red]")
        return 1

    console.print(f"[blue]📁 Loading results from: {args.results_file}[/blue]")

    try:
        with open(args.results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]❌ Error loading file: {e}[/red]")
        return 1

    # Analyze data
    console.print("[blue]🔄 Analyzing attack patterns...[/blue]")
    df = analyze_attack_data(data)

    if len(df) == 0:
        console.print("[red]❌ No data found to analyze[/red]")
        return 1

    console.print(f"[green]✅ Analyzed {len(df)} attack configurations[/green]")

    # Perform comparisons
    console.print("[blue]📊 Comparing prompt types...[/blue]")
    comparisons = compare_prompt_types(df)

    # Print results
    print_summary_statistics(df)
    print_prompt_type_comparison(comparisons)
    print_failed_attacks_analysis(df)
    print_statistical_insights(comparisons)

    # Save detailed results
    output_file = args.results_file.replace(".json", "_analysis.csv")
    df.to_csv(output_file, index=False)
    console.print(f"\n[green]💾 Detailed analysis saved to: {output_file}[/green]")

    return 0


if __name__ == "__main__":
    exit(main())
