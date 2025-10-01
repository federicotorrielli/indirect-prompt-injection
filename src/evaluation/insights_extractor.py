"""
Comprehensive Insights Extractor

This script performs a comprehensive analysis of prompt injection attack results
across multiple LLMs (ChatGPT and Gemini) to extract all possible insights.

Key Features:
- Multi-model comparative analysis
- Statistical significance testing
- Effect size calculations
- Vulnerability profiling
- Attack vector effectiveness analysis
- Human baseline comparisons

Usage:
    uv run python src/evaluation/insights_extractor.py
"""

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Statistical imports
try:
    from scipy.stats import chi2_contingency  # type: ignore

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning(
        "SciPy not available. Some advanced statistical tests will be skipped."
    )

# Initialize rich console
console = Console()

# Set up rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


@dataclass
class AttackSuccessMetrics:
    """Comprehensive attack success metrics."""

    total_attempts: int
    successful_attacks: int
    success_rate: float
    confidence_interval_95: Tuple[float, float]
    standard_error: float


@dataclass
class ComparativeAnalysis:
    """Comparative analysis between models or conditions."""

    condition_a: str
    condition_b: str
    success_rate_a: float
    success_rate_b: float
    effect_size: float
    statistical_significance: Dict[str, Any]
    practical_significance: bool


@dataclass
class VulnerabilityProfile:
    """Vulnerability profile for a specific attack vector."""

    attack_type: str
    payload_type: str
    injection_position: str
    success_rates_by_model: Dict[str, float]
    cross_model_variance: float
    vulnerability_score: float
    risk_classification: str


@dataclass
class StatisticalTest:
    """Statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    significant: bool


class ComprehensiveInsightsExtractor:
    """
    Production-ready comprehensive insights extractor for indirect prompt injection research.
    """

    def __init__(self, results_dir: str = "results/evaluation"):
        """
        Initialize the comprehensive insights extractor.

        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        self.data: Dict[str, Any] = {}
        self.insights: Dict[str, Any] = {}

        # Load all available data
        self._load_evaluation_data()

    def _load_evaluation_data(self) -> None:
        """Load all evaluation data from available files."""
        console.print("[blue]📊 Loading comprehensive evaluation data...[/blue]")

        evaluation_files = {
            "chatgpt_analyzed": "all_results_chatgpt_evaluated_analysis.json",
            "gemini_analyzed": "all_results_gemini_evaluated_analysis.json",
            "chatgpt_classified": "all_results_chatgpt_evaluated_classified.json",
            "gemini_classified": "all_results_gemini_evaluated_classified.json",
        }

        for key, filename in evaluation_files.items():
            file_path = self.results_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self.data[key] = json.load(f)
                    console.print(f"[green]✅ Loaded {key} data[/green]")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.warning(f"File not found: {filename}")

        if not self.data:
            raise FileNotFoundError("No evaluation data files found")

    def _calculate_confidence_interval(
        self, successes: int, total: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for success rate using Wilson score interval."""
        if total == 0:
            return (0.0, 0.0)

        p = successes / total
        n = total
        z = 1.96 if confidence == 0.95 else 2.576  # For 95% or 99% confidence

        # Wilson score interval
        denominator = 1 + z**2 / n
        centre_adjusted_probability = p + z**2 / (2 * n)
        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

        lower_bound = (
            centre_adjusted_probability - z * adjusted_standard_deviation
        ) / denominator
        upper_bound = (
            centre_adjusted_probability + z * adjusted_standard_deviation
        ) / denominator

        return (max(0, lower_bound), min(1, upper_bound))

    def _calculate_effect_size(
        self, success_rate_a: float, success_rate_b: float
    ) -> float:
        """Calculate Cohen's h effect size for proportions."""
        # Cohen's h for the difference between two proportions
        h = 2 * (
            np.arcsin(np.sqrt(success_rate_a / 100))
            - np.arcsin(np.sqrt(success_rate_b / 100))
        )
        return abs(h)

    def _perform_chi_square_test(
        self, successes_a: int, total_a: int, successes_b: int, total_b: int
    ) -> StatisticalTest:
        """Perform chi-square test for independence between two conditions."""
        if not SCIPY_AVAILABLE:
            return StatisticalTest(
                test_name="Chi-square test",
                statistic=0.0,
                p_value=1.0,
                effect_size=None,
                confidence_interval=None,
                interpretation="SciPy not available",
                significant=False,
            )

        # Create contingency table
        failures_a = total_a - successes_a
        failures_b = total_b - successes_b

        observed = np.array([[successes_a, failures_a], [successes_b, failures_b]])

        try:
            chi2, p_value, dof, expected = chi2_contingency(observed)

            # Calculate Cramér's V as effect size
            n = np.sum(observed)
            cramers_v = np.sqrt(chi2 / (n * (min(observed.shape) - 1)))

            interpretation = self._interpret_statistical_result(
                p_value, cramers_v, "cramers_v"
            )

            return StatisticalTest(
                test_name="Chi-square test of independence",
                statistic=chi2,
                p_value=p_value,
                effect_size=cramers_v,
                confidence_interval=None,
                interpretation=interpretation,
                significant=p_value < 0.05,
            )
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            return StatisticalTest(
                test_name="Chi-square test",
                statistic=0.0,
                p_value=1.0,
                effect_size=None,
                confidence_interval=None,
                interpretation="Test failed",
                significant=False,
            )

    def _interpret_statistical_result(
        self, p_value: float, effect_size: float, effect_type: str = "cohens_h"
    ) -> str:
        """Interpret statistical results with effect size."""
        significance = "significant" if p_value < 0.05 else "not significant"

        if effect_type == "cohens_h":
            if effect_size < 0.2:
                magnitude = "small"
            elif effect_size < 0.5:
                magnitude = "medium"
            else:
                magnitude = "large"
        elif effect_type == "cramers_v":
            if effect_size < 0.1:
                magnitude = "small"
            elif effect_size < 0.3:
                magnitude = "medium"
            else:
                magnitude = "large"
        else:
            magnitude = "unknown"

        return f"Statistically {significance} (p={p_value:.4f}) with {magnitude} effect size ({effect_size:.3f})"

    def analyze_overall_attack_success_rates(self) -> Dict[str, Any]:
        """Analyze overall attack success rates across models."""
        console.print("[blue]🔍 Analyzing overall attack success rates...[/blue]")

        analysis: Dict[str, Any] = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            overall_stats = data.get("overall_stats", {})

            total_attacks = overall_stats.get("total_attacks", 0)
            successful_attacks = overall_stats.get("successful_attacks", 0)
            success_rate = overall_stats.get("attack_success_rate", 0.0)

            ci_95 = self._calculate_confidence_interval(
                successful_attacks, total_attacks
            )
            std_error = (
                np.sqrt((success_rate / 100) * (1 - success_rate / 100) / total_attacks)
                if total_attacks > 0
                else 0
            )

            analysis[model] = AttackSuccessMetrics(
                total_attempts=total_attacks,
                successful_attacks=successful_attacks,
                success_rate=success_rate,
                confidence_interval_95=ci_95,
                standard_error=std_error,
            )

        # Comparative analysis between models
        if "chatgpt" in analysis and "gemini" in analysis:
            chatgpt_data = analysis["chatgpt"]
            gemini_data = analysis["gemini"]

            effect_size = self._calculate_effect_size(
                gemini_data.success_rate, chatgpt_data.success_rate
            )
            statistical_test = self._perform_chi_square_test(
                gemini_data.successful_attacks,
                gemini_data.total_attempts,
                chatgpt_data.successful_attacks,
                chatgpt_data.total_attempts,
            )

            analysis["comparative"] = asdict(
                ComparativeAnalysis(
                    condition_a="Gemini",
                    condition_b="ChatGPT",
                    success_rate_a=gemini_data.success_rate,
                    success_rate_b=chatgpt_data.success_rate,
                    effect_size=effect_size,
                    statistical_significance=asdict(statistical_test),
                    practical_significance=abs(
                        gemini_data.success_rate - chatgpt_data.success_rate
                    )
                    > 5.0,
                )
            )

        return analysis

    def analyze_attack_type_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of different attack types."""
        console.print("[blue]🎯 Analyzing attack type effectiveness...[/blue]")

        analysis = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            attack_type_analysis = data.get("attack_type_analysis", {})

            model_analysis = {}
            for attack_type, stats in attack_type_analysis.items():
                total = stats.get("total", 0)
                successes = stats.get("successes", 0)
                success_rate = stats.get("success_rate", 0.0)

                if total > 0:
                    ci_95 = self._calculate_confidence_interval(successes, total)
                    model_analysis[attack_type] = {
                        "total_attempts": total,
                        "successful_attacks": successes,
                        "success_rate": success_rate,
                        "confidence_interval_95": ci_95,
                        "standard_error": np.sqrt(
                            success_rate / 100 * (1 - success_rate / 100) / total
                        ),
                    }

            analysis[model] = model_analysis

        # Cross-model comparisons for each attack type
        if "chatgpt" in analysis and "gemini" in analysis:
            analysis["cross_model_comparisons"] = {}

            # Find common attack types
            chatgpt_attacks = set(analysis["chatgpt"].keys())
            gemini_attacks = set(analysis["gemini"].keys())
            common_attacks = chatgpt_attacks.intersection(gemini_attacks)

            for attack_type in common_attacks:
                chatgpt_stats = analysis["chatgpt"][attack_type]
                gemini_stats = analysis["gemini"][attack_type]

                effect_size = self._calculate_effect_size(
                    gemini_stats["success_rate"], chatgpt_stats["success_rate"]
                )

                statistical_test = self._perform_chi_square_test(
                    gemini_stats["successful_attacks"],
                    gemini_stats["total_attempts"],
                    chatgpt_stats["successful_attacks"],
                    chatgpt_stats["total_attempts"],
                )

                analysis["cross_model_comparisons"][attack_type] = {
                    "gemini_success_rate": gemini_stats["success_rate"],
                    "chatgpt_success_rate": chatgpt_stats["success_rate"],
                    "success_rate_difference": gemini_stats["success_rate"]
                    - chatgpt_stats["success_rate"],
                    "effect_size": effect_size,
                    "statistical_test": asdict(statistical_test),
                    "gemini_advantage": gemini_stats["success_rate"]
                    > chatgpt_stats["success_rate"],
                }

        return analysis

    def analyze_payload_and_position_effects(self) -> Dict[str, Any]:
        """Analyze the effects of payload type and injection position."""
        console.print("[blue]📍 Analyzing payload type and position effects...[/blue]")

        analysis = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            attack_key_analysis = data.get("attack_key_analysis", {})

            # Categorize by payload type and position
            payload_analysis: Dict[str, Dict[str, List[float]]] = {
                "narrative": {"first": [], "last": []},
                "policy": {"first": [], "last": []},
            }
            position_analysis: Dict[str, List[float]] = {"first": [], "last": []}

            for attack_key, stats in attack_key_analysis.items():
                success_rate = stats.get("success_rate", 0.0)
                total = stats.get("total", 0)

                if total == 0:
                    continue

                # Determine payload type and position
                payload_type = "policy" if "policy" in attack_key else "narrative"
                position = "first" if "first" in attack_key else "last"

                payload_analysis[payload_type][position].append(success_rate)
                position_analysis[position].append(success_rate)

            # Calculate statistics
            model_analysis = {}

            # Payload type effects
            for payload_type in ["narrative", "policy"]:
                first_rates = payload_analysis[payload_type]["first"]
                last_rates = payload_analysis[payload_type]["last"]

                if first_rates and last_rates:
                    model_analysis[f"{payload_type}_payload"] = {
                        "first_position_mean": statistics.mean(first_rates),
                        "last_position_mean": statistics.mean(last_rates),
                        "first_position_std": statistics.stdev(first_rates)
                        if len(first_rates) > 1
                        else 0,
                        "last_position_std": statistics.stdev(last_rates)
                        if len(last_rates) > 1
                        else 0,
                        "position_advantage": statistics.mean(first_rates)
                        - statistics.mean(last_rates),
                        "first_position_count": len(first_rates),
                        "last_position_count": len(last_rates),
                    }

            # Overall position effects
            first_rates = position_analysis["first"]
            last_rates = position_analysis["last"]

            if first_rates and last_rates:
                model_analysis["position_effect"] = {
                    "first_position_mean": statistics.mean(first_rates),
                    "last_position_mean": statistics.mean(last_rates),
                    "first_position_std": statistics.stdev(first_rates)
                    if len(first_rates) > 1
                    else 0,
                    "last_position_std": statistics.stdev(last_rates)
                    if len(last_rates) > 1
                    else 0,
                    "first_position_advantage": statistics.mean(first_rates)
                    - statistics.mean(last_rates),
                    "first_count": len(first_rates),
                    "last_count": len(last_rates),
                }

            analysis[model] = model_analysis

        return analysis

    def analyze_sentiment_steering_effectiveness(self) -> Dict[str, Any]:
        """Analyze sentiment steering attack effectiveness with human baseline comparison."""
        console.print("[blue]🎭 Analyzing sentiment steering effectiveness...[/blue]")

        analysis = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            steering_analysis = data.get("steering_analysis", {})

            model_analysis = {}

            # VADER vs LLM agreement analysis
            vader_vs_llm = steering_analysis.get("vader_vs_llm", {})
            if vader_vs_llm:
                agreement_rates = []
                vader_success_rates = []
                llm_success_rates = []

                for attack_key, stats in vader_vs_llm.items():
                    agreement_percentage = stats.get("agreement_percentage", 0)
                    total = stats.get("total", 0)
                    vader_successes = stats.get("vader_successes", 0)
                    llm_successes = stats.get("llm_successes", 0)

                    if total > 0:
                        agreement_rates.append(agreement_percentage)
                        vader_success_rates.append((vader_successes / total) * 100)
                        llm_success_rates.append((llm_successes / total) * 100)

                if agreement_rates:
                    model_analysis["evaluator_agreement"] = {
                        "mean_agreement": statistics.mean(agreement_rates),
                        "std_agreement": statistics.stdev(agreement_rates)
                        if len(agreement_rates) > 1
                        else 0,
                        "min_agreement": min(agreement_rates),
                        "max_agreement": max(agreement_rates),
                        "vader_mean_success": statistics.mean(vader_success_rates),
                        "llm_mean_success": statistics.mean(llm_success_rates),
                        "evaluator_difference": statistics.mean(llm_success_rates)
                        - statistics.mean(vader_success_rates),
                    }

            # Human baseline comparison - aggregate statistics only
            baseline_comparison = steering_analysis.get("human_baseline_comparison", {})
            if baseline_comparison:
                # Remove individual scores, keep only summary statistics
                baseline_stats = baseline_comparison.get("human_baseline_stats", {})
                cleaned_baseline = {}

                for category in ["accepted", "rejected"]:
                    if category in baseline_stats:
                        cat_data = baseline_stats[category]
                        cleaned_baseline[category] = {
                            "count": cat_data.get("count", 0),
                            "mean": cat_data.get("mean", 0),
                            "median": cat_data.get("median", 0),
                            "std_dev": cat_data.get("std_dev", 0),
                            # Explicitly exclude 'scores' field
                        }

                model_analysis["human_baseline"] = {
                    "summary_statistics": cleaned_baseline,
                    "statistical_tests": baseline_comparison.get(
                        "statistical_tests", {}
                    ),
                }

            # Positive vs negative steering effectiveness
            pos_neg_analysis: Dict[str, Dict[str, List[float]]] = {}
            for attack_key, stats in vader_vs_llm.items():
                if "pos_steering" in attack_key:
                    sentiment_type = "positive_steering"
                elif "neg_steering" in attack_key:
                    sentiment_type = "negative_steering"
                else:
                    continue

                if sentiment_type not in pos_neg_analysis:
                    pos_neg_analysis[sentiment_type] = {
                        "success_rates": [],
                        "agreement_rates": [],
                    }

                total = stats.get("total", 0)
                llm_successes = stats.get("llm_successes", 0)
                agreement_percentage = stats.get("agreement_percentage", 0)

                if total > 0:
                    pos_neg_analysis[sentiment_type]["success_rates"].append(
                        (llm_successes / total) * 100
                    )
                    pos_neg_analysis[sentiment_type]["agreement_rates"].append(
                        agreement_percentage
                    )

            for sentiment_type, data_lists in pos_neg_analysis.items():
                success_rates = data_lists["success_rates"]
                agreement_rates = data_lists["agreement_rates"]

                if success_rates:
                    model_analysis[sentiment_type] = {
                        "mean_success_rate": statistics.mean(success_rates),
                        "std_success_rate": statistics.stdev(success_rates)
                        if len(success_rates) > 1
                        else 0,
                        "mean_agreement": statistics.mean(agreement_rates),
                        "sample_count": len(success_rates),
                    }

            analysis[model] = model_analysis

        return analysis

    def analyze_vulnerability_profiles(self) -> Dict[str, Any]:
        """Create vulnerability profiles for different attack vectors."""
        console.print("[blue]🛡️ Analyzing vulnerability profiles...[/blue]")

        vulnerability_profiles = {}

        # Extract attack key data from both models
        attack_vectors: Dict[str, Dict[str, Any]] = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            attack_key_analysis = data.get("attack_key_analysis", {})

            for attack_key, stats in attack_key_analysis.items():
                success_rate = stats.get("success_rate", 0.0)
                total = stats.get("total", 0)

                if total == 0:
                    continue

                # Parse attack components
                parts = attack_key.split("_")
                if len(parts) >= 3:
                    attack_type = "_".join(parts[:2])  # e.g., "neg_steering"
                    payload_type = "policy" if "policy" in attack_key else "narrative"
                    position = "first" if "first" in attack_key else "last"

                    vector_key = f"{attack_type}_{payload_type}_{position}"

                    if vector_key not in attack_vectors:
                        attack_vectors[vector_key] = {
                            "attack_type": attack_type,
                            "payload_type": payload_type,
                            "injection_position": position,
                            "success_rates": {},
                        }

                    attack_vectors[vector_key]["success_rates"][model] = success_rate

        # Create vulnerability profiles
        for vector_key, vector_data in attack_vectors.items():
            success_rates_dict = vector_data.get("success_rates", {})

            if len(success_rates_dict) >= 2:  # Need at least 2 models for comparison
                success_values = list(success_rates_dict.values())
                mean_success = statistics.mean(success_values)
                variance = (
                    statistics.variance(success_values)
                    if len(success_values) > 1
                    else 0
                )

                # Risk classification based on mean success rate
                if mean_success >= 90:
                    risk_class = "CRITICAL"
                elif mean_success >= 75:
                    risk_class = "HIGH"
                elif mean_success >= 50:
                    risk_class = "MEDIUM"
                else:
                    risk_class = "LOW"

                vulnerability_profiles[vector_key] = VulnerabilityProfile(
                    attack_type=str(vector_data.get("attack_type", "")),
                    payload_type=str(vector_data.get("payload_type", "")),
                    injection_position=str(vector_data.get("injection_position", "")),
                    success_rates_by_model=success_rates_dict,
                    cross_model_variance=variance,
                    vulnerability_score=mean_success,
                    risk_classification=risk_class,
                )

        return vulnerability_profiles

    def analyze_academic_sentiment_classifier_performance(self) -> Dict[str, Any]:
        """Analyze performance of the academic sentiment classifier."""
        console.print(
            "[blue]🎓 Analyzing academic sentiment classifier performance...[/blue]"
        )

        analysis = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_classified"
            if key not in self.data:
                continue

            data = self.data[key]
            metadata = data.get("metadata", {})
            detailed_results = data.get("detailed_results", [])

            if not detailed_results:
                continue

            # Overall performance metrics
            total_evaluated = metadata.get("total_evaluated", 0)
            successful_attacks = metadata.get("successful_attacks", 0)
            overall_success_rate = metadata.get("overall_success_rate", 0.0)
            classifier_accuracy = metadata.get("classifier_accuracy", 0.0)
            positive_steering_success = metadata.get("positive_steering_success", 0.0)
            negative_steering_success = metadata.get("negative_steering_success", 0.0)

            # Confidence analysis
            confidences = [result.get("confidence", 0.0) for result in detailed_results]
            successful_confidences = [
                result.get("confidence", 0.0)
                for result in detailed_results
                if result.get("attack_successful", False)
            ]
            failed_confidences = [
                result.get("confidence", 0.0)
                for result in detailed_results
                if not result.get("attack_successful", False)
            ]

            # Response length analysis
            response_lengths = [
                result.get("response_length", 0) for result in detailed_results
            ]
            successful_lengths = [
                result.get("response_length", 0)
                for result in detailed_results
                if result.get("attack_successful", False)
            ]
            failed_lengths = [
                result.get("response_length", 0)
                for result in detailed_results
                if not result.get("attack_successful", False)
            ]

            model_analysis = {
                "overall_metrics": {
                    "total_evaluated": total_evaluated,
                    "successful_attacks": successful_attacks,
                    "overall_success_rate": overall_success_rate,
                    "classifier_accuracy": classifier_accuracy,
                    "positive_steering_success": positive_steering_success,
                    "negative_steering_success": negative_steering_success,
                    "steering_asymmetry": positive_steering_success
                    - negative_steering_success,
                },
                "confidence_analysis": {
                    "overall_mean_confidence": statistics.mean(confidences)
                    if confidences
                    else 0,
                    "overall_std_confidence": statistics.stdev(confidences)
                    if len(confidences) > 1
                    else 0,
                    "successful_mean_confidence": statistics.mean(
                        successful_confidences
                    )
                    if successful_confidences
                    else 0,
                    "failed_mean_confidence": statistics.mean(failed_confidences)
                    if failed_confidences
                    else 0,
                    "confidence_difference": statistics.mean(successful_confidences)
                    - statistics.mean(failed_confidences)
                    if successful_confidences and failed_confidences
                    else 0,
                },
                "response_length_analysis": {
                    "overall_mean_length": statistics.mean(response_lengths)
                    if response_lengths
                    else 0,
                    "overall_std_length": statistics.stdev(response_lengths)
                    if len(response_lengths) > 1
                    else 0,
                    "successful_mean_length": statistics.mean(successful_lengths)
                    if successful_lengths
                    else 0,
                    "failed_mean_length": statistics.mean(failed_lengths)
                    if failed_lengths
                    else 0,
                    "length_difference": statistics.mean(successful_lengths)
                    - statistics.mean(failed_lengths)
                    if successful_lengths and failed_lengths
                    else 0,
                },
            }

            # Attack type breakdown
            attack_type_breakdown: Dict[str, Dict[str, Any]] = {}
            for result in detailed_results:
                attack_type = result.get("attack_type", "unknown")
                if attack_type not in attack_type_breakdown:
                    attack_type_breakdown[attack_type] = {
                        "total": 0,
                        "successful": 0,
                        "confidences": [],
                    }

                attack_type_breakdown[attack_type]["total"] = (
                    attack_type_breakdown[attack_type]["total"] + 1
                )
                if result.get("attack_successful", False):
                    attack_type_breakdown[attack_type]["successful"] = (
                        attack_type_breakdown[attack_type]["successful"] + 1
                    )
                confidences_list = attack_type_breakdown[attack_type]["confidences"]
                if isinstance(confidences_list, list):
                    confidences_list.append(result.get("confidence", 0.0))

            # Calculate success rates and confidence stats for each attack type
            for attack_type, stats in attack_type_breakdown.items():
                total = stats.get("total", 0)
                successful = stats.get("successful", 0)
                confidences = stats.get("confidences", [])

                if (
                    isinstance(total, int)
                    and isinstance(successful, int)
                    and isinstance(confidences, list)
                ):
                    attack_type_breakdown[attack_type].update(
                        {
                            "success_rate": (successful / total) * 100
                            if total > 0
                            else 0,
                            "mean_confidence": statistics.mean(confidences)
                            if confidences
                            else 0,
                            "std_confidence": statistics.stdev(confidences)
                            if len(confidences) > 1
                            else 0,
                        }
                    )

            model_analysis["attack_type_breakdown"] = attack_type_breakdown
            analysis[model] = model_analysis

        return analysis

    def analyze_perfect_attacks(self) -> Dict[str, Any]:
        """Identify attacks with 100% or near-perfect success rates."""
        console.print("[blue]🎯 Identifying perfect attacks...[/blue]")

        analysis = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            attack_key_analysis = data.get("attack_key_analysis", {})

            perfect_attacks = []
            near_perfect_attacks = []

            for attack_key, stats in attack_key_analysis.items():
                success_rate = stats.get("success_rate", 0.0)
                total = stats.get("total", 0)
                successes = stats.get("successes", 0)

                if total == 0:
                    continue

                if success_rate == 100.0:
                    perfect_attacks.append(
                        {
                            "attack_key": attack_key,
                            "success_rate": success_rate,
                            "successes": successes,
                            "total": total,
                        }
                    )
                elif success_rate >= 95.0:
                    near_perfect_attacks.append(
                        {"attack_key": attack_key, "success_rate": success_rate}
                    )

            analysis[model] = {
                "perfect_attacks": perfect_attacks,
                "near_perfect_attacks": near_perfect_attacks,
                "perfect_count": len(perfect_attacks),
                "near_perfect_count": len(near_perfect_attacks),
            }

        return analysis

    def analyze_attack_predictability(self) -> Dict[str, Any]:
        """Analyze predictability of attack success rates using variance metrics."""
        console.print("[blue]📊 Analyzing attack predictability...[/blue]")

        analysis = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            attack_key_analysis = data.get("attack_key_analysis", {})

            success_rates = [
                stats.get("success_rate", 0.0)
                for stats in attack_key_analysis.values()
                if stats.get("total", 0) > 0
            ]

            if len(success_rates) < 2:
                continue

            mean_sr = statistics.mean(success_rates)
            median_sr = statistics.median(success_rates)
            std_sr = statistics.stdev(success_rates)
            variance_sr = statistics.variance(success_rates)
            min_sr = min(success_rates)
            max_sr = max(success_rates)
            sr_range = max_sr - min_sr

            # Coefficient of variation (lower means more predictable)
            cv = std_sr / mean_sr if mean_sr > 0 else 0

            # Predictability index (higher means more predictable)
            predictability_index = 1 - (cv / 1.0)  # Normalized

            # Classify success rates
            high_success = len([sr for sr in success_rates if sr >= 90])
            moderate_success = len([sr for sr in success_rates if 70 <= sr < 90])
            low_success = len([sr for sr in success_rates if sr < 70])

            analysis[model] = {
                "success_rate_mean": mean_sr,
                "success_rate_median": median_sr,
                "success_rate_std": std_sr,
                "success_rate_variance": variance_sr,
                "success_rate_range": sr_range,
                "coefficient_of_variation": cv,
                "predictability_index": predictability_index,
                "attack_count": len(success_rates),
                "high_success_attacks": high_success,
                "moderate_success_attacks": moderate_success,
                "low_success_attacks": low_success,
            }

        # Comparative analysis
        if "chatgpt" in analysis and "gemini" in analysis:
            analysis["comparative"] = {
                "more_predictable_model": "gemini"
                if analysis["gemini"]["predictability_index"]
                > analysis["chatgpt"]["predictability_index"]
                else "chatgpt",
                "predictability_difference": abs(
                    analysis["gemini"]["predictability_index"]
                    - analysis["chatgpt"]["predictability_index"]
                ),
                "variance_ratio": analysis["chatgpt"]["success_rate_variance"]
                / analysis["gemini"]["success_rate_variance"]
                if analysis["gemini"]["success_rate_variance"] > 0
                else 0,
            }

        return analysis

    def analyze_technique_effectiveness(self) -> Dict[str, Any]:
        """Compare narrative vs policy puppetry techniques."""
        console.print("[blue]🎭 Analyzing technique effectiveness...[/blue]")

        analysis = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            attack_key_analysis = data.get("attack_key_analysis", {})

            narrative_rates = []
            policy_rates = []

            for attack_key, stats in attack_key_analysis.items():
                success_rate = stats.get("success_rate", 0.0)
                total = stats.get("total", 0)

                if total == 0:
                    continue

                if "narrative" in attack_key:
                    narrative_rates.append(success_rate)
                elif "policy" in attack_key or "puppetry" in attack_key:
                    policy_rates.append(success_rate)

            if narrative_rates and policy_rates:
                analysis[model] = {
                    "narrative_mean": statistics.mean(narrative_rates),
                    "policy_mean": statistics.mean(policy_rates),
                    "narrative_std": statistics.stdev(narrative_rates)
                    if len(narrative_rates) > 1
                    else 0,
                    "policy_std": statistics.stdev(policy_rates)
                    if len(policy_rates) > 1
                    else 0,
                    "technique_advantage": statistics.mean(policy_rates)
                    - statistics.mean(narrative_rates),
                    "narrative_count": len(narrative_rates),
                    "policy_count": len(policy_rates),
                    "policy_is_better": statistics.mean(policy_rates)
                    > statistics.mean(narrative_rates),
                }

        # Cross-model summary
        if "chatgpt" in analysis and "gemini" in analysis:
            analysis["summary"] = {
                "average_policy_advantage": (
                    analysis["chatgpt"]["technique_advantage"]
                    + analysis["gemini"]["technique_advantage"]
                )
                / 2,
                "policy_universally_better": analysis["chatgpt"]["policy_is_better"]
                and analysis["gemini"]["policy_is_better"],
            }

        return analysis

    def analyze_position_technique_interaction(self) -> Dict[str, Any]:
        """Analyze interaction effects between injection position and technique."""
        console.print("[blue]🔄 Analyzing position-technique interactions...[/blue]")

        analysis = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            attack_key_analysis = data.get("attack_key_analysis", {})

            # Collect data by position and technique
            interactions: Dict[str, List[float]] = {
                "narrative_first": [],
                "narrative_last": [],
                "policy_first": [],
                "policy_last": [],
            }

            for attack_key, stats in attack_key_analysis.items():
                success_rate = stats.get("success_rate", 0.0)
                total = stats.get("total", 0)

                if total == 0:
                    continue

                if "narrative" in attack_key and "first" in attack_key:
                    interactions["narrative_first"].append(success_rate)
                elif "narrative" in attack_key and "last" in attack_key:
                    interactions["narrative_last"].append(success_rate)
                elif "policy" in attack_key and "first" in attack_key:
                    interactions["policy_first"].append(success_rate)
                elif "policy" in attack_key and "last" in attack_key:
                    interactions["policy_last"].append(success_rate)

            # Calculate interaction effects
            model_analysis: Dict[str, Any] = {}
            for key, rates in interactions.items():
                if rates:
                    model_analysis[key] = {
                        "mean": statistics.mean(rates),
                        "std": statistics.stdev(rates) if len(rates) > 1 else 0,
                        "count": len(rates),
                    }

            # Calculate best combination
            if model_analysis:
                best_combination = max(
                    model_analysis.items(), key=lambda x: x[1].get("mean", 0)
                )
                worst_combination = min(
                    model_analysis.items(), key=lambda x: x[1].get("mean", 0)
                )

                best_mean = float(best_combination[1].get("mean", 0))
                worst_mean = float(worst_combination[1].get("mean", 0))

                model_analysis["best_combination"] = {
                    "configuration": str(best_combination[0]),
                    "success_rate": best_mean,
                }
                model_analysis["worst_combination"] = {
                    "configuration": str(worst_combination[0]),
                    "success_rate": worst_mean,
                }
                model_analysis["interaction_range"] = best_mean - worst_mean

            analysis[model] = model_analysis

        return analysis

    def analyze_attack_type_vulnerabilities(self) -> Dict[str, Any]:
        """Analyze vulnerabilities by attack type with severity classification."""
        console.print("[blue]🔍 Analyzing attack type vulnerabilities...[/blue]")

        analysis: Dict[str, Any] = {}

        # Collect success rates by attack type for both models
        attack_type_data: Dict[str, Dict[str, float]] = {}

        for model in ["chatgpt", "gemini"]:
            key = f"{model}_analyzed"
            if key not in self.data:
                continue

            data = self.data[key]
            attack_type_analysis = data.get("attack_type_analysis", {})

            for attack_type, stats in attack_type_analysis.items():
                success_rate = stats.get("success_rate", 0.0)
                if attack_type not in attack_type_data:
                    attack_type_data[attack_type] = {}
                attack_type_data[attack_type][model] = success_rate

        # Analyze each attack type
        for attack_type, model_rates in attack_type_data.items():
            gemini_sr = model_rates.get("gemini", 0)
            chatgpt_sr = model_rates.get("chatgpt", 0)
            difference = gemini_sr - chatgpt_sr
            combined_avg = (gemini_sr + chatgpt_sr) / 2

            # Vulnerability severity classification
            if combined_avg >= 85:
                severity = "high"
            elif combined_avg >= 70:
                severity = "medium"
            else:
                severity = "low"

            analysis[attack_type] = {
                "gemini_success_rate": round(gemini_sr, 2),
                "chatgpt_success_rate": round(chatgpt_sr, 2),
                "success_rate_difference": round(difference, 2),
                "gemini_more_vulnerable": difference > 0,
                "vulnerability_gap": round(abs(difference), 2),
                "combined_average": round(combined_avg, 2),
                "vulnerability_severity": severity,
            }

        # Overall comparison
        if attack_type_data:
            all_gemini_rates = [
                rates.get("gemini", 0) for rates in attack_type_data.values()
            ]
            all_chatgpt_rates = [
                rates.get("chatgpt", 0) for rates in attack_type_data.values()
            ]

            analysis["overall_comparison"] = {
                "gemini_average_vulnerability": round(
                    statistics.mean(all_gemini_rates), 2
                ),
                "chatgpt_average_vulnerability": round(
                    statistics.mean(all_chatgpt_rates), 2
                ),
                "overall_difference": round(
                    statistics.mean(all_gemini_rates)
                    - statistics.mean(all_chatgpt_rates),
                    2,
                ),
                "gemini_more_vulnerable_overall": statistics.mean(all_gemini_rates)
                > statistics.mean(all_chatgpt_rates),
            }

        return analysis

    def generate_publication_ready_insights(self) -> Dict[str, Any]:
        """Generate comprehensive, publication-ready insights."""
        console.print(
            "[bold blue]📝 Generating publication-ready insights...[/bold blue]"
        )

        insights = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_version": "3.0.0",
                "methodology": "Comprehensive statistical analysis with multiple evaluation metrics",
                "models_analyzed": list(self.data.keys()),
                "total_attack_vectors_analyzed": 0,
            }
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # 1. Overall Attack Success Analysis
            task1 = progress.add_task(
                "Analyzing overall attack success rates...", total=None
            )
            insights["overall_attack_success_analysis"] = (
                self.analyze_overall_attack_success_rates()
            )
            progress.update(task1, completed=True)

            # 2. Attack Type Effectiveness
            task2 = progress.add_task(
                "Analyzing attack type effectiveness...", total=None
            )
            insights["attack_type_effectiveness_analysis"] = (
                self.analyze_attack_type_effectiveness()
            )
            progress.update(task2, completed=True)

            # 3. Payload and Position Effects
            task3 = progress.add_task(
                "Analyzing payload and position effects...", total=None
            )
            insights["payload_position_effects_analysis"] = (
                self.analyze_payload_and_position_effects()
            )
            progress.update(task3, completed=True)

            # 4. Sentiment Steering Analysis
            task4 = progress.add_task(
                "Analyzing sentiment steering effectiveness...", total=None
            )
            insights["sentiment_steering_analysis"] = (
                self.analyze_sentiment_steering_effectiveness()
            )
            progress.update(task4, completed=True)

            # 5. Vulnerability Profiles
            task5 = progress.add_task("Creating vulnerability profiles...", total=None)
            insights["vulnerability_profiles"] = self.analyze_vulnerability_profiles()
            progress.update(task5, completed=True)

            # 6. Academic Sentiment Classifier Performance
            task6 = progress.add_task("Analyzing classifier performance...", total=None)
            insights["academic_classifier_performance"] = (
                self.analyze_academic_sentiment_classifier_performance()
            )
            progress.update(task6, completed=True)

            # 7. Perfect Attacks Analysis
            task7 = progress.add_task("Identifying perfect attacks...", total=None)
            insights["perfect_attacks_analysis"] = self.analyze_perfect_attacks()
            progress.update(task7, completed=True)

            # 8. Predictability Analysis
            task8 = progress.add_task("Analyzing attack predictability...", total=None)
            insights["predictability_analysis"] = self.analyze_attack_predictability()
            progress.update(task8, completed=True)

            # 9. Technique Effectiveness Analysis
            task9 = progress.add_task("Analyzing technique effectiveness...", total=None)
            insights["technique_effectiveness_analysis"] = (
                self.analyze_technique_effectiveness()
            )
            progress.update(task9, completed=True)

            # 10. Position-Technique Interaction
            task10 = progress.add_task(
                "Analyzing position-technique interactions...", total=None
            )
            insights["position_technique_interaction"] = (
                self.analyze_position_technique_interaction()
            )
            progress.update(task10, completed=True)

            # 11. Attack Type Vulnerabilities
            task11 = progress.add_task(
                "Analyzing attack type vulnerabilities...", total=None
            )
            insights["attack_type_vulnerabilities"] = (
                self.analyze_attack_type_vulnerabilities()
            )
            progress.update(task11, completed=True)

        # Update metadata
        total_vectors = len(insights.get("vulnerability_profiles", {}))
        insights["metadata"]["total_attack_vectors_analyzed"] = total_vectors

        # Generate executive summary
        insights["executive_summary"] = self._generate_executive_summary(insights)

        # Generate research implications
        insights["research_implications"] = self._generate_research_implications(
            insights
        )

        # Generate statistical summary
        insights["statistical_summary"] = self._generate_statistical_summary(insights)

        return insights

    def _generate_executive_summary(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of key findings."""
        summary: Dict[str, Any] = {
            "key_findings": [],
            "critical_vulnerabilities": [],
            "model_differences": [],
            "methodological_insights": [],
        }

        # Overall metrics summary
        overall_analysis = insights.get("overall_attack_success_analysis", {})
        gemini_metrics = overall_analysis.get("gemini", {})
        chatgpt_metrics = overall_analysis.get("chatgpt", {})

        if gemini_metrics and chatgpt_metrics:
            # Handle both dataclass and dict
            if hasattr(gemini_metrics, "success_rate"):
                gemini_sr = gemini_metrics.success_rate
                gemini_total = gemini_metrics.total_attempts
            else:
                gemini_sr = gemini_metrics.get("success_rate", 0)
                gemini_total = gemini_metrics.get("total_attempts", 0)

            if hasattr(chatgpt_metrics, "success_rate"):
                chatgpt_sr = chatgpt_metrics.success_rate
                chatgpt_total = chatgpt_metrics.total_attempts
            else:
                chatgpt_sr = chatgpt_metrics.get("success_rate", 0)
                chatgpt_total = chatgpt_metrics.get("total_attempts", 0)

            summary["overall_metrics"] = {
                "total_attacks_analyzed": gemini_total + chatgpt_total,
                "gemini_overall_success_rate": round(gemini_sr, 2),
                "chatgpt_overall_success_rate": round(chatgpt_sr, 2),
                "vulnerability_gap": round(abs(gemini_sr - chatgpt_sr), 2),
            }

        # Position effect summary
        position_analysis = insights.get("payload_position_effects_analysis", {})
        if "chatgpt" in position_analysis and "gemini" in position_analysis:
            chatgpt_pos = position_analysis["chatgpt"].get("position_effect", {})
            gemini_pos = position_analysis["gemini"].get("position_effect", {})

            summary["position_effect"] = {
                "gemini_first_advantage": round(
                    gemini_pos.get("first_position_advantage", 0), 2
                ),
                "chatgpt_first_advantage": round(
                    chatgpt_pos.get("first_position_advantage", 0), 2
                ),
                "average_first_advantage": round(
                    (
                        gemini_pos.get("first_position_advantage", 0)
                        + chatgpt_pos.get("first_position_advantage", 0)
                    )
                    / 2,
                    2,
                ),
                "first_universally_better": gemini_pos.get("first_position_advantage", 0)
                > 0
                and chatgpt_pos.get("first_position_advantage", 0) > 0,
            }

        # Technique effectiveness summary
        technique_analysis = insights.get("technique_effectiveness_analysis", {})
        if "chatgpt" in technique_analysis and "gemini" in technique_analysis:
            summary["technique_effectiveness"] = {
                "gemini_policy_advantage": round(
                    technique_analysis["gemini"].get("technique_advantage", 0), 2
                ),
                "chatgpt_policy_advantage": round(
                    technique_analysis["chatgpt"].get("technique_advantage", 0), 2
                ),
                "average_policy_advantage": round(
                    (
                        technique_analysis["gemini"].get("technique_advantage", 0)
                        + technique_analysis["chatgpt"].get("technique_advantage", 0)
                    )
                    / 2,
                    2,
                ),
                "policy_universally_better": technique_analysis["gemini"].get(
                    "policy_is_better", False
                )
                and technique_analysis["chatgpt"].get("policy_is_better", False),
            }

        # Perfect attacks summary
        perfect_attacks = insights.get("perfect_attacks_analysis", {})
        if "chatgpt" in perfect_attacks and "gemini" in perfect_attacks:
            summary["perfect_attacks"] = {
                "gemini_perfect_count": perfect_attacks["gemini"].get(
                    "perfect_count", 0
                ),
                "chatgpt_perfect_count": perfect_attacks["chatgpt"].get(
                    "perfect_count", 0
                ),
                "gemini_near_perfect_count": perfect_attacks["gemini"].get(
                    "near_perfect_count", 0
                ),
                "chatgpt_near_perfect_count": perfect_attacks["chatgpt"].get(
                    "near_perfect_count", 0
                ),
            }

        # Predictability summary
        predictability = insights.get("predictability_analysis", {})
        if "comparative" in predictability:
            comp = predictability["comparative"]
            summary["predictability"] = {
                "gemini_predictability_index": round(
                    predictability.get("gemini", {}).get("predictability_index", 0), 3
                ),
                "chatgpt_predictability_index": round(
                    predictability.get("chatgpt", {}).get("predictability_index", 0), 3
                ),
                "more_predictable_model": comp.get("more_predictable_model", "unknown"),
                "predictability_difference": round(
                    comp.get("predictability_difference", 0), 3
                ),
            }

        # Critical vulnerabilities from attack type analysis
        attack_vuln = insights.get("attack_type_vulnerabilities", {})
        if attack_vuln:
            # Find specific high-impact vulnerabilities
            critical_attacks = {}
            for attack_type, vuln_data in attack_vuln.items():
                if attack_type == "overall_comparison":
                    continue
                if vuln_data.get("vulnerability_severity") in ["high", "medium"]:
                    critical_attacks[attack_type] = vuln_data

            summary["critical_vulnerabilities_by_type"] = critical_attacks

        # Extract key findings from overall analysis
        if "comparative" in overall_analysis:
            comp = overall_analysis["comparative"]
            if isinstance(comp, dict):
                summary["key_findings"].append(
                    {
                        "finding": f"Cross-model vulnerability disparity of {abs(comp.get('success_rate_a', 0) - comp.get('success_rate_b', 0)):.2f}%",
                        "details": f"{comp.get('condition_a', 'Unknown')} shows {comp.get('success_rate_a', 0):.2f}% success rate vs {comp.get('condition_b', 'Unknown')} at {comp.get('success_rate_b', 0):.2f}%",
                        "statistical_significance": comp.get(
                            "statistical_significance", {}
                        ).get("significant", False),
                        "effect_size": comp.get("effect_size", 0),
                    }
                )
            else:
                # Handle dataclass object
                summary["key_findings"].append(
                    {
                        "finding": f"Cross-model vulnerability disparity of {abs(comp.success_rate_a - comp.success_rate_b):.2f}%",
                        "details": f"{comp.condition_a} shows {comp.success_rate_a:.2f}% success rate vs {comp.condition_b} at {comp.success_rate_b:.2f}%",
                        "statistical_significance": comp.statistical_significance.get(
                            "significant", False
                        )
                        if isinstance(comp.statistical_significance, dict)
                        else False,
                        "effect_size": comp.effect_size,
                    }
                )

        # Extract critical vulnerabilities
        vulnerability_profiles = insights.get("vulnerability_profiles", {})
        for vector_key, profile in vulnerability_profiles.items():
            profile_dict = asdict(profile) if hasattr(profile, "__dict__") else profile
            if profile_dict.get("risk_classification") == "CRITICAL":
                summary["critical_vulnerabilities"].append(
                    {
                        "attack_vector": vector_key,
                        "vulnerability_score": profile_dict.get(
                            "vulnerability_score", 0
                        ),
                        "cross_model_consistency": profile_dict.get(
                            "cross_model_variance", 0
                        )
                        < 10,
                        "attack_components": {
                            "type": profile_dict.get("attack_type"),
                            "payload": profile_dict.get("payload_type"),
                            "position": profile_dict.get("injection_position"),
                        },
                    }
                )

        return summary

    def _generate_research_implications(
        self, insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate research implications for publication."""
        implications: Dict[str, List[Any]] = {
            "theoretical_contributions": [],
            "practical_implications": [],
            "methodological_contributions": [],
            "future_research_directions": [],
        }

        # Analyze cross-model differences for theoretical insights
        attack_type_analysis = insights.get("attack_type_effectiveness_analysis", {})
        cross_model_comparisons = attack_type_analysis.get(
            "cross_model_comparisons", {}
        )

        significant_differences = []
        for attack_type, comparison in cross_model_comparisons.items():
            if comparison.get("effect_size", 0) > 0.5:  # Large effect size
                significant_differences.append(
                    {
                        "attack_type": attack_type,
                        "difference": comparison.get("success_rate_difference", 0),
                        "effect_size": comparison.get("effect_size", 0),
                    }
                )

        if significant_differences:
            implications["theoretical_contributions"].append(
                {
                    "contribution": "Differential vulnerability patterns across LLM architectures",
                    "evidence": f"Found {len(significant_differences)} attack types with large effect sizes (>0.5)",
                    "significance": "Suggests architectural differences in prompt processing affect security",
                }
            )

        return implications

    def _generate_statistical_summary(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        summary: Dict[str, Any] = {
            "sample_sizes": {},
            "effect_sizes": {},
            "confidence_intervals": {},
            "statistical_tests_performed": [],
            "methodological_notes": [],
        }

        # Extract sample sizes
        overall_analysis = insights.get("overall_attack_success_analysis", {})
        for model, metrics in overall_analysis.items():
            if isinstance(metrics, dict) and hasattr(metrics, "total_attempts"):
                summary["sample_sizes"][model] = getattr(metrics, "total_attempts", 0)
            elif isinstance(metrics, dict) and "total_attempts" in metrics:
                summary["sample_sizes"][model] = metrics["total_attempts"]

        # Extract statistical tests
        attack_type_analysis = insights.get("attack_type_effectiveness_analysis", {})
        cross_model_comparisons = attack_type_analysis.get(
            "cross_model_comparisons", {}
        )

        for attack_type, comparison in cross_model_comparisons.items():
            statistical_test = comparison.get("statistical_test", {})
            if statistical_test:
                summary["statistical_tests_performed"].append(
                    {
                        "test_context": f"Cross-model comparison for {attack_type}",
                        "test_name": statistical_test.get("test_name", "Unknown"),
                        "p_value": statistical_test.get("p_value", 1.0),
                        "effect_size": statistical_test.get("effect_size", 0.0),
                        "significant": statistical_test.get("significant", False),
                    }
                )

        return summary

    def save_insights(
        self, output_file: str = "results/analyses/comprehensive_insights.json"
    ) -> None:
        """Save comprehensive insights to JSON file."""
        console.print(
            f"[blue]💾 Saving comprehensive insights to {output_file}...[/blue]"
        )

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate insights
        insights = self.generate_publication_ready_insights()

        # Convert dataclass objects to dictionaries for JSON serialization
        def convert_dataclass_to_dict(obj):
            if hasattr(obj, "__dict__"):
                return asdict(obj)
            elif isinstance(obj, dict):
                return {k: convert_dataclass_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()  # Convert numpy types to Python native types
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        insights_serializable = convert_dataclass_to_dict(insights)

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(insights_serializable, f, indent=2, ensure_ascii=False)

        console.print(
            f"[green]✅ Comprehensive insights saved to {output_file}[/green]"
        )

        # Print summary statistics
        self._print_insights_summary(insights_serializable)

    def _print_insights_summary(self, insights: Dict[str, Any]) -> None:
        """Print a summary of the extracted insights."""
        console.print("\n[bold blue]📊 Comprehensive Analysis Summary[/bold blue]")

        # Overall statistics
        metadata = insights.get("metadata", {})
        console.print(
            f"[cyan]Analysis completed: {metadata.get('analysis_timestamp', 'Unknown')}[/cyan]"
        )
        console.print(
            f"[cyan]Total attack vectors analyzed: {metadata.get('total_attack_vectors_analyzed', 0)}[/cyan]"
        )

        # Key findings table
        exec_summary = insights.get("executive_summary", {})
        key_findings = exec_summary.get("key_findings", [])

        if key_findings:
            findings_table = Table(
                title="🎯 Key Research Findings",
                show_header=True,
                header_style="bold magenta",
            )
            findings_table.add_column("Finding", style="cyan")
            findings_table.add_column(
                "Statistical Significance", style="green", justify="center"
            )
            findings_table.add_column("Effect Size", style="yellow", justify="right")

            for finding in key_findings:
                significance = (
                    "✅ Yes"
                    if finding.get("statistical_significance", False)
                    else "❌ No"
                )
                effect_size = f"{finding.get('effect_size', 0):.3f}"
                findings_table.add_row(
                    finding.get("finding", "Unknown"), significance, effect_size
                )

            console.print(findings_table)

        # Critical vulnerabilities
        critical_vulns = exec_summary.get("critical_vulnerabilities", [])
        if critical_vulns:
            vuln_table = Table(
                title="🚨 Critical Vulnerabilities Identified",
                show_header=True,
                header_style="bold red",
            )
            vuln_table.add_column("Attack Vector", style="red")
            vuln_table.add_column(
                "Vulnerability Score", style="bold red", justify="right"
            )
            vuln_table.add_column(
                "Cross-Model Consistent", style="yellow", justify="center"
            )

            for vuln in critical_vulns:
                consistency = (
                    "✅ Yes" if vuln.get("cross_model_consistency", False) else "❌ No"
                )
                vuln_table.add_row(
                    vuln.get("attack_vector", "Unknown"),
                    f"{vuln.get('vulnerability_score', 0):.1f}%",
                    consistency,
                )

            console.print(vuln_table)

        console.print("\n[bold green]✅ Comprehensive analysis complete![/bold green]")


def main() -> None:
    """Main execution function."""
    try:
        console.print(
            "[bold cyan]🔬 Comprehensive Insights Extractor for Indirect Prompt Injection Research[/bold cyan]\n"
        )

        # Initialize extractor
        extractor = ComprehensiveInsightsExtractor()

        # Generate and save insights
        extractor.save_insights()

        console.print("\n[bold green]🎉 Analysis completed successfully![/bold green]")
        console.print(
            "[dim]Results saved to: results/analyses/comprehensive_insights.json[/dim]"
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Analysis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error during analysis: {e}[/red]")
        logger.exception("Analysis failed")
        raise


if __name__ == "__main__":
    main()
