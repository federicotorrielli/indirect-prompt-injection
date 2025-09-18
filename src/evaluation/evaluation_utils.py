"""
Utility functions for academic sentiment evaluation.

This module contains helper functions and utilities used across
the academic sentiment evaluation pipeline.
"""

import json
import os
from typing import Any, Dict, List, Optional

from rich.console import Console

console = Console()


def ensure_output_directory(output_path: str) -> None:
    """
    Ensure the output directory exists.

    Args:
        output_path: Path to output file
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Safely load JSON file with error handling.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Safely save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to output file
        indent: JSON indentation
    """
    ensure_output_directory(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def calculate_success_rate(successful: int, total: int) -> float:
    """
    Calculate success rate percentage.

    Args:
        successful: Number of successful items
        total: Total number of items

    Returns:
        Success rate as percentage
    """
    if total == 0:
        return 0.0

    return (successful / total) * 100


def filter_steering_attacks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter and extract steering attack records from evaluation data.

    Args:
        data: Raw evaluation data

    Returns:
        List of steering attack records
    """
    steering_records = []

    for attack_key, attack_data in data.items():
        if not isinstance(attack_data, dict):
            continue

        for request_type, results in attack_data.items():
            if not isinstance(results, list):
                continue

            for result in results:
                attack_type = result.get("attack_type", "")

                if "steering" in attack_type:
                    result["attack_key"] = attack_key
                    result["request_type"] = request_type
                    steering_records.append(result)

    return steering_records


def determine_expected_sentiment(attack_type: str, attack_key: str) -> Optional[str]:
    """
    Determine expected sentiment based on attack type and key.

    Args:
        attack_type: Type of attack (e.g., 'pos_steering_attack')
        attack_key: Attack key for additional context

    Returns:
        Expected sentiment ('positive', 'negative', or None if unknown)
    """
    if "pos_steering" in attack_type or "pos_steering" in attack_key:
        return "positive"
    elif "neg_steering" in attack_type or "neg_steering" in attack_key:
        return "negative"

    return None


def group_results_by_attack_type(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group evaluation results by attack type.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with attack types as keys and results as values
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for result in results:
        attack_type = result.get("attack_type", "unknown")

        if attack_type not in grouped:
            grouped[attack_type] = []

        grouped[attack_type].append(result)

    return grouped


def calculate_confidence_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculate distribution of prediction confidence levels.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with confidence levels and counts
    """
    distribution = {"high": 0, "medium": 0, "low": 0}

    for result in results:
        confidence = result.get("confidence", 0.0)

        if confidence >= 0.8:
            distribution["high"] += 1
        elif confidence >= 0.6:
            distribution["medium"] += 1
        else:
            distribution["low"] += 1

    return distribution


def validate_evaluation_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate evaluation data structure and return any issues found.

    Args:
        data: Evaluation data to validate

    Returns:
        List of validation error messages
    """
    errors = []

    if not isinstance(data, dict):
        errors.append("Top-level data must be a dictionary")
        return errors

    for attack_key, attack_data in data.items():
        if not isinstance(attack_data, dict):
            errors.append(f"Attack data for '{attack_key}' must be a dictionary")
            continue

        for request_type, results in attack_data.items():
            if not isinstance(results, list):
                errors.append(
                    f"Results for '{attack_key}.{request_type}' must be a list"
                )
                continue

            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    errors.append(
                        f"Result {i} in '{attack_key}.{request_type}' must be a dictionary"
                    )
                    continue

                # Check for required fields
                required_fields = ["attack_type", "response"]
                for field in required_fields:
                    if field not in result:
                        errors.append(
                            f"Missing required field '{field}' in result {i} of '{attack_key}.{request_type}'"
                        )

    return errors


def print_validation_report(errors: List[str]) -> bool:
    """
    Print validation report and return whether data is valid.

    Args:
        errors: List of validation errors

    Returns:
        True if no errors, False otherwise
    """
    if not errors:
        console.print("[green]✅ Data validation passed[/green]")
        return True

    console.print(f"[red]❌ Data validation failed with {len(errors)} errors:[/red]")
    for i, error in enumerate(errors, 1):
        console.print(f"  {i}. {error}")

    return False


def create_evaluation_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a comprehensive summary of evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Summary dictionary
    """
    if not results:
        return {"total": 0, "summary": "No results to summarize"}

    total_results = len(results)
    successful_attacks = sum(1 for r in results if r.get("attack_successful", False))

    # Group by attack type
    by_attack_type = group_results_by_attack_type(results)

    # Calculate confidence distribution
    confidence_dist = calculate_confidence_distribution(results)

    summary = {
        "total_results": total_results,
        "successful_attacks": successful_attacks,
        "success_rate": calculate_success_rate(successful_attacks, total_results),
        "by_attack_type": {
            attack_type: {
                "total": len(attack_results),
                "successful": sum(
                    1 for r in attack_results if r.get("attack_successful", False)
                ),
                "success_rate": calculate_success_rate(
                    sum(1 for r in attack_results if r.get("attack_successful", False)),
                    len(attack_results),
                ),
            }
            for attack_type, attack_results in by_attack_type.items()
        },
        "confidence_distribution": confidence_dist,
        "average_confidence": sum(r.get("confidence", 0) for r in results)
        / total_results,
    }

    return summary
