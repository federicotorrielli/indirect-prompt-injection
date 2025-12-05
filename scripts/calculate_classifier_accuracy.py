"""
Calculate Academic Sentiment Classifier Accuracy

This script evaluates the trained academic sentiment classifier on the OpenReview
dataset test set and generates accuracy metrics for reporting in the paper.

Usage:
    uv run python scripts/calculate_classifier_accuracy.py
"""

import json
import logging
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Suppress warnings
warnings.filterwarnings("ignore")

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

MODEL_PATH = "data/models/academic-sentiment-classifier"
MAX_LENGTH = 512


def extract_review_text(review_data: dict) -> str:
    """Extract and combine all text fields from a review."""
    if not isinstance(review_data, dict):
        return ""

    review_obj = review_data.get("review", {})
    if not isinstance(review_obj, dict):
        return ""

    text_fields = [
        "main_review",
        "paper_summary",
        "questions",
        "strength_weakness",
        "limitations",
        "review_summary",
    ]

    review_parts = []
    for field in text_fields:
        field_value = review_obj.get(field)
        if field_value and isinstance(field_value, str) and field_value.strip():
            cleaned_value = field_value
            if ": " in cleaned_value and field in cleaned_value.lower():
                parts = cleaned_value.split(": ", 1)
                if len(parts) == 2:
                    cleaned_value = parts[1]
            review_parts.append(cleaned_value.strip())

    return " ".join(review_parts)


def prepare_dataset(min_review_length: int = 100) -> Tuple[List[str], List[int]]:
    """Prepare dataset from OpenReview."""
    console.print("[blue]📚 Loading OpenReview dataset...[/blue]")

    dataset = load_dataset("nhop/OpenReview", split="train")

    review_texts = []
    labels = []

    console.print("[cyan]🔄 Processing reviews...[/cyan]")

    for item in track(dataset, description="Processing papers"):
        if not isinstance(item, dict):
            continue

        decision = item.get("decision")
        if decision is None:
            continue

        reviews = item.get("reviews", [])
        if not reviews:
            continue

        for review in reviews:
            if not isinstance(review, dict):
                continue

            review_text = extract_review_text(review)

            if len(review_text) < min_review_length:
                continue

            label = 1 if decision else 0
            review_texts.append(review_text)
            labels.append(label)

    console.print(f"[green]✅ Processed {len(review_texts)} reviews[/green]")

    return review_texts, labels


def create_balanced_dataset(
    review_texts: List[str], labels: List[int], max_samples_per_class: int = 10000
) -> Tuple[List[str], List[int]]:
    """Create a balanced dataset with equal numbers of accept/reject reviews."""
    console.print("[yellow]⚖️  Creating balanced dataset...[/yellow]")

    accepted_reviews = []
    rejected_reviews = []

    for text, label in zip(review_texts, labels):
        if label == 1:
            accepted_reviews.append(text)
        else:
            rejected_reviews.append(text)

    min_class_size = min(len(accepted_reviews), len(rejected_reviews))
    target_size = min(min_class_size, max_samples_per_class)

    console.print(
        f"[cyan]  • Original - Accepted: {len(accepted_reviews)}, Rejected: {len(rejected_reviews)}[/cyan]"
    )
    console.print(f"[cyan]  • Balanced - Using {target_size} samples per class[/cyan]")

    np.random.seed(42)
    accepted_indices = np.random.choice(len(accepted_reviews), target_size, replace=False)
    rejected_indices = np.random.choice(len(rejected_reviews), target_size, replace=False)

    balanced_texts = []
    balanced_labels = []

    for idx in accepted_indices:
        balanced_texts.append(accepted_reviews[idx])
        balanced_labels.append(1)

    for idx in rejected_indices:
        balanced_texts.append(rejected_reviews[idx])
        balanced_labels.append(0)

    combined = list(zip(balanced_texts, balanced_labels))
    np.random.shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)

    console.print(f"[green]✅ Created balanced dataset with {len(balanced_texts)} samples[/green]")

    return list(balanced_texts), list(balanced_labels)


def evaluate_classifier() -> dict:
    """Evaluate the academic sentiment classifier and return metrics."""
    console.print("\n[bold cyan]🔬 Academic Sentiment Classifier Evaluation[/bold cyan]\n")

    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        console.print(f"[red]❌ Model not found at {MODEL_PATH}[/red]")
        console.print("[yellow]Run 'uv run python src/evaluation/train.py' first[/yellow]")
        return {}

    # Prepare dataset (same as training to ensure consistent splits)
    review_texts, labels = prepare_dataset(min_review_length=100)
    balanced_texts, balanced_labels = create_balanced_dataset(
        review_texts, labels, max_samples_per_class=15000
    )

    # Split dataset using same random state as training
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        balanced_texts, balanced_labels, test_size=0.2, random_state=42, stratify=balanced_labels
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    console.print(f"\n[cyan]📊 Dataset splits:[/cyan]")
    console.print(f"  • Training set: {len(train_texts)} samples")
    console.print(f"  • Validation set: {len(val_texts)} samples")
    console.print(f"  • Test set: {len(test_texts)} samples")

    # Load model
    console.print(f"\n[blue]🤖 Loading model from {MODEL_PATH}[/blue]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1

    classifier = pipeline(
        task="text-classification",
        model=str(model_path),
        device=device_id,
        return_all_scores=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    console.print(f"[green]✅ Model loaded on {device}[/green]")

    # Predict on test set
    console.print("\n[blue]🔮 Running predictions on test set...[/blue]")

    label_mapping = {
        "LABEL_0": 0,
        "LABEL_1": 1,
    }

    predictions = []
    batch_size = 32

    for i in track(range(0, len(test_texts), batch_size), description="Predicting"):
        batch_texts = test_texts[i : i + batch_size]
        batch_results = classifier(batch_texts)

        for result in batch_results:
            top_pred = max(result, key=lambda x: x["score"])
            pred_label = label_mapping.get(top_pred["label"], int(top_pred["label"][-1]))
            predictions.append(pred_label)

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="weighted")
    recall = recall_score(test_labels, predictions, average="weighted")
    f1 = f1_score(test_labels, predictions, average="weighted")

    precision_macro = precision_score(test_labels, predictions, average="macro")
    recall_macro = recall_score(test_labels, predictions, average="macro")
    f1_macro = f1_score(test_labels, predictions, average="macro")

    # Classification report
    report = classification_report(
        test_labels, predictions, target_names=["Reject", "Accept"], output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)

    # Display results
    console.print("\n[bold green]📊 Evaluation Results[/bold green]\n")

    # Main metrics table
    metrics_table = Table(title="🎯 Overall Metrics", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right", style="green")
    metrics_table.add_row("Accuracy", f"{accuracy:.4f} ({accuracy * 100:.2f}%)")
    metrics_table.add_row("Precision (weighted)", f"{precision:.4f}")
    metrics_table.add_row("Recall (weighted)", f"{recall:.4f}")
    metrics_table.add_row("F1-Score (weighted)", f"{f1:.4f}")
    metrics_table.add_row("---", "---")
    metrics_table.add_row("Precision (macro)", f"{precision_macro:.4f}")
    metrics_table.add_row("Recall (macro)", f"{recall_macro:.4f}")
    metrics_table.add_row("F1-Score (macro)", f"{f1_macro:.4f}")
    console.print(metrics_table)

    # Per-class metrics
    class_table = Table(
        title="\n📋 Per-Class Metrics", show_header=True, header_style="bold magenta"
    )
    class_table.add_column("Class", style="cyan")
    class_table.add_column("Precision", justify="right", style="green")
    class_table.add_column("Recall", justify="right", style="yellow")
    class_table.add_column("F1-Score", justify="right", style="blue")
    class_table.add_column("Support", justify="right", style="white")

    for class_name in ["Reject", "Accept"]:
        key = class_name
        if key in report:
            metrics = report[key]
            class_table.add_row(
                class_name,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1-score']:.4f}",
                str(int(metrics["support"])),
            )

    console.print(class_table)

    # Confusion matrix display
    console.print("\n[bold cyan]🔢 Confusion Matrix[/bold cyan]")
    console.print(f"              Predicted")
    console.print(f"            Reject  Accept")
    console.print(f"Actual Reject  {cm[0][0]:5d}   {cm[0][1]:5d}")
    console.print(f"       Accept  {cm[1][0]:5d}   {cm[1][1]:5d}")

    # Summary for paper
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]📝 For Paper Reporting:[/bold yellow]")
    console.print("=" * 60)
    console.print(
        f"\nThe academic sentiment classifier achieves an accuracy of "
        f"[bold green]{accuracy * 100:.2f}%[/bold green] on the test set "
        f"(n={len(test_texts)})."
    )
    console.print(f"\nWeighted F1-Score: [bold]{f1:.4f}[/bold]")
    console.print(f"Macro F1-Score: [bold]{f1_macro:.4f}[/bold]")
    console.print(
        f"\nPer-class performance:"
        f"\n  • Reject: P={report['Reject']['precision']:.3f}, R={report['Reject']['recall']:.3f}, F1={report['Reject']['f1-score']:.3f}"
        f"\n  • Accept: P={report['Accept']['precision']:.3f}, R={report['Accept']['recall']:.3f}, F1={report['Accept']['f1-score']:.3f}"
    )

    # Save results
    results = {
        "test_set_size": len(test_texts),
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    output_path = Path("results/evaluation/classifier_accuracy_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]💾 Results saved to: {output_path}[/green]")

    return results


if __name__ == "__main__":
    evaluate_classifier()
