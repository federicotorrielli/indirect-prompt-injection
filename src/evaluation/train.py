"""
Train an academic review sentiment classifier on OpenReview data.

This script fine-tunes a transformer model (DistilBERT) on the full OpenReview dataset
to classify academic reviews as recommending acceptance or rejection. The trained model
will replace VADER for academic sentiment analysis in the evaluation pipeline.

Usage:
    uv run python src/evaluation/train_academic_classifier.py
"""

import json
import logging
import os
import warnings
from typing import Dict, List, Tuple, Any
import torch
import numpy as np
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns

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


class AcademicSentimentTrainer:
    """Trainer for academic review sentiment classification."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        output_dir: str = "models/academic-sentiment-classifier",
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: HuggingFace model to fine-tune
            max_length: Maximum sequence length for tokenization
            output_dir: Directory to save the trained model
        """
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = output_dir
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        console.print(f"[green]✅ Initialized trainer with model: {model_name}[/green]")

    def extract_review_text(self, review_data: Dict[str, Any]) -> str:
        """
        Extract and combine all text fields from a review.
        
        Args:
            review_data: Review dictionary from OpenReview dataset
            
        Returns:
            Combined review text
        """
        if not isinstance(review_data, dict):
            return ""
            
        review_obj = review_data.get("review", {})
        if not isinstance(review_obj, dict):
            return ""
            
        # Text fields to extract
        text_fields = [
            "main_review",
            "paper_summary", 
            "questions",
            "strength_weakness",
            "limitations",
            "review_summary"
        ]
        
        review_parts = []
        for field in text_fields:
            field_value = review_obj.get(field)
            if field_value and isinstance(field_value, str) and field_value.strip():
                # Remove field labels if they exist (e.g., "paper_summary: ")
                cleaned_value = field_value
                if ": " in cleaned_value and field in cleaned_value.lower():
                    parts = cleaned_value.split(": ", 1)
                    if len(parts) == 2:
                        cleaned_value = parts[1]
                review_parts.append(cleaned_value.strip())
        
        return " ".join(review_parts)

    def prepare_dataset(self, min_review_length: int = 100) -> Tuple[List[str], List[int]]:
        """
        Prepare training dataset from OpenReview.
        
        Args:
            min_review_length: Minimum review length to include
            
        Returns:
            Tuple of (review_texts, labels) where label 0=reject, 1=accept
        """
        console.print("[blue]📚 Loading OpenReview dataset...[/blue]")
        
        # Load dataset
        dataset = load_dataset("nhop/OpenReview", split="train")
        
        review_texts = []
        labels = []
        paper_decisions = []
        
        console.print("[cyan]🔄 Processing reviews...[/cyan]")
        
        processed_count = 0
        accepted_count = 0
        rejected_count = 0
        
        for item in track(dataset, description="Processing papers"):
            if not isinstance(item, dict):
                continue
                
            # Get paper decision
            decision = item.get("decision")
            if decision is None:
                continue
                
            # Get reviews
            reviews = item.get("reviews", [])
            if not reviews:
                continue
                
            # Process each review in the paper
            for review in reviews:
                if not isinstance(review, dict):
                    continue
                    
                # Extract review text
                review_text = self.extract_review_text(review)
                
                # Filter by minimum length
                if len(review_text) < min_review_length:
                    continue
                    
                # Convert decision to label (True=accept=1, False=reject=0)
                label = 1 if decision else 0
                
                review_texts.append(review_text)
                labels.append(label)
                paper_decisions.append("accepted" if decision else "rejected")
                
                if label == 1:
                    accepted_count += 1
                else:
                    rejected_count += 1
                    
                processed_count += 1
        
        console.print(f"[green]✅ Processed {processed_count} reviews[/green]")
        console.print(f"[cyan]  • Accepted: {accepted_count} ({accepted_count/processed_count*100:.1f}%)[/cyan]")
        console.print(f"[cyan]  • Rejected: {rejected_count} ({rejected_count/processed_count*100:.1f}%)[/cyan]")
        
        return review_texts, labels

    def create_balanced_dataset(
        self, review_texts: List[str], labels: List[int], max_samples_per_class: int = 10000
    ) -> Tuple[List[str], List[int]]:
        """
        Create a balanced dataset with equal numbers of accept/reject reviews.
        
        Args:
            review_texts: All review texts
            labels: All labels
            max_samples_per_class: Maximum samples per class
            
        Returns:
            Balanced review texts and labels
        """
        console.print("[yellow]⚖️  Creating balanced dataset...[/yellow]")
        
        # Separate by class
        accepted_reviews = []
        rejected_reviews = []
        
        for text, label in zip(review_texts, labels):
            if label == 1:
                accepted_reviews.append(text)
            else:
                rejected_reviews.append(text)
        
        # Balance the dataset
        min_class_size = min(len(accepted_reviews), len(rejected_reviews))
        target_size = min(min_class_size, max_samples_per_class)
        
        console.print(f"[cyan]  • Original - Accepted: {len(accepted_reviews)}, Rejected: {len(rejected_reviews)}[/cyan]")
        console.print(f"[cyan]  • Balanced - Using {target_size} samples per class[/cyan]")
        
        # Sample equal amounts from each class
        np.random.seed(42)
        accepted_indices = np.random.choice(len(accepted_reviews), target_size, replace=False)
        rejected_indices = np.random.choice(len(rejected_reviews), target_size, replace=False)
        
        balanced_texts = []
        balanced_labels = []
        
        # Add accepted reviews
        for idx in accepted_indices:
            balanced_texts.append(accepted_reviews[idx])
            balanced_labels.append(1)
            
        # Add rejected reviews
        for idx in rejected_indices:
            balanced_texts.append(rejected_reviews[idx])
            balanced_labels.append(0)
            
        # Shuffle the balanced dataset
        combined = list(zip(balanced_texts, balanced_labels))
        np.random.shuffle(combined)
        balanced_texts, balanced_labels = zip(*combined)
        
        console.print(f"[green]✅ Created balanced dataset with {len(balanced_texts)} samples[/green]")
        
        return list(balanced_texts), list(balanced_labels)

    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    def train_model(self, review_texts: List[str], labels: List[int]) -> None:
        """
        Train the academic sentiment classifier.
        
        Args:
            review_texts: Review texts for training
            labels: Labels (0=reject, 1=accept)
        """
        console.print("[blue]🚀 Starting model training...[/blue]")
        
        # Split dataset
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            review_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        console.print(f"[cyan]  • Training set: {len(train_texts)} samples[/cyan]")
        console.print(f"[cyan]  • Validation set: {len(val_texts)} samples[/cyan]")
        console.print(f"[cyan]  • Test set: {len(test_texts)} samples[/cyan]")
        
        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
        val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_labels})
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        # Create data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            report_to=None,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        console.print("[green]🎯 Starting training...[/green]")
        trainer.train()
        
        # Evaluate on test set
        console.print("[blue]📊 Evaluating on test set...[/blue]")
        test_results = trainer.evaluate(test_dataset)
        
        console.print(f"[green]✅ Test Accuracy: {test_results['eval_accuracy']:.4f}[/green]")
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        console.print(f"[green]💾 Model saved to: {self.output_dir}[/green]")
        
        # Generate detailed evaluation report
        self.generate_evaluation_report(trainer, test_dataset, test_labels)
        
        # Store test dataset for later analysis
        self.test_texts = test_texts
        self.test_labels = test_labels

    def generate_evaluation_report(self, trainer, test_dataset, test_labels):
        """Generate detailed evaluation report."""
        console.print("[cyan]📋 Generating evaluation report...[/cyan]")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)
        
        # Classification report
        report = classification_report(
            test_labels, predicted_labels, 
            target_names=["Reject", "Accept"],
            output_dict=True
        )
        
        # Create table
        table = Table(title="📊 Classification Report", show_header=True, header_style="bold magenta")
        table.add_column("Class", style="cyan", no_wrap=True)
        table.add_column("Precision", justify="right", style="green")
        table.add_column("Recall", justify="right", style="yellow")
        table.add_column("F1-Score", justify="right", style="blue")
        table.add_column("Support", justify="right", style="white")
        
        for class_name in ["Reject", "Accept"]:
            metrics = report[class_name.lower()]
            table.add_row(
                class_name,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                str(int(metrics['support']))
            )
        
        # Add macro and weighted averages
        table.add_row("---", "---", "---", "---", "---")
        for avg_type in ["macro avg", "weighted avg"]:
            metrics = report[avg_type]
            table.add_row(
                avg_type.title(),
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                str(int(metrics['support']))
            )
        
        console.print(table)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predicted_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Reject', 'Accept'],
            yticklabels=['Reject', 'Accept']
        )
        plt.title('Confusion Matrix - Academic Review Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Ensure output directory exists
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig("visualizations/academic_classifier_confusion_matrix.png", dpi=300, bbox_inches='tight')
        console.print("[green]📈 Confusion matrix saved to visualizations/academic_classifier_confusion_matrix.png[/green]")
        
        # Save evaluation report
        report_file = f"{self.output_dir}/evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        console.print(f"[green]📄 Evaluation report saved to: {report_file}[/green]")

    def test_model_on_examples(self):
        """Test the trained model on some example reviews."""
        console.print("\n[bold blue]🧪 Testing Model on Example Reviews[/bold blue]")
        
        # Load the trained model
        model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        
        # Example reviews (crafted to test understanding)
        examples = [
            {
                "text": "This paper presents a novel approach with strong theoretical foundations. The experimental results are convincing and the writing is clear. I recommend acceptance.",
                "expected": "Accept"
            },
            {
                "text": "The methodology suffers from significant limitations. The experimental setup is inadequate and the results are not convincing. The paper lacks novelty and rigor.",
                "expected": "Reject"
            },
            {
                "text": "While the paper addresses an interesting problem, the proposed solution is not sufficiently novel. The experimental evaluation is limited and several important baselines are missing.",
                "expected": "Reject"
            },
            {
                "text": "This work makes a valuable contribution to the field. The approach is well-motivated, the experiments are comprehensive, and the results demonstrate clear improvements over existing methods.",
                "expected": "Accept"
            }
        ]
        
        # Create results table
        results_table = Table(title="🔍 Model Predictions on Example Reviews", show_header=True)
        results_table.add_column("Review (Truncated)", style="cyan", width=50)
        results_table.add_column("Expected", style="yellow", justify="center")
        results_table.add_column("Predicted", style="green", justify="center")
        results_table.add_column("Confidence", style="blue", justify="right")
        results_table.add_column("Correct", style="bold", justify="center")
        
        model.eval()
        correct = 0
        
        with torch.no_grad():
            for example in examples:
                # Tokenize
                inputs = tokenizer(
                    example["text"], 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.max_length
                )
                
                # Predict
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                confidence = probabilities.max().item()
                
                predicted_label = "Accept" if prediction == 1 else "Reject"
                is_correct = predicted_label == example["expected"]
                
                if is_correct:
                    correct += 1
                
                # Truncate text for display
                display_text = example["text"][:80] + "..." if len(example["text"]) > 80 else example["text"]
                
                results_table.add_row(
                    display_text,
                    example["expected"],
                    predicted_label,
                    f"{confidence:.3f}",
                    "✅" if is_correct else "❌"
                )
        
        console.print(results_table)
        console.print(f"\n[bold green]🎯 Accuracy on examples: {correct}/{len(examples)} ({correct/len(examples)*100:.1f}%)[/bold green]")


def main():
    """Main training function."""
    console.print("[bold cyan]🚀 Academic Review Sentiment Classifier Training[/bold cyan]\n")
    
    # Initialize trainer
    trainer = AcademicSentimentTrainer(
        model_name="distilbert-base-uncased",
        max_length=512,
        output_dir="models/academic-sentiment-classifier"
    )
    
    try:
        # Prepare dataset
        review_texts, labels = trainer.prepare_dataset(min_review_length=100)
        
        # Create balanced dataset
        balanced_texts, balanced_labels = trainer.create_balanced_dataset(
            review_texts, labels, max_samples_per_class=15000
        )
        
        # Train the model
        trainer.train_model(balanced_texts, balanced_labels)
        
        # Test on examples
        trainer.test_model_on_examples()
        
        console.print("\n[bold green]🎉 Training completed successfully![/bold green]")
        console.print(f"[cyan]💾 Model saved to: {trainer.output_dir}[/cyan]")
        console.print("[yellow]🔄 You can now use this model in evaluate_results.py by replacing VADER[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Training failed: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
