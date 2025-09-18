"""
Configuration module for Academic Sentiment Evaluator.

This module contains configuration settings and constants used throughout
the academic sentiment evaluation process.
"""

from pathlib import Path
from typing import Any, Dict

# Model Configuration
DEFAULT_MODEL_PATH = "data/models/academic-sentiment-classifier"
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "auto"

# Label Mappings for Different Model Types
LABEL_MAPPINGS = {
    "distilbert": {
        "LABEL_0": "negative",
        "LABEL_1": "positive",
        0: "negative",
        1: "positive",
    },
    "bert": {
        "LABEL_0": "negative",
        "LABEL_1": "positive",
        0: "negative",
        1: "positive",
    },
    "custom": {
        # Add your custom label mappings here
        "negative": "negative",
        "positive": "positive",
    },
}

# Attack Type Configurations
STEERING_ATTACK_TYPES = ["pos_steering_attack", "neg_steering_attack"]

# Expected sentiment mappings based on attack types
ATTACK_SENTIMENT_MAPPING = {"pos_steering": "positive", "neg_steering": "negative"}

# Output Configuration
OUTPUT_FORMATS = {
    "json": {"extension": ".json", "indent": 2, "ensure_ascii": False},
    "csv": {"extension": ".csv", "index": False},
}

# Evaluation Thresholds
CONFIDENCE_THRESHOLDS = {"high": 0.8, "medium": 0.6, "low": 0.4}

# Text Processing Configuration
MAX_TEXT_LENGTH = 512  # For most transformer models
PREVIEW_LENGTH = 200  # Length of text preview in results

# Progress Display Configuration
PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N batches

# Logging Configuration
LOG_FORMAT = "%(message)s"
LOG_DATE_FORMAT = "[%X]"


def get_model_config(model_type: str = "distilbert") -> Dict[str, Any]:
    """
    Get model-specific configuration.

    Args:
        model_type: Type of model being used

    Returns:
        Dictionary with model configuration
    """
    return {
        "label_mapping": LABEL_MAPPINGS.get(model_type, LABEL_MAPPINGS["distilbert"]),
        "max_length": MAX_TEXT_LENGTH,
        "batch_size": DEFAULT_BATCH_SIZE,
    }


def get_output_config(output_format: str = "json") -> Dict[str, Any]:
    """
    Get output format configuration.

    Args:
        output_format: Desired output format

    Returns:
        Dictionary with output configuration
    """
    return OUTPUT_FORMATS.get(output_format, OUTPUT_FORMATS["json"])


def validate_model_path(model_path: str) -> bool:
    """
    Validate that the model path exists and contains required files.

    Args:
        model_path: Path to model directory

    Returns:
        True if valid, False otherwise
    """
    path = Path(model_path)

    if not path.exists():
        return False

    # Check for required model files
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]

    for file_name in required_files:
        if not (path / file_name).exists():
            return False

    return True
