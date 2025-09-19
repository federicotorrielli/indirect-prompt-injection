"""
Publish the Academic Sentiment Classifier to the Hugging Face Hub.

This script uploads the model artifacts located under models/academic-sentiment-classifier
to a user/org repository on the Hugging Face Hub. It expects an access token provided via
the environment (HF_TOKEN) or a local huggingface-cli login.

Usage (run from repo root):
  uv run python scripts/publish_to_hf.py --repo YOUR_USERNAME/academic-sentiment-classifier

Environment:
  HF_TOKEN: Optional. If not present, the script will rely on cached login.

Notes:
  - Ensure README.md exists in the model directory; it will be used as the model card.
  - The script preserves directory structure and uploads safely via the hub.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfFolder, create_repo, upload_folder

LOGGER = logging.getLogger("publish_to_hf")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    # fall back to stored token from huggingface-cli login
    return HfFolder.get_token()


def publish(model_dir: Path, repo_id: str, private: bool, token: Optional[str]) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    required = ["config.json", "model.safetensors", "tokenizer.json"]
    for fname in required:
        if not (model_dir / fname).exists():
            raise FileNotFoundError(f"Missing required file in model dir: {fname}")

    LOGGER.info(f"Ensuring Hub repo exists: {repo_id} (private={private})")
    create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)

    LOGGER.info("Uploading model folder to the Hub (this may take a while)...")
    upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Initial upload of Academic Sentiment Classifier",
        ignore_patterns=["checkpoint-*/**", "*.bin"],
    )

    # Set recommended tags/metadata if README card is present.
    readme = model_dir / "README.md"
    if readme.exists():
        LOGGER.info("Model card found and uploaded as README.md")
    else:
        LOGGER.warning(
            "No README.md found; consider adding a model card for better discoverability."
        )

    LOGGER.info(f"Done. View your model at: https://huggingface.co/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish Academic Sentiment Classifier to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Target repo id on the Hub, e.g. YOUR_USERNAME/academic-sentiment-classifier",
    )
    parser.add_argument(
        "--model-dir",
        default="models/academic-sentiment-classifier",
        help="Path to local model directory",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update a private repository (default: public)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token. If omitted, will use HF_TOKEN env or cached login.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    token = get_hf_token(args.token)
    if token is None:
        LOGGER.warning(
            "No Hugging Face token found. If upload fails, login with 'huggingface-cli login' or set HF_TOKEN."
        )
    publish(Path(args.model_dir), args.repo, args.private, token)


if __name__ == "__main__":
    main()
