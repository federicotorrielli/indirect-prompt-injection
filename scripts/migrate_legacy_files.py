#!/usr/bin/env python3
"""
Migration script to convert legacy files to model-specific files.
This ensures backwards compatibility when upgrading.
"""

import json
from pathlib import Path


def clean_chatgpt_data(data):
    """Remove llm_service field and normalize batch keys for ChatGPT data."""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Handle special progress tracking keys
            if key in [
                "session_start",
                "last_updated",
                "total_pdfs_processed",
                "total_requests_sent",
            ]:
                cleaned[key] = value
                continue

            # Handle completed_pdfs, completed_batches, and failed_pdfs - normalize batch keys
            if key in ["completed_pdfs", "completed_batches", "failed_pdfs"]:
                if key == "completed_pdfs" and isinstance(value, dict):
                    # Transform batch keys: remove chatgpt_ prefix
                    cleaned_completed_pdfs = {}
                    for batch_key, pdf_data in value.items():
                        if batch_key.startswith("chatgpt_"):
                            # Remove chatgpt_ prefix to get clean batch key
                            clean_batch_key = batch_key[8:]  # Remove "chatgpt_"
                            cleaned_completed_pdfs[clean_batch_key] = pdf_data
                        else:
                            # Keep non-chatgpt prefixed keys as-is (shouldn't happen in legacy)
                            cleaned_completed_pdfs[batch_key] = pdf_data
                    cleaned[key] = cleaned_completed_pdfs
                elif key == "completed_batches" and isinstance(value, list):
                    # Transform batch keys in completed_batches list
                    cleaned_batches = []
                    for batch_key in value:
                        if isinstance(batch_key, str) and batch_key.startswith(
                            "chatgpt_"
                        ):
                            clean_batch_key = batch_key[8:]  # Remove "chatgpt_"
                            cleaned_batches.append(clean_batch_key)
                        else:
                            cleaned_batches.append(batch_key)
                    cleaned[key] = cleaned_batches
                elif key == "failed_pdfs" and isinstance(value, dict):
                    # Transform batch keys: remove chatgpt_ prefix
                    cleaned_failed_pdfs = {}
                    for batch_key, pdf_data in value.items():
                        if batch_key.startswith("chatgpt_"):
                            # Remove chatgpt_ prefix to get clean batch key
                            clean_batch_key = batch_key[8:]  # Remove "chatgpt_"
                            cleaned_failed_pdfs[clean_batch_key] = pdf_data
                        else:
                            # Keep non-chatgpt prefixed keys as-is (shouldn't happen in legacy)
                            cleaned_failed_pdfs[batch_key] = pdf_data
                    cleaned[key] = cleaned_failed_pdfs
                else:
                    cleaned[key] = value
                continue

            # Skip non-ChatGPT entries at top level (for results data) and normalize keys
            if (
                isinstance(value, dict)
                and not key.startswith("chatgpt_")
                and key
                not in [
                    "session_start",
                    "last_updated",
                    "completed_pdfs",
                    "failed_pdfs",
                    "completed_batches",
                    "total_pdfs_processed",
                    "total_requests_sent",
                ]
            ):
                continue

            # Handle top-level chatgpt_ keys (for results data) - normalize them
            if key.startswith("chatgpt_") and isinstance(value, dict):
                # Remove chatgpt_ prefix to get clean batch key
                clean_batch_key = key[8:]  # Remove "chatgpt_"
                cleaned[clean_batch_key] = clean_chatgpt_data(value)
                continue

            if isinstance(value, list):
                # Clean list items (removing llm_service field)
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = {
                            k: v for k, v in item.items() if k != "llm_service"
                        }
                        # Only keep ChatGPT entries
                        if (
                            "llm_service" not in item
                            or item.get("llm_service") == "chatgpt"
                        ):
                            cleaned_list.append(cleaned_item)
                    else:
                        cleaned_list.append(item)
                cleaned[key] = cleaned_list
            elif isinstance(value, dict):
                cleaned[key] = clean_chatgpt_data(value)
            else:
                cleaned[key] = value
        return cleaned
    return data


def migrate_legacy_files():
    """Migrate legacy all_results.json and automation_progress.json to model-specific files."""

    results_dir = Path("results")

    # Legacy files
    legacy_results = results_dir / "all_results.json"
    legacy_progress = results_dir / "automation_progress.json"

    # Check if legacy files exist
    if legacy_results.exists():
        print(f"Found legacy results file: {legacy_results}")

        # Load, clean, and save to ChatGPT-specific file
        with open(legacy_results, "r") as f:
            data = json.load(f)

        cleaned_data = clean_chatgpt_data(data)

        chatgpt_results = results_dir / "all_results_chatgpt.json"
        if not chatgpt_results.exists():
            with open(chatgpt_results, "w") as f:
                json.dump(cleaned_data, f, indent=2)
            print(f"Migrated and cleaned to: {chatgpt_results}")
        else:
            print(f"ChatGPT results file already exists: {chatgpt_results}")

    if legacy_progress.exists():
        print(f"Found legacy progress file: {legacy_progress}")

        # Load, clean, and save to ChatGPT-specific file
        with open(legacy_progress, "r") as f:
            data = json.load(f)

        cleaned_data = clean_chatgpt_data(data)

        chatgpt_progress = results_dir / "automation_progress_chatgpt.json"
        if not chatgpt_progress.exists():
            with open(chatgpt_progress, "w") as f:
                json.dump(cleaned_data, f, indent=2)
            print(f"Migrated and cleaned to: {chatgpt_progress}")
        else:
            print(f"ChatGPT progress file already exists: {chatgpt_progress}")

    print("Migration completed!")
    print("\nCleaned data by:")
    print("- Removing 'llm_service' fields from individual entries")
    print("- Filtering out any non-ChatGPT entries (if present)")
    print("- Normalizing batch keys by removing 'chatgpt_' prefix")
    print("- Ensuring compatibility with model-specific progress tracker")
    print("\nNow each model will use its own files:")
    print("- ChatGPT: all_results_chatgpt.json, automation_progress_chatgpt.json")
    print("- Gemini: all_results_gemini.json, automation_progress_gemini.json")
    print("- Copilot: all_results_copilot.json, automation_progress_copilot.json")
    print("\nBatch keys are now normalized (e.g., 'refusal_attack_narrative_first')")
    print("Progress tracker uses normalized keys consistently for all operations.")


if __name__ == "__main__":
    migrate_legacy_files()
