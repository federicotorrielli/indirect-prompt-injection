#!/usr/bin/env python3
"""
Script to clean up unsuccessful (success: false) results from model-specific results files
and the corresponding entries from automation_progress files before restarting the automation.
"""

import json
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def clean_unsuccessful_results(results_dir="results", llm_service="chatgpt"):
    """
    Remove all unsuccessful results (where success=False) from model-specific results files
    and the corresponding entries from automation_progress files.
    """
    # File paths
    results_file = os.path.join(results_dir, f"all_results_{llm_service}.json")
    progress_file = os.path.join(results_dir, f"automation_progress_{llm_service}.json")

    if not os.path.exists(results_file) or not os.path.exists(progress_file):
        logger.error(
            f"Results file or progress file not found for {llm_service} in {results_dir}"
        )
        logger.error(f"Expected files: {results_file}, {progress_file}")
        return False

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # Load progress
    with open(progress_file, "r", encoding="utf-8") as f:
        progress_data = json.load(f)

    # Collect items to remove (batch_key, pdf_name, request_type)
    items_to_remove = []
    original_result_count = count_results(all_results)
    successful_results = {}

    # Filter results and collect items to remove
    for batch_key, batch_data in all_results.items():
        successful_results[batch_key] = {}
        for request_type, results_list in batch_data.items():
            successful_results[batch_key][request_type] = []
            for result in results_list:
                if result.get("success", False):
                    # Keep successful results
                    successful_results[batch_key][request_type].append(result)
                else:
                    # Add to removal list
                    items_to_remove.append(
                        (
                            batch_key,
                            result.get("pdf_file"),
                            result.get("request_type", request_type),
                        )
                    )

            # Clean up empty lists
            if not successful_results[batch_key][request_type]:
                del successful_results[batch_key][request_type]

        # Clean up empty dictionaries
        if not successful_results[batch_key]:
            del successful_results[batch_key]

    # Clean up progress data - reset completed_pdfs for unsuccessful items
    for batch_key, pdf_name, request_type in items_to_remove:
        if (
            batch_key in progress_data.get("completed_pdfs", {})
            and pdf_name in progress_data["completed_pdfs"].get(batch_key, {})
            and request_type
            in progress_data["completed_pdfs"][batch_key].get(pdf_name, {})
        ):
            # Remove the processed flag
            del progress_data["completed_pdfs"][batch_key][pdf_name][request_type]

            # Clean up empty structures
            if not progress_data["completed_pdfs"][batch_key][pdf_name]:
                del progress_data["completed_pdfs"][batch_key][pdf_name]
            if not progress_data["completed_pdfs"][batch_key]:
                del progress_data["completed_pdfs"][batch_key]

    # Clean up failed_pdfs (reset all failure records)
    progress_data["failed_pdfs"] = {}

    # Clean up completed_batches (batches are only complete if ALL items succeeded)
    # This forces a reprocessing of batches with any failures
    # Get the list of batch keys from results
    all_batch_keys = set(all_results.keys())
    successful_batch_keys = set(successful_results.keys())

    # Batches with any failures need to be reprocessed
    batches_with_failures = all_batch_keys - successful_batch_keys

    # Remove those batches from completed_batches
    progress_data["completed_batches"] = [
        batch
        for batch in progress_data.get("completed_batches", [])
        if batch not in batches_with_failures
    ]

    # Update counts in progress data
    progress_data["total_requests_sent"] = sum(
        len(request_types)
        for batch_data in progress_data.get("completed_pdfs", {}).values()
        for request_types in batch_data.values()
    )

    # Update progress_data["total_pdfs_processed"]
    # (count unique PDFs across all batches)
    processed_pdfs = set()
    for batch_key, pdf_dict in progress_data.get("completed_pdfs", {}).items():
        processed_pdfs.update(pdf_dict.keys())
    progress_data["total_pdfs_processed"] = len(processed_pdfs)

    # Save filtered results
    new_result_count = count_results(successful_results)
    logger.info(
        f"Removing {original_result_count - new_result_count} unsuccessful results"
    )

    # Backup original files
    os.rename(results_file, f"{results_file}.bak")
    os.rename(progress_file, f"{progress_file}.bak")

    # Save new files
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(successful_results, f, indent=2, ensure_ascii=False)

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully cleaned up {llm_service} results and progress files")
    logger.info(f"Original result count: {original_result_count}")
    logger.info(f"New result count: {new_result_count}")
    logger.info(
        f"Removed {original_result_count - new_result_count} unsuccessful results"
    )

    return True


def count_results(results_dict):
    """Count total number of results in the nested dictionary."""
    count = 0
    for batch_data in results_dict.values():
        for results_list in batch_data.values():
            count += len(results_list)
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean unsuccessful results from model-specific files"
    )
    parser.add_argument(
        "--llm-service",
        choices=["chatgpt", "copilot", "gemini"],
        default="chatgpt",
        help="LLM service to clean results for (default: chatgpt)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory path (default: results)",
    )

    args = parser.parse_args()

    logger.info(f"Cleaning unsuccessful results for {args.llm_service}")
    success = clean_unsuccessful_results(args.results_dir, args.llm_service)
    exit(0 if success else 1)
