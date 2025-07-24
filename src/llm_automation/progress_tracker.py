"""
Progress tracking for ChatGPT PDF review automation.
Enables resuming from the last processed PDF if the script crashes.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks and persists automation progress."""

    def __init__(
        self, progress_file: Optional[str] = None, llm_service: str = "chatgpt"
    ):
        if progress_file is None:
            # Default to model-specific progress file
            progress_file = f"results/automation_progress_{llm_service}.json"
        self.progress_file = progress_file
        self.llm_service = llm_service
        self.progress_data = self._load_progress()

    def _load_progress(self) -> Dict:
        """Load existing progress from file."""
        default_structure: Dict = {
            "session_start": datetime.now().isoformat(),
            "last_updated": None,
            "completed_pdfs": {},  # batch_key -> {pdf_name -> {request_type -> True}}
            "failed_pdfs": {},  # batch_key -> {pdf_name -> {request_type -> {"attempts": N, "last_error": str}}}
            "completed_batches": [],  # List of fully completed batch keys
            "total_pdfs_processed": 0,
            "total_requests_sent": 0,
        }

        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)

                # Ensure backward compatibility - add missing keys
                for key, default_value in default_structure.items():
                    if key not in loaded_data:
                        loaded_data[key] = default_value
                        logger.info(
                            f"Added missing key '{key}' to progress data for backward compatibility"
                        )

                return loaded_data
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")

        return default_structure

    def _save_progress(self):
        """Save current progress to file."""
        try:
            self.progress_data["last_updated"] = datetime.now().isoformat()
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def is_pdf_processed(
        self,
        batch_key: str,
        pdf_name: str,
        request_type: str,
        llm_service: str = "chatgpt",
    ) -> bool:
        """Check if a specific PDF/request combination has been processed."""
        return (
            batch_key in self.progress_data["completed_pdfs"]
            and pdf_name in self.progress_data["completed_pdfs"][batch_key]
            and request_type
            in self.progress_data["completed_pdfs"][batch_key][pdf_name]
        )

    def mark_pdf_processed(
        self,
        batch_key: str,
        pdf_name: str,
        request_type: str,
        llm_service: str = "chatgpt",
    ):
        """Mark a PDF/request combination as processed."""
        if batch_key not in self.progress_data["completed_pdfs"]:
            self.progress_data["completed_pdfs"][batch_key] = {}

        if pdf_name not in self.progress_data["completed_pdfs"][batch_key]:
            self.progress_data["completed_pdfs"][batch_key][pdf_name] = {}

        self.progress_data["completed_pdfs"][batch_key][pdf_name][request_type] = True
        self.progress_data["total_requests_sent"] += 1

        # Check if this is a new PDF (first request type for this PDF)
        if len(self.progress_data["completed_pdfs"][batch_key][pdf_name]) == 1:
            self.progress_data["total_pdfs_processed"] += 1

        self._save_progress()

    def mark_batch_completed(self, batch_key: str, llm_service: str = "chatgpt"):
        """Mark an entire batch as completed."""
        if batch_key not in self.progress_data["completed_batches"]:
            self.progress_data["completed_batches"].append(batch_key)
            self._save_progress()

    def is_batch_completed(self, batch_key: str, llm_service: str = "chatgpt") -> bool:
        """Check if a batch has been fully completed."""
        return batch_key in self.progress_data["completed_batches"]

    def get_resume_point(
        self, directories: List[Tuple[str, str, str, str]], llm_service: str = "chatgpt"
    ) -> List[Tuple[str, str, str, str]]:
        """Get the list of directories that still need processing."""
        remaining_directories = []

        for attack_type, prompt_type, injection_locus, pdf_dir in directories:
            batch_key = f"{attack_type}_{prompt_type}_{injection_locus}"
            if not self.is_batch_completed(batch_key, llm_service):
                remaining_directories.append(
                    (attack_type, prompt_type, injection_locus, pdf_dir)
                )

        if remaining_directories:
            total_dirs = len(directories)
            remaining_count = len(remaining_directories)
            completed_count = total_dirs - remaining_count

            logger.info(
                f"Resuming {llm_service} automation: {completed_count}/{total_dirs} batches already completed"
            )
            logger.info(f"Remaining batches to process: {remaining_count}")

        return remaining_directories

    def get_unprocessed_pdfs(
        self,
        batch_key: str,
        pdf_files: List[str],
        request_types: List[str],
        max_retries: int = 3,
        llm_service: str = "chatgpt",
    ) -> List[Tuple[str, str]]:
        """Get PDF/request combinations that haven't been processed yet or should be retried."""
        logger.info(
            f"DEBUG: get_unprocessed_pdfs called with batch_key='{batch_key}', llm_service='{llm_service}'"
        )
        logger.info(
            f"DEBUG: Checking {len(pdf_files)} PDFs with {len(request_types)} request types"
        )

        unprocessed = []
        debug_stats = {"not_processed": 0, "should_retry": 0, "fully_processed": 0}

        for pdf_file in pdf_files:
            pdf_name = os.path.basename(pdf_file)
            for request_type in request_types:
                is_processed = self.is_pdf_processed(
                    batch_key, pdf_name, request_type, llm_service
                )
                should_retry = self.should_retry_failed_pdf(
                    batch_key, pdf_name, request_type, max_retries, llm_service
                )

                logger.debug(
                    f"DEBUG: {pdf_name} ({request_type}) - processed: {is_processed}, should_retry: {should_retry}"
                )

                # Include if never processed OR failed but under retry limit
                if not is_processed or should_retry:
                    unprocessed.append((pdf_file, request_type))
                    if not is_processed:
                        debug_stats["not_processed"] += 1
                    if should_retry:
                        debug_stats["should_retry"] += 1
                        logger.debug(
                            f"DEBUG: {pdf_name} ({request_type}) - ADDING TO RETRY due to failure check"
                        )
                else:
                    debug_stats["fully_processed"] += 1

        logger.info(
            f"DEBUG: Stats - not_processed: {debug_stats['not_processed']}, should_retry: {debug_stats['should_retry']}, fully_processed: {debug_stats['fully_processed']}"
        )
        logger.info(f"DEBUG: Total unprocessed items: {len(unprocessed)}")

        return unprocessed

    def get_statistics(self) -> Dict:
        """Get progress statistics."""
        # Count total failed items
        total_failed_items = 0
        for batch_key in self.progress_data.get("failed_pdfs", {}):
            for pdf_name in self.progress_data["failed_pdfs"][batch_key]:
                total_failed_items += len(
                    self.progress_data["failed_pdfs"][batch_key][pdf_name]
                )

        return {
            "session_start": self.progress_data["session_start"],
            "last_updated": self.progress_data["last_updated"],
            "total_pdfs_processed": self.progress_data["total_pdfs_processed"],
            "total_requests_sent": self.progress_data["total_requests_sent"],
            "completed_batches": len(self.progress_data["completed_batches"]),
            "completed_batch_keys": self.progress_data["completed_batches"],
            "total_failed_items": total_failed_items,
        }

    def reset_progress(self):
        """Reset all progress (start fresh)."""
        self.progress_data = {
            "session_start": datetime.now().isoformat(),
            "last_updated": None,
            "completed_pdfs": {},
            "completed_batches": [],
            "total_pdfs_processed": 0,
            "total_requests_sent": 0,
        }
        self._save_progress()
        logger.info("Progress reset - starting fresh")

    def cleanup_progress_file(self):
        """Remove the progress file."""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                logger.info("Progress file cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup progress file: {e}")

    def mark_pdf_failed(
        self,
        batch_key: str,
        pdf_name: str,
        request_type: str,
        error_msg: str,
        max_retries: int,
        llm_service: str = "chatgpt",
    ):
        """Mark a PDF/request combination as failed after max retries."""
        if batch_key not in self.progress_data["failed_pdfs"]:
            self.progress_data["failed_pdfs"][batch_key] = {}

        if pdf_name not in self.progress_data["failed_pdfs"][batch_key]:
            self.progress_data["failed_pdfs"][batch_key][pdf_name] = {}

        self.progress_data["failed_pdfs"][batch_key][pdf_name][request_type] = {
            "attempts": max_retries,
            "last_error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }

        self._save_progress()

    def get_failure_count(
        self,
        batch_key: str,
        pdf_name: str,
        request_type: str,
        llm_service: str = "chatgpt",
    ) -> int:
        """Get the number of previous failure attempts for a PDF/request combination."""
        if (
            batch_key in self.progress_data["failed_pdfs"]
            and pdf_name in self.progress_data["failed_pdfs"][batch_key]
            and request_type in self.progress_data["failed_pdfs"][batch_key][pdf_name]
        ):
            attempts = self.progress_data["failed_pdfs"][batch_key][pdf_name][
                request_type
            ]["attempts"]
            logger.info(
                f"DEBUG: get_failure_count - {pdf_name} ({request_type}) has {attempts} failure attempts"
            )
            return attempts
        return 0

    def clear_pdf_failure(
        self,
        batch_key: str,
        pdf_name: str,
        request_type: str,
        llm_service: str = "chatgpt",
    ):
        """Clear failure record when a PDF is successfully processed."""
        if (
            batch_key in self.progress_data["failed_pdfs"]
            and pdf_name in self.progress_data["failed_pdfs"][batch_key]
            and request_type in self.progress_data["failed_pdfs"][batch_key][pdf_name]
        ):
            del self.progress_data["failed_pdfs"][batch_key][pdf_name][request_type]

            # Clean up empty structures
            if not self.progress_data["failed_pdfs"][batch_key][pdf_name]:
                del self.progress_data["failed_pdfs"][batch_key][pdf_name]
            if not self.progress_data["failed_pdfs"][batch_key]:
                del self.progress_data["failed_pdfs"][batch_key]

            self._save_progress()

    def should_retry_failed_pdf(
        self,
        batch_key: str,
        pdf_name: str,
        request_type: str,
        max_retries: int,
        llm_service: str = "chatgpt",
    ) -> bool:
        """Check if a failed PDF should be retried (hasn't exceeded max retries)."""
        failure_count = self.get_failure_count(
            batch_key, pdf_name, request_type, llm_service
        )

        # Only retry if there ARE failures (failure_count > 0) AND it's under the retry limit
        should_retry = failure_count > 0 and failure_count < max_retries

        if failure_count > 0:
            logger.info(
                f"DEBUG: should_retry_failed_pdf - {pdf_name} ({request_type}) has {failure_count} failures, max_retries={max_retries}, should_retry={should_retry}"
            )

        return should_retry

    def sync_with_results_file(
        self, results_file_path: str, llm_service: str = "chatgpt"
    ):
        """Synchronize progress tracker with actual results in the consolidated file.

        This method checks the results file and marks batches as completed if all
        expected PDFs have been processed, fixing any discrepancies.
        """
        try:
            import json

            if not os.path.exists(results_file_path):
                logger.warning(f"Results file not found: {results_file_path}")
                return

            with open(results_file_path, "r", encoding="utf-8") as f:
                results_data = json.load(f)

            logger.info(
                f"Synchronizing progress with results file: {results_file_path}"
            )

            for batch_key, batch_data in results_data.items():
                # Check if this batch is already marked as completed
                if self.is_batch_completed(batch_key, llm_service):
                    continue

                # Count total results for this batch across all request types
                total_results = 0
                for request_type, results_list in batch_data.items():
                    total_results += len(results_list)

                    # Mark individual PDFs as processed
                    for result in results_list:
                        pdf_name = result.get("pdf_file")
                        if pdf_name and result.get("success", False):
                            self.mark_pdf_processed(
                                batch_key, pdf_name, request_type, llm_service
                            )

                # If we have results, check if batch should be marked as completed
                if total_results > 0:
                    logger.info(
                        f"Found {total_results} results for batch {batch_key}, checking completion status"
                    )

                    # For now, if there are any results, we assume the batch was attempted
                    # The main automation logic will determine if it's truly complete
                    # by checking against expected PDF counts

            self._save_progress()
            logger.info("Progress synchronization completed")

        except Exception as e:
            logger.error(f"Failed to sync with results file: {e}")

    def mark_batch_completed_if_all_processed(
        self,
        batch_key: str,
        expected_pdf_count: int,
        expected_request_types: List[str],
        llm_service: str = "chatgpt",
    ) -> bool:
        """Mark batch as completed only if all expected PDFs and request types are processed.

        Returns True if batch was marked as completed, False otherwise.
        """
        if batch_key not in self.progress_data["completed_pdfs"]:
            return False

        batch_pdfs = self.progress_data["completed_pdfs"][batch_key]

        # Count processed PDFs and request types
        processed_pdfs = len(batch_pdfs)

        # Check if all request types are covered for each PDF
        total_processed_requests = 0
        for pdf_name, requests in batch_pdfs.items():
            for request_type in expected_request_types:
                if request_type in requests:
                    total_processed_requests += 1

        expected_total_requests = expected_pdf_count * len(expected_request_types)

        if (
            processed_pdfs >= expected_pdf_count
            and total_processed_requests >= expected_total_requests
        ):
            if not self.is_batch_completed(batch_key, llm_service):
                self.mark_batch_completed(batch_key, llm_service)
                logger.info(
                    f"Batch {batch_key} marked as completed: {processed_pdfs} PDFs, {total_processed_requests} requests"
                )
                return True
        else:
            logger.debug(
                f"Batch {batch_key} incomplete: {processed_pdfs}/{expected_pdf_count} PDFs, {total_processed_requests}/{expected_total_requests} requests"
            )

        return False

    def clear_attack_type_progress(
        self,
        attack_type: str,
        llm_service: str = "chatgpt",
        results_file_path: Optional[str] = None,
    ):
        """Clear all progress for a specific attack type.

        This removes all batches, completed PDFs, and failed PDFs for the given attack type.
        Also removes results from the results file to prevent re-synchronization.
        Useful when you want to reprocess a specific attack type from scratch.

        Args:
            attack_type: The attack type to clear progress for
            llm_service: The LLM service being used
            results_file_path: Path to the results file to also clear (optional)
        """
        cleared_batches = 0
        cleared_pdfs = 0
        cleared_failed = 0
        cleared_results = 0

        # Remove completed batches for this attack type
        batches_to_remove = [
            batch_key
            for batch_key in self.progress_data["completed_batches"]
            if batch_key.startswith(f"{attack_type}_")
        ]
        for batch_key in batches_to_remove:
            self.progress_data["completed_batches"].remove(batch_key)
            cleared_batches += 1

        # Remove completed PDFs for this attack type
        pdfs_to_remove = [
            batch_key
            for batch_key in self.progress_data["completed_pdfs"]
            if batch_key.startswith(f"{attack_type}_")
        ]
        for batch_key in pdfs_to_remove:
            pdf_count = len(self.progress_data["completed_pdfs"][batch_key])
            del self.progress_data["completed_pdfs"][batch_key]
            cleared_pdfs += pdf_count

        # Remove failed PDFs for this attack type
        failed_to_remove = [
            batch_key
            for batch_key in self.progress_data["failed_pdfs"]
            if batch_key.startswith(f"{attack_type}_")
        ]
        for batch_key in failed_to_remove:
            failed_count = len(self.progress_data["failed_pdfs"][batch_key])
            del self.progress_data["failed_pdfs"][batch_key]
            cleared_failed += failed_count

        # Also clear results from the results file to prevent re-synchronization
        if results_file_path and os.path.exists(results_file_path):
            try:
                with open(results_file_path, "r", encoding="utf-8") as f:
                    results_data = json.load(f)

                # Remove batches for this attack type from results
                results_to_remove = [
                    batch_key
                    for batch_key in results_data
                    if batch_key.startswith(f"{attack_type}_")
                ]

                for batch_key in results_to_remove:
                    del results_data[batch_key]
                    cleared_results += 1

                # Save updated results file
                if cleared_results > 0:
                    with open(results_file_path, "w", encoding="utf-8") as f:
                        json.dump(results_data, f, indent=2, ensure_ascii=False)
                    logger.info(
                        f"Cleared {cleared_results} result batches from {results_file_path}"
                    )

            except Exception as e:
                logger.warning(f"Failed to clear results from {results_file_path}: {e}")

        if cleared_batches > 0 or cleared_pdfs > 0 or cleared_failed > 0:
            self._save_progress()
            logger.info(
                f"Cleared progress for attack type '{attack_type}': "
                f"{cleared_batches} batches, {cleared_pdfs} completed PDFs, {cleared_failed} failed PDFs"
                + (f", {cleared_results} result batches" if cleared_results > 0 else "")
            )
        else:
            logger.info(f"No progress found for attack type '{attack_type}' to clear")

    def get_available_attack_types(self) -> List[str]:
        """Get list of all attack types that have progress recorded."""
        attack_types = set()

        # Extract from completed batches
        for batch_key in self.progress_data["completed_batches"]:
            parts = batch_key.split("_")
            if len(parts) >= 3:
                # Handle attack types with underscores
                attack_type = "_".join(parts[:-2])
                attack_types.add(attack_type)

        # Extract from completed PDFs
        for batch_key in self.progress_data["completed_pdfs"]:
            parts = batch_key.split("_")
            if len(parts) >= 3:
                attack_type = "_".join(parts[:-2])
                attack_types.add(attack_type)

        # Extract from failed PDFs
        for batch_key in self.progress_data["failed_pdfs"]:
            parts = batch_key.split("_")
            if len(parts) >= 3:
                attack_type = "_".join(parts[:-2])
                attack_types.add(attack_type)

        return sorted(list(attack_types))
