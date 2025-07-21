"""
Progress tracking for ChatGPT PDF review automation.
Enables resuming from the last processed PDF if the script crashes.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks and persists automation progress."""

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
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
        full_batch_key = f"{llm_service}_{batch_key}"
        return (
            full_batch_key in self.progress_data["completed_pdfs"]
            and pdf_name in self.progress_data["completed_pdfs"][full_batch_key]
            and request_type
            in self.progress_data["completed_pdfs"][full_batch_key][pdf_name]
        )

    def mark_pdf_processed(
        self,
        batch_key: str,
        pdf_name: str,
        request_type: str,
        llm_service: str = "chatgpt",
    ):
        """Mark a PDF/request combination as processed."""
        full_batch_key = f"{llm_service}_{batch_key}"

        if full_batch_key not in self.progress_data["completed_pdfs"]:
            self.progress_data["completed_pdfs"][full_batch_key] = {}

        if pdf_name not in self.progress_data["completed_pdfs"][full_batch_key]:
            self.progress_data["completed_pdfs"][full_batch_key][pdf_name] = {}

        self.progress_data["completed_pdfs"][full_batch_key][pdf_name][request_type] = (
            True
        )
        self.progress_data["total_requests_sent"] += 1

        # Check if this is a new PDF (first request type for this PDF)
        if len(self.progress_data["completed_pdfs"][full_batch_key][pdf_name]) == 1:
            self.progress_data["total_pdfs_processed"] += 1

        self._save_progress()

    def mark_batch_completed(self, batch_key: str, llm_service: str = "chatgpt"):
        """Mark an entire batch as completed."""
        full_batch_key = f"{llm_service}_{batch_key}"
        if full_batch_key not in self.progress_data["completed_batches"]:
            self.progress_data["completed_batches"].append(full_batch_key)
            self._save_progress()

    def is_batch_completed(self, batch_key: str, llm_service: str = "chatgpt") -> bool:
        """Check if a batch has been fully completed."""
        full_batch_key = f"{llm_service}_{batch_key}"
        return full_batch_key in self.progress_data["completed_batches"]

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
        logger.info(f"DEBUG: get_unprocessed_pdfs called with batch_key='{batch_key}', llm_service='{llm_service}'")
        logger.info(f"DEBUG: Checking {len(pdf_files)} PDFs with {len(request_types)} request types")
        
        unprocessed = []
        debug_stats = {
            "not_processed": 0,
            "should_retry": 0,
            "fully_processed": 0
        }

        for pdf_file in pdf_files:
            pdf_name = os.path.basename(pdf_file)
            for request_type in request_types:
                is_processed = self.is_pdf_processed(
                    batch_key, pdf_name, request_type, llm_service
                )
                should_retry = self.should_retry_failed_pdf(
                    batch_key, pdf_name, request_type, max_retries, llm_service
                )
                
                logger.info(f"DEBUG: {pdf_name} ({request_type}) - processed: {is_processed}, should_retry: {should_retry}")
                
                # Include if never processed OR failed but under retry limit
                if not is_processed or should_retry:
                    unprocessed.append((pdf_file, request_type))
                    if not is_processed:
                        debug_stats["not_processed"] += 1
                    if should_retry:
                        debug_stats["should_retry"] += 1
                        logger.info(f"DEBUG: {pdf_name} ({request_type}) - ADDING TO RETRY due to failure check")
                else:
                    debug_stats["fully_processed"] += 1

        logger.info(f"DEBUG: Stats - not_processed: {debug_stats['not_processed']}, should_retry: {debug_stats['should_retry']}, fully_processed: {debug_stats['fully_processed']}")
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
        full_batch_key = f"{llm_service}_{batch_key}"

        if full_batch_key not in self.progress_data["failed_pdfs"]:
            self.progress_data["failed_pdfs"][full_batch_key] = {}

        if pdf_name not in self.progress_data["failed_pdfs"][full_batch_key]:
            self.progress_data["failed_pdfs"][full_batch_key][pdf_name] = {}

        self.progress_data["failed_pdfs"][full_batch_key][pdf_name][request_type] = {
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
        full_batch_key = f"{llm_service}_{batch_key}"

        if (
            full_batch_key in self.progress_data["failed_pdfs"]
            and pdf_name in self.progress_data["failed_pdfs"][full_batch_key]
            and request_type
            in self.progress_data["failed_pdfs"][full_batch_key][pdf_name]
        ):
            attempts = self.progress_data["failed_pdfs"][full_batch_key][pdf_name][
                request_type
            ]["attempts"]
            logger.info(f"DEBUG: get_failure_count - {pdf_name} ({request_type}) has {attempts} failure attempts")
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
        full_batch_key = f"{llm_service}_{batch_key}"

        if (
            full_batch_key in self.progress_data["failed_pdfs"]
            and pdf_name in self.progress_data["failed_pdfs"][full_batch_key]
            and request_type
            in self.progress_data["failed_pdfs"][full_batch_key][pdf_name]
        ):
            del self.progress_data["failed_pdfs"][full_batch_key][pdf_name][
                request_type
            ]

            # Clean up empty structures
            if not self.progress_data["failed_pdfs"][full_batch_key][pdf_name]:
                del self.progress_data["failed_pdfs"][full_batch_key][pdf_name]
            if not self.progress_data["failed_pdfs"][full_batch_key]:
                del self.progress_data["failed_pdfs"][full_batch_key]

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
            logger.info(f"DEBUG: should_retry_failed_pdf - {pdf_name} ({request_type}) has {failure_count} failures, max_retries={max_retries}, should_retry={should_retry}")
        
        return should_retry
