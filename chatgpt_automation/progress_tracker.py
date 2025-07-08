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
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")

        return {
            "session_start": datetime.now().isoformat(),
            "last_updated": None,
            "completed_pdfs": {},  # batch_key -> {pdf_name -> {request_type -> True}}
            "completed_batches": [],  # List of fully completed batch keys
            "total_pdfs_processed": 0,
            "total_requests_sent": 0,
        }

    def _save_progress(self):
        """Save current progress to file."""
        try:
            self.progress_data["last_updated"] = datetime.now().isoformat()
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def is_pdf_processed(
        self, batch_key: str, pdf_name: str, request_type: str
    ) -> bool:
        """Check if a specific PDF/request combination has been processed."""
        return (
            batch_key in self.progress_data["completed_pdfs"]
            and pdf_name in self.progress_data["completed_pdfs"][batch_key]
            and request_type
            in self.progress_data["completed_pdfs"][batch_key][pdf_name]
        )

    def mark_pdf_processed(self, batch_key: str, pdf_name: str, request_type: str):
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

    def mark_batch_completed(self, batch_key: str):
        """Mark an entire batch as completed."""
        if batch_key not in self.progress_data["completed_batches"]:
            self.progress_data["completed_batches"].append(batch_key)
            self._save_progress()

    def is_batch_completed(self, batch_key: str) -> bool:
        """Check if a batch has been fully completed."""
        return batch_key in self.progress_data["completed_batches"]

    def get_resume_point(
        self, directories: List[Tuple[str, str, str, str]]
    ) -> List[Tuple[str, str, str, str]]:
        """Get the list of directories that still need processing."""
        remaining_directories = []

        for attack_type, prompt_type, injection_locus, pdf_dir in directories:
            batch_key = f"{attack_type}_{prompt_type}_{injection_locus}"
            if not self.is_batch_completed(batch_key):
                remaining_directories.append(
                    (attack_type, prompt_type, injection_locus, pdf_dir)
                )

        if remaining_directories:
            total_dirs = len(directories)
            remaining_count = len(remaining_directories)
            completed_count = total_dirs - remaining_count

            logger.info(
                f"Resuming automation: {completed_count}/{total_dirs} batches already completed"
            )
            logger.info(f"Remaining batches to process: {remaining_count}")

        return remaining_directories

    def get_unprocessed_pdfs(
        self, batch_key: str, pdf_files: List[str], request_types: List[str]
    ) -> List[Tuple[str, str]]:
        """Get PDF/request combinations that haven't been processed yet."""
        unprocessed = []

        for pdf_file in pdf_files:
            pdf_name = os.path.basename(pdf_file)
            for request_type in request_types:
                if not self.is_pdf_processed(batch_key, pdf_name, request_type):
                    unprocessed.append((pdf_file, request_type))

        return unprocessed

    def get_statistics(self) -> Dict:
        """Get progress statistics."""
        return {
            "session_start": self.progress_data["session_start"],
            "last_updated": self.progress_data["last_updated"],
            "total_pdfs_processed": self.progress_data["total_pdfs_processed"],
            "total_requests_sent": self.progress_data["total_requests_sent"],
            "completed_batches": len(self.progress_data["completed_batches"]),
            "completed_batch_keys": self.progress_data["completed_batches"],
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
