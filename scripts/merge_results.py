#!/usr/bin/env python3
"""Merge automation result & progress JSON files with newer homonym files.

This script merges:
  - results/all_results_<service>.json
  - results/automation_progress_<service>.json
with newer JSON files of identical structure located in another directory
(e.g. ~/Downloads) and writes the merged artifacts back in-place (with
timestamped backups) unless --dry-run is specified.

Result file structure:
    all_results_<service>.json => { batch_key: { request_type: [ entry, ... ] } }
Each entry keys encountered include (non-exhaustive):
    pdf_file, attack_type, injection_locus, prompt_type, request_text,
    request_type, response, success, timestamp (plus possible future keys).

Progress file structure:
  automation_progress_<service>.json => {
      session_start: str,
      last_updated: str,
      completed_pdfs: { batch: { pdf: { request_flag: true, ... } } },
      failed_pdfs: { ... },
      completed_batches: [ ... ],
      total_pdfs_processed: int,
      total_requests_sent: int
  }

Conflict resolution strategies for result entries:
  prefer-existing: Keep existing entry when identity collision occurs.
  prefer-new: Always replace with new entry on collision.
  prefer-success (default): Replace only if new.success=True and existing.success=False.

Identity heuristic (in order):
    1. entry['id'] if present.
    2. Concatenation of salient fields among:
             pdf_file, attack_type, injection_locus, prompt_type,
             request_type, prompt_label, prompt_variant (only those present).
         This better differentiates multiple prompt/injection variants for the same PDF.
    3. Fallback: SHA256 hash of sorted key/value pairs excluding volatile keys.

Volatile keys excluded from identity hashing: ['timestamp', 'model_latency', 'raw_response'].

Outputs a concise summary of additions, replacements, skips, and newly covered
request flags.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(
    level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ----------------------------- Utility Functions -----------------------------


def load_json(path: Path) -> dict:
    """Load JSON from path with helpful error messages."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"File not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {path}: {e}")


def save_json(path: Path, data: dict) -> None:
    """Write JSON with UTF-8 and indentation."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def timestamp() -> str:
    # Use timezone-aware UTC datetime
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def backup_file(path: Path) -> Path:
    """Create a timestamped backup next to the file."""
    backup = path.with_suffix(
        path.suffix
        + f".bak.{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )
    path.rename(backup)
    return backup


# --------------------------- Result Merge Algorithm --------------------------

VOLATILE_IDENTITY_KEYS = {"timestamp", "model_latency", "raw_response"}


def derive_identity(entry: Mapping) -> str:
    """Produce a stable identity string for a result entry.

    Priority:
      - Explicit 'id' field
      - Concatenated tuple of salient fields
      - Hash of non-volatile sorted key/value pairs
    """

    if "id" in entry and entry["id"]:
        return f"id::{entry['id']}"

    salient_fields = []
    for key in (
        "pdf_file",
        "attack_type",
        "injection_locus",
        "prompt_type",
        "request_type",
        "prompt_label",
        "prompt_variant",
    ):
        if key in entry and entry[key] is not None:
            salient_fields.append(f"{key}={entry[key]}")
    if salient_fields:
        return "|".join(salient_fields)

    # Fallback: hash of deterministic serialization
    filtered_items = [
        (k, entry[k]) for k in sorted(entry.keys()) if k not in VOLATILE_IDENTITY_KEYS
    ]
    serialized = json.dumps(filtered_items, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ResultMerger:
    """Merge two all_results_<service>.json structures."""

    def __init__(self, strategy: str = "prefer-success") -> None:
        self.strategy = strategy
        self.added = 0
        self.replaced = 0
        self.skipped = 0

    def merge(self, base: dict, newer: dict) -> dict:
        for batch_key, batch_data in newer.items():
            if not isinstance(batch_data, Mapping):
                logger.debug("Skipping non-mapping batch '%s'", batch_key)
                continue
            base.setdefault(batch_key, {})
            for request_type, entries in batch_data.items():
                if not isinstance(entries, list):
                    continue
                base.setdefault(batch_key, {}).setdefault(request_type, [])
                existing_list: List[dict] = base[batch_key][request_type]  # type: ignore[assignment]

                # Build index of existing entries by identity
                existing_index = {
                    derive_identity(e): i
                    for i, e in enumerate(existing_list)
                    if isinstance(e, Mapping)
                }

                for entry in entries:
                    if not isinstance(entry, Mapping):
                        continue
                    ident = derive_identity(entry)
                    if ident in existing_index:
                        existing_entry = existing_list[existing_index[ident]]
                        if self._should_replace(existing_entry, entry):
                            existing_list[existing_index[ident]] = dict(
                                entry
                            )  # ensure concrete dict
                            self.replaced += 1
                        else:
                            self.skipped += 1
                    else:
                        existing_list.append(dict(entry))  # ensure concrete dict
                        self.added += 1

        # Normalize ordering for deterministic output
        for batch_key in base:
            for request_type in base[batch_key]:
                base[batch_key][request_type] = sorted(
                    base[batch_key][request_type],
                    key=lambda e: (
                        str(e.get("pdf_file")),
                        str(e.get("attack_type")),
                        str(e.get("injection_locus")),
                        str(e.get("prompt_type")),
                        str(e.get("request_type")),
                    ),
                )
        return base

    def _should_replace(self, existing: Mapping, new: Mapping) -> bool:
        if self.strategy == "prefer-existing":
            return False
        if self.strategy == "prefer-new":
            return True
        # prefer-success (default)
        existing_success = bool(existing.get("success"))
        new_success = bool(new.get("success"))
        if new_success and not existing_success:
            return True
        # If both same success, optionally use timestamp recency
        new_ts = existing_ts = None
        for obj, var in ((existing, "existing_ts"), (new, "new_ts")):
            ts_val = obj.get("timestamp")
            try:
                if ts_val:
                    parsed = dt.datetime.fromisoformat(ts_val.replace("Z", ""))
                    if obj is existing:
                        existing_ts = parsed
                    else:
                        new_ts = parsed
            except Exception:
                pass
        if new_ts and existing_ts and new_ts > existing_ts:
            return True
        return False


# -------------------------- Progress Merge Algorithm -------------------------


class ProgressMerger:
    """Merge two automation_progress_<service>.json structures."""

    def __init__(self) -> None:
        self.new_flags = 0

    def merge(self, base: dict, newer: dict) -> dict:
        # completed_pdfs merging
        newer_cp = newer.get("completed_pdfs", {})
        base.setdefault("completed_pdfs", {})

        for batch_key, pdf_map in newer_cp.items():
            if not isinstance(pdf_map, Mapping):
                continue
            base["completed_pdfs"].setdefault(batch_key, {})
            for pdf_name, flags in pdf_map.items():
                if not isinstance(flags, Mapping):
                    continue
                base["completed_pdfs"][batch_key].setdefault(pdf_name, {})
                for flag, value in flags.items():
                    if (
                        value
                        and flag not in base["completed_pdfs"][batch_key][pdf_name]
                    ):
                        self.new_flags += 1
                    base["completed_pdfs"][batch_key][pdf_name][flag] = bool(value)

        # Merge failed_pdfs (simple union) but drop those now completed
        base.setdefault("failed_pdfs", {})
        for batch_key, pdf_map in newer.get("failed_pdfs", {}).items():
            base["failed_pdfs"].setdefault(batch_key, {})
            for pdf_name, meta in pdf_map.items():
                if (
                    batch_key in base["completed_pdfs"]
                    and pdf_name in base["completed_pdfs"][batch_key]
                ):
                    continue  # completed now
                base["failed_pdfs"][batch_key][pdf_name] = meta

        # Merge completed_batches (set union)
        existing_batches = set(base.get("completed_batches", []))
        newer_batches = set(newer.get("completed_batches", []))
        base["completed_batches"] = sorted(existing_batches | newer_batches)

        # Recalculate counters
        completed_pdfs = base.get("completed_pdfs", {})
        request_flag_count = 0
        unique_pdfs = set()
        for batch_key, pdf_map in completed_pdfs.items():
            for pdf_name, flags in pdf_map.items():
                unique_pdfs.add((batch_key, pdf_name))
                request_flag_count += sum(1 for v in flags.values() if v)
        base["total_requests_sent"] = request_flag_count
        base["total_pdfs_processed"] = len(unique_pdfs)

        # Keep earliest session_start
        session_start_candidates = [
            c
            for c in (base.get("session_start"), newer.get("session_start"))
            if isinstance(c, str)
        ]
        if session_start_candidates:
            try:
                # choose earliest isoformat
                base["session_start"] = min(session_start_candidates)
            except Exception:
                base["session_start"] = session_start_candidates[0]
        base["last_updated"] = timestamp()
        return base


# ----------------------------- High-Level Orchestration -----------------------------


def merge_files(
    llm_service: str,
    results_dir: Path,
    new_dir: Path,
    strategy: str,
    dry_run: bool,
    no_backup: bool,
) -> int:
    if strategy not in {"prefer-existing", "prefer-new", "prefer-success"}:
        raise SystemExit(
            "Invalid strategy. Choose from: prefer-existing, prefer-new, prefer-success"
        )

    results_filename = f"all_results_{llm_service}.json"
    progress_filename = f"automation_progress_{llm_service}.json"

    base_results_path = results_dir / results_filename
    base_progress_path = results_dir / progress_filename

    new_results_path = new_dir / results_filename
    new_progress_path = new_dir / progress_filename

    logger.info("Loading base files: %s, %s", base_results_path, base_progress_path)
    base_results = load_json(base_results_path)
    base_progress = load_json(base_progress_path)

    logger.info("Loading newer files: %s, %s", new_results_path, new_progress_path)
    newer_results = load_json(new_results_path)
    newer_progress = load_json(new_progress_path)

    # Merge results
    result_merger = ResultMerger(strategy=strategy)
    merged_results = result_merger.merge(base_results, newer_results)

    # Merge progress
    progress_merger = ProgressMerger()
    merged_progress = progress_merger.merge(base_progress, newer_progress)

    # Summaries
    logger.info(
        "Results merge summary: added=%d replaced=%d skipped=%d",
        result_merger.added,
        result_merger.replaced,
        result_merger.skipped,
    )
    logger.info(
        "Progress merge summary: new request flags added=%d", progress_merger.new_flags
    )
    logger.info(
        "Merged progress counters: total_pdfs_processed=%s total_requests_sent=%s",
        merged_progress.get("total_pdfs_processed"),
        merged_progress.get("total_requests_sent"),
    )

    if dry_run:
        logger.info("Dry run enabled: no files written.")
        return 0

    if not no_backup:
        br = backup_file(base_results_path)
        bp = backup_file(base_progress_path)
        logger.info("Backups created: %s, %s", br.name, bp.name)
    else:
        logger.info("No backups requested (--no-backup). Proceeding to overwrite.")

    save_json(base_results_path, merged_results)
    save_json(base_progress_path, merged_progress)
    logger.info("Merged files written: %s, %s", base_results_path, base_progress_path)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge existing automation result/progress JSON with newer versions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--llm-service",
        default="gemini",
        help="LLM service name suffix used in filenames.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        type=Path,
        help="Directory containing the base JSON files to merge into.",
    )
    parser.add_argument(
        "--new-dir",
        default=Path.home() / "Downloads",
        type=Path,
        help="Directory containing newer homonym JSON files.",
    )
    parser.add_argument(
        "--strategy",
        choices=["prefer-existing", "prefer-new", "prefer-success"],
        default="prefer-success",
        help="Conflict resolution strategy for result entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse & merge but do not write any files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create timestamped .bak files before overwriting.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    results_dir: Path = args.results_dir
    new_dir: Path = args.new_dir
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")
    if not new_dir.is_dir():
        raise SystemExit(f"New files directory not found: {new_dir}")

    return merge_files(
        llm_service=args.llm_service,
        results_dir=results_dir,
        new_dir=new_dir,
        strategy=args.strategy,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
