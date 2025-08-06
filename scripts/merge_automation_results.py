#!/usr/bin/env python3
"""
Intelligent Automation Results Merger

A sophisticated script to merge automation results from multiple computers using
advanced conflict resolution algorithms and beautiful CLI interface.

Author: GitHub Copilot
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ResultFingerprint:
    """Unique identifier for a result based on key attributes."""

    pdf_file: str
    attack_type: str
    prompt_type: str
    injection_locus: str
    request_type: str

    def __hash__(self) -> int:
        return hash(
            (
                self.pdf_file,
                self.attack_type,
                self.prompt_type,
                self.injection_locus,
                self.request_type,
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ResultFingerprint):
            return False
        return (
            self.pdf_file == other.pdf_file
            and self.attack_type == other.attack_type
            and self.prompt_type == other.prompt_type
            and self.injection_locus == other.injection_locus
            and self.request_type == other.request_type
        )


@dataclass
class ConflictResolution:
    """Information about how a conflict was resolved."""

    fingerprint: ResultFingerprint
    source_a_timestamp: str
    source_b_timestamp: str
    chosen_source: str
    resolution_reason: str
    source_a_success: bool
    source_b_success: bool


@dataclass
class MergeStatistics:
    """Statistics about the merge operation."""

    total_results_a: int = 0
    total_results_b: int = 0
    unique_to_a: int = 0
    unique_to_b: int = 0
    conflicts_found: int = 0
    conflicts_resolved: int = 0
    final_result_count: int = 0
    merge_duration: float = 0.0
    conflicts: List[ConflictResolution] = field(default_factory=list)


class AutomationResultsMerger:
    """Advanced merger for automation results with intelligent conflict resolution."""

    def __init__(self, console: Console):
        self.console = console
        self.logger = logger

    def create_fingerprint(self, result: Dict[str, Any]) -> ResultFingerprint:
        """Create a unique fingerprint for a result."""
        return ResultFingerprint(
            pdf_file=result.get("pdf_file", ""),
            attack_type=result.get("attack_type", ""),
            prompt_type=result.get("prompt_type", ""),
            injection_locus=result.get("injection_locus", ""),
            request_type=result.get("request_type", ""),
        )

    def assess_result_quality(self, result: Dict[str, Any]) -> float:
        """Assess the quality of a result for conflict resolution."""
        score = 0.0

        # Success is most important
        if result.get("success", False):
            score += 50.0

        # Response length indicates thoroughness
        response = result.get("response", "")
        if isinstance(response, str):
            response_length = len(response)
            score += min(response_length / 100, 30.0)  # Cap at 30 points

        # Presence of key fields
        if result.get("pdf_file"):
            score += 5.0
        if result.get("attack_type"):
            score += 5.0
        if result.get("timestamp"):
            score += 5.0

        # Response quality heuristics
        if isinstance(response, str):
            if "error" in response.lower() or "failed" in response.lower():
                score -= 20.0
            if len(response.split()) > 50:  # Substantial response
                score += 10.0

        return max(score, 0.0)

    def resolve_conflict(
        self,
        result_a: Dict[str, Any],
        result_b: Dict[str, Any],
        fingerprint: ResultFingerprint,
    ) -> Tuple[Dict[str, Any], ConflictResolution]:
        """Resolve a conflict between two results with the same fingerprint."""
        timestamp_a = result_a.get("timestamp", "")
        timestamp_b = result_b.get("timestamp", "")

        # Parse timestamps for comparison
        dt_a = None
        dt_b = None
        try:
            dt_a = datetime.fromisoformat(timestamp_a.replace("Z", "+00:00"))
            dt_b = datetime.fromisoformat(timestamp_b.replace("Z", "+00:00"))
            time_diff = abs((dt_a - dt_b).total_seconds())
        except (ValueError, AttributeError):
            time_diff = float("inf")

        quality_a = self.assess_result_quality(result_a)
        quality_b = self.assess_result_quality(result_b)

        success_a = result_a.get("success", False)
        success_b = result_b.get("success", False)

        # Resolution logic
        chosen_source = "unknown"
        reason = "default"

        if success_a and not success_b:
            chosen_result = result_a
            chosen_source = "source_a"
            reason = "source_a_successful"
        elif success_b and not success_a:
            chosen_result = result_b
            chosen_source = "source_b"
            reason = "source_b_successful"
        elif time_diff < 300:  # Within 5 minutes - use quality
            if quality_a > quality_b:
                chosen_result = result_a
                chosen_source = "source_a"
                reason = "higher_quality"
            elif quality_b > quality_a:
                chosen_result = result_b
                chosen_source = "source_b"
                reason = "higher_quality"
            elif dt_a and dt_b and dt_a < dt_b:
                chosen_result = result_a
                chosen_source = "source_a"
                reason = "chronologically_first"
            else:
                chosen_result = result_b
                chosen_source = "source_b"
                reason = "chronologically_first_or_default"
        else:
            # Different times - use earliest
            if dt_a and dt_b:
                if dt_a < dt_b:
                    chosen_result = result_a
                    chosen_source = "source_a"
                    reason = "chronologically_first"
                else:
                    chosen_result = result_b
                    chosen_source = "source_b"
                    reason = "chronologically_first"
            else:
                # Fallback to quality
                if quality_a >= quality_b:
                    chosen_result = result_a
                    chosen_source = "source_a"
                    reason = "fallback_quality"
                else:
                    chosen_result = result_b
                    chosen_source = "source_b"
                    reason = "fallback_quality"

        resolution = ConflictResolution(
            fingerprint=fingerprint,
            source_a_timestamp=timestamp_a,
            source_b_timestamp=timestamp_b,
            chosen_source=chosen_source,
            resolution_reason=reason,
            source_a_success=success_a,
            source_b_success=success_b,
        )

        return chosen_result, resolution

    def build_fingerprint_map(
        self, results_data: Dict[str, Any], source_name: str
    ) -> Dict[ResultFingerprint, Dict[str, Any]]:
        """Build a map of fingerprints to results for efficient lookup."""
        fingerprint_map = {}

        for batch_key, batch_data in results_data.items():
            for request_type, results_list in batch_data.items():
                for result in results_list:
                    # Ensure result has request_type if missing
                    if "request_type" not in result:
                        result["request_type"] = request_type

                    fingerprint = self.create_fingerprint(result)

                    if fingerprint in fingerprint_map:
                        self.logger.warning(
                            f"Duplicate fingerprint found within same source {source_name}: "
                            f"{fingerprint} in batch {batch_key}"
                        )

                    fingerprint_map[fingerprint] = result.copy()

        return fingerprint_map

    def merge_results_data(
        self,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        progress: Progress,
        task_id: TaskID,
    ) -> Tuple[Dict[str, Any], MergeStatistics]:
        """Merge two results datasets with intelligent conflict resolution."""
        stats = MergeStatistics()

        # Build fingerprint maps
        progress.update(task_id, description="Building fingerprint maps...")
        fingerprints_a = self.build_fingerprint_map(results_a, "source_a")
        fingerprints_b = self.build_fingerprint_map(results_b, "source_b")

        stats.total_results_a = len(fingerprints_a)
        stats.total_results_b = len(fingerprints_b)

        # Find conflicts and unique items
        progress.update(task_id, description="Analyzing conflicts...")
        conflicts = set(fingerprints_a.keys()) & set(fingerprints_b.keys())
        unique_to_a = set(fingerprints_a.keys()) - conflicts
        unique_to_b = set(fingerprints_b.keys()) - conflicts

        stats.conflicts_found = len(conflicts)
        stats.unique_to_a = len(unique_to_a)
        stats.unique_to_b = len(unique_to_b)

        # Start building merged results
        merged_fingerprints = {}

        # Add unique results from both sources
        progress.update(task_id, description="Adding unique results...")
        for fingerprint in unique_to_a:
            merged_fingerprints[fingerprint] = fingerprints_a[fingerprint]

        for fingerprint in unique_to_b:
            merged_fingerprints[fingerprint] = fingerprints_b[fingerprint]

        # Resolve conflicts
        progress.update(task_id, description="Resolving conflicts...")
        for fingerprint in conflicts:
            result_a = fingerprints_a[fingerprint]
            result_b = fingerprints_b[fingerprint]

            chosen_result, resolution = self.resolve_conflict(
                result_a, result_b, fingerprint
            )
            merged_fingerprints[fingerprint] = chosen_result
            stats.conflicts.append(resolution)
            stats.conflicts_resolved += 1

        # Rebuild hierarchical structure
        progress.update(task_id, description="Rebuilding result structure...")
        merged_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for fingerprint, result in merged_fingerprints.items():
            # Reconstruct batch key
            batch_key = f"{result['attack_type']}_{result['prompt_type']}_{result['injection_locus']}"
            request_type = result["request_type"]

            merged_results[batch_key][request_type].append(result)

        stats.final_result_count = len(merged_fingerprints)

        # Convert defaultdict to regular dict for JSON serialization
        final_results = {
            batch_key: dict(batch_data)
            for batch_key, batch_data in merged_results.items()
        }

        return final_results, stats

    def merge_progress_data(
        self,
        progress_a: Dict[str, Any],
        progress_b: Dict[str, Any],
        merged_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge automation progress data from two sources."""
        merged_progress = {}

        # Merge timestamps - use earliest session start and latest update
        session_starts = []
        if progress_a.get("session_start"):
            session_starts.append(progress_a["session_start"])
        if progress_b.get("session_start"):
            session_starts.append(progress_b["session_start"])

        if session_starts:
            merged_progress["session_start"] = min(session_starts)

        last_updates = []
        if progress_a.get("last_updated"):
            last_updates.append(progress_a["last_updated"])
        if progress_b.get("last_updated"):
            last_updates.append(progress_b["last_updated"])

        if last_updates:
            merged_progress["last_updated"] = max(last_updates)
        else:
            merged_progress["last_updated"] = datetime.now().isoformat()

        # Recalculate completed batches from merged results
        merged_progress["completed_batches"] = list(merged_results.keys())

        # Merge completed PDFs
        completed_pdfs_a = progress_a.get("completed_pdfs", {})
        completed_pdfs_b = progress_b.get("completed_pdfs", {})
        merged_completed_pdfs = {}

        # Merge PDF completion data
        all_batch_keys = set(completed_pdfs_a.keys()) | set(completed_pdfs_b.keys())
        for batch_key in all_batch_keys:
            pdfs_a = set(completed_pdfs_a.get(batch_key, []))
            pdfs_b = set(completed_pdfs_b.get(batch_key, []))
            merged_completed_pdfs[batch_key] = list(pdfs_a | pdfs_b)

        merged_progress["completed_pdfs"] = merged_completed_pdfs

        # Merge failed PDFs
        failed_a = set(progress_a.get("failed_pdfs", []))
        failed_b = set(progress_b.get("failed_pdfs", []))
        merged_progress["failed_pdfs"] = list(failed_a | failed_b)

        # Recalculate counters based on merged data
        total_pdfs = sum(len(pdfs) for pdfs in merged_completed_pdfs.values())
        merged_progress["total_pdfs_processed"] = total_pdfs

        # Calculate total requests from merged results
        total_requests = 0
        for batch_data in merged_results.values():
            for request_results in batch_data.values():
                total_requests += len(request_results)
        merged_progress["total_requests_sent"] = total_requests

        return merged_progress


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            # Check for Git merge conflict markers
            if any(marker in content for marker in ["<<<<<<<", "=======", ">>>>>>>"]):
                console.print(
                    f"[yellow]Warning: Git merge conflict markers found in {file_path}[/yellow]"
                )
                console.print(
                    "[yellow]The file appears to have unresolved merge conflicts. Skipping this file.[/yellow]"
                )
                return {}

            return json.loads(content)
    except FileNotFoundError:
        console.print(
            f"[yellow]Warning: File not found: {file_path} (will skip)[/yellow]"
        )
        return {}
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in {file_path}: {e}[/red]")
        raise


def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file with pretty formatting."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def backup_files(source_a: str, source_b: str, backup_dir: str) -> None:
    """Create backups of original files."""
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup source files
    if os.path.exists(source_a):
        backup_path = os.path.join(
            backup_dir, f"source_a_{timestamp}_{os.path.basename(source_a)}"
        )
        shutil.copy2(source_a, backup_path)
        console.print(f"[green]Backed up source A to: {backup_path}[/green]")

    if os.path.exists(source_b):
        backup_path = os.path.join(
            backup_dir, f"source_b_{timestamp}_{os.path.basename(source_b)}"
        )
        shutil.copy2(source_b, backup_path)
        console.print(f"[green]Backed up source B to: {backup_path}[/green]")


def display_merge_report(stats: MergeStatistics, console: Console) -> None:
    """Display a beautiful merge report using Rich."""

    # Create main statistics table
    stats_table = Table(
        title="Merge Statistics", show_header=True, header_style="bold magenta"
    )
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Results in Source A", str(stats.total_results_a))
    stats_table.add_row("Results in Source B", str(stats.total_results_b))
    stats_table.add_row("Unique to Source A", str(stats.unique_to_a))
    stats_table.add_row("Unique to Source B", str(stats.unique_to_b))
    stats_table.add_row("Conflicts Found", str(stats.conflicts_found))
    stats_table.add_row("Conflicts Resolved", str(stats.conflicts_resolved))
    stats_table.add_row("Final Result Count", str(stats.final_result_count))
    stats_table.add_row("Merge Duration", f"{stats.merge_duration:.2f}s")

    console.print(stats_table)
    console.print()

    # Conflict resolution details
    if stats.conflicts:
        console.print(Panel.fit("Conflict Resolution Details", style="bold yellow"))

        conflict_table = Table(show_header=True, header_style="bold blue")
        conflict_table.add_column("PDF File", style="cyan", max_width=20)
        conflict_table.add_column("Attack Type", style="yellow", max_width=15)
        conflict_table.add_column("Chosen Source", style="green")
        conflict_table.add_column("Reason", style="magenta")
        conflict_table.add_column("Success A/B", style="white")

        for conflict in stats.conflicts[:10]:  # Show first 10 conflicts
            success_indicator = (
                f"{conflict.source_a_success}/{conflict.source_b_success}"
            )
            conflict_table.add_row(
                conflict.fingerprint.pdf_file[:18] + "..."
                if len(conflict.fingerprint.pdf_file) > 18
                else conflict.fingerprint.pdf_file,
                conflict.fingerprint.attack_type.replace("_attack", ""),
                conflict.chosen_source.replace("source_", "").upper(),
                conflict.resolution_reason.replace("_", " ").title(),
                success_indicator,
            )

        if len(stats.conflicts) > 10:
            conflict_table.add_row(
                "...", "...", "...", f"+ {len(stats.conflicts) - 10} more", "..."
            )

        console.print(conflict_table)


def main():
    """Main function with beautiful CLI interface."""
    parser = argparse.ArgumentParser(
        description="Intelligent Automation Results Merger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge results from Downloads directory
  %(prog)s --source-a ~/Downloads/all_results_chatgpt.json --source-b results/all_results_chatgpt.json

  # Dry run to see what would be merged
  %(prog)s --source-a file1.json --source-b file2.json --dry-run

  # Merge with custom output location
  %(prog)s --source-a file1.json --source-b file2.json --output merged_results.json
        """,
    )

    parser.add_argument("--source-a", required=True, help="First results file to merge")
    parser.add_argument(
        "--source-b", required=True, help="Second results file to merge"
    )
    parser.add_argument(
        "--progress-a",
        help="First progress file to merge (auto-detected if not provided)",
    )
    parser.add_argument(
        "--progress-b",
        help="Second progress file to merge (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output", help="Output file path (default: merged_results.json)"
    )
    parser.add_argument(
        "--output-progress",
        help="Output progress file path (default: merged_progress.json)",
    )
    parser.add_argument(
        "--backup-dir",
        default="./backups",
        help="Backup directory (default: ./backups)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without making changes",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backups"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    # Beautiful header
    console.print(
        Panel.fit(
            Text("🔀 Intelligent Automation Results Merger", style="bold cyan"),
            subtitle="Advanced conflict resolution with temporal analysis",
            style="blue",
        )
    )

    start_time = datetime.now()

    try:
        # Auto-detect progress files if not provided
        if not args.progress_a:
            args.progress_a = args.source_a.replace(
                "all_results_", "automation_progress_"
            )
        if not args.progress_b:
            args.progress_b = args.source_b.replace(
                "all_results_", "automation_progress_"
            )

        # Set default output paths
        if not args.output:
            args.output = "merged_all_results.json"
        if not args.output_progress:
            args.output_progress = "merged_automation_progress.json"

        # Validate input files
        for file_path in [args.source_a, args.source_b]:
            if not os.path.exists(file_path):
                console.print(f"[red]Error: File not found: {file_path}[/red]")
                sys.exit(1)

        # Create backups unless disabled
        if not args.no_backup and not args.dry_run:
            backup_files(args.source_a, args.source_b, args.backup_dir)

        # Load data with progress indication
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            load_task = progress.add_task("Loading files...", total=4)

            progress.update(load_task, description="Loading results A...")
            results_a = load_json_file(args.source_a)
            progress.advance(load_task)

            progress.update(load_task, description="Loading results B...")
            results_b = load_json_file(args.source_b)
            progress.advance(load_task)

            progress.update(load_task, description="Loading progress A...")
            progress_a = (
                load_json_file(args.progress_a)
                if os.path.exists(args.progress_a)
                else {}
            )
            progress.advance(load_task)

            progress.update(load_task, description="Loading progress B...")
            progress_b = (
                load_json_file(args.progress_b)
                if os.path.exists(args.progress_b)
                else {}
            )
            progress.advance(load_task)

            # Merge results
            merger = AutomationResultsMerger(console)
            merge_task = progress.add_task("Merging results...", total=100)

            merged_results, stats = merger.merge_results_data(
                results_a, results_b, progress, merge_task
            )

            # Merge progress data
            progress.update(merge_task, description="Merging progress data...")
            merged_progress = merger.merge_progress_data(
                progress_a, progress_b, merged_results
            )

            progress.update(merge_task, advance=100)

        # Calculate merge duration
        stats.merge_duration = (datetime.now() - start_time).total_seconds()

        # Display results
        console.print()
        display_merge_report(stats, console)

        if args.dry_run:
            console.print("\n[yellow]🔍 DRY RUN MODE - No files were modified[/yellow]")
            console.print(f"[cyan]Would save merged results to: {args.output}[/cyan]")
            console.print(
                f"[cyan]Would save merged progress to: {args.output_progress}[/cyan]"
            )
        else:
            # Save merged files
            console.print(
                f"\n[green]💾 Saving merged results to: {args.output}[/green]"
            )
            save_json_file(merged_results, args.output)

            console.print(
                f"[green]💾 Saving merged progress to: {args.output_progress}[/green]"
            )
            save_json_file(merged_progress, args.output_progress)

            console.print("\n[bold green]✅ Merge completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Merge interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]❌ Error during merge: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
