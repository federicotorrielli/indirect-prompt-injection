"""
Results processing and analysis for ChatGPT PDF review automation.
"""

import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from config import Config

logger = logging.getLogger(__name__)


class ResultsProcessor:
    """Handles saving and processing of automation results."""

    def __init__(self, config: Config):
        self.config = config
        self.config.setup_directories()
        self.consolidated_file = os.path.join(
            self.config.results_dir, "all_results.json"
        )
        self.consolidated_csv = os.path.join(self.config.results_dir, "all_results.csv")

    def load_existing_results(self) -> Dict[str, Dict]:
        """Load existing consolidated results if they exist."""
        if os.path.exists(self.consolidated_file):
            try:
                with open(self.consolidated_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
        return {}

    def save_single_result(
        self,
        result: Dict,
        attack_type: str,
        prompt_type: str,
        injection_locus: str,
        request_type: str,
    ):
        """Save a single result to the consolidated file."""
        try:
            # Load existing results
            all_results = self.load_existing_results()

            # Create batch key
            batch_key = f"{attack_type}_{prompt_type}_{injection_locus}"

            # Initialize structure if needed
            if batch_key not in all_results:
                all_results[batch_key] = {}
            if request_type not in all_results[batch_key]:
                all_results[batch_key][request_type] = []

            # Add the new result
            all_results[batch_key][request_type].append(result)

            # Save back to file
            with open(self.consolidated_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            # Also update CSV if needed
            if self.config.results_format in ["csv", "both"]:
                self._save_consolidated_as_csv(all_results, self.consolidated_csv)

            logger.debug(f"Saved result for {batch_key}/{request_type}")

        except Exception as e:
            logger.error(f"Failed to save single result: {e}")

    def get_result_counts(self) -> Dict[str, int]:
        """Get counts of results in the consolidated file."""
        try:
            all_results = self.load_existing_results()
            total_batches = len(all_results)
            total_results = 0

            for batch_results in all_results.values():
                for request_results in batch_results.values():
                    total_results += len(request_results)

            return {"total_batches": total_batches, "total_results": total_results}
        except Exception as e:
            logger.error(f"Failed to get result counts: {e}")
            return {"total_batches": 0, "total_results": 0}

    def save_batch_results(
        self,
        results: Dict[str, List[Dict]],
        attack_type: str,
        prompt_type: str,
        injection_locus: str,
        request_type: str,
    ):
        """Save results for a specific batch."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{attack_type}_{prompt_type}_{injection_locus}_{request_type}_{timestamp}"

            # Save as JSON
            if self.config.results_format in ["json", "both"]:
                json_path = os.path.join(self.config.results_dir, f"{filename}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved batch results to {json_path}")

            # Save as CSV
            if self.config.results_format in ["csv", "both"]:
                csv_path = os.path.join(self.config.results_dir, f"{filename}.csv")
                self._save_results_as_csv(results, csv_path)
                logger.info(f"Saved batch results to {csv_path}")

        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")

    def save_consolidated_results(self, all_results: Dict[str, Dict]):
        """Save consolidated results from all batches."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save as JSON
            if self.config.results_format in ["json", "both"]:
                json_path = os.path.join(
                    self.config.results_dir, f"consolidated_results_{timestamp}.json"
                )
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved consolidated results to {json_path}")

            # Save as CSV
            if self.config.results_format in ["csv", "both"]:
                csv_path = os.path.join(
                    self.config.results_dir, f"consolidated_results_{timestamp}.csv"
                )
                self._save_consolidated_as_csv(all_results, csv_path)
                logger.info(f"Saved consolidated results to {csv_path}")

            # Also save as latest
            latest_json = os.path.join(self.config.results_dir, "latest_results.json")
            with open(latest_json, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save consolidated results: {e}")

    def _save_results_as_csv(self, results: Dict[str, List[Dict]], csv_path: str):
        """Save results as CSV file."""
        try:
            all_rows = []
            for request_type, batch_results in results.items():
                for result in batch_results:
                    row = result.copy()
                    row["request_type"] = request_type
                    all_rows.append(row)

            if all_rows:
                fieldnames = all_rows[0].keys()
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_rows)

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def _save_consolidated_as_csv(self, all_results: Dict[str, Dict], csv_path: str):
        """Save consolidated results as CSV."""
        try:
            all_rows = []
            for batch_key, batch_data in all_results.items():
                for request_type, results_list in batch_data.items():
                    for result in results_list:
                        row = result.copy()
                        row["batch_key"] = batch_key
                        all_rows.append(row)

            if all_rows:
                fieldnames = all_rows[0].keys()
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_rows)

        except Exception as e:
            logger.error(f"Failed to save consolidated CSV: {e}")

    def generate_analysis_report(self, all_results: Dict[str, Dict]):
        """Generate analysis report from results."""
        try:
            logger.info("Generating analysis report...")

            analysis = self._analyze_results(all_results)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.config.results_dir, f"analysis_report_{timestamp}.json"
            )

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            # Also save as markdown
            md_path = os.path.join(
                self.config.results_dir, f"analysis_report_{timestamp}.md"
            )
            self._save_analysis_as_markdown(analysis, md_path)

            logger.info(f"Saved analysis report to {report_path}")
            logger.info(f"Saved markdown report to {md_path}")

        except Exception as e:
            logger.error(f"Failed to generate analysis report: {e}")

    def _analyze_results(self, all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze results and generate statistics."""
        analysis: Dict[str, Any] = {
            "summary": {},
            "by_attack_type": defaultdict(dict),
            "by_prompt_type": defaultdict(dict),
            "by_injection_locus": defaultdict(dict),
            "by_request_type": defaultdict(dict),
            "success_rates": {},
            "response_analysis": {},
            "generated_at": datetime.now().isoformat(),
        }

        total_requests = 0
        successful_requests = 0

        attack_type_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "successful": 0}
        )
        prompt_type_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "successful": 0}
        )
        injection_locus_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "successful": 0}
        )
        request_type_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "successful": 0}
        )

        response_lengths = []

        for batch_key, batch_data in all_results.items():
            for request_type, results_list in batch_data.items():
                for result in results_list:
                    total_requests += 1

                    attack_type = result.get("attack_type", "unknown")
                    prompt_type = result.get("prompt_type", "unknown")
                    injection_locus = result.get("injection_locus", "unknown")
                    req_type = result.get("request_type", request_type)

                    # Update counters
                    attack_type_stats[attack_type]["total"] += 1
                    prompt_type_stats[prompt_type]["total"] += 1
                    injection_locus_stats[injection_locus]["total"] += 1
                    request_type_stats[req_type]["total"] += 1

                    if result.get("success", False):
                        successful_requests += 1
                        attack_type_stats[attack_type]["successful"] += 1
                        prompt_type_stats[prompt_type]["successful"] += 1
                        injection_locus_stats[injection_locus]["successful"] += 1
                        request_type_stats[req_type]["successful"] += 1

                        # Analyze response
                        response = result.get("response", "")
                        if response:
                            response_lengths.append(len(response))

        # Calculate summary statistics
        analysis["summary"] = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "overall_success_rate": successful_requests / total_requests
            if total_requests > 0
            else 0,
            "total_batches": len(all_results),
            "avg_response_length": sum(response_lengths) / len(response_lengths)
            if response_lengths
            else 0,
            "min_response_length": min(response_lengths) if response_lengths else 0,
            "max_response_length": max(response_lengths) if response_lengths else 0,
        }

        # Calculate success rates by category
        for category, stats in [
            ("attack_type", attack_type_stats),
            ("prompt_type", prompt_type_stats),
            ("injection_locus", injection_locus_stats),
            ("request_type", request_type_stats),
        ]:
            if "success_rates" not in analysis:
                analysis["success_rates"] = {}
            analysis["success_rates"][category] = {}
            for key, data in stats.items():
                success_rate = (
                    data["successful"] / data["total"] if data["total"] > 0 else 0
                )
                analysis["success_rates"][category][key] = {
                    "total": data["total"],
                    "successful": data["successful"],
                    "success_rate": success_rate,
                }

        return analysis

    def _save_analysis_as_markdown(self, analysis: Dict[str, Any], md_path: str):
        """Save analysis as markdown report."""
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# ChatGPT PDF Review Automation Analysis Report\n\n")
                f.write(f"Generated: {analysis['generated_at']}\n\n")

                # Summary
                f.write("## Summary\n\n")
                summary = analysis["summary"]
                f.write(f"- **Total Requests**: {summary['total_requests']}\n")
                f.write(
                    f"- **Successful Requests**: {summary['successful_requests']}\n"
                )
                f.write(
                    f"- **Overall Success Rate**: {summary['overall_success_rate']:.2%}\n"
                )
                f.write(f"- **Total Batches**: {summary['total_batches']}\n")
                f.write(
                    f"- **Average Response Length**: {summary['avg_response_length']:.0f} characters\n"
                )
                f.write(
                    f"- **Response Length Range**: {summary['min_response_length']} - {summary['max_response_length']} characters\n\n"
                )

                # Success rates by category
                for category, rates in analysis["success_rates"].items():
                    f.write(
                        f"## Success Rates by {category.replace('_', ' ').title()}\n\n"
                    )
                    f.write("| Category | Total | Successful | Success Rate |\n")
                    f.write("|----------|-------|------------|-------------|\n")
                    for key, data in rates.items():
                        f.write(
                            f"| {key} | {data['total']} | {data['successful']} | {data['success_rate']:.2%} |\n"
                        )
                    f.write("\n")

        except Exception as e:
            logger.error(f"Failed to save markdown report: {e}")

    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Load results from file."""
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results from {results_file}: {e}")
            return {}

    def export_responses_for_analysis(
        self, all_results: Dict[str, Dict], output_dir: str
    ):
        """Export responses in format suitable for further analysis."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Group responses by attack type and request type
            responses_by_type = defaultdict(list)

            for batch_key, batch_data in all_results.items():
                for request_type, results_list in batch_data.items():
                    for result in results_list:
                        if result.get("success") and result.get("response"):
                            key = f"{result['attack_type']}_{result['prompt_type']}_{request_type}"
                            responses_by_type[key].append(
                                {
                                    "pdf_file": result["pdf_file"],
                                    "response": result["response"],
                                    "injection_locus": result.get("injection_locus"),
                                    "timestamp": result.get("timestamp"),
                                }
                            )

            # Save each group to separate file
            for group_key, responses in responses_by_type.items():
                output_file = os.path.join(output_dir, f"{group_key}_responses.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(responses, f, indent=2, ensure_ascii=False)
                logger.info(f"Exported {len(responses)} responses to {output_file}")

        except Exception as e:
            logger.error(f"Failed to export responses: {e}")
