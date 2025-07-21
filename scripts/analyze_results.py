#!/usr/bin/env python3
"""
Comprehensive Analysis Script for all_results.json

This script provides detailed statistics and counts for the indirect prompt injection
experiment results, including attack success rates, response patterns, and more.
"""

import json
import os
from collections import defaultdict, Counter
from datetime import datetime
import re
from pathlib import Path


class ResultsAnalyzer:
    """Analyzer for indirect prompt injection results."""
    
    def __init__(self, results_file: str, model: str = "chatgpt"):
        self.model = model
        self.results_file = results_file
        self.data = self.load_data()
        self.all_pdf_files = self.get_all_pdf_files()
        
    def get_all_pdf_files(self) -> set:
        """Get all PDF files from the data/redacted_pdfs directory."""
        pdf_dir = Path("data/redacted_pdfs")
        if not pdf_dir.exists():
            print(f"Warning: {pdf_dir} directory not found!")
            return set()
        
        pdf_files = set()
        for pdf_file in pdf_dir.glob("*.pdf"):
            pdf_files.add(pdf_file.name)
        
        print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        return pdf_files
        
    def load_data(self) -> dict:
        """Load the results JSON file."""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {self.results_file}: {e}")
            return {}
    
    def save_data(self):
        """Save the modified data back to the JSON file."""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            print(f"Updated {self.results_file} with PDF completeness information")
        except Exception as e:
            print(f"Error saving {self.results_file}: {e}")
    
    def get_basic_statistics(self) -> dict:
        """Get basic counts and statistics."""
        stats = {
            'total_batch_keys': len(self.data),
            'total_entries': 0,
            'successful_entries': 0,
            'failed_entries': 0,
            'entries_with_responses': 0,
            'entries_with_errors': 0,
            'pdf_completeness': {
                'complete_experiments': 0,
                'incomplete_experiments': 0,
                'total_expected_pdfs': len(self.all_pdf_files)
            }
        }
        
        for batch_key, batch_data in self.data.items():
            # Track PDFs processed in this experiment
            processed_pdfs = set()
            experiment_entries = 0
            
            for request_type, entries in batch_data.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            stats['total_entries'] += 1
                            experiment_entries += 1
                            
                            # Count success/failure
                            if entry.get('success'):
                                stats['successful_entries'] += 1
                            else:
                                stats['failed_entries'] += 1
                            
                            # Count responses and errors
                            if entry.get('response'):
                                stats['entries_with_responses'] += 1
                            if entry.get('error'):
                                stats['entries_with_errors'] += 1
                            
                            # Track processed PDFs
                            if 'pdf_file' in entry:
                                processed_pdfs.add(entry['pdf_file'])
            
            # Check completeness for this experiment
            if experiment_entries > 0:  # Only check experiments that have entries
                unprocessed_pdfs = self.all_pdf_files - processed_pdfs
                
                if len(unprocessed_pdfs) == 0:
                    stats['pdf_completeness']['complete_experiments'] += 1
                else:
                    stats['pdf_completeness']['incomplete_experiments'] += 1
                    # Store unprocessed PDFs info for this experiment
                    stats['pdf_completeness'][f'{batch_key}_unprocessed'] = list(unprocessed_pdfs)
        
        return stats
    
    def get_attack_type_breakdown(self) -> dict:
        """Analyze results by attack type."""
        attack_stats = defaultdict(lambda: {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'request_types': defaultdict(int),
            'unique_pdfs': set(),
            'injection_loci': defaultdict(int),
            'prompt_types': defaultdict(int)
        })
        
        for batch_key, batch_data in self.data.items():
            for request_type, entries in batch_data.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and 'attack_type' in entry:
                            attack_type = entry['attack_type']
                            attack_stats[attack_type]['total'] += 1
                            attack_stats[attack_type]['request_types'][request_type] += 1
                            
                            if entry.get('success'):
                                attack_stats[attack_type]['successful'] += 1
                            else:
                                attack_stats[attack_type]['failed'] += 1
                            
                            if 'pdf_file' in entry:
                                attack_stats[attack_type]['unique_pdfs'].add(entry['pdf_file'])
                            
                            if 'injection_locus' in entry:
                                attack_stats[attack_type]['injection_loci'][entry['injection_locus']] += 1
                            
                            if 'prompt_type' in entry:
                                attack_stats[attack_type]['prompt_types'][entry['prompt_type']] += 1
        
        # Convert sets to counts
        for attack_type in attack_stats:
            attack_stats[attack_type]['unique_pdfs'] = len(attack_stats[attack_type]['unique_pdfs'])
        
        return dict(attack_stats)
    
    def get_request_type_analysis(self) -> dict:
        """Analyze results by request type."""
        request_stats = defaultdict(lambda: {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'attack_types': defaultdict(int),
            'avg_response_length': 0,
            'response_lengths': []
        })
        
        for batch_key, batch_data in self.data.items():
            for request_type, entries in batch_data.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            request_stats[request_type]['total'] += 1
                            
                            if entry.get('success'):
                                request_stats[request_type]['successful'] += 1
                            else:
                                request_stats[request_type]['failed'] += 1
                            
                            if 'attack_type' in entry:
                                request_stats[request_type]['attack_types'][entry['attack_type']] += 1
                            
                            # Response length analysis
                            if entry.get('response'):
                                length = len(entry['response'])
                                request_stats[request_type]['response_lengths'].append(length)
        
        # Calculate average response lengths
        for req_type in request_stats:
            lengths = request_stats[req_type]['response_lengths']
            if lengths:
                request_stats[req_type]['avg_response_length'] = sum(lengths) / len(lengths)
            del request_stats[req_type]['response_lengths']  # Remove raw data for cleaner output
        
        return dict(request_stats)
    
    def get_steering_attack_analysis(self) -> dict:
        """Specific analysis for steering attacks."""
        steering_stats = defaultdict(lambda: {
            'total_attempts': 0,
            'positive_requests': 0,
            'negative_requests': 0,
            'standard_requests': 0,
            'successful_positive': 0,
            'successful_negative': 0,
            'successful_standard': 0,
            'positive_compliance': 0,
            'negative_compliance': 0
        })
        
        for batch_key, batch_data in self.data.items():
            if 'steering_attack' in batch_key:
                attack_type = None
                if 'pos_steering_attack' in batch_key:
                    attack_type = 'pos_steering_attack'
                elif 'neg_steering_attack' in batch_key:
                    attack_type = 'neg_steering_attack'
                
                if attack_type:
                    for request_type, entries in batch_data.items():
                        if isinstance(entries, list):
                            for entry in entries:
                                if isinstance(entry, dict):
                                    steering_stats[attack_type]['total_attempts'] += 1
                                    
                                    # Count request types
                                    if request_type == 'positive_request':
                                        steering_stats[attack_type]['positive_requests'] += 1
                                        if entry.get('success'):
                                            steering_stats[attack_type]['successful_positive'] += 1
                                    elif request_type == 'negative_request':
                                        steering_stats[attack_type]['negative_requests'] += 1
                                        if entry.get('success'):
                                            steering_stats[attack_type]['successful_negative'] += 1
                                    elif request_type == 'standard_request':
                                        steering_stats[attack_type]['standard_requests'] += 1
                                        if entry.get('success'):
                                            steering_stats[attack_type]['successful_standard'] += 1
                                    
                                    # Analyze compliance (basic heuristic)
                                    response = entry.get('response', '') or ''
                                    if response:
                                        response_lower = response.lower()
                                        if any(word in response_lower for word in ['excellent', 'outstanding', 'recommend accept', 'strong accept']):
                                            steering_stats[attack_type]['positive_compliance'] += 1
                                        elif any(word in response_lower for word in ['poor', 'reject', 'significant flaws', 'major issues']):
                                            steering_stats[attack_type]['negative_compliance'] += 1
        
        return dict(steering_stats)
    
    def get_temporal_analysis(self) -> dict:
        """Analyze results by time patterns."""
        temporal_stats = {
            'dates': defaultdict(int),
            'hours': defaultdict(int),
            'first_timestamp': None,
            'last_timestamp': None,
            'total_duration_hours': 0
        }
        
        timestamps = []
        
        for batch_key, batch_data in self.data.items():
            for request_type, entries in batch_data.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and 'timestamp' in entry:
                            timestamp_str = entry['timestamp']
                            try:
                                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                timestamps.append(dt)
                                
                                date_str = dt.date().isoformat()
                                hour = dt.hour
                                
                                temporal_stats['dates'][date_str] += 1
                                temporal_stats['hours'][hour] += 1
                            except Exception:
                                continue
        
        if timestamps:
            timestamps.sort()
            temporal_stats['first_timestamp'] = timestamps[0].isoformat()
            temporal_stats['last_timestamp'] = timestamps[-1].isoformat()
            duration = timestamps[-1] - timestamps[0]
            temporal_stats['total_duration_hours'] = duration.total_seconds() / 3600
        
        return temporal_stats
    
    def get_pdf_analysis(self) -> dict:
        """Analyze PDF-related statistics."""
        pdf_stats = {
            'unique_pdfs': set(),
            'pdf_attack_combinations': defaultdict(set),
            'most_tested_pdfs': Counter(),
            'pdf_success_rates': defaultdict(lambda: {'total': 0, 'successful': 0})
        }
        
        for batch_key, batch_data in self.data.items():
            for request_type, entries in batch_data.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and 'pdf_file' in entry:
                            pdf_file = entry['pdf_file']
                            attack_type = entry.get('attack_type', 'unknown')
                            
                            pdf_stats['unique_pdfs'].add(pdf_file)
                            pdf_stats['pdf_attack_combinations'][pdf_file].add(attack_type)
                            pdf_stats['most_tested_pdfs'][pdf_file] += 1
                            
                            pdf_stats['pdf_success_rates'][pdf_file]['total'] += 1
                            if entry.get('success'):
                                pdf_stats['pdf_success_rates'][pdf_file]['successful'] += 1
        
        # Convert to counts and calculate success rates
        pdf_stats['unique_pdfs'] = len(pdf_stats['unique_pdfs'])
        pdf_stats['pdf_attack_combinations'] = {
            pdf: len(attacks) for pdf, attacks in pdf_stats['pdf_attack_combinations'].items()
        }
        
        # Calculate success rates
        success_rates = {}
        for pdf, stats in pdf_stats['pdf_success_rates'].items():
            if stats['total'] > 0:
                success_rates[pdf] = stats['successful'] / stats['total']
        pdf_stats['pdf_success_rates'] = success_rates
        
        return pdf_stats
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        print(f"Analyzing all_results_{self.model}.json...")
        
        # Gather all statistics
        basic_stats = self.get_basic_statistics()
        attack_stats = self.get_attack_type_breakdown()
        request_stats = self.get_request_type_analysis()
        steering_analysis = self.get_steering_attack_analysis()
        temporal_analysis = self.get_temporal_analysis()
        pdf_analysis = self.get_pdf_analysis()
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("INDIRECT PROMPT INJECTION EXPERIMENT RESULTS ANALYSIS")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Results file: {self.results_file}")
        report.append("")
        
        # Basic Statistics
        report.append("BASIC STATISTICS")
        report.append("-" * 40)
        report.append(f"Total batch keys: {basic_stats['total_batch_keys']}")
        report.append(f"Total entries: {basic_stats['total_entries']}")
        report.append(f"Successful entries: {basic_stats['successful_entries']} ({basic_stats['successful_entries']/basic_stats['total_entries']*100:.1f}%)")
        report.append(f"Failed entries: {basic_stats['failed_entries']} ({basic_stats['failed_entries']/basic_stats['total_entries']*100:.1f}%)")
        report.append(f"Entries with responses: {basic_stats['entries_with_responses']}")
        report.append(f"Entries with errors: {basic_stats['entries_with_errors']}")
        report.append("")
        
        # PDF Completeness
        report.append("PDF COMPLETENESS")
        report.append("-" * 40)
        pdf_comp = basic_stats['pdf_completeness']
        report.append(f"Total PDFs available: {pdf_comp['total_expected_pdfs']}")
        report.append(f"Complete experiments: {pdf_comp['complete_experiments']}")
        report.append(f"Incomplete experiments: {pdf_comp['incomplete_experiments']}")
        total_experiments = pdf_comp['complete_experiments'] + pdf_comp['incomplete_experiments']
        if total_experiments > 0:
            completeness_rate = (pdf_comp['complete_experiments'] / total_experiments) * 100
            report.append(f"Completeness rate: {completeness_rate:.1f}%")
        
        # Show unprocessed PDFs for incomplete experiments
        if pdf_comp['incomplete_experiments'] > 0:
            report.append("")
            report.append("Unprocessed PDFs by experiment:")
            for key, value in pdf_comp.items():
                if key.endswith('_unprocessed') and isinstance(value, list):
                    experiment_name = key.replace('_unprocessed', '')
                    report.append(f"  {experiment_name}: {len(value)} unprocessed PDFs")
                    if len(value) <= 10:  # Show all if 10 or fewer
                        report.append(f"    {', '.join(value)}")
                    else:  # Show first 10 if more
                        report.append(f"    {', '.join(value[:10])} ... (and {len(value)-10} more)")
        report.append("")
        
        # Attack Type Analysis
        report.append("ATTACK TYPE BREAKDOWN")
        report.append("-" * 40)
        for attack_type, stats in sorted(attack_stats.items()):
            success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
            report.append(f"{attack_type}:")
            report.append(f"  Total attempts: {stats['total']}")
            report.append(f"  Success rate: {success_rate:.1f}% ({stats['successful']}/{stats['total']})")
            report.append(f"  Unique PDFs: {stats['unique_pdfs']}")
            report.append(f"  Request types: {dict(stats['request_types'])}")
            report.append(f"  Injection loci: {dict(stats['injection_loci'])}")
            report.append(f"  Prompt types: {dict(stats['prompt_types'])}")
            report.append("")
        
        # Request Type Analysis
        report.append("REQUEST TYPE ANALYSIS")
        report.append("-" * 40)
        for req_type, stats in sorted(request_stats.items()):
            success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
            report.append(f"{req_type}:")
            report.append(f"  Total: {stats['total']}")
            report.append(f"  Success rate: {success_rate:.1f}%")
            report.append(f"  Avg response length: {stats['avg_response_length']:.0f} chars")
            report.append(f"  Attack types: {dict(stats['attack_types'])}")
            report.append("")
        
        # Steering Attack Analysis
        if steering_analysis:
            report.append("STEERING ATTACK ANALYSIS")
            report.append("-" * 40)
            for attack_type, stats in steering_analysis.items():
                report.append(f"{attack_type}:")
                report.append(f"  Total attempts: {stats['total_attempts']}")
                report.append(f"  Positive requests: {stats['positive_requests']} (success: {stats['successful_positive']})")
                report.append(f"  Negative requests: {stats['negative_requests']} (success: {stats['successful_negative']})")
                report.append(f"  Standard requests: {stats['standard_requests']} (success: {stats['successful_standard']})")
                report.append(f"  Positive compliance detected: {stats['positive_compliance']}")
                report.append(f"  Negative compliance detected: {stats['negative_compliance']}")
                report.append("")
        
        # Temporal Analysis
        report.append("TEMPORAL ANALYSIS")
        report.append("-" * 40)
        if temporal_analysis['first_timestamp']:
            report.append(f"First timestamp: {temporal_analysis['first_timestamp']}")
            report.append(f"Last timestamp: {temporal_analysis['last_timestamp']}")
            report.append(f"Total duration: {temporal_analysis['total_duration_hours']:.1f} hours")
            report.append("")
            report.append("Activity by date:")
            for date, count in sorted(temporal_analysis['dates'].items()):
                report.append(f"  {date}: {count} entries")
            report.append("")
            report.append("Activity by hour:")
            for hour in range(24):
                count = temporal_analysis['hours'][hour]
                if count > 0:
                    report.append(f"  {hour:02d}:00: {count} entries")
        report.append("")
        
        # PDF Analysis
        report.append("PDF ANALYSIS")
        report.append("-" * 40)
        report.append(f"Unique PDFs tested: {pdf_analysis['unique_pdfs']}")
        report.append("")
        report.append("Most tested PDFs:")
        for pdf, count in pdf_analysis['most_tested_pdfs'].most_common(10):
            success_rate = pdf_analysis['pdf_success_rates'].get(pdf, 0) * 100
            report.append(f"  {pdf}: {count} tests (success rate: {success_rate:.1f}%)")
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        overall_success_rate = basic_stats['successful_entries'] / basic_stats['total_entries'] * 100
        report.append(f"Overall success rate: {overall_success_rate:.1f}%")
        report.append(f"Most effective attack: {max(attack_stats.items(), key=lambda x: x[1]['successful']/x[1]['total'])[0] if attack_stats else 'N/A'}")
        report.append(f"Least effective attack: {min(attack_stats.items(), key=lambda x: x[1]['successful']/x[1]['total'])[0] if attack_stats else 'N/A'}")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_detailed_statistics(self, output_file: str):
        """Save detailed statistics to a JSON file."""
        detailed_stats = {
            'generated_at': datetime.now().isoformat(),
            'source_file': self.results_file,
            'basic_statistics': self.get_basic_statistics(),
            'attack_type_breakdown': self.get_attack_type_breakdown(),
            'request_type_analysis': self.get_request_type_analysis(),
            'steering_attack_analysis': self.get_steering_attack_analysis(),
            'temporal_analysis': self.get_temporal_analysis(),
            'pdf_analysis': self.get_pdf_analysis()
        }
        
        # Convert Counter objects to regular dicts for JSON serialization
        def convert_counters(obj):
            if isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, dict):
                return {k: convert_counters(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_counters(item) for item in obj]
            else:
                return obj
        
        detailed_stats = convert_counters(detailed_stats)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed statistics saved to: {output_file}")


def main():
    """Main function to run the analysis."""
    model = "chatgpt"
    results_file = f"results/all_results_{model}.json"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        return
    
    # Create analyzer
    analyzer = ResultsAnalyzer(results_file, model)
    
    # Generate and display summary report
    report = analyzer.generate_summary_report()
    print(report)
    
    # Save detailed statistics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_output = f"results/detailed_analysis_{timestamp}.json"
    analyzer.save_detailed_statistics(detailed_output)
    
    # Save summary report
    report_output = f"results/summary_report_{timestamp}.txt"
    with open(report_output, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Summary report saved to: {report_output}")


if __name__ == "__main__":
    main()
