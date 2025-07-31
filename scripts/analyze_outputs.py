#!/usr/bin/env python3
"""
Simple script to print basic statistics from all_results_chatgpt.json
"""

import json
import os


def print_basic_stats():
    """Print simple statistics from the results file."""
    results_file = "results/all_results_chatgpt.json"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        return
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return
    
    print("=" * 50)
    print("BASIC STATISTICS - all_results_chatgpt.json")
    print("=" * 50)
    
    # Count total entries and successes
    total_entries = 0
    successful_entries = 0
    attack_types = set()
    request_types = set()
    batch_counts = {}
    
    for batch_key, batch_data in data.items():
        batch_counts[batch_key] = {}
        batch_total = 0
        
        for request_type, entries in batch_data.items():
            if isinstance(entries, list):
                request_types.add(request_type)
                entry_count = len(entries)
                batch_counts[batch_key][request_type] = entry_count
                batch_total += entry_count
                
                for entry in entries:
                    if isinstance(entry, dict):
                        total_entries += 1
                        if entry.get('success'):
                            successful_entries += 1
                        if 'attack_type' in entry:
                            attack_types.add(entry['attack_type'])
        
        batch_counts[batch_key]['_total'] = batch_total
    
    print(f"Total batch keys: {len(data)}")
    print(f"Total entries: {total_entries}")
    print(f"Successful entries: {successful_entries}")
    print(f"Success rate: {(successful_entries/total_entries*100):.1f}%")
    print(f"Failed entries: {total_entries - successful_entries}")
    print()
    
    print("ENTRIES BY BATCH KEY AND REQUEST TYPE:")
    print("-" * 50)
    for batch_key in sorted(batch_counts.keys()):
        batch_data = batch_counts[batch_key]
        total = batch_data.get('_total', 0)
        print(f"{batch_key}: {total} total entries")
        
        for request_type in sorted(request_types):
            if request_type in batch_data:
                count = batch_data[request_type]
                print(f"  └─ {request_type}: {count} entries")
        print()
    
    print(f"Attack types found: {len(attack_types)}")
    for attack_type in sorted(attack_types):
        print(f"  - {attack_type}")
    print()
    print(f"Request types found: {len(request_types)}")
    for request_type in sorted(request_types):
        print(f"  - {request_type}")
    print("=" * 50)


if __name__ == "__main__":
    print_basic_stats()