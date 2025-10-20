import os
import sys
from datetime import datetime

import pandas as pd

import glob
import json
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm
import re
import numpy as np
from itertools import combinations
import argparse


def get_search_paths(base_dirs, model):
    search_paths = []
    safe_model_folder = model.replace(os.sep, '_')
    
    for base_dir in base_dirs:
        scoped_path = os.path.join(base_dir, safe_model_folder)
        if os.path.isdir(scoped_path):
            search_paths.append(scoped_path)
        else:
            search_paths.append(base_dir)
            
    return list(set(search_paths))


def normalize_task_key(task_key: str) -> str:
    """Normalize task keys so equivalent tasks map to a consistent canonical form."""
    key = task_key

    # remove common suffixes for backwards compatability
    for suffix in ("_custom", "_reasoning_letter", "_reasoning_custom_prompt_letter"):
        key = key.replace(suffix, "")

    return key

def find_model_specific_paths(all_paths, model):
    variations = {
        model,
        model.replace('_', os.sep),
        model.replace(os.sep, '_')
    }
    
    model_keys_to_check = set(variations)
    for var in variations:
        if os.sep in var:
            model_keys_to_check.add(os.path.basename(var))
    
    final_keys = list(model_keys_to_check)

    model_specific_paths = [
        p for p in all_paths 
        if any(os.path.join('', key, '') in p for key in final_keys)
    ]
    return model_specific_paths

def extract_task_key_from_filename(filename: str) -> str:
    return filename[len("details_") : filename.rfind("_")]

def extract_timestamp_from_path(path: str) -> str:
    basename = os.path.basename(path)
    if "results_" in basename:
        return basename[len("results_"):-len(".json")]
    if "details_" in basename:
        try:
            return basename.rsplit('_', 1)[-1].replace('.parquet', '')
        except IndexError:
            return None
    return None

def build_timestamp_to_seed_map(results_dirs, model):
    timestamp_to_seed = {}
    search_paths = get_search_paths(results_dirs, model)
    print(f"INFO: Searching for results files for model '{model}' in optimized paths: {search_paths}")
    
    all_json_paths = []
    for search_path in search_paths:
        all_json_paths.extend(glob.glob(os.path.join(search_path, '**', 'results_*.json'), recursive=True))
    
    model_specific_paths = find_model_specific_paths(all_json_paths, model)
    print(f"  [DEBUG] Found {len(model_specific_paths)} JSON files matching model '{model}'.")

    for results_path in model_specific_paths:
        timestamp = extract_timestamp_from_path(results_path)
        if timestamp:
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                seed = data.get("config_general", {}).get("generation_parameters", {}).get("seed")
                if seed is not None:
                    timestamp_to_seed[timestamp] = seed
                else:
                    print(f"WARNING: Seed not found in JSON file: {results_path}")
            except json.JSONDecodeError:
                print(f"WARNING: Could not decode JSON from file: {results_path}")
            except Exception as e:
                print(f"WARNING: An unexpected error occurred while processing {results_path}: {e}")
    return timestamp_to_seed

def build_parquet_index_by_seed(details_dirs, model, timestamp_to_seed, debug=False):
    temp_index = defaultdict(list)
    search_paths = get_search_paths(details_dirs, model)
    print(f"INFO: Searching for details parquet files for model '{model}' in optimized paths: {search_paths}")
    
    all_parquet_paths = []
    for search_path in search_paths:
        all_parquet_paths.extend(glob.glob(os.path.join(search_path, '**', 'details_*.parquet'), recursive=True))
    
    model_specific_paths = find_model_specific_paths(all_parquet_paths, model)
    print(f"  [DEBUG] Found {len(model_specific_paths)} Parquet files matching model '{model}'.")

    for parquet_path in model_specific_paths:
        task_key = extract_task_key_from_filename(os.path.basename(parquet_path))
        task_key = normalize_task_key(task_key)
        timestamp = extract_timestamp_from_path(parquet_path)
        
        if task_key and timestamp:
            if timestamp in timestamp_to_seed:
                seed = timestamp_to_seed[timestamp]
                temp_index[(task_key, seed)].append((timestamp, parquet_path))
            else:
                print(f"WARNING: Could not map timestamp for {parquet_path} (corresponding JSON may be missing or failed to parse)")
    
    # Keep only the latest file for each (task, seed) pair
    index = {}
    duplicates_found = 0
    for (task_key, seed), files_with_timestamps in temp_index.items():
        if len(files_with_timestamps) > 1:
            duplicates_found += 1

            files_with_timestamps.sort(key=lambda x: x[0], reverse=True)
            latest_timestamp, latest_file = files_with_timestamps[0]
            if debug:
                print(f"  Found {len(files_with_timestamps)} files for {task_key}, seed {seed}. Using latest: {os.path.basename(latest_file)}")
        else:
            latest_timestamp, latest_file = files_with_timestamps[0]
        
        index[(task_key, seed)] = [latest_file]
    
    if duplicates_found > 0:
        print(f"  INFO: Found {duplicates_found} duplicate (task, seed) pairs. Using latest timestamp for each.")
    
    return index

def compute_cross_seed_statistics(task_transitions_by_seed, max_group_size=3):
    available_seeds = list(task_transitions_by_seed.keys())
    stats_by_size = {}
    
    for k in range(2, min(len(available_seeds) + 1, max_group_size + 1)):
        all_combos = list(combinations(available_seeds, k))
        if not all_combos: continue
        combo_counts_list = []
        for seed_combo in all_combos:
            seed_data = [task_transitions_by_seed[s] for s in seed_combo]
            min_len = min(len(data) for data in seed_data)
            transition_counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
            for i in range(min_len):
                transitions = [seed_data[j][i] for j in range(len(seed_combo))]
                if len(set(transitions)) == 1:
                    transition = transitions[0]
                    if transition in transition_counts:
                        transition_counts[transition] += 1
            combo_counts_list.append(transition_counts)
        
        mean_counts, var_counts = {}, {}
        for trans_type in ["0->0", "0->1", "1->0", "1->1"]:
            counts = [c[trans_type] for c in combo_counts_list]
            mean_counts[trans_type] = np.mean(counts)
            var_counts[trans_type] = np.var(counts)
        stats_by_size[k] = {'mean_counts': mean_counts, 'var_counts': var_counts, 'num_combos': len(all_combos)}
    
    all_transitions = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
    for seed, transitions in task_transitions_by_seed.items():
        for trans in transitions:
            if trans in all_transitions: all_transitions[trans] += 1
    return {'by_size': stats_by_size, 'total_counts': all_transitions}

def compare_models(details_dirs, results_dirs, before_model, after_model, output, tasks=None, debug=False):
    print("INFO: Building timestamp maps and data indexes...")
    before_timestamp_to_seed = build_timestamp_to_seed_map(results_dirs, before_model)
    after_timestamp_to_seed = build_timestamp_to_seed_map(results_dirs, after_model)
    
    before_parquet_index = build_parquet_index_by_seed(details_dirs, before_model, before_timestamp_to_seed, debug)
    after_parquet_index = build_parquet_index_by_seed(details_dirs, after_model, after_timestamp_to_seed, debug)
    
    print(f"\nINFO: Found {len(before_parquet_index)} unique task-seed pairs for model {before_model}")
    print(f"INFO: Found {len(after_parquet_index)} unique task-seed pairs for model {after_model}")

    all_task_seed_pairs = set(before_parquet_index.keys()) & set(after_parquet_index.keys())
    print(f"INFO: Found {len(all_task_seed_pairs)} common (task, seed) pairs to compare.")
    
    rows, task_transitions = [], defaultdict(lambda: defaultdict(list))
    mismatches = []
    
    for (task_key, seed) in tqdm(sorted(all_task_seed_pairs), desc="Comparing task+seed groups"):
        task_key = normalize_task_key(task_key)
        if tasks and task_key not in tasks: continue
        
        before_parquets = before_parquet_index.get((task_key, seed), [])
        after_parquets = after_parquet_index.get((task_key, seed), [])
        if not before_parquets or not after_parquets: continue
        
        try:
            before_df = pq.read_table(before_parquets[0]).to_pandas() if len(before_parquets) == 1 else pd.concat([pq.read_table(p).to_pandas() for p in before_parquets], ignore_index=True)
            after_df = pq.read_table(after_parquets[0]).to_pandas() if len(after_parquets) == 1 else pd.concat([pq.read_table(p).to_pandas() for p in after_parquets], ignore_index=True)
                    
        except Exception as e:
            print(f"WARNING: Failed to load parquet for {task_key}, seed {seed}: {e}")
            continue
            
        if len(before_df) != len(after_df):
            mismatch_info = {
                'task': task_key,
                'seed': seed,
                'before_count': len(before_df),
                'after_count': len(after_df),
                'before_files': len(before_parquets),
                'after_files': len(after_parquets)
            }
            mismatches.append(mismatch_info)
            
            if debug:
                print(f"\nWARNING: Mismatched example count for {task_key}, seed {seed}")
                print(f"  {before_model}: {len(before_df)} examples")
                print(f"  {after_model}: {len(after_df)} examples")
                print(f"  Files used:")
                print(f"    {before_model}: {os.path.basename(before_parquets[0])}")
                print(f"    {after_model}: {os.path.basename(after_parquets[0])}")
            continue

        if 'full_prompt' in before_df.columns and 'full_prompt' in after_df.columns:
            before_df = before_df.sort_values("full_prompt").reset_index(drop=True)
            after_df = after_df.sort_values("full_prompt").reset_index(drop=True)
        
        merged = pd.concat([before_df["metrics"].rename("metrics_before"), after_df["metrics"].rename("metrics_after")], axis=1)
        
        transitions = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
        example_transitions = []
        for _, row_data in merged.iterrows():
            b = row_data["metrics_before"].get("extractive_match") if isinstance(row_data["metrics_before"], dict) else None
            a = row_data["metrics_after"].get("extractive_match") if isinstance(row_data["metrics_after"], dict) else None
            if b is None or a is None:
                example_transitions.append(None)
                continue
            key = f"{int(round(b))}->{int(round(a))}"
            if key in transitions:
                transitions[key] += 1
                example_transitions.append(key)
        
        task_transitions[task_key][seed] = example_transitions
        n = sum(transitions.values())
        if n == 0: continue
        
        rows.append({
            "task": task_key, "seed": seed, "model_before": before_model, "model_after": after_model,
            "n": n, "acc_before": (transitions["1->0"] + transitions["1->1"]) / n,
            "acc_after": (transitions["0->1"] + transitions["1->1"]) / n, **transitions
        })

    if mismatches:
        print(f"\n\nSUMMARY: Found {len(mismatches)} mismatched task-seed pairs:")
        mismatch_df = pd.DataFrame(mismatches)
        print(mismatch_df.to_string(index=False))
        
        mismatch_file = output.replace('.csv', '_mismatches.csv').replace('.txt', '_mismatches.csv')
        mismatch_df.to_csv(mismatch_file, index=False)
        print(f"\nMismatch details saved to: {mismatch_file}")

    if not rows:
        print("WARNING: No pairs compared; no CSV written.")
        return
    
    print("INFO: Computing cross-seed statistics...")
    stats_rows = []
    max_k_to_compute = 3
    for task_key, seeds_data in task_transitions.items():
        filtered_transitions = {s: [t for t in tl if t] for s, tl in seeds_data.items() if any(tl)}
        if len(filtered_transitions) >= 2:
            stats = compute_cross_seed_statistics(filtered_transitions, max_group_size=max_k_to_compute)
            if stats:
                stat_row = {'task': normalize_task_key(task_key)}
                for trans_type in ["0->0", "0->1", "1->0", "1->1"]:
                    col_key = trans_type.replace("->", "_")
                    stat_row[f'total_{col_key}'] = stats['total_counts'].get(trans_type)
                for k in range(2, max_k_to_compute + 1):
                    stats_for_k = stats['by_size'].get(k)
                    stat_row[f'num_combos_{k}_seeds'] = stats_for_k['num_combos'] if stats_for_k else None
                    for trans_type in ["0->0", "0->1", "1->0", "1->1"]:
                        col_key = trans_type.replace("->", "_")
                        stat_row[f'mean_count_{k}_seeds_{col_key}'] = stats_for_k['mean_counts'][trans_type] if stats_for_k else None
                        stat_row[f'var_count_{k}_seeds_{col_key}'] = stats_for_k['var_counts'][trans_type] if stats_for_k else None
                stats_rows.append(stat_row)

    df = pd.DataFrame(rows)
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        df = pd.merge(df, stats_df, on='task', how='left')
    
    df.to_csv(output, index=False)
    print(f"INFO: Wrote {len(df)} rows to {output}")


if __name__ == "__main__":
   os.makedirs('results/comparison_results_per_model', exist_ok=True)

   before_models = [
       "Qwen_Qwen2.5-32B-Instruct",
       "Qwen_Qwen2.5-7B-Instruct",
       "Qwen_QwQ-32B",
       "Qwen_Qwen3-8B-Base",
       "Qwen_Qwen3-8B-Base",
       "Qwen_Qwen2.5-32B",
       "Qwen_Qwen2.5-32B",
       "Qwen_Qwen2.5-32B",
       "Qwen_Qwen2.5-7B",
       "Qwen_Qwen2.5-Coder-32B",
       "Qwen_Qwen2.5-32B",
       "Qwen_Qwen2.5-Math-7B",
       "Qwen_Qwen2.5-7B",
       "meta-llama_Llama-3.1-8B",
       "nvidia_Llama-3.1-Minitron-4B-Width-Base",
       "meta-llama_Llama-3.1-8B",
       "Qwen_Qwen2.5-3B",
       "Qwen_Qwen2.5-7B",
       "Qwen_Qwen2.5-Coder-7B",
       "meta-llama_Llama-3.1-8B",
       "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
       "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
       "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B",
       "meta-llama_Llama-3.1-8B-Instruct",
       "meta-llama_Llama-3.1-8B-Instruct",
       "Qwen_Qwen2.5-14B-Instruct",
       "Qwen_Qwen2.5-7B-Instruct",
       "Qwen_Qwen2.5-Coder-3B",
       "meta-llama_Llama-3.1-8B",
       "meta-llama_Llama-3.1-8B",
       "meta-llama_Llama-3.1-8B",
       "Qwen_Qwen2.5-Math-1.5B",
       "Qwen_Qwen2.5-Math-7B",
       "google_gemma-3-12b-pt",
       "google_gemma-3-1b-pt",
       "google_gemma-3-4b-pt",
       "meta-llama_Llama-3.1-8B",
       "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B",
       "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
       "nvidia_Llama-3.1-Minitron-4B-Width-Base",
       "meta-llama_Llama-3.1-8B-Instruct",
       "meta-llama_Llama-3.1-8B",
       "Qwen_Qwen2.5-7B-Instruct",
       "Qwen_Qwen2.5-1.5B",
       "Qwen_Qwen2.5-7B",
       "Qwen_Qwen2.5-Coder-3B",
       "Qwen_Qwen2.5-Coder-7B",
       "Qwen_Qwen2.5-Math-7B",
       "Qwen_Qwen2.5-7B-Instruct",
       "Qwen_Qwen2.5-14B-Instruct",
       "Qwen_Qwen2.5-32B-Instruct",
       "Qwen_Qwen2.5-7B-Instruct",
       "Qwen_Qwen2.5-14B-Instruct",
       "Qwen_Qwen2.5-32B-Instruct",
       "Qwen_Qwen2.5-32B-Instruct",
       "Qwen_Qwen2.5-Coder-14B",
       "Qwen_Qwen2.5-14B",
       "Qwen_Qwen2.5-3B",
       "Qwen_Qwen2.5-14B",
       "Qwen_Qwen2.5-32B-Instruct",
   ]

   after_models = [
       "open-thoughts_OpenThinker2-32B",
       "open-thoughts_OpenThinker3-7B",
       "PrimeIntellect_INTELLECT-2",
       "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B",
       "Qwen_Qwen3-8B",
       "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
       "Qwen_QwQ-32B",
       "Qwen_Qwen2.5-32B-Instruct",
       "Qwen_Qwen2.5-7B-Instruct",
       "Qwen_Qwen2.5-Coder-32B-Instruct",
       "Qwen_Qwen2.5-Coder-32B",
       "Qwen_Qwen2.5-Math-7B-Instruct",
       "Qwen_Qwen2.5-Math-7B",
       "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
       "nvidia_Llama-3.1-Nemotron-Nano-4B-v1.1",
       "nvidia_OpenMath2-Llama3.1-8B",
       "Qwen_Qwen2.5-Coder-3B",
       "Qwen_Qwen2.5-Coder-7B",
       "Qwen_Qwen2.5-Coder-7B-Instruct",
       "NousResearch_Hermes-3-Llama-3.1-8B",
       "Skywork_Skywork-OR1-7B",
       "nvidia_AceReason-Nemotron-7B",
       "nvidia_AceReason-Nemotron-14B",
       "nvidia_llama-3.1-nemoguard-8b-content-safety",
       "nvidia_Llama-3.1-Nemotron-Nano-8B-v1",
       "nvidia_OpenCodeReasoning-Nemotron-1.1-14B",
       "nvidia_OpenCodeReasoning-Nemotron-1.1-7B",
       "Qwen_Qwen2.5-Coder-3B-Instruct",
       "meta-llama_Llama-3.1-8B-Instruct",
       "meta-llama_Llama-3.1-8B-Instruct",
       "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
       "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B",
       "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
       "google_gemma-3-12b-it",
       "google_gemma-3-1b-it",
       "google_gemma-3-4b-it",
       "meta-llama_Llama-3.1-8B-Instruct",
       "nvidia_AceReason-Nemotron-14B",
       "nvidia_AceReason-Nemotron-7B",
       "nvidia_Llama-3.1-Nemotron-Nano-4B-v1.1",
       "nvidia_Llama-3.1-Nemotron-Nano-8B-v1",
       "nvidia_OpenMath2-Llama3.1-8B",
       "open-thoughts_OpenThinker-7B",
       "Qwen_Qwen2.5-1.5B-Instruct",
       "Qwen_Qwen2.5-7B-Instruct",
       "Qwen_Qwen2.5-Coder-3B-Instruct",
       "Qwen_Qwen2.5-Coder-7B-Instruct",
       "Qwen_Qwen2.5-Math-7B-Instruct",
       "simplescaling_s1.1-7B",
       "simplescaling_s1.1-14B",
       "simplescaling_s1.1-32B",
       "simplescaling_s1.1-7B",
       "simplescaling_s1.1-14B",
       "simplescaling_s1.1-32B",
       "GAIR_LIMO",
       "Qwen_Qwen2.5-Coder-14B-Instruct",
       "Qwen_Qwen2.5-Coder-14B",
       "Qwen_Qwen2.5-3B-Instruct",
       "Qwen_Qwen2.5-14B-Instruct",
       "GAIR_LIMO-v2",
   ]

   for before_model, after_model in zip(before_models, after_models):
       # Create sanitized filenames (replace / with _)
       before_safe = before_model.replace('/', '_')
       after_safe = after_model.replace('/', '_')
       output_file = f"results/comparison_results_per_model/{before_safe}_vs_{after_safe}.csv"

       print("=" * 43)
       print(f"Comparing: {before_model} -> {after_model}")
       print(f"Output: {output_file}")
       print(f"Start time: {datetime.now()}")
       print("=" * 43)

       # Run the comparison
       details_dirs = ['results']
       results_dirs = ['results']
       output = output_file
       task_filter = None
       debug = None

       compare_models(details_dirs, results_dirs, before_model, after_model, output, tasks=task_filter, debug=debug)


# Combine all results
csv_files = glob.glob(f"results/comparison_results_per_model/*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)
merged_df.to_csv(f"results/comparison_results_all.csv", index=False)
