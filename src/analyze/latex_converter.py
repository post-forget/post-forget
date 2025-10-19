import pandas as pd
import os
import glob
import numpy as np
import re
import argparse
import sys
from typing import Dict, List

MODEL_ORDER: List[str] = [

    "Qwen_Qwen2.5-3B → Qwen_Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5-7B → Qwen_Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5-14B → Qwen_Qwen2.5-14B-Instruct",
    "Qwen_Qwen2.5-32B → Qwen_Qwen2.5-32B-Instruct",

    "Qwen_Qwen2.5-3B → Qwen_Qwen2.5-Coder-3B",
    "Qwen_Qwen2.5-7B → Qwen_Qwen2.5-Coder-7B",
    "Qwen_Qwen2.5-14B → Qwen_Qwen2.5-Coder-14B",
    "Qwen_Qwen2.5-32B → Qwen_Qwen2.5-Coder-32B",

    "Qwen_Qwen2.5-7B → Qwen_Qwen2.5-Math-7B",

    "Qwen_Qwen2.5-Coder-3B → Qwen_Qwen2.5-Coder-3B-Instruct",
    "Qwen_Qwen2.5-Coder-7B → Qwen_Qwen2.5-Coder-7B-Instruct",
    "Qwen_Qwen2.5-Coder-14B → Qwen_Qwen2.5-Coder-14B-Instruct",
    "Qwen_Qwen2.5-Coder-32B → Qwen_Qwen2.5-Coder-32B-Instruct",

    "Qwen_Qwen2.5-Math-7B → Qwen_Qwen2.5-Math-7B-Instruct",

    "Qwen_Qwen2.5-32B → Qwen/QwQ-32B",
    "Qwen/QwQ-32B → PrimeIntellect/INTELLECT-2",

    "meta-llama/Llama-3.1-8B → meta-llama_Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B → nvidia_OpenMath2-Llama3.1-8B",

    "Qwen_Qwen2.5-7B-Instruct → simplescaling_s1.1-7B",
    "Qwen_Qwen2.5-14B-Instruct → simplescaling_s1.1-14B",
    "Qwen_Qwen2.5-32B-Instruct → simplescaling_s1.1-32B",

    "Qwen_Qwen2.5-7B-Instruct → open-thoughts_OpenThinker-7B",
    "Qwen_Qwen2.5-32B-Instruct → open-thoughts/OpenThinker2-32B",
    "Qwen_Qwen2.5-7B-Instruct → open-thoughts/OpenThinker3-7B",

    "Qwen_Qwen2.5-7B-Instruct → nvidia_OpenCodeReasoning-Nemotron-1.1-7B",
    "Qwen_Qwen2.5-14B-Instruct → nvidia_OpenCodeReasoning-Nemotron-1.1-14B",

    "Qwen_Qwen2.5-32B-Instruct → GAIR/LIMO",
    "Qwen_Qwen2.5-32B-Instruct → GAIR_LIMO-v2",

    "Qwen_Qwen2.5-Math-7B → deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-3.1-8B → deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "Qwen_Qwen2.5-32B → deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

MODEL_NAME_MAP: Dict[str, str] = {
    "Qwen_Qwen2.5-3B → Qwen_Qwen2.5-3B-Instruct": "Q2.5 Inst. (3B)",
    "Qwen_Qwen2.5-7B → Qwen_Qwen2.5-7B-Instruct": "Q2.5 Inst. (7B)",
    "Qwen_Qwen2.5-14B → Qwen_Qwen2.5-14B-Instruct": "Q2.5 Inst. (14B)",
    "Qwen_Qwen2.5-32B → Qwen_Qwen2.5-32B-Instruct": "Q2.5 Inst. (32B)",
    "Qwen_Qwen2.5-3B → Qwen_Qwen2.5-Coder-3B": "Q2.5 Coder (3B)",
    "Qwen_Qwen2.5-7B → Qwen_Qwen2.5-Coder-7B": "Q2.5 Coder (7B)",
    "Qwen_Qwen2.5-14B → Qwen_Qwen2.5-Coder-14B": "Q2.5 Coder (14B)",
    "Qwen_Qwen2.5-32B → Qwen_Qwen2.5-Coder-32B": "Q2.5 Coder (32B)",
    "Qwen_Qwen2.5-Coder-3B → Qwen_Qwen2.5-Coder-3B-Instruct": "Q2.5 Coder Inst. (3B)",
    "Qwen_Qwen2.5-Coder-7B → Qwen_Qwen2.5-Coder-7B-Instruct": "Q2.5 Coder Inst. (7B)",
    "Qwen_Qwen2.5-Coder-14B → Qwen_Qwen2.5-Coder-14B-Instruct": "Q2.5 Coder Inst. (14B)",
    "Qwen_Qwen2.5-Coder-32B → Qwen_Qwen2.5-Coder-32B-Instruct": "Q2.5 Coder Inst. (32B)",
    "Qwen_Qwen2.5-7B → Qwen_Qwen2.5-Math-7B": "Q2.5 Math (7B)",
    "Qwen_Qwen2.5-Math-7B → Qwen_Qwen2.5-Math-7B-Instruct": "Q2.5 Math Inst. (7B)",
    "Qwen_Qwen2.5-32B → Qwen/QwQ-32B": "QwQ (32B)",
    "Qwen/QwQ-32B → PrimeIntellect/INTELLECT-2": "INTELLECT-2 (32B)",
    "meta-llama/Llama-3.1-8B → meta-llama_Llama-3.1-8B-Instruct": "Llama 3.1 Inst. (8B)",
    "meta-llama/Llama-3.1-8B → nvidia_OpenMath2-Llama3.1-8B": "Open Math 2 (8B)",
    "meta-llama/Llama-3.1-8B → deepseek-ai_DeepSeek-R1-Distill-Llama-8B": "R1 Distill Llama (8B)",
    "Qwen_Qwen2.5-7B-Instruct → simplescaling_s1.1-7B": "s1.1 (7B)",
    "Qwen_Qwen2.5-14B-Instruct → simplescaling_s1.1-14B": "s1.1 (14B)",
    "Qwen_Qwen2.5-32B-Instruct → simplescaling_s1.1-32B": "s1.1 (32B)",
    "Qwen_Qwen2.5-7B-Instruct → open-thoughts_OpenThinker-7B": "OpenThinker (7B)",
    "Qwen_Qwen2.5-32B-Instruct → open-thoughts/OpenThinker2-32B": "OpenThinker2 (32B)",
    "Qwen_Qwen2.5-7B-Instruct → open-thoughts/OpenThinker3-7B": "OpenThinker3 (7B)",
    "Qwen_Qwen2.5-7B-Instruct → nvidia_OpenCodeReasoning-Nemotron-1.1-7B": "Nemotron Code Reasoning (7B)",
    "Qwen_Qwen2.5-14B-Instruct → nvidia_OpenCodeReasoning-Nemotron-1.1-14B": "Nemotron Code Reasoning (14B)",
    "Qwen_Qwen2.5-32B-Instruct → GAIR/LIMO": "LIMO (32B)",
    "Qwen_Qwen2.5-32B-Instruct → GAIR_LIMO-v2": "LIMO v2 (32B)",
    "Qwen_Qwen2.5-32B → deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "R1 Distill Qwen (32B)",
    "Qwen_Qwen2.5-Math-7B → deepseek-ai_DeepSeek-R1-Distill-Qwen-7B": "R1 Distill Qwen (7B)",
}

CATEGORY_ABBREVIATIONS = {
    "commonsense": "Common Sense", "culture": "Culture", "deduction": "Logic",
    "knowledge_and_qa": "Knowledge/QA", "language_and_communication": "Language",
    "liberal_arts": "Liberal Arts", "math": "Math", "multi-linguality": "Multilingual",
    "safety_and_truthfulness": "Safety/Truth", "science,technology,engineering": "Science/Tech",
    "OVERALL": "\\textbf{Total}"
}

def format_model_name(name: str) -> str:
    """Looks up the full model name in the map, with a simple fallback."""
    if name in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[name]
    
    print(f"Warning: Model name '{name}' not found in MODEL_NAME_MAP. Using raw name.")
    return name.replace("_", "\\_")

def format_number(value) -> str:
    if isinstance(value, pd.Series):
        value = value.iloc[0] if len(value) == 1 else value.iloc[0]
    if isinstance(value, (tuple, list)):
        mean, std, third_val = (list(value) + [None, None, None])[:3]
        if pd.isna(mean): return "-"
        try: mean_pct = float(mean) * 100; mean_str = f"{mean_pct:.1f}"
        except (ValueError, TypeError): return "-"
        std_part = ""
        if not pd.isna(std):
            try: std_pct = float(std) * 100; std_part = f" {{\\scriptsize$\\pm${std_pct:.1f}}}"
            except (ValueError, TypeError): pass
        third_val_part = ""
        if not pd.isna(third_val):
            try: third_val_pct = float(third_val) * 100; third_val_part = f" ({max(third_val_pct, 0):.1f})"
            except (ValueError, TypeError): pass
        return f"{mean_str}{std_part}{third_val_part}"
    try:
        if pd.isna(value): return "-"
        percentage = float(value) * 100
        return "0.0" if abs(percentage) < 0.01 else f"{percentage:.1f}"
    except (ValueError, TypeError): return "-"

def load_metric_mean_and_std(csv_dir: str, group_name: str, metric: str):
    pattern = f"{csv_dir}/{group_name.replace(' ', '_')}_{metric.replace(' ', '_').replace('.', '')}*.csv"
    files = sorted(glob.glob(pattern))
    if not files: return None, None
    mean_file, std_file, var_file = None, None, None
    for f in files:
        fname_lower = os.path.basename(f).lower()
        if "_std" in fname_lower: std_file = f
        elif "_var" in fname_lower: var_file = f
        elif mean_file is None: mean_file = f
    if mean_file is None: return None, None
    mean_df = pd.read_csv(mean_file, index_col=0).apply(pd.to_numeric, errors="coerce")
    std_df = None
    if std_file: std_df = pd.read_csv(std_file, index_col=0).apply(pd.to_numeric, errors="coerce")
    elif var_file:
        var_df = pd.read_csv(var_file, index_col=0).apply(pd.to_numeric, errors="coerce")
        std_df = var_df.apply(np.sqrt)
    return mean_df, std_df

def try_load_first(csv_dir: str, group_name: str, candidates: list):
    for metric in candidates:
        mean_df, std_df = load_metric_mean_and_std(csv_dir, group_name, metric)
        if mean_df is not None: return mean_df, std_df
    return None, None

def create_combined_forgetting_improvement_table(csv_dir: str, group_name: str):
    forgetting_candidates = ["Adj Abs Forgetting", "Adj. Abs. Forgetting"]
    improvement_candidates = ["Adj Abs Improvement", "Adj. Abs. Improvement"]
    max_f_candidates = ["Max Possible Forgetting"]
    max_i_candidates = ["Max Possible Improvement"]
    mean_f, std_f = try_load_first(csv_dir, group_name, forgetting_candidates)
    mean_i, std_i = try_load_first(csv_dir, group_name, improvement_candidates)
    mean_max_f, _ = try_load_first(csv_dir, group_name, max_f_candidates)
    mean_max_i, _ = try_load_first(csv_dir, group_name, max_i_candidates)
    if mean_f is None and mean_i is None: return None
    base_df = mean_f if mean_f is not None else mean_i
    base_index, base_cols = base_df.index, base_df.columns
    cols_f = [f"{c} (F)" for c in base_cols]; cols_i = [f"{c} (I)" for c in base_cols]
    combined_f = pd.DataFrame(index=base_index, columns=cols_f, dtype=object)
    combined_i = pd.DataFrame(index=base_index, columns=cols_i, dtype=object)
    for idx in base_index:
        for col in base_cols:
            def safe_at(df, i, c):
                return df.at[i, c] if (df is not None and i in df.index and c in df.columns) else np.nan
            combined_f.at[idx, f"{col} (F)"] = (safe_at(mean_f, idx, col), safe_at(std_f, idx, col), safe_at(mean_max_f, idx, col))
            combined_i.at[idx, f"{col} (I)"] = (safe_at(mean_i, idx, col), safe_at(std_i, idx, col), safe_at(mean_max_i, idx, col))
    combined = pd.concat([combined_f, combined_i], axis=1)
    if "OVERALL" in combined.index:
        overall = combined.loc[["OVERALL"]]
        combined = pd.concat([combined.drop("OVERALL", errors='ignore'), overall])
    return combined

def escape_latex(text: str) -> str:
    return str(text).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("{", "\\{").replace("}", "\\}")

def generate_single_metric_table(df: pd.DataFrame, caption: str, label: str) -> str:
    df_filtered = df.copy()
    df_filtered = df_filtered.loc[~df_filtered.index.isin(["Missing", "ignore"])]
    if "OVERALL" in df.index:
        overall_row = df.loc[["OVERALL"]]
        df_filtered = pd.concat([df_filtered.drop("OVERALL", errors='ignore'), overall_row])
    num_models = len(df_filtered.columns)
    col_format = "l" + "r" * num_models
    lines = ["\\begin{table*}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    font_size = "\\small" if num_models <= 6 else "\\scriptsize"
    lines.append(font_size)
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")
    header_parts = ["Category"] + [format_model_name(col) for col in df_filtered.columns]
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")
    for idx in df_filtered.index:
        if idx == "OVERALL": lines.append("\\midrule")
        category_name = CATEGORY_ABBREVIATIONS.get(idx, escape_latex(idx))
        row_parts = [category_name]
        for model_name in df_filtered.columns:
            val = format_number(df_filtered.at[idx, model_name])
            row_parts.append(val)
        lines.append(" & ".join(row_parts) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)

def create_latex_document(tables_dict: dict, output_file: str):
    lines = ["% This file was auto-generated by the script.", "\\section{Quantifying Forgetting Accurately (Tables for referencing plots)}\n"]
    for group_name, group_tables in tables_dict.items():
        lines.append(f"\\subsection{{{escape_latex(group_name)}}}\n")
        for _, combined_df in group_tables.items():
            f_cols = [c for c in combined_df.columns if c.endswith(' (F)')]
            i_cols = [c for c in combined_df.columns if c.endswith(' (I)')]
            df_forgetting = combined_df[f_cols].copy()
            df_forgetting.columns = [c.replace(' (F)', '') for c in df_forgetting.columns]
            df_improvement = combined_df[i_cols].copy()
            df_improvement.columns = [c.replace(' (I)', '') for c in df_improvement.columns]

            # --- MODIFIED: Final, simplified sorting logic ---
            all_available_models = set(df_forgetting.columns) | set(df_improvement.columns)
            all_models_sorted = [model for model in MODEL_ORDER if model in all_available_models]
            remaining_models = sorted(list(all_available_models - set(all_models_sorted)))
            if remaining_models:
                print(f"Warning: The following models were found in the data for group '{group_name}' but not in MODEL_ORDER: {remaining_models}")
                all_models_sorted.extend(remaining_models)
            # --- End of sorting logic ---
            
            chunk_size = 4
            model_chunks = [all_models_sorted[i:i + chunk_size] for i in range(0, len(all_models_sorted), chunk_size)]
            num_chunks = len(model_chunks)

            if not df_forgetting.empty:
                for i, chunk in enumerate(model_chunks):
                    forgetting_models_in_chunk = [m for m in chunk if m in df_forgetting.columns]
                    if not forgetting_models_in_chunk: continue
                    df_chunk = df_forgetting[forgetting_models_in_chunk]
                    part_str = f" (Part {i + 1} of {num_chunks})" if num_chunks > 1 else ""
                    caption = f"{escape_latex(group_name)}: Forgetting{part_str}"
                    label = f"tab:{re.sub(r'[^a-zA-Z0-9]', '', group_name)}_forgetting_part{i+1}"
                    lines.append(generate_single_metric_table(df_chunk, caption, label))
                    lines.append("\n")

            if not df_improvement.empty:
                for i, chunk in enumerate(model_chunks):
                    improvement_models_in_chunk = [m for m in chunk if m in df_improvement.columns]
                    if not improvement_models_in_chunk: continue
                    df_chunk = df_improvement[improvement_models_in_chunk]
                    part_str = f" (Part {i + 1} of {num_chunks})" if num_chunks > 1 else ""
                    caption = f"{escape_latex(group_name)}: Backward Transfer{part_str}"
                    label = f"tab:{re.sub(r'[^a-zA-Z0-9]', '', group_name)}_btransfer_part{i+1}"
                    lines.append(generate_single_metric_table(df_chunk, caption, label))
                    lines.append("\n")
        lines.append("\\clearpage\n")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Created LaTeX include file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert pivot table CSVs to a LaTeX document.")
    parser.add_argument("--input", required=True, type=str, help="Directory with input pivot CSVs.")
    parser.add_argument("--output", required=True, type=str, help="Path for the output LaTeX file.")
    args = parser.parse_args()
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' not found."); sys.exit(1)
    all_csvs = glob.glob(os.path.join(args.input, "*_Adj_Abs_Forgetting.csv"))
    if not all_csvs:
        all_csvs = glob.glob(os.path.join(args.input, "*_Adj_Abs_Improvement.csv"))
        if not all_csvs:
            print(f"Error: No '*_Adj_Abs_Forgetting.csv' or '*_Adj_Abs_Improvement.csv' files found in '{args.input}'."); sys.exit(1)
        model_groups = sorted(list(set(f.replace("_Adj_Abs_Improvement.csv", "").replace('_', ' ') for f in [os.path.basename(csv) for csv in all_csvs])))
    else:
        model_groups = sorted(list(set(f.replace("_Adj_Abs_Forgetting.csv", "").replace('_', ' ') for f in [os.path.basename(csv) for csv in all_csvs])))
    print(f"Discovered {len(model_groups)} model groups to process: {model_groups}")
    
    tables_dict = {}
    all_discovered_model_names = set()
    for group_name in model_groups:
        combined_table = create_combined_forgetting_improvement_table(args.input, group_name)
        if combined_table is not None and not combined_table.empty:
            tables_dict.setdefault(group_name, {})["Forgetting and Backward Transfer"] = combined_table
            f_cols = [c.replace(' (F)', '') for c in combined_table.columns if c.endswith(' (F)')]
            i_cols = [c.replace(' (I)', '') for c in combined_table.columns if c.endswith(' (I)')]
            all_discovered_model_names.update(f_cols)
            all_discovered_model_names.update(i_cols)

    if not tables_dict:
        print("No data found to generate LaTeX tables. Exiting."); sys.exit(1)
    create_latex_document(tables_dict, args.output)

if __name__ == "__main__":
    main()
