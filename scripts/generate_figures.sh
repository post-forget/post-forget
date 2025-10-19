#!/bin/bash

python3 ./src/analyze/process_data.py

CSV_OUTPUT_BASE_DIR="processed_csv"
LATEX_OUTPUT_BASE_DIR="latex_output"
PLOT_OUTPUT_BASE_DIR="plots"

mkdir -p "$CSV_OUTPUT_BASE_DIR"
mkdir -p "$LATEX_OUTPUT_BASE_DIR"
mkdir -p "$PLOT_OUTPUT_BASE_DIR"

input_file="comparison_results_all.csv"
base_name=$(basename "$input_file" .csv)
echo "--- Processing file: $base_name ---"

csv_pivot_dir="$CSV_OUTPUT_BASE_DIR/$base_name"
main_csv_output="$CSV_OUTPUT_BASE_DIR/${base_name}.csv"
latex_output_dir="$LATEX_OUTPUT_BASE_DIR/$base_name"
latex_file="$latex_output_dir/${base_name}.tex"
plot_output_dir="$PLOT_OUTPUT_BASE_DIR/$base_name"
    
mkdir -p "$csv_pivot_dir"
mkdir -p "$latex_output_dir"
mkdir -p "$plot_output_dir"

echo "Creating intermediate pivot tables in -> $csv_pivot_dir"
python3 src/analyze/create_csv.py \
    --input "$input_file" \
    --output-prefix "$main_csv_output" \
    --pivot-dir "$csv_pivot_dir" && \

echo "Creating final LaTeX document -> $latex_file"
python3 src/analyze/latex_converter.py \
    --input "$csv_pivot_dir" \
    --output "$latex_file" && \
        
echo "Creating radar plots in -> $plot_output_dir"
python3 src/analyze/radar.py \
    --input-dir "$csv_pivot_dir" \
    --output-dir "$plot_output_dir" \

if [ $? -eq 0 ]; then
    echo "Successfully processed '$base_name'."
    echo "--------------------------------------------------"
else
    echo "Error processing '$base_name'. Halting script."
    echo "--------------------------------------------------"
    exit 1
fi

echo "All files processed successfully."
