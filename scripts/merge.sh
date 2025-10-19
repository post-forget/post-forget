#!/bin/bash

mkdir -p merge_models

for config in config/merge_configs/*.yaml; do
    if [[ ! -f "$config" ]]; then
        echo "No config files found"
        exit 1
    fi
    
    config_name=$(basename "$config" .yaml)
    output_dir="merge_models/${config_name}"
    
    echo "Processing: $config_name"
    
    mkdir -p "$output_dir"
    
    if mergekit-yaml "$config" "$output_dir" --cuda --allow-crimes --random-seed=1; then
        echo "Success: $config_name"
    else
        echo "Failed: $config_name"
        rm -rf "$output_dir"
    fi
    
    echo ""
done
