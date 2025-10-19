# Mapping Post-Training Forgetting in Language Models at Scale

This repository contains code and resources for analyzing post-training forgetting phenomena in large language models.

## Resources

- **Paper**: [Read the full paper here](https://github.com/post-forget/post-forget/blob/main/paper/paper.pdf)
- **Website**: [https://post-forget.github.io/](https://post-forget.github.io/)
- **Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/post-forget/post-forget)

## Requirements

- **Python**: 3.11.2
- **CUDA**: 12.2

## Installation

### 1. Install Base Requirements

```bash
pip3 install -r requirements.txt
```

### 2. Install Mergekit (for Model Merging Experiments)

For running model merging experiments, you need to install mergekit:

```bash
pip install mergekit
```

For detailed installation instructions, see the [mergekit repository](https://github.com/arcee-ai/mergekit/tree/main).

## Usage

### Running Experiments

To evaluate models and run the main experiments:

```bash
./scripts/eval_models.sh
```

### Generating Figures

To generate plots and visualizations from the experimental results:

```bash
./scripts/generate_figures.sh
```