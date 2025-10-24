# Mapping Post-Training Forgetting in Language Models at Scale

This repository contains code and resources for analyzing post-training forgetting phenomena in large language models.

## Resources

- **Paper**: [Read the full paper here](https://arxiv.org/abs/2510.17776)
- **Website**: [https://post-forget.github.io/](https://post-forget.github.io/)
- **Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/post-forget/post-forget)

## Requirements

- **Python**: 3.11.2
- **CUDA**: 12.2
- 4 Nvidia A100s or higher in the case of 32GB models

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

### 3. Base Model Setup

Locate `json_file_path` variable at the beginning of the `src/experiments/custom_tasks_extractor_few_shot.py` file under the TODO note.
Set it equal to the output of this shell command `realpath src/experiments/cot_template.json`. This manual step is needed as
this file is copied by LightEval to a temporary directory, thereby preventing the reliable use of relative paths.

## Usage

### Running Experiments

To evaluate models and run the main experiments:

```bash
./scripts/eval_models.sh <qwen_math_base|qwen_math_instruct|base|tuned|merge_base|merge_tuned>
```

Before running with `merge_base` or `merge_tuned`, be sure to generate the merged models by running:
```bash
./scripts/merge.sh
```

### Generating Figures

To generate plots and visualizations from the experimental results:

```bash
./scripts/generate_local_figures.sh
```

If you wish you to generate figures from the pre-computed sample-level results without needing to
re-run experiments, then instead run:

```bash
./scripts/generate_precomputed_figures.sh
```

## Citation

To cite our work:
```bibtex
@misc{harmon2025postforgetting,
      title={Mapping Post-Training Forgetting in Language Models at Scale}, 
      author={Jackson Harmon and Andreas Hochlehnert and Matthias Bethge and Ameya Prabhu},
      year={2025},
      eprint={2510.17776},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.17776}, 
}
```