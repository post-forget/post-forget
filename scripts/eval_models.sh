#!/usr/bin/env bash

source /home/bethge/bkr007/.bashrc

CODE_DIR=/mnt/lustre/work/bethge/bkr007/ReasoningForgettingLM-feature-reasoning
VENV_DIR=${CODE_DIR}/.venv
RESULT_DIR=post-forget/results

LIGHTEVAL_TASKS_FEW_SHOT=post-forget/src/experiments/custom_tasks_extractor_few_shot.py
LIGHTEVAL_TASKS_PROMPT=post-forget/src/experiments/custom_tasks_extractor_prompt.py

TENSOR_PARALLEL=4
DATA_PARALLEL=1

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <qwen_math_base|qwen_math_instruct|base|tuned>"
    exit 1
fi

MODE=$1
MODELS=()

if [ "$MODE" = "qwen_math_base" ]; then
    MODELS=(
	Qwen/Qwen2.5-Math-7B
    )
    COMMON_ARGS=( --custom_tasks_directory "${LIGHTEVAL_TASKS_FEW_SHOT}" \
					   --num_seeds 3 \
					   --max_new_tokens 4096 \
					   --max_model_length 4096 \
					   --temperature 0.6 \
					   --top_p 0.95 \
					   --tensor_parallel_size "${TENSOR_PARALLEL}" \
					   --data_parallel_size "${DATA_PARALLEL}" \
					   --venv_dir "${VENV_DIR}" \
					   --code_dir "${CODE_DIR}" )
elif [ "$MODE" = "qwen_math_instruct" ]; then
    MODELS=(
	Qwen/Qwen2.5-Math-7B-Instruct
    )
    COMMON_ARGS=( --custom_tasks_directory "${LIGHTEVAL_TASKS_PROMPT}" \
					   --num_seeds 3 \
					   --max_new_tokens 4096 \
					   --max_model_length 4096 \
					   --temperature 0.6 \
					   --top_p 0.95 \
					   --use_chat_template \
					   --tensor_parallel_size "${TENSOR_PARALLEL}" \
					   --data_parallel_size "${DATA_PARALLEL}" \
					   --venv_dir "${VENV_DIR}" \
					   --code_dir "${CODE_DIR}" )
elif [ "$MODE" = "base" ]; then
    MODELS=(
	Qwen/Qwen2.5-3B
	Qwen/Qwen2.5-7B
	Qwen/Qwen2.5-14B
	Qwen/Qwen2.5-32B

	Qwen/Qwen2.5-Coder-3B
	Qwen/Qwen2.5-Coder-7B
	Qwen/Qwen2.5-Coder-14B
	Qwen/Qwen2.5-Coder-32B
	meta-llama/Llama-3.1-8B
    )
    COMMON_ARGS=( --custom_tasks_directory "${LIGHTEVAL_TASKS_FEW_SHOT}" \
					   --num_seeds 3 \
					   --max_new_tokens 32768 \
					   --max_model_length 32768 \
					   --temperature 0.6 \
					   --top_p 0.95 \
					   --tensor_parallel_size "${TENSOR_PARALLEL}" \
					   --data_parallel_size "${DATA_PARALLEL}" \
					   --venv_dir "${VENV_DIR}" \
					   --code_dir "${CODE_DIR}" )
elif [ "$MODE" = "tuned" ]; then
    MODELS=(
	meta-llama/Llama-3.1-8B-Instruct

	Qwen/Qwen2.5-3B-Instruct
 	Qwen/Qwen2.5-7B-Instruct
	Qwen/Qwen2.5-14B-Instruct
	Qwen/Qwen2.5-32B-Instruct

	Qwen/Qwen2.5-Coder-3B-Instruct
	Qwen/Qwen2.5-Coder-7B-Instruct
	Qwen/Qwen2.5-Coder-14B-Instruct
	Qwen/Qwen2.5-Coder-32B-Instruct

	deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
	deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
	deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
	deepseek-ai/DeepSeek-R1-Distill-Llama-8B
	nvidia/OpenCodeReasoning-Nemotron-1.1-7B
	nvidia/OpenCodeReasoning-Nemotron-1.1-14B
	nvidia/OpenMath2-Llama3.1-8B
	open-thoughts/OpenThinker-7B
	open-thoughts/OpenThinker2-32B
	open-thoughts/OpenThinker3-7B
	GAIR/LIMO
	GAIR/LIMO-v2
	simplescaling/s1.1-7B
	simplescaling/s1.1-14B
	simplescaling/s1.1-32B
	Qwen/QwQ-32B
	PrimeIntellect/INTELLECT-2
	Skywork/Skywork-OR1-7B
    )
    COMMON_ARGS=( --custom_tasks_directory "${LIGHTEVAL_TASKS_PROMPT}" \
					   --num_seeds 3 \
					   --max_new_tokens 32768 \
					   --max_model_length 32768 \
					   --temperature 0.6 \
					   --top_p 0.95 \
					   --use_chat_template \
					   --tensor_parallel_size "${TENSOR_PARALLEL}" \
					   --data_parallel_size "${DATA_PARALLEL}" \
					   --venv_dir "${VENV_DIR}" \
					   --code_dir "${CODE_DIR}" )
# elif [ "$MODE" = "merge_models" ]; then
#     for model_path in $(ls -d merge_models/*/); do
#         if [[ -d "$model_path" ]]; then
# 	    MODELS+=("$(realpath "$model_path")")
#         fi
#     done
else
    echo "Error: Please enter a valid mode"
    exit 1
fi

source "${VENV_DIR}/bin/activate"
cd "${CODE_DIR}"


TASKS_BASE_DIR="${CODE_DIR}/post-forget/config/tasks"
if [ ! -d "${TASKS_BASE_DIR}" ]; then
  echo "ERROR: tasks base directory not found: ${TASKS_BASE_DIR}" >&2
  exit 2
fi

for model in "${MODELS[@]}"; do
  # sanitize model string for filesystem (replace / with _)
  MODEL_SAFE=${model//\//_}
  for task_dir in "${TASKS_BASE_DIR}"/*; do
      python post-forget/src/experiments/eval_models.py \
             --results_directory  "${RESULT_DIR}" \
             --tasks_directory "${task_dir}" \
             --model "${model}" \
             "${COMMON_ARGS[@]}"
  done
done
