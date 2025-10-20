import os
import argparse
import json
from glob import glob

import torch

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


def get_existing_results(results_directory, model):
    def convert_data(data):
        task_name = list(data["config_tasks"].keys())[0]
        return {
            "model": data["config_general"]["model_name"],
            "task": task_name,
            "seed": data["config_general"]["generation_parameters"]["seed"],
            "max_new_tokens": data["config_general"]["generation_parameters"]["max_new_tokens"],
            "temperature": data["config_general"]["generation_parameters"]["temperature"],
            "top_p": data["config_general"]["generation_parameters"]["top_p"],
        }
    results = glob(f"{results_directory}/results/{model}/*.json")
    dl = []
    for result in results:
        with open(result, "r") as f:
            data = json.load(f)
            task_name = convert_data(data)['task']
            dl.append(task_name)
    return dl


def get_tasks(tasks_directory):
    tasks = []
    for task_file in os.listdir(tasks_directory):
        if not task_file.endswith(".txt"):
            continue
        with open(os.path.join(tasks_directory, task_file), "r") as f:
            for line in f:
                task = line.split("#")[0].strip()
                if task:
                    tasks.append(task)
    return tasks


def run(results_directory,
        custom_tasks_directory,
        model,
        task,
        seed,
        dtype,
        use_chat_template,
        gpu_memory_utilization,
        max_new_tokens,
        max_model_length,
        max_num_batched_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        system_prompt,
        tensor_parallel_size,
        data_parallel_size,
        ):

    evaluation_tracker = EvaluationTracker(
        output_dir=results_directory,
        save_details=True,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        job_id=0,
        dataset_loading_processes=1,
        custom_tasks_directory=custom_tasks_directory,
        # override_batch_size=-1,  # Cannot override batch size when using VLLM
        max_samples=None,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        load_responses_from_details_date_id=None,
    )

    print(f"Evaluating {model} with seed {seed} on {torch.cuda.device_count()} GPUs")
    model_config = VLLMModelConfig(
        model_name=model,
        dtype=dtype,
        seed=seed,
        use_chat_template=use_chat_template,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_length=max_model_length,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        max_num_batched_tokens=max_model_length,
        generation_parameters=GenerationParameters(
            seed=seed,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        ),
    )

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_directory", type=str, required=True)
    parser.add_argument("--custom_tasks_directory", type=str, default="main_experiments/src/custom_tasks_extractor.py", help="Python file containing the lighteval tasks to evaluate")
    parser.add_argument("--tasks_directory", type=str, required=True, help="Directory containing txt files with the tasks to evaluate")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--data_parallel_size", type=int, default=2)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)

    args = parser.parse_args()

    print(args)

    existing_tasks = get_existing_results(args.results_directory, args.model)
    seeds = list(range(args.num_seeds))

    # Set environment variables
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['OMP_NUM_THREADS'] = '16'

    for task in get_tasks(args.tasks_directory):
        print(f"Generating {task} for {args.model}")

        for seed in seeds:
            # use_chat_template = "--use_chat_template" if args.use_chat_template else ""
            
            run(
                use_chat_template=args.use_chat_template,
                results_directory=args.results_directory,
                task=task,
                model=args.model,
                seed=seed,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_new_tokens=args.max_new_tokens,
                max_model_length=args.max_model_length,
                temperature=args.temperature,
                top_p=args.top_p,
                custom_tasks_directory=args.custom_tasks_directory,
                tensor_parallel_size=args.tensor_parallel_size,
                data_parallel_size=args.data_parallel_size,
                dtype='bfloat16',
                max_num_batched_tokens=None,
                top_k=None,
                repetition_penalty=None,
                system_prompt=None,
            )
            print(f"Running task '{task}' with seed {seed}")
