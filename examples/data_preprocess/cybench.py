"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
import json
import subprocess
from datetime import datetime
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple, Dict, Union
from tqdm import tqdm
import argparse
import sys
from pathlib import Path

# Now you can import from cybench
from agent.agent import SimpleAgent
from agent.prompt import END_TO_END_PROMPT as PROMPT, STOP_TOKEN
from run_task import TaskRunner, read_metadata, get_subtasks, DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS
from benchmark.dataclasses.task_run_input import TaskRunInput, TaskRunConfig, SubtaskConfig
from agent.dataclasses.agent_spec import AgentConfig, IterationConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/cybench')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=32768)
    parser.add_argument('--test_size', type=int, default=128)
    parser.add_argument('--from_docker', action='store_true', help='Set to true if running from docker')
    parser.add_argument('--task_dir', type=str, default='benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic')
    parser.add_argument('--task_objective', type=str, default='capture the flag', help='The objective of the task. Defaults to "capture the flag" if not specified.')

    args = parser.parse_args()

    data_source = args.task_dir
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    trte_dataset = train_dataset.select(range(len(test_dataset)))

    if args.from_docker:
        task_dir = os.path.join("/app", args.task_dir) #cybench
    else:
        task_dir = args.task_dir
    task_name = task_dir.split("/")[-1]
    
    # Read task metadata
    metadata = read_metadata(task_dir)

    # One subtask: set metadata to only have one subtask
    # If metadata has subtasks, use the first one; otherwise do nothing
    # This is specific to cyagent and not in cybench!
    if 'subtasks' in metadata and len(metadata['subtasks']) > 0:
        metadata['subtasks'] = [metadata['subtasks'][0]]
    
    # Create minimal TaskRunner instance just for prompt formatting
    # Note: The task run configuration below is only needed to instantiate TaskRunner
    # and is not used for the actual task execution
    task_run_config = TaskRunConfig(
        max_input_tokens_per_iteration=1000,  # Arbitrary value
        run_with_subtasks=True,
        use_easy_prompt=True,
        start_time_in_ms=int(round(datetime.now().timestamp() * 1000)),
        subtask_config=SubtaskConfig(
            tasks=get_subtasks(metadata),
            iteration_config=IterationConfig(
                max_iterations=10,
                iterations_until_hint=3
            )
        )
    )
    # We aren't actually using the agent, but we need to instantiate it to format the prompt correctly
    agent_config = AgentConfig(deployment_name="gpt-4") 
    task_runner = TaskRunner(
        config=TaskRunInput(
            task_run_config=task_run_config,  # Use the above task run config
            agent_config=agent_config,
            task_dir=task_dir,
        ),
        benchmark_id="", # Used to store a directory with the id, usually datetime. We ignore for now.
        metadata=metadata
    )

    # Setup environment and get formatted prompt
    # You need to set up environment to get the directory that the environment takes place in 
    # The files will be used for the task description
    task_runner.setup_environment()
    task_name = task_dir.split("/")[-1]
    target_host = metadata.get("target_host")
    task_description = task_runner._setup_prompt(target_host, args.task_objective)
    target_host_prompt = "\n"
    if target_host:
        target_host_prompt = f"""
Thought: ping the target host
Log: 2. python3 print_file.py - prints out a line in a file
Command: ping {target_host}
{STOP_TOKEN}
        """
        task_runner.host_health_check(target_host)

    prompt = PROMPT.format(
        task_name=task_name,
        work_dir=task_runner.agent_dir,
        task_description=task_description,
        answer_format=metadata["subtasks"][-1]["answer_format"],
        stop_token=STOP_TOKEN,
        target_host_prompt=target_host_prompt,
    )

    task_author = task_dir.split("benchmark/")[1].split("/")[0]
    competition_name = task_dir.split("benchmark/")[1].split("/")[1]
    task = Task(
        path=task_dir,
        name=task_name,
        categories=[
            CyberCategory.from_str(category)
            for category in metadata["categories"]
        ],
        competition=Competition(
            task_author=task_author,
            competition_name=competition_name,
        ),
        easy_prompt=metadata["easy_prompt"],
        challenging_prompt=metadata["hard_prompt"],
        difficulty=int(metadata["difficulty"]),
        subtasks=task_run_config.subtask_config.tasks,
    )

    agent = SimpleAgent(
        config=agent_config,
        subtask_config=task_run_config.subtask_config,
        work_dir=task_runner.agent_dir,
        mock_calls=False, # Set to True to mock API calls for testing purposes
        results_dir="", # Not used in this context
        interactive_mode=False,
        unguided_mode=False,
        task_name=task_name,
        prompt=prompt,
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS, 
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        responses_to_keep=3, # Default value from run_task.py, not used
        observations_to_keep=3, # Default value from run_task.py, not used
        helm=False,
        azure=False,
    )

    default_subtask_prompt = agent._get_subtask_input_text(agent.subtask_config.tasks[0], subtask_index=0, prime_with_prompt=True)
    print(f"Default subtask prompt:\n{default_subtask_prompt}\n")

    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": default_subtask_prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": data_source,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    trte_dataset = trte_dataset.map(function=make_map_fn('trte'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    trte_dataset.to_parquet(os.path.join(local_dir, 'trte.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 