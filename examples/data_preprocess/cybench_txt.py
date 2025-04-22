"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

def extract_answer_format(content: str):
    """Extract the Answer Format string from a text file.
    
    Args:
        filename (str): Path to the text file
        
    Returns:
        str: The extracted answer format string
    """
    pattern = r"Answer Format:\s*(.*?)(?=\n|$|\<)"
    match = re.finditer(pattern, content)
    matches = list(match)
    
    if matches:
        # Return the last occurrence
        return matches[-1].group(1).strip()
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/nlp/scr/kathli/repos/TinyZero/data/cybench')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=32768)
    parser.add_argument('--test_size', type=int, default=128)
    parser.add_argument('--task_name', type=str, default='hackthebox_dynastic')
    
    args = parser.parse_args()

    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    trte_dataset = train_dataset.select(range(len(test_dataset)))

    prompt_file = f'/nlp/scr/kathli/repos/TinyZero/examples/data_preprocess/{args.task_name}.txt'
    with open(prompt_file, 'r') as f:
        prompt = f.read()

    answer_format = extract_answer_format(prompt)

    ans_file = f'/nlp/scr/kathli/repos/TinyZero/examples/data_preprocess/{args.task_name}_ans.txt'
    with open(ans_file, 'r') as f:
        ans = f.read().strip()

    print(prompt)
    print(answer_format)
    print(ans)

    def make_map_fn(split):
        def process_fn(example, idx):
            solution = {
                "answer": ans, 
                "format": answer_format
            }
            data = {
                "data_source": "cybench",
                "task_name": args.task_name,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
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