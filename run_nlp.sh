#!/bin/bash

#SBATCH --account=nlp                 # Specify the account
##SBATCH --partition=sphinx               # Specify the partition
#SBATCH --partition=jag-standard
##SBATCH --nodelist=sphinx5       # Request the specific node
#SBATCH --nodelist=jagupard34
#SBATCH --gres=gpu:4                        # Request GPUs
#SBATCH --cpus-per-task=8
##SBATCH --mem=360GB                         # Memory request
#SBATCH --mem=160GB
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=run_nlp.out
#SBATCH --error=run_nlp.err

source /nlp/scr/kathli/miniconda3/etc/profile.d/conda.sh
conda activate cyagent

cd /nlp/scr/kathli/repos/TinyZero

export N_GPUS=1
# export BASE_MODEL=model/Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
export BASE_MODEL=model/Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987
export DATA_DIR=data/cybench_512
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=$DATA_DIR-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./wandb_login.sh
bash ./scripts/train_tiny_zero_a100.sh